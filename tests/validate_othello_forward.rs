// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `OthelloMDLM` forward + capture parity against the fp32
//! `PyTorch` oracle.
//!
//! Consumes the two safetensors fixtures produced by the askesis
//! `reference/othello_mdlm/export_fixtures.py`:
//!
//! - `weights.safetensors` — the `OthelloMDLM` `state_dict` under its original
//!   `PyTorch` key names (fp32), loaded directly by [`OthelloGpt`].
//! - `forward_capture.safetensors` — `input_ids [N, T]`, `logits [N, T, vocab]`,
//!   and `resid_post.{i} [N, T, d]` (the per-layer residual stream the board
//!   probes consume) for `N` fixed full games at `t = 0` (no masking).
//!
//! Acceptance bars (both are the dogfood's forward/capture parity gates):
//!
//! - **Forward parity**: max-abs `logits` diff `< 1e-3` (CPU `F32` vs CPU `F32`)
//!   or `< 5e-3` (GPU `F32` vs CPU `F32`).
//! - **Capture parity**: each `resid_post.{i}` matches the captured
//!   [`HookPoint::ResidPost`] to the same bar.
//!
//! The fixtures are git-ignored in askesis (regenerable), so point the test at
//! their directory with the `OTHELLO_MDLM_FIXTURES` environment variable; the
//! test skips cleanly when it is unset or the files are absent.
//!
//! Run CPU:
//!   `$env:OTHELLO_MDLM_FIXTURES="<dir>"; cargo test --test validate_othello_forward --features diffusion -- --ignored othello_forward_parity_cpu`
//!
//! Run GPU:
//!   `$env:OTHELLO_MDLM_FIXTURES="<dir>"; cargo test --test validate_othello_forward --features diffusion -- --ignored othello_forward_parity_gpu`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    unsafe_code,
    missing_docs
)]

mod common;

use common::cuda_device;

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_mi::{HookPoint, HookSpec, MIBackend, OthelloGpt, OthelloGptConfig};
use serial_test::serial;

/// The released world model: `n_head = 8` (the one dimension the capture
/// fixture does not encode; everything else is derived from the tensors).
const N_HEAD: usize = 8;
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Directory holding `weights.safetensors` + `forward_capture.safetensors`,
/// or `None` when the fixtures are unavailable.
fn fixtures_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("OTHELLO_MDLM_FIXTURES").ok()?);
    let weights = dir.join("weights.safetensors");
    let capture = dir.join("forward_capture.safetensors");
    (weights.is_file() && capture.is_file()).then_some(dir)
}

/// Max absolute element-wise difference between two tensors (computed in CPU
/// `F32`).
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b = b
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let diff = (a - b).unwrap().abs().unwrap();
    diff.flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Run the `OthelloMDLM` forward + capture parity check on `device`.
fn run_othello_parity(dir: &std::path::Path, device: &Device, device_name: &str, bar: f32) {
    // --- Load the capture fixture and derive the architecture from it. ---
    let capture = candle_core::safetensors::load(dir.join("forward_capture.safetensors"), device)
        .expect("load forward_capture.safetensors");
    let input_ids_i64 = capture.get("input_ids").expect("input_ids in fixture");
    let ref_logits = capture.get("logits").expect("logits in fixture");

    let (n_games, seq_len) = input_ids_i64.dims2().unwrap();
    let (_, _, vocab) = ref_logits.dims3().unwrap();

    // Per-layer residual targets: resid_post.0, resid_post.1, ...
    let mut resid_refs: Vec<Tensor> = Vec::new();
    while let Some(t) = capture.get(&format!("resid_post.{}", resid_refs.len())) {
        resid_refs.push(t.clone());
    }
    let n_layer = resid_refs.len();
    let n_embd = resid_refs[0].dims3().unwrap().2;

    assert!(n_layer > 0, "fixture has no resid_post.* tensors");
    assert!(
        n_embd.is_multiple_of(N_HEAD),
        "n_embd {n_embd} not divisible by {N_HEAD}"
    );

    println!("Validating OthelloMDLM forward + capture parity ({device_name}):");
    println!(
        "  n_games={n_games}, seq_len={seq_len}, vocab={vocab}, n_layer={n_layer}, n_embd={n_embd}"
    );
    println!("  abs-diff bar = {bar:.0e}");

    // --- Load the model from the verbatim-keyed weights. ---
    let config = OthelloGptConfig::new(vocab, seq_len, n_layer, N_HEAD, n_embd, false).unwrap();
    let weights = dir.join("weights.safetensors");
    // SAFETY: safetensors files are not modified during test execution.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device).unwrap()
    };
    let model = OthelloGpt::load(config, vb).unwrap();

    assert_eq!(model.num_layers(), n_layer);
    assert_eq!(model.hidden_size(), n_embd);
    assert_eq!(model.vocab_size(), vocab);
    assert_eq!(model.num_heads(), N_HEAD);

    // candle embeddings index with U32; the fixture stores token ids as i64.
    let input_ids = input_ids_i64.to_dtype(DType::U32).unwrap();

    let mut hooks = HookSpec::new();
    for i in 0..n_layer {
        hooks.capture(HookPoint::ResidPost(i));
    }
    let cache = model.forward(&input_ids, &hooks).unwrap();

    // --- Forward parity. ---
    let logits_diff = max_abs_diff(cache.output(), ref_logits);
    println!("  logits max-abs-diff = {logits_diff:.2e}");
    assert!(
        logits_diff < bar,
        "forward parity: logits max-abs-diff {logits_diff:.2e} >= {bar:.0e}"
    );

    // --- Capture parity (per layer). ---
    let mut worst = logits_diff;
    for (i, reference) in resid_refs.iter().enumerate() {
        let captured = cache.require(&HookPoint::ResidPost(i)).unwrap();
        let diff = max_abs_diff(captured, reference);
        assert!(
            diff < bar,
            "capture parity: resid_post.{i} max-abs-diff {diff:.2e} >= {bar:.0e}"
        );
        worst = worst.max(diff);
    }

    println!(
        "All parity checks passed on {device_name}; worst max-abs-diff = {worst:.2e} (bar {bar:.0e})"
    );
}

#[test]
#[ignore = "requires OTHELLO_MDLM_FIXTURES pointing at the askesis fixtures; run with --ignored"]
#[serial]
fn othello_forward_parity_cpu() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("SKIP: OTHELLO_MDLM_FIXTURES unset or fixtures missing");
        return;
    };
    run_othello_parity(&dir, &Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires OTHELLO_MDLM_FIXTURES and a CUDA device; run with --ignored"]
#[serial]
fn othello_forward_parity_gpu() {
    let Some(dir) = fixtures_dir() else {
        eprintln!("SKIP: OTHELLO_MDLM_FIXTURES unset or fixtures missing");
        return;
    };
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_othello_parity(&dir, &device, "CUDA", ABS_DIFF_BAR_GPU);
}
