// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `Qwen/Qwen3-1.7B-Base` forward-pass parity against the
//! from-first-principles Python oracle in `scripts/qwen3_forward_validation.py`.
//!
//! Consumes the frozen reference JSON (`scripts/qwen3_forward_reference.json`)
//! and verifies that candle-mi's [`GenericTransformer`] produces matching
//! output when fed the same input prompts.  Acceptance bar:
//!
//! - Detected config has `use_qk_norm == true`.
//! - `(hidden_size, num_layers, vocab_size, head_dim)` match the Python run.
//! - Per test case: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU
//!   `F32` — looser to absorb CUDA-vs-CPU rounding noise documented for
//!   RWKV-7).
//!
//! Two test wrappers (one CPU, one GPU), both `#[ignore]`-gated and serial.
//! GPU test skips cleanly when no `CUDA` device is available.
//!
//! Requires `Qwen/Qwen3-1.7B-Base` (~3.2 GiB) cached in
//! `~/.cache/huggingface/hub/`.
//!
//! Run CPU:
//!   `cargo test --test validate_qwen3_forward --features transformer -- --ignored qwen3_1_7b_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_qwen3_forward --features transformer -- --ignored qwen3_1_7b_forward_parity_gpu`

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
    unsafe_code,
    missing_docs
)]

mod common;

use common::{
    cuda_device, find_snapshot, json_f32, json_u32, json_usize, reference_path, safetensors_paths,
};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "Qwen/Qwen3-1.7B-Base";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Run the Qwen3 forward-parity check on the given `device`.  Panics on any
/// mismatch.  `abs_diff_bar` is the per-rank acceptance threshold for the
/// top-10 logit magnitudes (CPU is tighter than GPU).
#[allow(clippy::too_many_lines)] // Flat sequence — load → assert → iterate cases.
fn run_qwen3_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    // --- Load the frozen reference JSON ---
    let reference_str = std::fs::read_to_string(reference_path("qwen3_forward_reference.json")).expect(
        "failed to read qwen3_forward_reference.json — run scripts/qwen3_forward_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    let ref_hidden = json_usize(&reference["hidden_size"]);
    let ref_layers = json_usize(&reference["num_layers"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let ref_head_dim = json_usize(&reference["head_dim"]);
    let ref_use_qk_norm = reference["use_qk_norm"].as_bool().unwrap();
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");
    assert!(
        ref_use_qk_norm,
        "Qwen3 reference must report use_qk_norm == true"
    );

    println!("Validating Qwen3 forward parity ({device_name}) against Python oracle:");
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}",
        test_cases.len()
    );

    // --- Load model from HF cache ---
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    // Sanity: candle-mi must detect QK norm.
    assert!(
        config.use_qk_norm,
        "candle-mi config parsing must detect use_qk_norm=true for Qwen3"
    );
    assert_eq!(config.hidden_size, ref_hidden);
    assert_eq!(config.num_layers, ref_layers);
    assert_eq!(config.vocab_size, ref_vocab);
    assert_eq!(config.head_dim, ref_head_dim);

    let dtype = DType::F32;
    let paths = safetensors_paths(&snapshot);

    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config, device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_layers);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);

    // --- Run each test case ---
    let mut max_abs_diff: f32 = 0.0;
    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();

        println!("\nPrompt: {prompt:?}");
        println!("  expected tokens: {ref_tokens:?}");

        // Use the Python-tokenized IDs directly so any tokenizer-version drift
        // doesn't taint the forward-pass comparison.
        let input = Tensor::new(&ref_tokens[..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let hooks = HookSpec::new();
        let result = model.forward(&input, &hooks).unwrap();
        let logits = result.output();

        let (batch, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(out_seq, ref_tokens.len());
        assert_eq!(vocab, ref_vocab);

        // Last-token logits, F32 on CPU.
        let last_logits: Vec<f32> = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .i((0, out_seq - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        // Compute Rust's top-10.
        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Compare top-10 indices + magnitudes.
        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);

            let (rust_idx, rust_logit) = indexed[rank];

            assert_eq!(
                rust_idx, ref_idx,
                "rank {rank}: top-10 index mismatch (Rust {rust_idx}, Python {ref_idx}) \
                 for prompt {prompt:?}"
            );

            let diff = (rust_logit - ref_logit).abs();
            assert!(
                diff < abs_diff_bar,
                "rank {rank}: logit abs-diff {diff:.2e} >= {abs_diff_bar:.0e} \
                 (Rust {rust_logit}, Python {ref_logit}) for prompt {prompt:?}"
            );
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }

        println!(
            "  Rust top-1: ({}, {:.4}) — matches Python",
            indexed[0].0, indexed[0].1
        );
    }

    println!(
        "\nAll {} test cases passed on {device_name}; max abs-diff across all top-10 logits = {:.2e} (bar: {:.0e})",
        test_cases.len(),
        max_abs_diff,
        abs_diff_bar
    );
}

#[test]
#[ignore = "requires Qwen/Qwen3-1.7B-Base cached (~3.2 GiB); run with --ignored"]
#[serial]
fn qwen3_1_7b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_qwen3_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires Qwen/Qwen3-1.7B-Base cached (~3.2 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn qwen3_1_7b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_qwen3_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
