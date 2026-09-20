// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: the SUBS ancestral sampler runs on a **decoder-style**
//! masked-diffusion model loaded as a bidirectional `GenericTransformer`.
//!
//! Stage 3 makes the sampler reusable across backends — `diffusion::generate`
//! and `generate_trajectory` take `&dyn MIBackend` + a `mask_token_id`, so they
//! are not tied to the `GenericMdlm` `DiT` backend.  This test demonstrates that
//! end-to-end on `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`
//! (`model_type` `"a2d-qwen2"`, `<|mask|>` = 151665), asserting the same
//! structural invariants as the MDLM sampler test: determinism by seed, monotone
//! unmasking (carry-over), termination (no `[MASK]` left), and prompt-prefix
//! preservation.  (Token *values* can't be matched against `PyTorch` — different
//! RNGs — so we assert falsifiable structural properties.)
//!
//! Requires **both** `transformer` (to load the model) and `diffusion` (the
//! sampler).  No single-feature CI lane builds this; a dedicated compile-check
//! covers it (see `ci.yml` / `scripts/preflight.ps1`).
//!
//! Run:
//!   `cargo test --test validate_bidirectional_sampler --features transformer,diffusion,mmap -- --ignored`

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

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_mi::{DiffusionSamplingConfig, GenericTransformer, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1";
/// `<|mask|>` in the a2d-qwen2 tokenizer (from `added_tokens.json`) — note it is
/// **not** `vocab_size - 1`, which is the MDLM convention.
const MASK_ID: u32 = 151_665;

fn hf_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory");
}

fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Demonstrates that the masked-diffusion `SUBS` sampler works on a bidirectional
/// `GenericTransformer` (not just `GenericMdlm`), with the model's real
/// `<|mask|>` id.  Asserts determinism, monotone unmasking, termination, and
/// prompt-prefix preservation.
#[test]
#[ignore = "requires dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm cached (~1.2 GiB); run with --ignored"]
#[serial]
fn a2d_qwen2_sampler_invariants() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };
    let device = Device::Cpu;
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();
    assert!(config.bidirectional, "a2d-qwen2 must load as bidirectional");

    let weights = snapshot.join("model.safetensors");
    // SAFETY: safetensors files are not modified during test execution.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap()
    };
    let model = GenericTransformer::load(config, &device, DType::F32, vb).unwrap();

    // Carry over a 3-token prompt prefix; denoise the rest.
    let prompt: Vec<u32> = vec![785, 6722, 315]; // arbitrary valid Qwen2 ids
    let cfg = DiffusionSamplingConfig::default()
        .with_seq_len(16)
        .with_num_steps(16)
        .with_top_k(Some(50));

    // `&model` coerces to `&dyn MIBackend` — the sampler is backend-agnostic.
    let traj1 =
        candle_mi::diffusion::generate_trajectory(&model, &device, MASK_ID, &prompt, &cfg).unwrap();
    let traj2 =
        candle_mi::diffusion::generate_trajectory(&model, &device, MASK_ID, &prompt, &cfg).unwrap();

    // Determinism: same seed -> identical trajectory.
    assert_eq!(traj1, traj2, "same seed must reproduce the trajectory");

    // num_steps + 1 states; each full-length and keeping the prompt prefix.
    assert_eq!(traj1.len(), cfg.num_steps + 1, "wrong trajectory length");
    for state in &traj1 {
        assert_eq!(state.len(), cfg.seq_len, "wrong state length");
        for (i, &p) in prompt.iter().enumerate() {
            assert_eq!(state[i], p, "prompt prefix changed at position {i}");
        }
    }

    // Monotone unmasking (carry-over); final state fully revealed.
    let revealed = |s: &[u32]| s.iter().filter(|&&t| t != MASK_ID).count();
    let mut prev = revealed(&traj1[0]);
    for state in &traj1[1..] {
        let now = revealed(state);
        assert!(now >= prev, "unmasking not monotone: {prev} -> {now}");
        prev = now;
    }
    assert_eq!(
        revealed(traj1.last().unwrap()),
        cfg.seq_len,
        "final state still has [MASK] tokens"
    );

    println!(
        "a2d-qwen2 sampler reuse OK; final state: {:?}",
        traj1.last().unwrap()
    );
}
