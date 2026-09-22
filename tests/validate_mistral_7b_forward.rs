// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `mistralai/Mistral-7B-v0.1` forward-pass parity against
//! the from-first-principles Python oracle in `scripts/mistral_7b_validation.py`.
//!
//! Mistral is `LLaMA`-like plus **sliding-window attention** (window 4096 on
//! every layer); no soft-capping.  Its only prior guard was a "`Paris` in
//! top-5" smoke test in `validate_models.rs`; this is the exact-logit guard.
//!
//! 7B in F32 is ~27 GiB — larger than a 16 GiB GPU — which makes Mistral the
//! one family worth validating on **three** tiers, all against the same F32
//! oracle.  All three were exercised and pass; keeping all three documents that
//! both GPU strategies (fast/fully-resident and exact/oversubscribed) work:
//!
//! - **CPU (F32)** — exact, `abs diff < 1e-3` (same tier as the other
//!   families). Needs ~27 GiB RAM; slow.
//! - **GPU (F32)** — exact, `abs diff < 5e-3` (CUDA-vs-CPU F32 rounding).
//!   F32 7B exceeds 16 GiB VRAM, so this relies on CUDA memory oversubscription
//!   (weights spill to host RAM over PCIe): correct but slower (~18s observed).
//!   Matches to ~1e-5.
//! - **GPU (BF16)** — fast and fully GPU-resident (~13.5 GiB, ~7s observed).
//!   BF16 carries ~3 significant figures, so the bar is looser (`< 0.1`; the
//!   observed diff is ~5e-2, BF16's granularity at these magnitudes) and only
//!   the top-1 index is required to match (bf16 can reorder near-tied ranks).
//!
//! All wrappers are `#[ignore]`-gated and serial; each skips cleanly when its
//! prerequisite (the cached model / a CUDA device) is missing.
//!
//! Requires `mistralai/Mistral-7B-v0.1` (gated; ~13.5 GiB bf16) cached.
//!
//! Run CPU (F32):
//!   `cargo test --test validate_mistral_7b_forward --features transformer -- --ignored mistral_7b_forward_parity_cpu`
//!
//! Run GPU (F32, exact, oversubscribed):
//!   `cargo test --test validate_mistral_7b_forward --features transformer -- --ignored mistral_7b_forward_parity_gpu`
//!
//! Run GPU (BF16, fast, resident):
//!   `cargo test --test validate_mistral_7b_forward --features transformer -- --ignored mistral_7b_forward_parity_gpu_bf16`

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

const MODEL_ID: &str = "mistralai/Mistral-7B-v0.1";
const ABS_DIFF_BAR_CPU_F32: f32 = 1e-3;
const ABS_DIFF_BAR_GPU_F32: f32 = 5e-3;
/// BF16 carries ~3 significant figures; candle-bf16 vs the F32 truth diverges
/// by ~5e-2 (BF16's representational granularity at these logit magnitudes).
/// This 0.1 bar gives ~2x headroom for that while still catching gross errors.
const ABS_DIFF_BAR_GPU_BF16: f32 = 1e-1;

/// Run the Mistral forward-parity check on the given `device`/`dtype`.
///
/// `strict_indices`: when true (F32), every top-10 index must match; when false
/// (BF16), only the top-1 index is required (bf16 reorders near-tied ranks).
/// Magnitudes are always compared per *token* (the Rust logit for the
/// reference token), so a reordering never inflates the magnitude diff.
#[allow(clippy::too_many_lines)]
fn run_mistral_forward_parity(
    device: &Device,
    device_name: &str,
    dtype: DType,
    abs_diff_bar: f32,
    strict_indices: bool,
) {
    let reference_str = std::fs::read_to_string(reference_path("mistral_7b_forward_reference.json")).expect(
        "failed to read mistral_7b_forward_reference.json — run scripts/mistral_7b_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    let ref_hidden = json_usize(&reference["hidden_size"]);
    let ref_layers = json_usize(&reference["num_layers"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let ref_head_dim = json_usize(&reference["head_dim"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");

    println!(
        "Validating Mistral 7B forward parity ({device_name}, {dtype:?}) against F32 Python oracle:"
    );
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}, strict_indices={strict_indices}",
        test_cases.len()
    );

    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, ref_hidden);
    assert_eq!(config.num_layers, ref_layers);
    assert_eq!(config.vocab_size, ref_vocab);
    assert_eq!(config.head_dim, ref_head_dim);

    // Mistral's distinguishing trait: a fixed sliding window on every layer.
    assert_eq!(
        config.sliding_window,
        Some(4096),
        "Mistral must detect the 4096 sliding window"
    );

    let paths = safetensors_paths(&snapshot);
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config, device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_layers);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);

    let mut max_abs_diff: f32 = 0.0;
    let mut failures: Vec<String> = Vec::new();

    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();

        // Use the Python-tokenized IDs directly so tokenizer drift can't
        // taint the forward-pass comparison.
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

        let last_logits: Vec<f32> = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .i((0, out_seq - 1))
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut indexed: Vec<(usize, f32)> = last_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ref_top1_idx = json_usize(&ref_top10[0]["index"]);
        let ref_top1_logit = json_f32(&ref_top10[0]["logit"]);
        // INDEX: `indexed` has `vocab` entries (> 10), so index 0 is valid.
        println!("\nPrompt: {prompt:?}  ({} tokens)", ref_tokens.len());
        println!(
            "  Python top-1: ({ref_top1_idx}, {ref_top1_logit:.4})   \
             Rust top-1: ({}, {:.4})",
            indexed[0].0, indexed[0].1
        );

        // Top-1 index must always match, on every tier.
        if indexed[0].0 != ref_top1_idx {
            failures.push(format!(
                "{prompt:?}: top-1 index mismatch (Rust {}, Python {ref_top1_idx})",
                indexed[0].0
            ));
        }

        let mut prompt_max_diff: f32 = 0.0;
        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);
            // INDEX: rank < ref_top10.len() (== 10) and `indexed` has `vocab`
            // entries, so both lookups are in bounds.
            let rust_idx = indexed[rank].0;

            // Strict (F32): every top-10 index must match.  Loose (BF16): skip
            // lower-rank index checks (bf16 can reorder near-tied logits).
            if strict_indices && rust_idx != ref_idx {
                failures.push(format!(
                    "{prompt:?} rank {rank}: index mismatch (Rust {rust_idx}, Python {ref_idx})"
                ));
            }

            // Magnitude: compare the reference logit to the Rust logit for the
            // *same token*, so a reordering doesn't inflate the diff.
            // INDEX: ref_idx is an oracle vocab index; vocab was asserted equal
            // to the model's, so it is in bounds for `last_logits`.
            let rust_logit_for_ref = last_logits[ref_idx];
            let diff = (rust_logit_for_ref - ref_logit).abs();
            if diff >= abs_diff_bar {
                failures.push(format!(
                    "{prompt:?} rank {rank} (token {ref_idx}): logit abs-diff {diff:.3e} >= {abs_diff_bar:.0e} \
                     (Rust {rust_logit_for_ref:.4}, Python {ref_logit:.4})"
                ));
            }
            prompt_max_diff = prompt_max_diff.max(diff);
            max_abs_diff = max_abs_diff.max(diff);
        }
        println!("  max abs-diff over top-10 tokens: {prompt_max_diff:.3e}");
    }

    println!(
        "\n{} test cases on {device_name} ({dtype:?}); max abs-diff across all top-10 tokens = {:.3e} (bar: {:.0e})",
        test_cases.len(),
        max_abs_diff,
        abs_diff_bar
    );

    assert!(
        failures.is_empty(),
        "Mistral 7B forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires mistralai/Mistral-7B-v0.1 cached; F32 on CPU needs ~27 GiB RAM and is slow; run with --ignored"]
#[serial]
fn mistral_7b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_mistral_forward_parity(&Device::Cpu, "CPU", DType::F32, ABS_DIFF_BAR_CPU_F32, true);
}

#[test]
#[ignore = "requires mistralai/Mistral-7B-v0.1 cached and a CUDA device; F32 7B (~27 GiB) exceeds 16 GiB VRAM and relies on CUDA memory oversubscription; run with --ignored"]
#[serial]
fn mistral_7b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_mistral_forward_parity(&device, "CUDA", DType::F32, ABS_DIFF_BAR_GPU_F32, true);
}

#[test]
#[ignore = "requires mistralai/Mistral-7B-v0.1 cached (~13.5 GiB bf16) and a CUDA device; run with --ignored"]
#[serial]
fn mistral_7b_forward_parity_gpu_bf16() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_mistral_forward_parity(&device, "CUDA", DType::BF16, ABS_DIFF_BAR_GPU_BF16, false);
}
