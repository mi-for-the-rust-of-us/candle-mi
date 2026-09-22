// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: **longrope** `rope_scaling` (Phi-3.5-mini) exact forward
//! parity against a `PyTorch` oracle, exercising both the short- and long-factor
//! regimes.
//!
//! longrope (Phi-3.5-mini, Phi-3-medium-128k) divides the `RoPE` inverse
//! frequencies by per-dimension `short_factor`/`long_factor` arrays and scales
//! cos/sin by `attention_factor` (mscale).  candle-mi builds two caches; a
//! forward longer than `original_max_position_embeddings` (4096) selects the
//! long-factor cache, matching `HuggingFace`'s seq-length branch.
//!
//! Validated against the **non-quantized** `microsoft/Phi-3.5-mini-instruct`,
//! whose config leaves `attention_factor` unset → HF derives the paper value
//! (~1.19), so the model behaves sanely and gives clean exact parity (the cached
//! unsloth bnb-4bit re-save hard-codes `attention_factor: 32`, whose 1024×
//! attention scaling makes the model chaotically sensitive to the bnb weight
//! rounding — unusable for exact parity; the longrope math itself is the same and
//! is unit-tested in `src/transformer/rope.rs` against that model's ground truth).
//!
//! Acceptance bar:
//! - Detected config is `RopeScaling::Longrope`; its `attention_factor` matches
//!   the oracle's `attention_scaling` ground truth.
//! - Per case: top-10 indices match exactly; magnitudes within `< 5e-3`
//!   (GPU F32 vs the oracle's F32).
//! - A short-regime *and* a long-regime (>4096-token) case both pass.
//!
//! `#[ignore]`-gated; requires the model cached.  F32 (~15 GiB) + the long-case
//! activations exceed 16 GiB VRAM and run via CUDA memory oversubscription.
//!
//! Run:
//!   `cargo test --test validate_longrope --features transformer,mmap -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    missing_docs
)]

mod common;

use common::{hf_cache_dir, json_f32, json_u32, json_usize, reference_path};

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{HookSpec, MIModel, RopeScaling, TransformerConfig};

const MODEL_ID: &str = "microsoft/Phi-3.5-mini-instruct";
const ABS_DIFF_BAR: f32 = 5e-3;

fn snapshot_dir() -> Option<PathBuf> {
    let dir = hf_cache_dir()
        .join(format!("models--{}", MODEL_ID.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let p = e.path();
        (p.join("config.json").exists()
            && (p.join("model.safetensors").exists()
                || p.join("model.safetensors.index.json").exists()))
        .then_some(p)
    })
}

#[test]
#[ignore = "requires microsoft/Phi-3.5-mini-instruct cached and a CUDA device; run with --ignored,--features mmap"]
// EXPLICIT: a single parity oracle read top to bottom; its setup, forward and
// per-tensor comparisons belong in one place so a failure names its own stage.
#[allow(clippy::too_many_lines)]
fn phi35_longrope_forward_parity() {
    let Some(snapshot) = snapshot_dir() else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };

    // --- Oracle ---
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(reference_path("phi35_longrope_forward_reference.json")).unwrap(),
    )
    .expect("run scripts/phi35_longrope_validation.py first");
    assert_eq!(reference["model_repo"].as_str().unwrap(), MODEL_ID);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let ref_mscale = json_f32(&reference["attention_scaling"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    // --- Structural: longrope parses; its mscale matches the oracle's ---
    let cfg_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let config = TransformerConfig::from_hf_config(&cfg_json).unwrap();
    match &config.rope_scaling {
        Some(RopeScaling::Longrope {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            attention_factor,
        }) => {
            assert_eq!(short_factor.len(), config.head_dim / 2);
            assert_eq!(long_factor.len(), config.head_dim / 2);
            assert_eq!(*original_max_position_embeddings, 4096);
            assert!(
                // CAST: f64 → f32, the oracle stores the scale as a Python double and the
                // comparison runs at the test's F32 bar
                (*attention_factor as f32 - ref_mscale).abs() < 1e-4,
                "candle mscale {attention_factor} != oracle attention_scaling {ref_mscale}"
            );
        }
        other => panic!("expected Longrope, got {other:?}"),
    }

    let model = MIModel::from_pretrained(MODEL_ID)
        .unwrap_or_else(|e| panic!("load {MODEL_ID} (longrope): {e}"));
    println!(
        "Validating Phi-3.5 longrope forward parity (mscale={ref_mscale:.4}, short + long regimes):"
    );
    assert_eq!(model.vocab_size(), ref_vocab);

    let device = model.device().clone();
    let mut max_abs_diff: f32 = 0.0;
    let mut failures: Vec<String> = Vec::new();
    let mut saw_long = false;

    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let regime = tc["regime"].as_str().unwrap_or("short");
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();
        if regime == "long" {
            saw_long = true;
            assert!(
                ref_tokens.len() > 4096,
                "long case must exceed original_max"
            );
        }

        let input = Tensor::new(&ref_tokens[..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hooks = HookSpec::new();
        let cache = model.forward(&input, &hooks).unwrap();
        let logits = cache.output();
        let (_, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(vocab, ref_vocab);

        let last: Vec<f32> = logits
            .i((0, out_seq - 1))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        let mut indexed: Vec<(usize, f32)> =
            last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ref_top1_idx = json_usize(&ref_top10[0]["index"]);
        let ref_top1_logit = json_f32(&ref_top10[0]["logit"]);
        println!(
            "  {prompt:?} [{regime}, {} tok]: oracle top-1 ({ref_top1_idx}, {ref_top1_logit:.4})  candle ({}, {:.4})",
            ref_tokens.len(),
            indexed[0].0,
            indexed[0].1
        );

        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);
            let (rust_idx, _) = indexed[rank];
            if rust_idx != ref_idx {
                failures.push(format!(
                    "{prompt:?} [{regime}] rank {rank}: index mismatch (candle {rust_idx}, oracle {ref_idx})"
                ));
            }
            let diff = (last[ref_idx] - ref_logit).abs();
            if diff >= ABS_DIFF_BAR {
                failures.push(format!(
                    "{prompt:?} [{regime}] token {ref_idx}: abs-diff {diff:.3e} >= {ABS_DIFF_BAR:.0e}"
                ));
            }
            max_abs_diff = max_abs_diff.max(diff);
        }
    }

    println!(
        "max abs-diff across all top-10 tokens = {max_abs_diff:.3e} (bar: {ABS_DIFF_BAR:.0e})"
    );
    assert!(saw_long, "oracle must include a long-regime (>4096) case");
    assert!(
        failures.is_empty(),
        "Phi-3.5 longrope parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
