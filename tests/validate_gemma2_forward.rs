// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `google/gemma-2-2b` forward-pass parity against the
//! from-first-principles Python oracle in `scripts/gemma2_validation.py`.
//!
//! Gemma 2 is the highest-quirk decoder-only family candle-mi covers:
//! attention + final logit soft-capping (`tanh(x/cap)*cap`), four norms per
//! layer, `sqrt(hidden_size)` embedding scaling, GELU-tanh approximation, a
//! custom `query_pre_attn_scalar`, and a required BOS token.  Its only prior
//! guard was a "`Paris` in top-10" smoke test in `validate_models.rs`, which a
//! subtly-wrong soft-capping/4-norm path still passes.  This is the exact-logit
//! regression guard the smoke test could not be.
//!
//! Consumes the frozen reference JSON (`scripts/gemma2_forward_reference.json`)
//! and verifies that candle-mi's [`GenericTransformer`] produces matching
//! output when fed the same input prompts.  Acceptance bar:
//!
//! - Detected config carries non-`None` `attn_logit_softcapping` /
//!   `final_logit_softcapping`, `use_post_norms`, and `embedding_scale`.
//! - `(hidden_size, num_layers, vocab_size, head_dim)` match the Python run.
//! - Per test case: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU
//!   `F32`).
//!
//! Two test wrappers (one CPU, one GPU), both `#[ignore]`-gated and serial.
//! GPU test skips cleanly when no `CUDA` device is available.
//!
//! Requires `google/gemma-2-2b` (gated; ~5 GiB) cached in
//! `~/.cache/huggingface/hub/`.
//!
//! Run CPU:
//!   `cargo test --test validate_gemma2_forward --features transformer -- --ignored gemma2_2b_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_gemma2_forward --features transformer -- --ignored gemma2_2b_forward_parity_gpu`

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

const MODEL_ID: &str = "google/gemma-2-2b";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Run the Gemma 2 forward-parity check on the given `device`.  Prints a
/// per-prompt comparison, then asserts top-10 index + magnitude parity across
/// all cases at the end.
#[allow(clippy::too_many_lines)]
fn run_gemma2_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    let reference_str = std::fs::read_to_string(reference_path("gemma2_forward_reference.json"))
        .expect(
            "failed to read gemma2_forward_reference.json, run scripts/gemma2_validation.py first",
        );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    let ref_hidden = json_usize(&reference["hidden_size"]);
    let ref_layers = json_usize(&reference["num_layers"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let ref_head_dim = json_usize(&reference["head_dim"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");

    println!("Validating Gemma 2 forward parity ({device_name}) against Python oracle:");
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}",
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

    // candle-mi must detect the Gemma 2 soft-capping + 4-norm + scaled-embedding
    // stack; a wrong path here is the latent class this guards against.
    assert!(
        config.attn_logit_softcapping.is_some(),
        "Gemma 2 must detect attn_logit_softcapping"
    );
    assert!(
        config.final_logit_softcapping.is_some(),
        "Gemma 2 must detect final_logit_softcapping"
    );
    assert!(config.use_post_norms, "Gemma 2 must use 4 norms per layer");
    assert!(
        config.embedding_scale.is_some(),
        "Gemma 2 must scale embeddings by sqrt(hidden_size)"
    );

    let dtype = DType::F32;
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

        // INDEX: ref_top10 has TOP_K = 10 entries by construction of the oracle.
        let ref_top1_idx = json_usize(&ref_top10[0]["index"]);
        let ref_top1_logit = json_f32(&ref_top10[0]["logit"]);
        println!("\nPrompt: {prompt:?}  ({} tokens)", ref_tokens.len());
        // INDEX: `indexed` is the full vocabulary sorted by logit, so index 0
        // always exists.
        println!(
            "  Python top-1: ({ref_top1_idx}, {ref_top1_logit:.4})   \
             Rust top-1: ({}, {:.4})",
            indexed[0].0, indexed[0].1
        );

        let mut prompt_max_diff: f32 = 0.0;
        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);
            // INDEX: rank < ref_top10.len() = TOP_K, and `indexed` covers the whole
            // vocabulary, which is larger.
            let (rust_idx, rust_logit) = indexed[rank];

            if rust_idx != ref_idx {
                failures.push(format!(
                    "{prompt:?} rank {rank}: index mismatch (Rust {rust_idx}, Python {ref_idx})"
                ));
            }
            let diff = (rust_logit - ref_logit).abs();
            if diff >= abs_diff_bar {
                failures.push(format!(
                    "{prompt:?} rank {rank}: logit abs-diff {diff:.3e} >= {abs_diff_bar:.0e} \
                     (Rust {rust_logit:.4}, Python {ref_logit:.4})"
                ));
            }
            prompt_max_diff = prompt_max_diff.max(diff);
            max_abs_diff = max_abs_diff.max(diff);
        }
        println!("  max abs-diff over top-10: {prompt_max_diff:.3e}");
    }

    println!(
        "\n{} test cases on {device_name}; max abs-diff across all top-10 logits = {:.3e} (bar: {:.0e})",
        test_cases.len(),
        max_abs_diff,
        abs_diff_bar
    );

    assert!(
        failures.is_empty(),
        "Gemma 2 forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires google/gemma-2-2b cached (~5 GiB); run with --ignored"]
#[serial]
fn gemma2_2b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_gemma2_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires google/gemma-2-2b cached (~5 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn gemma2_2b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_gemma2_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
