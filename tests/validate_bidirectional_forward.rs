// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: bidirectional decoder forward-pass parity against the fp32
//! Python oracle in `scripts/bidirectional_forward_validation.py`.
//!
//! Validates Stage 3 — candle-mi's **bidirectional** [`GenericTransformer`]
//! path, which loads decoder-style masked-diffusion LMs (Dream, a2d-qwen2, ...)
//! and runs the decoder non-causally.  Loads
//! `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` (a standard
//! `Qwen2.5` layout under `model_type` `"a2d-qwen2"`, the cheapest exact oracle)
//! and checks raw logits at several positions — including *early* ones, where
//! bidirectional attention provably differs from causal.  Acceptance bar:
//!
//! - `(hidden_size, num_hidden_layers, num_attention_heads, vocab_size)` match
//!   the Python run, and `config.bidirectional` is set.
//! - Per position: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU `F32`).
//!
//! Run CPU:
//!   `cargo test --test validate_bidirectional_forward --features transformer,mmap -- --ignored a2d_qwen2_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_bidirectional_forward --features transformer,mmap -- --ignored a2d_qwen2_forward_parity_gpu`

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
    clippy::single_match_else,
    clippy::manual_let_else,
    unsafe_code,
    missing_docs
)]

mod common;

use common::{cuda_device, find_snapshot, json_f32, json_u32, json_usize, reference_path};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Run the bidirectional forward-parity check on `device`.  Panics on any
/// mismatch.  `abs_diff_bar` is the per-rank acceptance threshold.
fn run_bidirectional_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    let reference_str =
        std::fs::read_to_string(reference_path("bidirectional_forward_reference.json")).expect(
            "failed to read bidirectional_forward_reference.json — run \
         scripts/bidirectional_forward_validation.py first",
        );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let ref_hidden = json_usize(&reference["hidden_size"]);
    let ref_layers = json_usize(&reference["num_hidden_layers"]);
    let ref_heads = json_usize(&reference["num_attention_heads"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    println!("Validating bidirectional forward parity ({device_name}) against the fp32 oracle:");
    println!("  model: {MODEL_ID} (model_type a2d-qwen2)");
    println!(
        "  hidden_size={ref_hidden}, layers={ref_layers}, heads={ref_heads}, vocab={ref_vocab}"
    );
    println!(
        "  {} prompts, abs-diff bar = {abs_diff_bar:.0e}",
        test_cases.len()
    );

    // --- Load model from HF cache via the dispatch path (model_type -> config) ---
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert!(
        config.bidirectional,
        "a2d-qwen2 must parse as a bidirectional decoder"
    );
    assert_eq!(config.hidden_size, ref_hidden);
    assert_eq!(config.num_layers, ref_layers);
    assert_eq!(config.num_attention_heads, ref_heads);
    assert_eq!(config.vocab_size, ref_vocab);

    let dtype = DType::F32;
    let weights = snapshot.join("model.safetensors");
    // SAFETY: safetensors files are not modified during test execution.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, device).unwrap()
    };
    let model = GenericTransformer::load(config, device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_layers);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);

    let mut max_abs_diff: f32 = 0.0;
    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_u32)
            .collect();
        let positions = tc["positions"].as_array().unwrap();

        println!("\nPrompt: {prompt:?}  ({} tokens)", ref_tokens.len());

        let input = Tensor::new(&ref_tokens[..], device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let result = model.forward(&input, &HookSpec::new()).unwrap();
        let logits = result.output();

        let (batch, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(out_seq, ref_tokens.len());
        assert_eq!(vocab, ref_vocab);

        // Logits on CPU `F32` for an order-stable comparison.
        let logits_cpu = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        for pos_entry in positions {
            let pos = json_usize(&pos_entry["position"]);
            let ref_top10 = pos_entry["top_10"].as_array().unwrap();

            let at_pos: Vec<f32> = logits_cpu.i((0, pos)).unwrap().to_vec1().unwrap();
            let mut indexed: Vec<(usize, f32)> =
                at_pos.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (rank, ref_item) in ref_top10.iter().enumerate() {
                let ref_idx = json_usize(&ref_item["index"]);
                let ref_logit = json_f32(&ref_item["logit"]);
                let (rust_idx, rust_logit) = indexed[rank];

                assert_eq!(
                    rust_idx, ref_idx,
                    "pos {pos} rank {rank}: top-10 index mismatch (Rust {rust_idx}, \
                     Python {ref_idx}) for prompt {prompt:?}"
                );
                let diff = (rust_logit - ref_logit).abs();
                assert!(
                    diff < abs_diff_bar,
                    "pos {pos} rank {rank}: logit abs-diff {diff:.2e} >= {abs_diff_bar:.0e} \
                     (Rust {rust_logit}, Python {ref_logit}) for prompt {prompt:?}"
                );
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }
            println!(
                "  pos {pos}: top-1 ({}, {:.4}) — matches Python",
                indexed[0].0, indexed[0].1
            );
        }
    }

    println!(
        "\nAll {} prompts passed on {device_name}; max abs-diff across all top-10 logits \
         = {max_abs_diff:.2e} (bar: {abs_diff_bar:.0e})",
        test_cases.len()
    );
}

#[test]
#[ignore = "requires dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm cached (~1.2 GiB); run with --ignored"]
#[serial]
fn a2d_qwen2_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_bidirectional_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires dllm-hub/Qwen2.5-Coder-0.5B-...-mdlm cached (~1.2 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn a2d_qwen2_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let device = match cuda_device() {
        Some(d) => d,
        None => {
            eprintln!("SKIP: no CUDA device available");
            return;
        }
    };
    run_bidirectional_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
