// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `deepseek-ai/deepseek-coder-1.3b-base` forward-pass
//! parity against the from-first-principles Python oracle in
//! `scripts/deepseek_coder_validation.py`.
//!
//! DeepSeek-Coder declares `model_type: "llama"` but ships a
//! `rope_scaling` block (`{"type": "linear", "factor": 4.0}`) that extends
//! its context from 4 096 to 16 384.  Linear scaling divides *every*
//! position by the factor before the rotary rotation, so an implementation
//! that ignores `rope_scaling` diverges even on short prompts.  This test
//! is the regression guard for candle-mi's linear `rope_scaling` support.
//!
//! Consumes the frozen reference JSON
//! (`scripts/deepseek_coder_forward_reference.json`) and verifies that
//! candle-mi's [`GenericTransformer`] produces matching output when fed the
//! same input prompts.  Acceptance bar:
//!
//! - `(hidden_size, num_layers, vocab_size, head_dim)` match the Python run.
//! - Per test case: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU
//!   `F32` — looser to absorb CUDA-vs-CPU rounding noise documented for
//!   RWKV-7).
//!
//! The repository ships only `pytorch_model.bin` (no safetensors), so
//! weights are loaded via [`candle_nn::VarBuilder::from_pth`] rather than
//! the mmaped-safetensors path used by the other forward-parity tests.
//!
//! Two test wrappers (one CPU, one GPU), both `#[ignore]`-gated and serial.
//! GPU test skips cleanly when no `CUDA` device is available.
//!
//! Requires `deepseek-ai/deepseek-coder-1.3b-base` (~2.5 GiB) cached in
//! `~/.cache/huggingface/hub/`.
//!
//! Run CPU:
//!   `cargo test --test validate_deepseek_forward --features transformer -- --ignored deepseek_coder_1_3b_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_deepseek_forward --features transformer -- --ignored deepseek_coder_1_3b_forward_parity_gpu`

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

use common::{cuda_device, find_snapshot, json_f32, json_u32, json_usize, reference_path};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, MIModel, RopeScaling, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "deepseek-ai/deepseek-coder-1.3b-base";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Run the DeepSeek-Coder forward-parity check on the given `device`.
/// Prints a per-prompt comparison, then asserts top-10 index + magnitude
/// parity across all cases at the end (so a failure shows every prompt's
/// divergence, not just the first).
#[allow(clippy::too_many_lines)]
fn run_deepseek_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    // --- Load the frozen reference JSON ---
    let reference_str = std::fs::read_to_string(reference_path("deepseek_coder_forward_reference.json")).expect(
        "failed to read deepseek_coder_forward_reference.json — run scripts/deepseek_coder_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    let ref_hidden = json_usize(&reference["hidden_size"]);
    let ref_layers = json_usize(&reference["num_layers"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let ref_head_dim = json_usize(&reference["head_dim"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");

    println!("Validating DeepSeek-Coder forward parity ({device_name}) against Python oracle:");
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}",
        test_cases.len()
    );

    // --- Load model from HF cache (pytorch_model.bin via from_pth) ---
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, ref_hidden);
    assert_eq!(config.num_layers, ref_layers);
    assert_eq!(config.vocab_size, ref_vocab);
    assert_eq!(config.head_dim, ref_head_dim);

    // candle-mi must detect the linear rope_scaling that defines this family;
    // without it the forward pass diverges grossly (the regression this guards).
    assert_eq!(
        config.rope_scaling,
        Some(RopeScaling::Linear { factor: 4.0 }),
        "candle-mi must parse DeepSeek-Coder's linear rope_scaling (factor 4.0)"
    );

    let dtype = DType::F32;
    let pth_path = snapshot.join("pytorch_model.bin");
    assert!(
        pth_path.exists(),
        "pytorch_model.bin not found in {}",
        snapshot.display()
    );

    let vb = candle_nn::VarBuilder::from_pth(&pth_path, dtype, device).unwrap();
    let model = GenericTransformer::load(config, device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_layers);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);

    // --- Run each test case; collect failures, print, then assert ---
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
        println!("\nPrompt: {prompt:?}  ({} tokens)", ref_tokens.len());
        println!(
            "  Python top-1: ({ref_top1_idx}, {ref_top1_logit:.4})   \
             Rust top-1: ({}, {:.4})",
            indexed[0].0, indexed[0].1
        );

        let mut prompt_max_diff: f32 = 0.0;
        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);
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
        "DeepSeek-Coder forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires deepseek-ai/deepseek-coder-1.3b-base cached (~2.5 GiB); run with --ignored"]
#[serial]
fn deepseek_coder_1_3b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_deepseek_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

/// End-to-end check that the public `MIModel::from_pretrained` API can load a
/// repository that ships **only** `pytorch_model.bin` (no safetensors).  Prior
/// to `.bin` support this returned an error ("model.safetensors not found").
/// A successful load with matching dims proves every pickle tensor mapped: the
/// underlying `GenericTransformer::load` validates each weight by name/shape.
#[test]
#[ignore = "requires deepseek-ai/deepseek-coder-1.3b-base cached (~2.5 GiB); hits HF for metadata; run with --ignored"]
#[serial]
fn deepseek_coder_from_pretrained_loads_bin() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let model = MIModel::from_pretrained(MODEL_ID)
        .expect("from_pretrained must load a pytorch_model.bin-only repo");
    assert_eq!(model.num_layers(), 24);
    assert_eq!(model.hidden_size(), 2048);
    assert_eq!(model.vocab_size(), 32256);
    println!("from_pretrained loaded DeepSeek-Coder 1.3B from pytorch_model.bin");
}

#[test]
#[ignore = "requires deepseek-ai/deepseek-coder-1.3b-base cached (~2.5 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn deepseek_coder_1_3b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_deepseek_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
