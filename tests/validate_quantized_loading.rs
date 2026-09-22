// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests: loading + running **quantized** checkpoints
//! (bitsandbytes NF4, AWQ, GPTQ) via candle-mi's anamnesis-backed dequant path
//! (`quantized` feature), validated against `PyTorch` oracles that use each
//! scheme's real library.
//!
//! candle-mi cannot consume quantized weights directly (`candle_nn` loads plain
//! float safetensors).  With the `quantized` feature, `from_pretrained` detects
//! `quantization_config` and routes the weights through anamnesis
//! (`parse` → `remember_to_bytes(BF16)`) before building the `VarBuilder`.  One
//! code path covers all three schemes (anamnesis auto-detects); these tests
//! close the loop on each.
//!
//! All three targets are **llama** models (a family already exact-parity
//! validated), so a mismatch isolates to the dequant, not the forward:
//! - bnb NF4  — `medmekk/Llama-3.2-1B-Instruct-bnb-nf4` (oracle: bitsandbytes, F32)
//! - AWQ      — `casperhansen/llama-3.2-1b-instruct-awq` (oracle: `AutoAWQ`, fp16)
//! - GPTQ     — `shuyuej/Llama-3.2-1B-Instruct-GPTQ`     (oracle: `GPTQModel`, fp16)
//!
//! anamnesis dequantizes to **BF16**, while the oracles compute in F32 (bnb) or
//! fp16 (AWQ/GPTQ), so the bar is a **weight-precision tier** (not the ~1e-5 of
//! full-precision families): observed max diff ~1.0 for bnb (its F32 oracle
//! exposes the full BF16-weight gap) and ~2–3e-2 for AWQ/GPTQ (their fp16
//! oracles are themselves reduced-precision).  anamnesis's own cross-validation
//! proves each decode is bit-exact to the real library; the residual here is
//! dtype precision through 16 layers, not a dequant error.  The headline
//! correctness guarantee is the **exact top-1 match** on every prompt.
//!
//! The oracle records the token ids; the Rust side feeds them directly (some
//! quant repos ship no tokenizer), keeping tokenization out of the comparison.
//!
//! Each test is `#[ignore]`-gated; requires the model cached, a CUDA device, and
//! the `quantized` feature.
//!
//! Run all:
//!   `cargo test --test validate_quantized_loading --features transformer,quantized -- --ignored --nocapture`

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

use common::{json_f32, json_u32, json_usize};

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{HookSpec, MIModel};

/// bnb dequant → BF16 vs an **F32** oracle (the checkpoint's compute dtype):
/// residual is the full BF16-vs-F32 weight-precision gap (observed max ~1.0).
const BAR_BNB: f32 = 1.5;
/// AWQ/GPTQ dequant → BF16 vs an **fp16** oracle: much tighter than bnb because
/// the fp16 oracle is itself reduced-precision (observed max ~3e-2 AWQ /
/// ~2e-2 GPTQ). 0.1 keeps ~3x headroom while staying meaningful.
const BAR_AWQ: f32 = 1e-1;
const BAR_GPTQ: f32 = 1e-1;

fn reference_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(filename)
}

fn model_is_cached(model_id: &str) -> bool {
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return false;
    };
    let dir = std::path::Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    dir.exists()
}

/// Load a quantized checkpoint through candle-mi's anamnesis dequant path and
/// compare its forward to a frozen oracle: exact top-1 per prompt, magnitudes
/// within `bar` (per-token, robust to near-tied reordering).
fn run_quantized_parity(scheme: &str, model_id: &str, oracle_file: &str, bar: f32) {
    if !model_is_cached(model_id) {
        eprintln!("SKIP: {model_id} not in HF cache");
        return;
    }

    let reference_str = std::fs::read_to_string(reference_path(oracle_file)).unwrap_or_else(|_| {
        panic!("read {oracle_file} — run the matching scripts/*_validation.py")
    });
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();
    assert_eq!(reference["model_repo"].as_str().unwrap(), model_id);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    // Loads via the anamnesis dequant path (quantization_config → dequant → BF16).
    let model = MIModel::from_pretrained(model_id)
        .unwrap_or_else(|e| panic!("load {scheme} model {model_id} via anamnesis dequant: {e}"));

    println!("Validating {scheme} forward parity (anamnesis dequant) vs real-library oracle:");
    println!(
        "  model: {model_id}  (num_layers={}, hidden_size={}, vocab_size={})",
        model.num_layers(),
        model.hidden_size(),
        model.vocab_size()
    );
    assert_eq!(model.vocab_size(), ref_vocab);

    let device = model.device().clone();
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

        let input = Tensor::new(&ref_tokens[..], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let hooks = HookSpec::new();
        let cache = model.forward(&input, &hooks).unwrap();
        let logits = cache.output();

        let (batch, out_seq, vocab) = logits.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(out_seq, ref_tokens.len());
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
            "  {prompt:?}: oracle top-1 ({ref_top1_idx}, {ref_top1_logit:.4})  candle ({}, {:.4})",
            indexed[0].0, indexed[0].1
        );

        if indexed[0].0 != ref_top1_idx {
            failures.push(format!(
                "{prompt:?}: top-1 mismatch (candle {}, oracle {ref_top1_idx})",
                indexed[0].0
            ));
        }

        // Per-token magnitude (candle's logit for the oracle's token), so a
        // near-tied reordering does not inflate the diff.
        for ref_item in ref_top10 {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);
            let diff = (last[ref_idx] - ref_logit).abs();
            if diff >= bar {
                failures.push(format!(
                    "{prompt:?} token {ref_idx}: logit abs-diff {diff:.3e} >= {bar:.2} \
                     (candle {:.4}, oracle {ref_logit:.4})",
                    last[ref_idx]
                ));
            }
            max_abs_diff = max_abs_diff.max(diff);
        }
    }

    println!(
        "{} test cases; max abs-diff across all top-10 tokens = {:.3e} (bar: {:.2})",
        test_cases.len(),
        max_abs_diff,
        bar
    );

    assert!(
        failures.is_empty(),
        "{scheme} forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires medmekk/Llama-3.2-1B-Instruct-bnb-nf4 cached, a CUDA device, and the `quantized` feature; run with --ignored"]
fn bnb_nf4_llama_forward_parity() {
    run_quantized_parity(
        "bnb-NF4",
        "medmekk/Llama-3.2-1B-Instruct-bnb-nf4",
        "bnb_nf4_forward_reference.json",
        BAR_BNB,
    );
}

#[test]
#[ignore = "requires casperhansen/llama-3.2-1b-instruct-awq cached, a CUDA device, and the `quantized` feature; run with --ignored"]
fn awq_llama_forward_parity() {
    run_quantized_parity(
        "AWQ",
        "casperhansen/llama-3.2-1b-instruct-awq",
        "awq_forward_reference.json",
        BAR_AWQ,
    );
}

#[test]
#[ignore = "requires shuyuej/Llama-3.2-1B-Instruct-GPTQ cached, a CUDA device, and the `quantized` feature; run with --ignored"]
fn gptq_llama_forward_parity() {
    run_quantized_parity(
        "GPTQ",
        "shuyuej/Llama-3.2-1B-Instruct-GPTQ",
        "gptq_forward_reference.json",
        BAR_GPTQ,
    );
}
