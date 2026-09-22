// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `google/gemma-2b` forward-pass parity against the
//! from-first-principles Python oracle in `scripts/gemma_validation.py`.
//!
//! Gemma 1 was the one entry in `SUPPORTED_MODEL_TYPES` with no
//! forward-parity record in `RESURRECTION.md`: `parse_gemma` had
//! config-level unit tests and a support claim in `README.md`, but its
//! forward pass had never been checked against an oracle.
//!
//! It is not covered by the Gemma 2 arm, because it is that arm's *inverse*.
//! Gemma 1 keeps `GemmaRmsNorm`, `sqrt(hidden_size)` embedding scaling,
//! `GeluApprox` and the required BOS token, but has **no** attention or final
//! logit soft-capping, **no** post-attention / post-feedforward norms, and
//! **no** sliding window.  A passing `validate_gemma2_forward` says nothing
//! about any of those being correctly *absent*.
//!
//! `google/gemma-2b` additionally uses multi-query attention (a single KV
//! head), which no other validated family exercises, so this is also the
//! narrowest test of the GQA path at its degenerate end.
//!
//! Consumes the frozen reference JSON (`scripts/gemma_forward_reference.json`)
//! and verifies that candle-mi's [`GenericTransformer`] produces matching
//! output when fed the same input prompts.  Acceptance bar:
//!
//! - Detected config carries `None` `attn_logit_softcapping` /
//!   `final_logit_softcapping`, `use_post_norms == false`,
//!   `sliding_window == None`, and a non-`None` `embedding_scale`.
//! - `(hidden_size, num_layers, vocab_size, head_dim, num_kv_heads)` match
//!   the Python run.
//! - Per test case: top-10 logit indices match exactly; magnitudes within
//!   `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3` (GPU `F32` vs CPU
//!   `F32`).
//!
//! Two test wrappers (one CPU, one GPU), both `#[ignore]`-gated and serial.
//! GPU test skips cleanly when no `CUDA` device is available.
//!
//! Requires `google/gemma-2b` (gated; ~4.7 GiB) cached in
//! `~/.cache/huggingface/hub/`.
//!
//! Run CPU:
//!   `cargo test --test validate_gemma_forward --features transformer -- --ignored gemma_2b_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_gemma_forward --features transformer -- --ignored gemma_2b_forward_parity_gpu`

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

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, TransformerConfig};
use serial_test::serial;

const MODEL_ID: &str = "google/gemma-2b";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

fn reference_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("gemma_forward_reference.json")
}

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

/// Locate the snapshot directory carrying `config.json` and safetensors
/// weights (single-file or sharded).  A repo can have several snapshot dirs.
fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    for entry in std::fs::read_dir(snapshots_dir).ok()?.flatten() {
        let path = entry.path();
        let has_weights = path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists();
        if path.join("config.json").exists() && has_weights {
            return Some(path);
        }
    }
    None
}

/// Collect safetensors paths for a snapshot (single file or sharded index).
fn safetensors_paths(snapshot: &Path) -> Vec<PathBuf> {
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }
    let index_path = snapshot.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path).unwrap_or_else(|_| {
        panic!(
            "no model.safetensors or index.json in {}",
            snapshot.display()
        )
    });
    let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let mut shard_names: Vec<String> = weight_map
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    shard_names.sort();
    shard_names.dedup();
    shard_names.iter().map(|name| snapshot.join(name)).collect()
}

fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0).ok().filter(Device::is_cuda)
}

/// Run the Gemma 1 forward-parity check on the given `device`.  Prints a
/// per-prompt comparison, then asserts top-10 index + magnitude parity across
/// all cases at the end.
#[allow(clippy::too_many_lines)]
fn run_gemma_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    let reference_str = std::fs::read_to_string(reference_path()).expect(
        "failed to read gemma_forward_reference.json, run scripts/gemma_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let model_repo = reference["model_repo"].as_str().unwrap();
    // CAST: u64 → usize, covering the five reads below. serde_json stores every
    // integer as u64; these are model dimensions from our own oracle JSON, all
    // far below usize::MAX on any supported target.
    let ref_hidden = reference["hidden_size"].as_u64().unwrap() as usize;
    let ref_layers = reference["num_layers"].as_u64().unwrap() as usize;
    let ref_vocab = reference["vocab_size"].as_u64().unwrap() as usize;
    let ref_head_dim = reference["head_dim"].as_u64().unwrap() as usize;
    let ref_kv_heads = reference["num_kv_heads"].as_u64().unwrap() as usize;
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(model_repo, MODEL_ID, "oracle JSON model_repo mismatch");

    println!("Validating Gemma 1 forward parity ({device_name}) against Python oracle:");
    println!("  model:  {model_repo}");
    println!(
        "  hidden_size={ref_hidden}, num_layers={ref_layers}, \
         vocab_size={ref_vocab}, head_dim={ref_head_dim}, num_kv_heads={ref_kv_heads}"
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
    assert_eq!(config.num_kv_heads, ref_kv_heads);

    // Gemma 1 is Gemma 2's inverse: the shared pieces must be detected, and
    // every Gemma 2 extension must be correctly ABSENT.  A parser that leaks a
    // soft-cap or a post-norm into the Gemma 1 path is the latent class this
    // guards against, and no Gemma 2 run can catch it.
    assert!(
        config.embedding_scale.is_some(),
        "Gemma 1 must scale embeddings by sqrt(hidden_size)"
    );
    assert!(
        config.attn_logit_softcapping.is_none(),
        "Gemma 1 must NOT have attn_logit_softcapping"
    );
    assert!(
        config.final_logit_softcapping.is_none(),
        "Gemma 1 must NOT have final_logit_softcapping"
    );
    assert!(
        !config.use_post_norms,
        "Gemma 1 must use 2 norms per layer, not 4"
    );
    assert!(
        config.sliding_window.is_none(),
        "Gemma 1 must NOT use a sliding window"
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
            // CAST: u64 → u32, token ids from our own oracle JSON; the vocabulary
            // is 256000, so every id fits in u32.
            .map(|v| v.as_u64().unwrap() as u32)
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
        // CAST: u64 → usize and f64 → f32, a vocabulary index and a logit read
        // back for display; the oracle writes F32 values as JSON f64.
        let ref_top1_idx = ref_top10[0]["index"].as_u64().unwrap() as usize;
        let ref_top1_logit = ref_top10[0]["logit"].as_f64().unwrap() as f32;
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
            // CAST: u64 → usize and f64 → f32, as above, per rank.
            let ref_idx = ref_item["index"].as_u64().unwrap() as usize;
            let ref_logit = ref_item["logit"].as_f64().unwrap() as f32;
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
        "Gemma 1 forward parity FAILED ({} divergences):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
#[ignore = "requires google/gemma-2b cached (~4.7 GiB); run with --ignored"]
#[serial]
fn gemma_2b_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_gemma_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires google/gemma-2b cached (~4.7 GiB) and a CUDA device; run with --ignored"]
#[serial]
fn gemma_2b_forward_parity_gpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    run_gemma_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
