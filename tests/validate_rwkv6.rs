// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests: load RWKV-6 Finch model from the `HuggingFace` cache
//! and validate forward-pass outputs against Python reference data.
//!
//! These tests require `RWKV/v6-Finch-1B6-HF` in the local HF cache.
//!
//! Run CPU tests:
//!   `cargo test --test validate_rwkv6 --no-default-features --features rwkv,rwkv-tokenizer`
//!
//! Run all (CPU + GPU):
//!   `cargo test --test validate_rwkv6 --features rwkv,rwkv-tokenizer`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    unsafe_code,
    missing_docs
)]

mod common;

use common::{cuda_device, find_snapshot, json_f32, json_u32, safetensors_paths};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::rwkv::{GenericRwkv, RwkvConfig};
use candle_mi::{HookPoint, HookSpec, MIBackend, MITokenizer};
use serial_test::serial;

const MODEL_ID: &str = "RWKV/v6-Finch-1B6-HF";
const VOCAB_FILE: &str = "rwkv_vocab_v20230424.txt";

// ---------------------------------------------------------------------------
// Reference data
// ---------------------------------------------------------------------------

/// Parsed reference data from `scripts/rwkv6_reference.json`.
struct ReferenceData {
    test_prompt: String,
    token_ids: Vec<u32>,
    top_predictions: Vec<(u32, String, f32)>, // (token_id, token_str, logit)
    generated_token_ids: Vec<u32>,            // 20-token greedy generation
}

fn load_reference() -> ReferenceData {
    let json_str =
        std::fs::read_to_string("scripts/rwkv6_reference.json").expect("reference JSON not found");
    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let test_prompt = json["test_prompt"].as_str().unwrap().to_string();

    let token_ids: Vec<u32> = json["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_u32)
        .collect();

    let top_predictions: Vec<(u32, String, f32)> = json["top_predictions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let id = json_u32(&p["token_id"]);
            let token = p["token"].as_str().unwrap().to_string();
            let logit = json_f32(&p["logit"]);
            (id, token, logit)
        })
        .collect();

    let generated_token_ids: Vec<u32> = json["generated_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(json_u32)
        .collect();

    ReferenceData {
        test_prompt,
        token_ids,
        top_predictions,
        generated_token_ids,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load the RWKV-6 model and tokenizer from the local HF cache.
fn load_rwkv6_on(device: &Device) -> (GenericRwkv, MITokenizer, RwkvConfig) {
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));

    // Parse config
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = RwkvConfig::from_hf_config(&json).unwrap();

    // F32 everywhere: research-grade precision matching Python/PyTorch.
    let dtype = DType::F32;

    // Resolve safetensors paths
    let paths = safetensors_paths(&snapshot);

    // Load weights (mmap for both single and sharded)
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };

    // Build model
    let model = GenericRwkv::load(config.clone(), device, dtype, vb).unwrap();

    // Load RWKV World tokenizer
    let vocab_path = snapshot.join(VOCAB_FILE);
    let tokenizer = MITokenizer::from_rwkv_path(&vocab_path).unwrap();

    (model, tokenizer, config)
}

/// Run a forward pass and return top-k `(token_id, token_string, logit)` for the last position.
fn top_k_last_token(
    model: &GenericRwkv,
    tokenizer: &MITokenizer,
    device: &Device,
    prompt: &str,
    k: usize,
) -> Vec<(u32, String, f32)> {
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();

    let input = Tensor::new(&token_ids[..], device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks).unwrap();

    let logits = result.output();
    let (batch, out_seq, _vocab) = logits.dims3().unwrap();
    assert_eq!(batch, 1);
    assert_eq!(out_seq, seq_len);

    // Move to CPU F32 for inspection
    let logits_cpu = logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Get logits for the last token position
    let last_logits: Vec<f32> = logits_cpu.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();

    // Sort by logit value descending
    let mut indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Decode top-k
    indexed
        .iter()
        .take(k)
        .map(|(idx, logit)| {
            // CAST: usize → u32, a vocabulary index being handed back to `tokenizers`,
            // whose ids are u32; the largest vocabulary candle-mi loads is 256000
            let token = tokenizer.decode(&[*idx as u32]).unwrap();
            // CAST: usize → u32, a vocabulary index; candle takes u32 token ids and no
            // supported vocabulary approaches u32::MAX
            (*idx as u32, token, *logit)
        })
        .collect()
}

/// Greedy-decode `n_new` tokens from `prompt_ids`, recomputing the full sequence
/// each step (RWKV has no incremental state API) — exactly how the Python
/// reference generated `generated_token_ids`. Returns the generated ids.
fn greedy_generate(
    model: &GenericRwkv,
    device: &Device,
    prompt_ids: &[u32],
    n_new: usize,
) -> Vec<u32> {
    let hooks = HookSpec::new();
    let mut ids: Vec<u32> = prompt_ids.to_vec();
    let mut generated = Vec::with_capacity(n_new);

    for _ in 0..n_new {
        let input = Tensor::new(&ids[..], device).unwrap().unsqueeze(0).unwrap();
        let result = model.forward(&input, &hooks).unwrap();
        let seq_len = ids.len();
        let last: Vec<f32> = result
            .output()
            .i((0, seq_len - 1))
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        // Greedy argmax over the vocabulary.
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in last.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        // CAST: usize → u32, a vocabulary index; candle takes u32 token ids and no
        // supported vocabulary approaches u32::MAX
        let next = best_idx as u32;
        ids.push(next);
        generated.push(next);
    }
    generated
}

fn print_top_k(device_name: &str, prompt: &str, top_k: &[(u32, String, f32)]) {
    println!(
        "RWKV-6 ({device_name}) — Top {} for '{prompt}':",
        top_k.len()
    );
    for (rank, (id, token, logit)) in top_k.iter().enumerate() {
        println!("  {}: id={id} '{}' (logit={logit:.4})", rank + 1, token);
    }
}

// ===========================================================================
// Config parsing
// ===========================================================================

#[test]
fn rwkv6_config_parse() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = RwkvConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.num_heads, 32); // 2048 / 64
    assert_eq!(config.vocab_size, 65536);
    assert_eq!(config.intermediate_size, 7168); // (2048 * 7/2) / 32 * 32
}

// ===========================================================================
// Tokenizer validation
// ===========================================================================

#[test]
fn rwkv6_tokenizer() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };

    let vocab_path = snapshot.join(VOCAB_FILE);
    let tokenizer = MITokenizer::from_rwkv_path(&vocab_path).unwrap();

    let reference = load_reference();
    let tokens = tokenizer.encode(&reference.test_prompt).unwrap();

    assert_eq!(
        tokens, reference.token_ids,
        "Token IDs don't match reference for '{}'",
        reference.test_prompt
    );
}

// ===========================================================================
// CPU forward pass
// ===========================================================================

#[test]
fn rwkv6_forward_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv6_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("CPU", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // Check logit is close to reference (4.798)
    // F32 CPU should be very close to the Python F32 reference
    let ref_logit = reference.top_predictions[0].2;
    let logit_diff = (*top1_logit - ref_logit).abs();
    assert!(
        logit_diff < 1.0,
        "Top-1 logit {top1_logit:.4} differs from reference {ref_logit:.4} by {logit_diff:.4}"
    );

    // Validate top-5 token IDs match reference
    for (rank, (ref_id, ref_token, _ref_logit)) in
        reference.top_predictions.iter().take(5).enumerate()
    {
        let (got_id, got_token, _) = &top_k[rank];
        assert_eq!(
            got_id,
            ref_id,
            "Rank {}: expected token {ref_id} ('{ref_token}'), got {got_id} ('{got_token}')",
            rank + 1
        );
    }
}

// ===========================================================================
// GPU forward pass
// ===========================================================================

#[test]
#[serial]
fn rwkv6_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };

    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let (model, tokenizer, _config) = load_rwkv6_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("GPU", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // BF16 has lower precision, so allow wider tolerance
    let ref_logit = reference.top_predictions[0].2;
    let logit_diff = (*top1_logit - ref_logit).abs();
    assert!(
        logit_diff < 2.0,
        "Top-1 logit {top1_logit:.4} differs from reference {ref_logit:.4} by {logit_diff:.4}"
    );

    // Top-5 token IDs should match
    for (rank, (ref_id, ref_token, _ref_logit)) in
        reference.top_predictions.iter().take(5).enumerate()
    {
        let (got_id, got_token, _) = &top_k[rank];
        assert_eq!(
            got_id,
            ref_id,
            "Rank {}: expected token {ref_id} ('{ref_token}'), got {got_id} ('{got_token}')",
            rank + 1
        );
    }
}

// ===========================================================================
// Hook capture: RwkvState shape
// ===========================================================================

#[test]
fn rwkv6_hook_capture_state() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, config) = load_rwkv6_on(&device);
    let reference = load_reference();

    let token_ids = tokenizer.encode(&reference.test_prompt).unwrap();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Capture RwkvState, RwkvDecay, and ResidPre at layer 0
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::RwkvState(0));
    hooks.capture(HookPoint::RwkvDecay(0));
    hooks.capture(HookPoint::ResidPre(0));

    let result = model.forward(&input, &hooks).unwrap();

    // RwkvState should be [batch, num_heads, head_dim, head_dim]
    let state = result.require(&HookPoint::RwkvState(0)).unwrap();
    let state_dims = state.dims4().unwrap();
    assert_eq!(state_dims.0, 1, "batch");
    assert_eq!(state_dims.1, config.num_heads, "num_heads");
    assert_eq!(state_dims.2, config.head_dim, "head_dim");
    assert_eq!(state_dims.3, config.head_dim, "head_dim");

    // RwkvDecay should be [batch, seq_len, num_heads, head_dim]
    let decay = result.require(&HookPoint::RwkvDecay(0)).unwrap();
    let decay_dims = decay.dims4().unwrap();
    assert_eq!(decay_dims.0, 1, "batch");
    assert_eq!(decay_dims.1, token_ids.len(), "seq_len");
    assert_eq!(decay_dims.2, config.num_heads, "num_heads");
    assert_eq!(decay_dims.3, config.head_dim, "head_dim");

    // ResidPre should be [batch, seq_len, hidden_size]
    let resid = result.require(&HookPoint::ResidPre(0)).unwrap();
    let resid_dims = resid.dims3().unwrap();
    assert_eq!(resid_dims.0, 1, "batch");
    assert_eq!(resid_dims.1, token_ids.len(), "seq_len");
    assert_eq!(resid_dims.2, config.hidden_size, "hidden_size");

    // --- Value checks (not just shapes) ---
    // State: the accumulated WKV recurrence must be finite and non-trivial
    // (a real prompt leaves a nonzero state). Catches a NaN/blow-up.
    let state_vals: Vec<f32> = state
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        state_vals.iter().all(|v| v.is_finite()),
        "RwkvState has non-finite values"
    );
    assert!(
        state_vals.iter().any(|&v| v != 0.0),
        "RwkvState is all zero"
    );

    // Decay (RWKV-6): the captured value is `decay = exp(−exp(w))`, so every
    // element lies in (0, 1). A sign flip, a NaN, or a regression to the V7-style
    // negative log-decay fails here — a shape check can't. (Version-specific: V7
    // captures the raw log-decay in (−0.6065, 0).)
    let decay_vals: Vec<f32> = decay
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        decay_vals.iter().all(|v| v.is_finite()),
        "RwkvDecay has non-finite values"
    );
    let dmin = decay_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let dmax = decay_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        dmin >= 0.0 && dmax <= 1.0,
        "RWKV-6 decay out of range (0, 1): [{dmin}, {dmax}]"
    );
}

// ===========================================================================
// Multi-step greedy generation — the only intermediate-timestep parity check
// ===========================================================================

/// Greedy-decode 20 tokens and compare to the Python reference's
/// `generated_token_ids`. Unlike the single-last-token forward tests, this
/// exercises the WKV recurrence across many timesteps: each step's argmax feeds
/// the next, so a recurrence/decay bug that only manifests after several tokens
/// diverges the sequence here. `#[ignore]` because it runs ~20 full forwards of a
/// 1.6B model — heavier than the other lanes; run via `scripts/resurrect.ps1`
/// (or `--include-ignored`).
#[test]
#[ignore = "20-step generation on a 1.6B model — run via resurrect.ps1 / --include-ignored"]
fn rwkv6_greedy_generation_matches_python() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv6_on(&device);
    let reference = load_reference();

    let token_ids = tokenizer.encode(&reference.test_prompt).unwrap();
    assert_eq!(
        token_ids, reference.token_ids,
        "tokenizer mismatch vs reference"
    );

    let n_new = reference.generated_token_ids.len();
    let generated = greedy_generate(&model, &device, &token_ids, n_new);

    assert_eq!(
        generated, reference.generated_token_ids,
        "greedy generation diverged from the Python reference over {n_new} steps"
    );
    println!("RWKV-6 greedy generation matches Python reference ({n_new} tokens)");
}
