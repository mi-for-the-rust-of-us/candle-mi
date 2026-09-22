// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests: load RWKV-7 Goose model from the `HuggingFace` cache
//! and validate forward-pass outputs against Python reference data.
//!
//! These tests require `RWKV/RWKV7-Goose-World3-1.5B-HF` in the local HF cache.
//!
//! Run CPU tests:
//!   `cargo test --test validate_rwkv7 --no-default-features --features rwkv,rwkv-tokenizer`
//!
//! Run all (CPU + GPU):
//!   `cargo test --test validate_rwkv7 --features rwkv,rwkv-tokenizer`

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
use candle_mi::rwkv::{GenericRwkv, RwkvConfig, RwkvVersion};
use candle_mi::{HookPoint, HookSpec, MIBackend, MIModel, MITokenizer};
use serial_test::serial;

const MODEL_ID: &str = "RWKV/RWKV7-Goose-World3-1.5B-HF";
const VOCAB_FILE: &str = "rwkv_vocab_v20230424.txt";

// ---------------------------------------------------------------------------
// Reference data
// ---------------------------------------------------------------------------

/// Parsed reference data from `scripts/rwkv7_reference.json`.
struct ReferenceData {
    test_prompt: String,
    token_ids: Vec<u32>,
    top_predictions: Vec<(u32, String, f32)>, // (token_id, token_str, logit)
    generated_token_ids: Vec<u32>,            // 20-token greedy generation
}

fn load_reference() -> ReferenceData {
    let json_str =
        std::fs::read_to_string("scripts/rwkv7_reference.json").expect("reference JSON not found");
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

/// Load the RWKV-7 model and tokenizer from the local HF cache.
///
/// Uses F32 everywhere for research-grade precision matching Python/PyTorch.
fn load_rwkv7_on(device: &Device) -> (GenericRwkv, MITokenizer, RwkvConfig) {
    load_rwkv7_on_with_dtype(device, DType::F32)
}

/// Load the RWKV-7 model with a specific dtype.
fn load_rwkv7_on_with_dtype(
    device: &Device,
    dtype: DType,
) -> (GenericRwkv, MITokenizer, RwkvConfig) {
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));

    // Parse config
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = RwkvConfig::from_hf_config(&json).unwrap();

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
        "RWKV-7 ({device_name}) — Top {} for '{prompt}':",
        top_k.len()
    );
    for (rank, (id, token, logit)) in top_k.iter().enumerate() {
        println!("  {}: id={id} '{}' (logit={logit:.4})", rank + 1, token);
    }
}

// ===========================================================================
// Intervention conformance (BACKENDS.md testing checklist)
// ===========================================================================

/// `Intervention::Zero` at `ResidPost(0)` must change the output.
///
/// This is `BACKENDS.md`'s conformance item, and until v0.2.0 `GenericRwkv`
/// failed it silently: it captured correctly but never consulted
/// `interventions_at` at any point except `Embed`, so a causal experiment
/// returned the untouched baseline and measured an effect of exactly zero.
/// Nothing asserted it, which is why it survived several releases.
#[test]
#[serial]
fn rwkv7_intervention_at_resid_post_changes_output() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv7_on_with_dtype(&device, DType::F32);

    let ids = tokenizer.encode("The capital of France is").unwrap();
    let input = Tensor::new(ids.as_slice(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let baseline: Vec<f32> = model
        .forward(&input, &HookSpec::new())
        .unwrap()
        .output()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut hooks = HookSpec::new();
    hooks.intervene(HookPoint::ResidPost(0), candle_mi::Intervention::Zero);
    let treated: Vec<f32> = model
        .forward(&input, &hooks)
        .unwrap()
        .output()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_ne!(
        baseline, treated,
        "Intervention::Zero at ResidPost(0) must change GenericRwkv's output"
    );
}

/// A diagnostic read-out refuses an intervention instead of dropping it.
///
/// `RwkvEffectiveAttn` is reconstructed only when captured, so an intervention
/// registered without a capture had nothing to refuse and vanished silently.
#[test]
#[serial]
fn rwkv7_diagnostic_read_out_refuses_an_intervention() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv7_on_with_dtype(&device, DType::F32);

    let ids = tokenizer.encode("Paris").unwrap();
    let input = Tensor::new(ids.as_slice(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    // Intervened, never captured: the case that used to disappear.
    let mut hooks = HookSpec::new();
    hooks.intervene(
        HookPoint::RwkvEffectiveAttn(0),
        candle_mi::Intervention::Zero,
    );
    let err = model
        .forward(&input, &hooks)
        .expect_err("a diagnostic read-out must refuse an intervention");
    let msg = err.to_string();
    assert!(msg.contains("diagnostic read-out"), "{msg}");
    // Each read-out names its own alternative; effective attention is not a
    // state edit, so it points at the residual stream rather than at
    // `set_state_knockout`.
    assert!(
        msg.contains("steer `ResidPre`/`ResidPost` instead"),
        "must name the alternative: {msg}"
    );
    // The message is spliced from a const that once carried a stray newline
    // from a botched line continuation; it must read as one line.
    assert!(
        !msg.contains('\n'),
        "error message must be a single line: {msg}"
    );
}

// ===========================================================================
// Config parsing
// ===========================================================================

#[test]
fn rwkv7_config_parse() {
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
    assert_eq!(config.version, RwkvVersion::V7);
}

// ===========================================================================
// Tokenizer validation
// ===========================================================================

#[test]
fn rwkv7_tokenizer() {
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
fn rwkv7_forward_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv7_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("CPU", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // Check logit is close to reference (7.559)
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
fn rwkv7_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };

    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let (model, tokenizer, _config) = load_rwkv7_on(&device);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("GPU-F32", &reference.test_prompt, &top_k);

    // Top-1 should be "if" (token 1942)
    let (top1_id, top1_token, top1_logit) = &top_k[0];
    assert_eq!(
        *top1_id, 1942,
        "Expected top-1 token ID 1942 ('if'), got {top1_id} ('{top1_token}')"
    );

    // F32 on GPU should be very close to the Python F32 reference
    let ref_logit = reference.top_predictions[0].2;
    let logit_diff = (*top1_logit - ref_logit).abs();
    println!(
        "GPU-F32 top-1 logit diff from Python reference: {logit_diff:.6} (ref={ref_logit:.4})"
    );
    assert!(
        logit_diff < 1.0,
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
// GPU forward pass (BF16) — regression test for reduced-precision mode
// ===========================================================================

#[test]
#[serial]
fn rwkv7_forward_gpu_bf16() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };

    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let (model, tokenizer, _config) = load_rwkv7_on_with_dtype(&device, DType::BF16);
    let reference = load_reference();

    let top_k = top_k_last_token(&model, &tokenizer, &device, &reference.test_prompt, 10);
    print_top_k("GPU-BF16", &reference.test_prompt, &top_k);

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
// Hook capture: RwkvState + RwkvDecay shape
// ===========================================================================

#[test]
fn rwkv7_hook_capture_state() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, config) = load_rwkv7_on(&device);
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

    // Decay (RWKV-7): the captured value is the RAW log-decay
    // `w = sigmoid(...)·(−0.6065)`, so every element lies strictly in
    // (−0.6065, 0). A regression to a V6-style (0,1) decay, a sign flip, or a
    // NaN fails here — a shape check can't. (Version-specific: V6 differs.)
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
        dmin >= -0.6066 && dmax <= 1e-4,
        "RWKV-7 log-decay out of range (−0.6065, 0): [{dmin}, {dmax}]"
    );
}

// ===========================================================================
// MIModel::from_pretrained
// ===========================================================================

#[test]
fn rwkv7_from_pretrained() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let model = MIModel::from_pretrained(MODEL_ID).unwrap();

    // Verify metadata
    assert_eq!(model.num_layers(), 24);
    assert_eq!(model.hidden_size(), 2048);
    assert_eq!(model.vocab_size(), 65536);
    assert_eq!(model.num_heads(), 32);

    // Quick forward pass
    let device = model.device();
    let input = Tensor::new(&[1942u32, 98, 5170], device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks).unwrap();
    let logits = result.output();
    let dims = logits.dims3().unwrap();
    assert_eq!(dims.0, 1, "batch");
    assert_eq!(dims.1, 3, "seq_len");
    assert_eq!(dims.2, 65536, "vocab_size");
}

// ===========================================================================
// Multi-step greedy generation — the only intermediate-timestep parity check
// ===========================================================================

/// Greedy-decode 20 tokens and compare to the Python reference's
/// `generated_token_ids`. Unlike the single-last-token forward tests, this
/// exercises the WKV recurrence across many timesteps: each step's argmax feeds
/// the next, so a recurrence/decay bug that only manifests after several tokens
/// diverges the sequence here. `#[ignore]` because it runs ~20 full forwards of a
/// 1.5B model — heavier than the other lanes; run via `scripts/resurrect.ps1`
/// (or `--include-ignored`).
#[test]
#[ignore = "20-step generation on a 1.5B model — run via resurrect.ps1 / --include-ignored"]
fn rwkv7_greedy_generation_matches_python() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_rwkv7_on(&device);
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
    println!("RWKV-7 greedy generation matches Python reference ({n_new} tokens)");
}
