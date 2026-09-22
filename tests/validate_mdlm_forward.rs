// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: MDLM (`TheQweaker/mdlm-owt-noflash`) forward-pass parity against the
//! fp32 Python oracle in `scripts/mdlm_forward_validation.py`.
//!
//! Consumes the frozen reference JSON (`scripts/mdlm_forward_reference.json`)
//! and verifies that candle-mi's [`GenericMdlm`] produces matching raw logits
//! at the masked positions when fed the same token sequences.  Acceptance bar:
//!
//! - `(hidden_dim, n_blocks, n_heads, vocab_size)` match the Python run.
//! - Per test case: top-10 logit indices at the masked position match exactly;
//!   magnitudes within `abs diff < 1e-3` (CPU vs CPU `F32`) or `< 5e-3`
//!   (GPU `F32` vs CPU `F32`).
//!
//! Both the fp32 oracle and the weights this test loads come from the
//! flash-attn-free `TheQweaker/mdlm-owt-noflash` — a byte-identical-weights
//! reimplementation of `kuleshov-group/mdlm-owt` (only the modeling code differs).
//!
//! Run CPU:
//!   `cargo test --test validate_mdlm_forward --features diffusion,mmap -- --ignored mdlm_owt_forward_parity_cpu`
//!
//! Run GPU:
//!   `cargo test --test validate_mdlm_forward --features diffusion,mmap -- --ignored mdlm_owt_forward_parity_gpu`

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
use candle_mi::{DiffusionSamplingConfig, GenericMdlm, HookSpec, MIBackend, MdlmConfig};
use serial_test::serial;

const MODEL_ID: &str = "TheQweaker/mdlm-owt-noflash";
const ABS_DIFF_BAR_CPU: f32 = 1e-3;
const ABS_DIFF_BAR_GPU: f32 = 5e-3;

/// Run the MDLM forward-parity check on the given `device`.  Panics on any
/// mismatch.  `abs_diff_bar` is the per-rank acceptance threshold.
fn run_mdlm_forward_parity(device: &Device, device_name: &str, abs_diff_bar: f32) {
    let reference_str = std::fs::read_to_string(reference_path("mdlm_forward_reference.json")).expect(
        "failed to read mdlm_forward_reference.json — run scripts/mdlm_forward_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let weights_repo = reference["weights_repo"].as_str().unwrap();
    let ref_hidden = json_usize(&reference["hidden_dim"]);
    let ref_blocks = json_usize(&reference["n_blocks"]);
    let ref_heads = json_usize(&reference["n_heads"]);
    let ref_vocab = json_usize(&reference["vocab_size"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(weights_repo, MODEL_ID, "oracle JSON weights_repo mismatch");

    println!("Validating MDLM forward parity ({device_name}) against the fp32 Python oracle:");
    println!("  weights: {weights_repo}");
    println!(
        "  hidden_dim={ref_hidden}, n_blocks={ref_blocks}, n_heads={ref_heads}, vocab_size={ref_vocab}"
    );
    println!(
        "  {} test cases, abs-diff bar = {abs_diff_bar:.0e}",
        test_cases.len()
    );

    // --- Load model from HF cache ---
    let snapshot =
        find_snapshot(MODEL_ID).unwrap_or_else(|| panic!("{MODEL_ID} not found in HF cache"));
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = MdlmConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_dim, ref_hidden);
    assert_eq!(config.n_blocks, ref_blocks);
    assert_eq!(config.n_heads, ref_heads);
    assert_eq!(config.vocab_size, ref_vocab);

    let dtype = DType::F32;
    let weights = snapshot.join("model.safetensors");
    // SAFETY: safetensors files are not modified during test execution.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, device).unwrap()
    };
    let model = GenericMdlm::load(config, device, dtype, vb).unwrap();

    assert_eq!(model.num_layers(), ref_blocks);
    assert_eq!(model.hidden_size(), ref_hidden);
    assert_eq!(model.vocab_size(), ref_vocab);
    assert_eq!(model.num_heads(), ref_heads);

    let mut max_abs_diff: f32 = 0.0;
    for tc in test_cases {
        let prompt = tc["prompt"].as_str().unwrap();
        let mask_position = json_usize(&tc["mask_position"]);
        let ref_tokens: Vec<u32> = tc["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_u32)
            .collect();
        let ref_top10 = tc["top_10"].as_array().unwrap();

        println!("\nPrompt: {prompt:?}  (mask at position {mask_position})");

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

        // Raw logits at the masked position, F32 on CPU.
        let at_mask: Vec<f32> = logits
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .i((0, mask_position))
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut indexed: Vec<(usize, f32)> =
            at_mask.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, ref_item) in ref_top10.iter().enumerate() {
            let ref_idx = json_usize(&ref_item["index"]);
            let ref_logit = json_f32(&ref_item["logit"]);

            let (rust_idx, rust_logit) = indexed[rank];

            assert_eq!(
                rust_idx, ref_idx,
                "rank {rank}: top-10 index mismatch (Rust {rust_idx}, Python {ref_idx}) \
                 for prompt {prompt:?}"
            );

            let diff = (rust_logit - ref_logit).abs();
            assert!(
                diff < abs_diff_bar,
                "rank {rank}: logit abs-diff {diff:.2e} >= {abs_diff_bar:.0e} \
                 (Rust {rust_logit}, Python {ref_logit}) for prompt {prompt:?}"
            );
            if diff > max_abs_diff {
                max_abs_diff = diff;
            }
        }

        println!(
            "  Rust top-1 at mask: ({}, {:.4}) — matches Python",
            indexed[0].0, indexed[0].1
        );
    }

    println!(
        "\nAll {} test cases passed on {device_name}; max abs-diff across all top-10 logits = {:.2e} (bar: {:.0e})",
        test_cases.len(),
        max_abs_diff,
        abs_diff_bar
    );
}

/// Real invariants of the SUBS ancestral sampler on the actual model:
/// determinism by seed, monotone unmasking (carry-over), termination (no
/// `[MASK]` left), and prompt-prefix preservation. (Token *values* can't be
/// matched against `PyTorch` — different RNGs — so we assert falsifiable
/// structural properties instead, on top of the already-exact forward pass.)
#[test]
#[ignore = "requires TheQweaker/mdlm-owt-noflash cached (~648 MiB); run with --ignored"]
#[serial]
fn mdlm_sampler_invariants() {
    let Some(snapshot) = find_snapshot(MODEL_ID) else {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    };
    let device = Device::Cpu;
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let config = MdlmConfig::from_hf_config(&json).unwrap();
    let mask_id = config.mask_token_id;
    let weights = snapshot.join("model.safetensors");
    // SAFETY: safetensors files are not modified during test execution.
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap()
    };
    let model = GenericMdlm::load(config, &device, DType::F32, vb).unwrap();

    // Carry over a 3-token prompt prefix; denoise the rest.
    let prompt: Vec<u32> = vec![464, 3139, 286]; // "The capital of"
    let cfg = DiffusionSamplingConfig::default()
        .with_seq_len(16)
        .with_num_steps(16)
        .with_top_k(Some(50));

    let traj1 =
        candle_mi::diffusion::generate_trajectory(&model, &device, mask_id, &prompt, &cfg).unwrap();
    let traj2 =
        candle_mi::diffusion::generate_trajectory(&model, &device, mask_id, &prompt, &cfg).unwrap();

    // Determinism: same seed -> identical trajectory.
    assert_eq!(traj1, traj2, "same seed must reproduce the trajectory");

    // Trajectory has num_steps + 1 states; each is full-length and keeps the prefix.
    assert_eq!(traj1.len(), cfg.num_steps + 1, "wrong trajectory length");
    for state in &traj1 {
        assert_eq!(state.len(), cfg.seq_len, "wrong state length");
        for (i, &p) in prompt.iter().enumerate() {
            assert_eq!(state[i], p, "prompt prefix changed at position {i}");
        }
    }

    // Monotone unmasking (carry-over): the revealed count never decreases, and
    // the final state is fully revealed.
    let revealed = |s: &[u32]| s.iter().filter(|&&t| t != mask_id).count();
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
        "sampler invariants OK; final state: {:?}",
        traj1.last().unwrap()
    );
}

#[test]
#[ignore = "requires TheQweaker/mdlm-owt-noflash cached (~648 MiB); run with --ignored"]
#[serial]
fn mdlm_owt_forward_parity_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in HF cache");
        return;
    }
    run_mdlm_forward_parity(&Device::Cpu, "CPU", ABS_DIFF_BAR_CPU);
}

#[test]
#[ignore = "requires TheQweaker/mdlm-owt-noflash cached (~648 MiB) and a CUDA device; run with --ignored"]
#[serial]
fn mdlm_owt_forward_parity_gpu() {
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
    run_mdlm_forward_parity(&device, "CUDA", ABS_DIFF_BAR_GPU);
}
