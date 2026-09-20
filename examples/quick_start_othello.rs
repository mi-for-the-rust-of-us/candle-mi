// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quick start: `OthelloGpt` masked-diffusion fill-in (a plain GPT-2 backbone).
//!
//! ```bash
//! # Convert a checkpoint first (or point at an export_fixtures.py output dir):
//! python scripts/convert_othello_mdlm.py CKPT.pt out_dir
//! $env:OTHELLO_MDLM_DIR="out_dir"
//! cargo run --features diffusion --release --example quick_start_othello
//! ```
//!
//! **What it does:**
//!
//! 1. Loads `OthelloGpt` from a local converted directory (`model.safetensors` +
//!    `config.json`).  Unlike `MDLM`, the `OthelloMDLM` world model is a bare
//!    `torch.save` checkpoint — not a `HuggingFace` repo with a `model_type` —
//!    so it loads via [`OthelloGpt::load`](candle_mi::OthelloGpt::load) rather
//!    than `MIModel::from_pretrained`.
//! 2. Masks one cell in an illustrative move sequence, runs a single
//!    bidirectional forward pass, applies the `SUBS` rule (forbid `[MASK]`), and
//!    prints the model's fill-in for the masked position.
//! 3. Demonstrates the diffusion `SUBS` sampler
//!    ([`generate_trajectory`](candle_mi::diffusion::generate_trajectory)) reused
//!    **unchanged** on this backbone — the denoising path the commitment study
//!    builds on.
//!
//! The move ids here are illustrative (real games come from the Othello engine);
//! the point is the loader + hook + sampler API surface, not move legality.

use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{
    DiffusionSamplingConfig, HookPoint, HookSpec, MIBackend, OthelloGpt, OthelloGptConfig,
};

fn main() -> candle_mi::Result<()> {
    // 0. Locate a converted OthelloMDLM directory.
    let Some(dir) = std::env::var("OTHELLO_MDLM_DIR").ok().map(PathBuf::from) else {
        println!("Set OTHELLO_MDLM_DIR to a converted OthelloMDLM directory");
        println!("(model.safetensors + config.json from scripts/convert_othello_mdlm.py).");
        return Ok(());
    };
    let weights = dir.join("model.safetensors");
    let config_path = dir.join("config.json");
    if !weights.is_file() || !config_path.is_file() {
        println!(
            "Expected model.safetensors + config.json in {}",
            dir.display()
        );
        return Ok(());
    }

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // 1. Parse the companion config.json and load the weights (buffered/safe —
    //    the world model is ~100 MB, so no mmap is needed).
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| candle_mi::MIError::Config(format!("read config.json: {e}")))?;
    let json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| candle_mi::MIError::Config(format!("parse config.json: {e}")))?;
    let config = OthelloGptConfig::from_hf_config(&json)?;

    let bytes = std::fs::read(&weights)
        .map_err(|e| candle_mi::MIError::Config(format!("read weights: {e}")))?;
    let vb = candle_nn::VarBuilder::from_buffered_safetensors(bytes, DType::F32, &device)?;
    let model = OthelloGpt::load(config, vb)?;

    println!(
        "OthelloGpt: {} blocks, {} hidden, {} heads, vocab {}, device {device:?}",
        model.num_layers(),
        model.hidden_size(),
        model.num_heads(),
        model.vocab_size(),
    );

    // [MASK] is the final vocab id (vocab_size - 1).
    let mask_id = model.vocab_size() - 1;
    let mask_u32 = u32::try_from(mask_id).map_err(|e| {
        candle_mi::MIError::Model(candle_core::Error::Msg(format!("mask id overflow: {e}")))
    })?;

    // 2. Masked forward: mask one cell, capture the final-layer residual, decode.
    //    Illustrative move cells (1..=8); position 4 is masked.
    let mut ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let pos = 4;
    if let Some(slot) = ids.get_mut(pos) {
        *slot = mask_u32;
    }
    println!("\nMasked sequence: {ids:?}  ([MASK] = {mask_id} at position {pos})");

    let input = Tensor::new(&ids[..], &device)?.unsqueeze(0)?; // [1, seq]
    let mut hooks = HookSpec::new();
    let last_layer = model.num_layers().saturating_sub(1);
    hooks.capture(HookPoint::ResidPost(last_layer));
    let cache = model.forward(&input, &hooks)?;
    let logits = cache.output(); // [1, seq, vocab]

    // SUBS: forbid the [MASK] token, then greedily decode the masked position.
    let at_mask = logits.i((0, pos))?; // [vocab]
    let vocab = model.vocab_size();
    let suppress: Vec<f32> = (0..vocab)
        .map(|i| if i == mask_id { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    let suppress = Tensor::new(suppress, &device)?;
    let masked_logits = at_mask.broadcast_add(&suppress)?;
    let pred = candle_mi::sample_token(&masked_logits, 0.0)?;
    println!("OthelloGpt fills [MASK] -> cell {pred}");

    // The per-layer residual stream is what board probes consume.
    let resid = cache.require(&HookPoint::ResidPost(last_layer))?;
    println!("Captured ResidPost({last_layer}): shape {:?}", resid.dims());

    // 3. Diffusion SUBS sampler reuse: denoise a short fully-masked board.
    let cfg = DiffusionSamplingConfig::default()
        .with_seq_len(model.config().block_size.min(16))
        .with_num_steps(8)
        .with_top_k(Some(20));
    let trajectory =
        candle_mi::diffusion::generate_trajectory(&model, &device, mask_u32, &[], &cfg)?;
    let revealed = |state: &[u32]| state.iter().filter(|&&t| t != mask_u32).count();
    let progression: Vec<usize> = trajectory.iter().map(|s| revealed(s)).collect();
    println!(
        "\nSUBS sampler: {} steps over a {}-cell board; revealed-count progression {progression:?}",
        cfg.num_steps, cfg.seq_len,
    );

    Ok(())
}
