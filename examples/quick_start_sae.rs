// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quick start: load a Sparse Autoencoder, encode model activations, print
//! top features and reconstruction error.
//!
//! ```bash
//! cargo run --release --features sae,transformer --example quick_start_sae
//! ```
//!
//! **What it does:**
//!
//! 1. Loads Gemma 2 2B from the `HuggingFace` Hub cache.
//! 2. Loads a Gemma Scope SAE targeting `resid_post` at layer 0.
//! 3. Runs a forward pass capturing the residual stream at the SAE's hook point.
//! 4. Encodes the last-token residual into sparse features.
//! 5. Prints the top-10 active features and reconstruction MSE.
//!
//! Requires `google/gemma-2-2b` cached locally. The Gemma Scope SAE
//! (`google/gemma-scope-2b-pt-res`) is downloaded automatically via
//! `hf-fetch-model`.

// Test/example target: these lints are denied crate-wide for library code,
// where a panic is a bug. Here a failed unwrap IS the failure signal, and
// indexing a fixture whose shape the test itself fixes cannot go out of
// bounds. Same allowance as the other 30+ files under tests/ and examples/.
#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]
#![allow(clippy::indexing_slicing)]

use candle_core::{DType, Device, IndexOp};
use candle_mi::sae::SparseAutoencoder;
use candle_mi::{
    GenericTransformer, HookPoint, HookSpec, MIBackend, MITokenizer, TransformerConfig,
};
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Load an SAE, encode a forward pass, and report top features plus reconstruction error.
fn run() -> candle_mi::Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Device: {device:?}");

    // --- Load Gemma 2 2B ---
    let snapshot =
        find_snapshot("google/gemma-2-2b").expect("google/gemma-2-2b not found in HF cache");

    let config_str =
        std::fs::read_to_string(snapshot.join("config.json")).expect("cannot read config.json");
    let json: serde_json::Value =
        serde_json::from_str(&config_str).expect("cannot parse config.json");
    let config = TransformerConfig::from_hf_config(&json)?;

    let dtype = DType::F32;
    let paths = safetensors_paths(&snapshot);
    // SAFETY: safetensors files are not modified during execution.
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };
    let model = GenericTransformer::load(config, &device, dtype, vb)?;

    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json"))?;

    println!(
        "Model: {} layers, {} hidden",
        model.num_layers(),
        model.hidden_size()
    );

    // --- Load SAE (Gemma Scope, NPZ format) ---
    let sae = SparseAutoencoder::from_pretrained_npz(
        "google/gemma-scope-2b-pt-res",
        "layer_0/width_16k/average_l0_105/params.npz",
        0, // hook_layer
        &device,
    )?;
    println!(
        "SAE: d_in={}, d_sae={}, arch={:?}, hook={}",
        sae.d_in(),
        sae.d_sae(),
        sae.config().architecture,
        sae.config().hook_name,
    );

    // --- Forward pass with hook capture ---
    let prompt = "The capital of France is";
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    println!("\nPrompt: \"{prompt}\" ({seq_len} tokens)");

    let input = candle_core::Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::ResidPost(0));
    let result = model.forward(&input, &hooks)?;

    let resid_post = result.require(&HookPoint::ResidPost(0))?; // [1, seq, 2304]

    // --- Encode last-token residual ---
    let resid_last = resid_post.i((0, seq_len - 1))?; // [2304]
    let sparse = sae.encode_sparse(&resid_last)?;

    println!("\nActive features: {}", sparse.len());
    println!("Top-10 features:");
    for (rank, (fid, val)) in sparse.features.iter().take(10).enumerate() {
        println!("  #{}: feature {} = {val:.4}", rank + 1, fid.index);
    }

    // --- Reconstruction error ---
    let mse = sae.reconstruction_error(resid_post)?;
    println!("\nReconstruction MSE: {mse:.6}");

    // --- Decoder vector for top feature ---
    if let Some((top_fid, _)) = sparse.features.first() {
        let dec_vec = sae.decoder_vector(top_fid.index)?;
        let norm: f32 = dec_vec.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        println!("Top feature {} decoder norm: {norm:.4}", top_fid.index);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the `HuggingFace` Hub cache directory.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(cache).join("hub"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Find the first snapshot directory for a cached model.
fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let cache_dir = hf_cache_dir()?;
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache_dir.join(dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Collect safetensors paths for a snapshot (single file or sharded index).
fn safetensors_paths(snapshot: &std::path::Path) -> Vec<PathBuf> {
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
