// SPDX-License-Identifier: MIT OR Apache-2.0

//! Autoregressive text generation with greedy decoding.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example generate -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example generate
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example generate -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer via
//!    [`MIModel::from_pretrained`](candle_mi::MIModel::from_pretrained).
//! 2. Tokenizes the prompt *"The capital of France is"*.
//! 3. Runs a greedy autoregressive generation loop (temperature 0) for
//!    up to 20 tokens, printing each token as it is produced.
//! 4. Builds a [`GenerationResult`](candle_mi::GenerationResult) and
//!    prints a summary with timing.
//!
//! The forward pass recomputes the full sequence at every step (no KV
//! cache).  This is intentional: candle-mi prioritises interpretability
//! (all activations available for analysis) over inference speed.
//!
//! Pass a model ID to run a single model; omit to run all cached models.
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::{
    GenerationResult, HookSpec, MIModel, MITokenizer, SUPPORTED_MODEL_TYPES, sample_token,
};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    // BORROW: an optional `GEN_PROMPT` env var overrides the default prompt
    // (lets callers pass a multi-line prompt without a CLI flag); `.as_deref()`
    // hands the inner `&str` to the generation loop without cloning.
    let prompt_owned = std::env::var("GEN_PROMPT").ok();
    let prompt = prompt_owned
        .as_deref()
        .unwrap_or("The capital of France is");
    let max_new_tokens: usize = 20;
    let args: Vec<String> = std::env::args().collect();

    // If a model ID is provided, run only that model
    if args.len() > 1 {
        // INDEX: args[1] is safe — checked len() > 1 above
        #[allow(clippy::indexing_slicing)]
        let model_id = &args[1];
        return run_single_model(model_id, prompt, max_new_tokens);
    }

    // Otherwise, discover and run all cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot, prompt, max_new_tokens) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model by ID, generate tokens, and print results.
fn run_single_model(model_id: &str, prompt: &str, max_new_tokens: usize) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB");
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    let prompt_tokens = tokenizer.encode(prompt)?;

    let t1 = Instant::now();
    let result = generate(&model, tokenizer, &prompt_tokens, max_new_tokens)?;
    let gen_time = t1.elapsed();

    println!("\n  --- Generation Result ---");
    println!("  Prompt tokens : {}", result.prompt_tokens.len());
    println!("  Generated     : {}", result.generated_tokens.len());
    println!("  Generation time: {gen_time:.2?}");
    println!("  Full text     : \"{}\"", result.full_text);

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as quick_start_transformer)
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
fn find_snapshot(cache_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache_dir.join(dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Read `model_type` from a cached `config.json`.
fn read_model_type(snapshot: &Path) -> Option<String> {
    let config_path = snapshot.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    // BORROW: explicit .as_str() — serde_json::Value → &str
    json.get("model_type")?.as_str().map(String::from)
}

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)`.
fn discover_cached_models() -> Vec<(String, String, PathBuf)> {
    let Some(cache_dir) = hf_cache_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return Vec::new();
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(dir_name) = name.to_str() else {
            continue;
        };
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let model_id = rest.replacen("--", "/", 1);
        let Some(snapshot) = find_snapshot(&cache_dir, &model_id) else {
            continue;
        };
        let Some(model_type) = read_model_type(&snapshot) else {
            continue;
        };
        // BORROW: explicit .as_str() — String → &str for slice lookup
        if SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            models.push((model_id, model_type, snapshot));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ---------------------------------------------------------------------------
// Per-model generation (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, generate tokens, and print results.
fn run_model(
    model_id: &str,
    snapshot: &Path,
    prompt: &str,
    max_new_tokens: usize,
) -> candle_mi::Result<()> {
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!(
        "  {} layers, {} heads, {} hidden, device: {:?}",
        n_layers,
        n_heads,
        hidden,
        model.device()
    );
    println!("  Estimated F32 weight size: {weight_mb:.0} MB  |  Load: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Memory");
    }

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    let prompt_tokens = tokenizer.encode(prompt)?;

    let t1 = Instant::now();
    let result = generate(&model, &tokenizer, &prompt_tokens, max_new_tokens)?;
    let gen_time = t1.elapsed();

    println!("  --- Result (generation: {gen_time:.2?}) ---");
    println!(
        "  {} prompt + {} generated = {} total tokens",
        result.prompt_tokens.len(),
        result.generated_tokens.len(),
        result.total_tokens
    );
    println!("  Full text: \"{}\"", result.full_text);
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------

/// Run greedy autoregressive generation.
///
/// Recomputes the full sequence at every step so that all intermediate
/// activations remain available for mechanistic interpretability analyses.
fn generate(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> candle_mi::Result<GenerationResult> {
    let mut tokens = prompt_tokens.to_vec();
    let hooks = HookSpec::new();
    let prompt_text = tokenizer.decode(prompt_tokens)?;

    print!("  {prompt_text}");

    for _ in 0..max_new_tokens {
        // Build input: full sequence each step — shape [1, seq_len]
        let input = candle_core::Tensor::new(tokens.as_slice(), model.device())?.unsqueeze(0)?;

        // Forward pass (empty hooks = zero overhead)
        let cache = model.forward(&input, &hooks)?;
        let logits = cache.output(); // [1, seq_len, vocab]

        // Extract last-token logits — shape [vocab]
        let seq_len = tokens.len();
        let last_logits = logits.get(0)?.get(seq_len - 1)?;

        // Greedy decode (temperature 0)
        let next_token = sample_token(&last_logits, 0.0)?;

        // Print token as it is produced
        let token_text = tokenizer.decode(&[next_token])?;
        print!("{token_text}");

        tokens.push(next_token);
    }

    println!();

    // Build result
    let prompt_len = prompt_tokens.len();
    let generated_tokens = tokens.get(prompt_len..).unwrap_or_default().to_vec();
    let full_text = tokenizer.decode(&tokens)?;
    let generated_text = tokenizer.decode(&generated_tokens)?;

    Ok(GenerationResult::new(
        prompt_text,
        full_text,
        generated_text,
        prompt_tokens.to_vec(),
        generated_tokens,
    ))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough estimate of F32 weight memory in MB.
///
/// Uses the formula: `n_layers * (12 * hidden^2) * 4 bytes / 1e6`.
/// This approximates the dominant weight matrices (QKV, O, gate, up, down)
/// per transformer layer. The actual size varies by architecture.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}
