// SPDX-License-Identifier: MIT OR Apache-2.0

//! Steering dose-response: calibrate intervention strength and sweep dose levels.
//!
//! ```bash
//! # Run on a specific model
//! cargo run --release --features transformer --example steering_dose_response -- "meta-llama/Llama-3.2-1B"
//!
//! # Run on all cached models (no argument)
//! cargo run --release --features transformer,mmap --example steering_dose_response
//!
//! # With real memory reporting (RAM + VRAM)
//! cargo run --release --features transformer,memory --example steering_dose_response -- "meta-llama/Llama-3.2-1B"
//! ```
//!
//! **What it does:**
//!
//! 1. Loads a transformer, runs a **baseline** forward pass, and captures the
//!    post-softmax attention pattern at a target layer.
//! 2. Measures how strongly the **last token** attends to **position 0**
//!    (the first token) using [`measure_attention_to_targets`].
//! 3. Builds a [`SteeringCalibration`] from the measured baseline, then
//!    sweeps [`DOSE_LEVELS`] — at each dose the attention is rescaled via
//!    [`apply_steering`], the modified pattern is injected via
//!    [`Intervention::Replace`], and the resulting logit shift is measured.
//! 4. Records each point in a [`DoseResponseCurve`] and uses
//!    [`scale_for_target()`](DoseResponseCurve::scale_for_target) to find
//!    the interpolated scale factor for a specific attention target.
//!
//! [`measure_attention_to_targets`]: candle_mi::interp::intervention::measure_attention_to_targets
//! [`apply_steering`]: candle_mi::interp::intervention::apply_steering
//! [`SteeringCalibration`]: candle_mi::SteeringCalibration
//! [`DOSE_LEVELS`]: candle_mi::interp::steering::DOSE_LEVELS
//! [`DoseResponseCurve`]: candle_mi::DoseResponseCurve
//! [`Intervention::Replace`]: candle_mi::Intervention::Replace
//!
//! Pass a model ID to run a single model; omit to run all cached models.
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]

use candle_mi::interp::intervention::{apply_steering, measure_attention_to_targets};
use candle_mi::interp::steering::DOSE_LEVELS;
use candle_mi::{
    DoseResponseCurve, HookPoint, HookSpec, Intervention, MIModel, MITokenizer,
    SUPPORTED_MODEL_TYPES, SteeringCalibration, SteeringResult, SteeringSpec,
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
    let prompt = "The capital of France is";
    let args: Vec<String> = std::env::args().collect();

    // If a model ID is provided, run only that model
    if args.len() > 1 {
        // INDEX: args[1] is safe — checked len() > 1 above
        #[allow(clippy::indexing_slicing)]
        let model_id = &args[1];
        return run_single_model(model_id, prompt);
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
        if let Err(e) = run_model(model_id, snapshot, prompt) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single model (by ID)
// ---------------------------------------------------------------------------

/// Load a model by ID, run steering dose-response, and print results.
fn run_single_model(model_id: &str, prompt: &str) -> candle_mi::Result<()> {
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

    run_steering(&model, tokenizer, prompt)
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
// Per-model steering (cache discovery mode)
// ---------------------------------------------------------------------------

/// Load a model from a snapshot, run steering, and print the analysis.
fn run_model(model_id: &str, snapshot: &Path, prompt: &str) -> candle_mi::Result<()> {
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

    run_steering(&model, &tokenizer, prompt)
}

// ---------------------------------------------------------------------------
// Core steering logic
// ---------------------------------------------------------------------------

/// Run steering dose-response analysis.
fn run_steering(model: &MIModel, tokenizer: &MITokenizer, prompt: &str) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();
    let n_heads = model.num_heads();

    // Encode prompt
    let token_ids = tokenizer.encode(prompt)?;
    let seq_len = token_ids.len();
    let input = candle_core::Tensor::new(&token_ids[..], model.device())?.unsqueeze(0)?; // [1, seq]
    println!("  Prompt: \"{prompt}\" ({seq_len} tokens)");

    // Target a middle layer for steering
    let target_layer = n_layers / 2;
    let last_pos = seq_len - 1;
    let to_positions: Vec<usize> = vec![0]; // attend to first token
    println!("  Steering: layer {target_layer}, edge (pos {last_pos} → pos 0)\n");

    // --- Baseline forward pass ---
    let mut baseline_hooks = HookSpec::new();
    baseline_hooks.capture(HookPoint::AttnPattern(target_layer));

    let t1 = Instant::now();
    let baseline_cache = model.forward(&input, &baseline_hooks)?;
    let baseline_time = t1.elapsed();
    let baseline_logits = baseline_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

    // Retrieve baseline attention pattern: [1, heads, seq, seq]
    let baseline_attn = baseline_cache.require(&HookPoint::AttnPattern(target_layer))?;

    // Measure baseline attention from last position to first position
    let baseline_mean = measure_attention_to_targets(baseline_attn, last_pos, &to_positions)?;
    println!("  Baseline forward: {baseline_time:.2?}");
    println!("  Baseline attention (last → pos 0): {baseline_mean:.6}");

    // --- Calibration ---
    // Use 2× baseline as the "source" reference (a higher-attention condition)
    let source_reference = baseline_mean * 2.0;
    let calibration = SteeringCalibration::new(
        source_reference,
        baseline_mean,
        target_layer,
        1, // single sample
        1,
    );
    println!("  Calibration: source={source_reference:.6}, target={baseline_mean:.6}");
    println!(
        "  Scale to source: {:.2}, attention ratio: {:.2}",
        calibration.scale_factor_to_source(),
        calibration.attention_ratio,
    );

    // --- Dose-response sweep ---
    let mut curve = DoseResponseCurve::new(
        "last→first".to_string(),
        "factual_recall".to_string(),
        target_layer,
        baseline_mean,
    );

    println!(
        "\n  {:>6}  {:>12}  {:>12}  {:>10}",
        "Dose", "Attn Level", "KL Div", "Top-1 Δ"
    );
    println!("  {:->6}  {:->12}  {:->12}  {:->10}", "", "", "", "");

    // Find "Paris" token for logit diff
    let paris_tokens = tokenizer.encode(" Paris")?;
    let paris_id = paris_tokens.last().copied();

    for &dose in DOSE_LEVELS {
        // Build steering spec for this dose level
        let spec = SteeringSpec::scale(dose)
            .layer(target_layer)
            .edge(last_pos, 0);

        // Apply steering to baseline attention pattern (with renormalization)
        let steered_attn = apply_steering(baseline_attn, &spec, n_heads, seq_len)?;

        // Measure steered attention
        let steered_mean = measure_attention_to_targets(&steered_attn, last_pos, &to_positions)?;

        // Forward pass with steered attention injected
        let mut steered_hooks = HookSpec::new();
        steered_hooks.intervene(
            HookPoint::AttnPattern(target_layer),
            Intervention::Replace(steered_attn),
        );

        let steered_cache = model.forward(&input, &steered_hooks)?;
        let steered_logits = steered_cache.output().get(0)?.get(seq_len - 1)?; // [vocab]

        // Build SteeringResult for analysis
        let result = SteeringResult::new(baseline_logits.clone(), steered_logits, spec)
            .with_attention_measurements(baseline_mean, steered_mean);

        let kl = result.kl_divergence()?;

        // Get logit diff for "Paris" if available
        let logit_diff_str = if let Some(pid) = paris_id {
            let diff = result.logit_diff(pid)?;
            format!("{diff:+.4}")
        } else {
            "—".to_string()
        };

        println!("  {dose:>6.1}  {steered_mean:>12.6}  {kl:>12.6}  {logit_diff_str:>10}");

        // Record in dose-response curve
        curve.add_point(dose, steered_mean, kl);
    }

    // --- Interpolation query ---
    // Find the scale needed to double the baseline attention
    let target_attention = baseline_mean * 2.0;
    println!();
    if let Some(scale) = curve.scale_for_target(target_attention) {
        println!("  Interpolation: scale {scale:.2} achieves attention {target_attention:.6}");
    } else {
        println!("  Interpolation: target {target_attention:.6} is outside the measured range");
    }

    // Show dose levels as absolute attention values
    let dose_abs = calibration.dose_levels_absolute();
    println!("\n  Dose levels (absolute attention targets):");
    for (dose, abs_attn) in &dose_abs {
        println!("    dose {dose:.1} → target attention {abs_attn:.6}");
    }

    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rough estimate of F32 weight memory in MB.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}
