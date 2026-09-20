// SPDX-License-Identifier: MIT OR Apache-2.0

//! Steering convergence: does residual stream injection converge to natural
//! computation, or does the model take a different internal path?
//!
//! ```bash
//! # Factual recall (default: France → Paris)
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B"
//!
//! # Custom prompts — rhyme planning
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --prompt "Twinkle twinkle little star, how I wonder what you" --contrastive "Twinkle twinkle little star, how I wonder where you" --target-token " are"
//!
//! # Batch mode — 20 rhyme groups from plip-rs corpus (~30s on GPU)
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --batch-file examples/results/steering_convergence/batch_rhyme_groups.json --output examples/results/steering_convergence/batch_llama
//!
//! # CLT decoder vector steering (requires clt feature)
//! cargo run --release --features transformer,clt,mmap --example steering_convergence -- "google/gemma-2-2b" --clt "mntss/clt-gemma-2-2b-426k" --feature "L22:10243" --inject-position auto --prompt "..." --contrastive "..."
//!
//! # Auto-detect planning site position
//! cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B" --inject-position auto --prompt "..." --contrastive "..."
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **baseline** forward pass on "The capital of France is", capturing
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint) at every layer.
//! 2. Computes **steering vectors** — either contrastive (France − Germany
//!    residual subtraction) or CLT decoder vectors (`--clt` mode).
//! 3. For each injection layer (0..n_layers), injects the steering vector via
//!    [`Intervention::Add`](candle_mi::Intervention) and captures all layers —
//!    producing an N×N **convergence matrix** of cosine similarities between
//!    steered and natural activations.
//! 4. Identifies the **absorption boundary** — the earliest layer after
//!    injection where cosine similarity exceeds a threshold (default 0.95).
//! 5. Runs a **strength sweep** at the most effective injection layer to show
//!    how increasing perturbation strength shifts the absorption boundary.
//!
//! This answers: when externally controlled, does the model converge to its
//! natural attractor state, or does it find an alternative internal path?
//!
//! Inspired by Jyothir S V, Siddhartha Jalagam, Yann LeCun, and Vlad Sobal.
//! "Gradient-based Planning with World Models." arXiv:2312.17227, 2023.
//! — reframed as MI observation of external control in language models.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_core::{DType, Device, Tensor};
#[cfg(feature = "clt")]
use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::interp::intervention::kl_divergence;
use candle_mi::interp::logit_lens::format_probability;
use candle_mi::{HookPoint, HookSpec, Intervention, MIModel, MITokenizer};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "steering_convergence")]
#[command(
    about = "Steering convergence: does residual stream injection converge to natural computation?"
)]
struct Args {
    /// `HuggingFace` model ID
    #[arg(default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Cosine similarity threshold for absorption boundary
    #[arg(long, default_value_t = 0.95)]
    threshold: f32,

    /// Maximum steering strength for the strength sweep
    #[arg(long, default_value_t = 6.0)]
    max_strength: f32,

    /// Number of strength steps in the sweep
    #[arg(long, default_value_t = 12)]
    strength_steps: usize,

    /// Custom clean prompt (overrides the default "The capital of France is")
    #[arg(long)]
    prompt: Option<String>,

    /// Custom contrastive prompt (must have same token count as --prompt)
    #[arg(long)]
    contrastive: Option<String>,

    /// Custom target token to track (e.g., " are", " mat"); default " Paris"
    #[arg(long)]
    target_token: Option<String>,

    /// Token position to inject at (default: last token). Use "auto" to find
    /// the first differing token between clean and contrastive prompts, or a
    /// number for an explicit position.
    #[arg(long)]
    inject_position: Option<String>,

    /// Write structured JSON output to this file (or directory for --batch-file)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Run all experiments from a batch JSON file (overrides --prompt/--contrastive/--target-token)
    #[arg(long)]
    batch_file: Option<PathBuf>,

    /// CLT repository for feature-based steering (e.g., "mntss/clt-gemma-2-2b-426k").
    /// When set, uses a CLT decoder vector instead of contrastive residual subtraction.
    #[arg(long)]
    clt: Option<String>,

    /// CLT feature to use as steering direction, format "L<layer>:<index>" (e.g., "L22:10243").
    /// Requires --clt.
    #[arg(long)]
    feature: Option<String>,

    /// Target layer for CLT decoder vector extraction. The decoder vector at this layer
    /// is the direction the feature adds to the residual stream. Defaults to n_layers - 1.
    #[arg(long)]
    decoder_layer: Option<usize>,
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    prompt: String,
    contrastive_prompt: String,
    n_layers: usize,
    hidden_size: usize,
    target_token: String,
    target_token_id: u32,
    baseline_p_target: f32,
    baseline_top5: Vec<JsonPrediction>,
    /// `convergence_matrix[inj][obs]` = cosine similarity
    convergence_matrix: Vec<Vec<f32>>,
    layer_summaries: Vec<JsonLayerSummary>,
    best_injection_layer: usize,
    strength_sweep: Vec<JsonStrengthPoint>,
    threshold: f32,
}

#[derive(Serialize)]
struct JsonPrediction {
    token: String,
    token_id: u32,
    probability: f32,
}

#[derive(Serialize)]
struct JsonLayerSummary {
    injection_layer: usize,
    p_target: f32,
    kl_divergence: f32,
    absorption_layer: Option<usize>,
}

#[derive(Serialize)]
struct JsonStrengthPoint {
    strength: f32,
    p_target: f32,
    kl_divergence: f32,
    absorption_layer: Option<usize>,
}

// ---------------------------------------------------------------------------
// Batch file types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct BatchFile {
    experiments: Vec<BatchExperiment>,
}

#[derive(serde::Deserialize)]
struct BatchExperiment {
    group: String,
    prompt: String,
    contrastive: String,
    target_token: String,
    /// Optional inject position: "auto", a number, or absent for last token
    inject_position: Option<String>,
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

/// Parse a CLT feature ID from "L<layer>:<index>" format.
#[cfg(feature = "clt")]
fn parse_clt_feature(s: &str) -> candle_mi::Result<CltFeatureId> {
    let s = s.trim();
    if !s.starts_with('L') {
        return Err(candle_mi::MIError::Config(format!(
            "CLT feature must start with 'L', got \"{s}\""
        )));
    }
    let rest = &s[1..];
    // `.next()` twice rather than `collect()` + `[0]`/`[1]`: CONVENTIONS.md
    // prefers explicit handling over an indexing allow, and `splitn(2, ..)`
    // yields one item when there is no ':', which the `else` arm rejects with
    // the same message the length check used to.
    let mut parts = rest.splitn(2, ':');
    let (Some(layer_str), Some(index_str)) = (parts.next(), parts.next()) else {
        return Err(candle_mi::MIError::Config(format!(
            "CLT feature must be \"L<layer>:<index>\", got \"{s}\""
        )));
    };
    let layer: usize = layer_str
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid layer number in \"{s}\"")))?;
    let index: usize = index_str
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid feature index in \"{s}\"")))?;
    Ok(CltFeatureId { layer, index })
}

const CLEAN_PROMPT: &str = "The capital of France is";

/// Contrastive candidates — first one whose token count matches is used.
const CONTRASTIVE_CANDIDATES: &[&str] = &[
    "The capital of Germany is",
    "The capital of Poland is",
    "The capital of Brazil is",
    "The capital of Russia is",
    "The capital of Canada is",
];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();

    if let Some(ref batch_path) = args.batch_file {
        return run_batch(&args, batch_path);
    }

    run_model(&args.model, &args, None)
}

/// Load model once, then run all experiments from the batch file.
fn run_batch(args: &Args, batch_path: &Path) -> candle_mi::Result<()> {
    let batch_text = std::fs::read_to_string(batch_path).map_err(candle_mi::MIError::Io)?;
    let batch: BatchFile = serde_json::from_str(&batch_text).map_err(|e| {
        candle_mi::MIError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    })?;

    println!(
        "=== Batch mode: {} experiments from {} ===\n",
        batch.experiments.len(),
        batch_path.display()
    );

    // Determine output directory (--output is treated as a directory in batch mode)
    let output_dir = args.output.as_deref();
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir).map_err(candle_mi::MIError::Io)?;
    }

    // Load model once
    let model = MIModel::from_pretrained(&args.model)?;
    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();
    println!(
        "  Model: {}, Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}\n",
        args.model,
        model.device()
    );

    let mut successes = 0_usize;
    let mut skipped = 0_usize;

    for (i, exp) in batch.experiments.iter().enumerate() {
        println!(
            "--- [{}/{}] group: {} ---",
            i + 1,
            batch.experiments.len(),
            exp.group
        );

        // Build per-experiment args overlay
        let exp_args = Args {
            model: args.model.clone(),
            threshold: args.threshold,
            max_strength: args.max_strength,
            strength_steps: args.strength_steps,
            prompt: Some(exp.prompt.clone()),
            contrastive: Some(exp.contrastive.clone()),
            target_token: Some(exp.target_token.clone()),
            inject_position: exp
                .inject_position
                .clone()
                .or_else(|| args.inject_position.clone()),
            output: output_dir.map(|dir| dir.join(format!("{}.json", exp.group))),
            batch_file: None,
            clt: args.clt.clone(),
            feature: args.feature.clone(),
            decoder_layer: args.decoder_layer,
        };

        match run_model_with(&model, &exp_args) {
            Ok(()) => successes += 1,
            Err(e) => {
                println!("  Skipped: {e}\n");
                skipped += 1;
            }
        }
    }

    println!("\n=== Batch complete: {successes} succeeded, {skipped} skipped ===");
    Ok(())
}

// ---------------------------------------------------------------------------
// Core experiment
// ---------------------------------------------------------------------------

fn run_model(model_id: &str, args: &Args, _label: Option<&str>) -> candle_mi::Result<()> {
    println!("=== {model_id} ===");

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(&Device::cuda_if_available(0).unwrap_or(Device::Cpu))?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(model_id)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let hidden = model.hidden_size();

    println!(
        "  Layers: {n_layers}, heads: {n_heads}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    println!(
        "  Estimated F32 weight size: {:.0} MB",
        estimate_weight_mb(n_layers, hidden)
    );
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    run_model_with(&model, args)
}

/// Run the experiment on a pre-loaded model (used by both single and batch modes).
fn run_model_with(model: &MIModel, args: &Args) -> candle_mi::Result<()> {
    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    let device = model.device().clone();

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    // Resolve prompt, contrastive, and target token
    let clean_prompt = args.prompt.as_deref().unwrap_or(CLEAN_PROMPT);
    let clean_tokens = tokenizer.encode(clean_prompt)?;
    let seq_len = clean_tokens.len();

    let (contrastive_prompt, contrastive_tokens) = if let Some(ref c) = args.contrastive {
        let tokens = tokenizer.encode(c)?;
        if tokens.len() != seq_len {
            return Err(candle_mi::MIError::Tokenizer(format!(
                "--contrastive has {} tokens but --prompt has {} tokens (must match)",
                tokens.len(),
                seq_len
            )));
        }
        // BORROW: explicit &str from String for return type
        (c.as_str(), tokens)
    } else {
        find_contrastive_prompt(tokenizer, &clean_tokens)?
    };

    println!("  Prompt: \"{clean_prompt}\" ({seq_len} tokens)");
    println!("  Contrastive: \"{contrastive_prompt}\"");

    // Find target token (custom or default " Paris")
    let target_str = args.target_token.as_deref().unwrap_or(" Paris");
    let target_tokens = tokenizer.encode(target_str)?;
    let target_id = *target_tokens
        .last()
        .ok_or(candle_mi::MIError::Tokenizer(format!(
            "could not encode target token \"{target_str}\""
        )))?;
    let target_text = tokenizer.decode(&[target_id])?;
    println!("  Target token: \"{target_text}\" (id {target_id})");

    // Build input tensors
    let clean_input = Tensor::new(&clean_tokens[..], &device)?.unsqueeze(0)?;
    let contrastive_input = Tensor::new(&contrastive_tokens[..], &device)?.unsqueeze(0)?;

    // Set up capture hooks for all layers
    let mut capture_hooks = HookSpec::new();
    for layer in 0..n_layers {
        capture_hooks.capture(HookPoint::ResidPost(layer));
    }

    // Injection position: auto (first differing token), explicit number, or last token
    let inject_pos = resolve_inject_position(
        args.inject_position.as_deref(),
        &clean_tokens,
        &contrastive_tokens,
        seq_len,
    )?;
    if args.inject_position.is_some() {
        println!("  Inject position: {inject_pos} (planning site mode)");
    }

    let t1 = Instant::now();

    // -----------------------------------------------------------------------
    // Step 1: Baseline run
    // -----------------------------------------------------------------------
    println!("\n  Step 1: Baseline forward pass...");
    let baseline_cache = model.forward(&clean_input, &capture_hooks)?;
    let baseline_logits = last_token_logits(baseline_cache.output(), seq_len)?;
    let baseline_probs = softmax_1d(&baseline_logits)?;
    let baseline_p_target = extract_prob(&baseline_probs, target_id)?;
    let baseline_top5 = top_k_predictions(&baseline_probs, tokenizer, 5)?;

    // Store baseline residuals at last token (for convergence) and inject position (for steering)
    let mut baseline_resid: Vec<Tensor> = Vec::with_capacity(n_layers);
    let mut baseline_at_inject: Vec<Tensor> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = baseline_cache.require(&HookPoint::ResidPost(layer))?;
        // resid: [1, seq_len, hidden] → [hidden]
        baseline_resid.push(resid.get(0)?.get(seq_len - 1)?);
        baseline_at_inject.push(resid.get(0)?.get(inject_pos)?);
    }

    println!(
        "    P(\"{target_text}\") = {}",
        format_probability(baseline_p_target)
    );
    print!("    Top-5:");
    for p in &baseline_top5 {
        print!(
            "  \"{}\" {}",
            p.token.trim(),
            format_probability(p.probability)
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Step 2: Steering vectors (CLT decoder or contrastive subtraction)
    // -----------------------------------------------------------------------
    let steering_vectors: Vec<Tensor> = compute_steering_vectors(
        model,
        args,
        &capture_hooks,
        &contrastive_input,
        &baseline_at_inject,
        n_layers,
        inject_pos,
        &device,
    )?;

    // -----------------------------------------------------------------------
    // Step 3: Injection layer sweep
    // -----------------------------------------------------------------------
    println!("  Step 3: Injection layer sweep ({n_layers} forward passes)...");

    // steered_resids[inj_layer][obs_layer] = Tensor [hidden]
    let mut steered_resids: Vec<Vec<Tensor>> = Vec::with_capacity(n_layers);
    let mut steered_logits_per_layer: Vec<Tensor> = Vec::with_capacity(n_layers);

    let clt_multilayer = args.clt.is_some();

    // INDEX: `inj_layer` and `target` are both in `0..n_layers`, and
    // `steering_vectors` holds exactly one entry per layer.
    #[allow(clippy::indexing_slicing)]
    for inj_layer in 0..n_layers {
        let mut hooks = HookSpec::new();

        if clt_multilayer {
            // CLT mode: inject at all layers from inj_layer to n_layers-1
            // simultaneously, matching how CLT features write to all downstream layers
            // EXPLICIT: the loop variable is a LAYER index, not an incidental container index --
            // it selects `HookPoint::ResidPost(..)` as well as the parallel per-layer vectors, so
            // iterating one of those slices directly would not remove it (CONVENTIONS Rule 9).
            #[allow(clippy::needless_range_loop)]
            for target in inj_layer..n_layers {
                let delta = build_position_delta(
                    &steering_vectors[target],
                    seq_len,
                    hidden,
                    inject_pos,
                    &device,
                )?;
                hooks.intervene(HookPoint::ResidPost(target), Intervention::Add(delta));
            }
        } else {
            // Contrastive mode: single-layer injection
            let delta = build_position_delta(
                &steering_vectors[inj_layer],
                seq_len,
                hidden,
                inject_pos,
                &device,
            )?;
            hooks.intervene(HookPoint::ResidPost(inj_layer), Intervention::Add(delta));
        }

        for layer in 0..n_layers {
            hooks.capture(HookPoint::ResidPost(layer));
        }

        let steered_cache = model.forward(&clean_input, &hooks)?;
        let steered_logits = last_token_logits(steered_cache.output(), seq_len)?;
        steered_logits_per_layer.push(steered_logits);

        // Diagnostic: on first CLT injection, check if residual actually changed
        if clt_multilayer && inj_layer == 0 {
            for check_layer in [n_layers - 4, n_layers - 1] {
                let resid = steered_cache.require(&HookPoint::ResidPost(check_layer))?;
                let at_inject = resid.get(0)?.get(inject_pos)?.to_dtype(DType::F32)?;
                let at_last = resid.get(0)?.get(seq_len - 1)?.to_dtype(DType::F32)?;
                let base_inject = baseline_cache
                    .require(&HookPoint::ResidPost(check_layer))?
                    .get(0)?
                    .get(inject_pos)?
                    .to_dtype(DType::F32)?;
                let base_last = baseline_cache
                    .require(&HookPoint::ResidPost(check_layer))?
                    .get(0)?
                    .get(seq_len - 1)?
                    .to_dtype(DType::F32)?;
                let diff_inject: f32 = (&at_inject - &base_inject)?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                let diff_last: f32 = (&at_last - &base_last)?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                let norm_inject: f32 = base_inject.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                let norm_last: f32 = base_last.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                println!(
                    "    [DIAG] Layer {check_layer}: pos {inject_pos} diff={diff_inject:.6} (norm={norm_inject:.1}), pos {} diff={diff_last:.6} (norm={norm_last:.1})",
                    seq_len - 1
                );
            }
        }

        let mut layer_resids = Vec::with_capacity(n_layers);
        for obs_layer in 0..n_layers {
            let resid = steered_cache.require(&HookPoint::ResidPost(obs_layer))?;
            layer_resids.push(resid.get(0)?.get(seq_len - 1)?);
        }
        steered_resids.push(layer_resids);
    }

    // -----------------------------------------------------------------------
    // Step 4: Compute convergence matrix
    // -----------------------------------------------------------------------
    println!("  Step 4: Computing convergence matrix...");

    let mut convergence_matrix: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    // EXPLICIT: the loop variable is a LAYER index, not an incidental container index --
    // it selects `HookPoint::ResidPost(..)` as well as the parallel per-layer vectors, so
    // iterating one of those slices directly would not remove it (CONVENTIONS Rule 9).
    // INDEX: `inj_layer` and `obs_layer` are both in `0..n_layers`;
    // `baseline_resid` and every row of `steered_resids` hold one entry per layer.
    #[allow(clippy::needless_range_loop, clippy::indexing_slicing)]
    for inj_layer in 0..n_layers {
        let mut row = Vec::with_capacity(n_layers);
        for obs_layer in 0..n_layers {
            let sim = cosine_similarity(
                &baseline_resid[obs_layer],
                &steered_resids[inj_layer][obs_layer],
            )?;
            row.push(sim);
        }
        convergence_matrix.push(row);
    }

    // -----------------------------------------------------------------------
    // Step 5: Absorption boundary + per-layer summary
    // -----------------------------------------------------------------------
    println!("  Step 5: Analyzing absorption boundaries...");

    let mut layer_summaries: Vec<JsonLayerSummary> = Vec::with_capacity(n_layers);
    #[allow(unused_assignments)]
    let mut best_layer = 0_usize;

    // INDEX: `inj_layer` is in `0..n_layers`; `steered_logits_per_layer` and
    // `convergence_matrix` were each pushed once per layer just above.
    #[allow(clippy::indexing_slicing)]
    for inj_layer in 0..n_layers {
        let steered_probs = softmax_1d(&steered_logits_per_layer[inj_layer])?;
        let p_target = extract_prob(&steered_probs, target_id)?;
        let kl = kl_divergence(&baseline_logits, &steered_logits_per_layer[inj_layer])?;

        // Find absorption boundary: first layer AFTER injection where sim >= threshold
        let absorption =
            find_absorption_boundary(&convergence_matrix[inj_layer], inj_layer, args.threshold);

        layer_summaries.push(JsonLayerSummary {
            injection_layer: inj_layer,
            p_target,
            kl_divergence: kl,
            absorption_layer: absorption,
        });
    }

    // Best layer for strength sweep = deepest layer that still achieves absorption.
    // This is the most informative site: right at the absorption boundary, where
    // increasing strength is most likely to push past the attractor's basin.
    best_layer = 0;
    for s in &layer_summaries {
        if s.absorption_layer.is_some() {
            best_layer = s.injection_layer;
        }
    }
    // Fallback: if no layer absorbs, pick the one with lowest KL divergence
    if layer_summaries.iter().all(|s| s.absorption_layer.is_none()) {
        let mut min_kl = f32::MAX;
        for s in &layer_summaries {
            if s.kl_divergence < min_kl {
                min_kl = s.kl_divergence;
                best_layer = s.injection_layer;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 6: Strength sweep at best injection layer
    // -----------------------------------------------------------------------
    println!(
        "  Step 6: Strength sweep at layer {best_layer} ({} steps)...",
        args.strength_steps
    );

    let mut strength_sweep: Vec<JsonStrengthPoint> = Vec::with_capacity(args.strength_steps);
    // CAST: usize -> f32, strength_steps is a small CLI-supplied count
    #[allow(clippy::as_conversions)]
    let step_size = args.max_strength / args.strength_steps as f32;

    // INDEX: `target` and `best_layer` are both in `0..n_layers`, and
    // `steering_vectors` holds exactly one entry per layer.
    #[allow(clippy::indexing_slicing)]
    for step in 1..=args.strength_steps {
        // CAST: usize → f32, step is small (≤12)
        #[allow(clippy::as_conversions)]
        let strength = step as f32 * step_size;

        let mut hooks = HookSpec::new();

        if clt_multilayer {
            // CLT mode: inject at all layers from best_layer to n_layers-1
            // EXPLICIT: the loop variable is a LAYER index, not an incidental container index --
            // it selects `HookPoint::ResidPost(..)` as well as the parallel per-layer vectors, so
            // iterating one of those slices directly would not remove it (CONVENTIONS Rule 9).
            #[allow(clippy::needless_range_loop)]
            for target in best_layer..n_layers {
                let scaled = (&steering_vectors[target] * f64::from(strength))?;
                let delta = build_position_delta(&scaled, seq_len, hidden, inject_pos, &device)?;
                hooks.intervene(HookPoint::ResidPost(target), Intervention::Add(delta));
            }
        } else {
            let scaled = (&steering_vectors[best_layer] * f64::from(strength))?;
            let delta = build_position_delta(&scaled, seq_len, hidden, inject_pos, &device)?;
            hooks.intervene(HookPoint::ResidPost(best_layer), Intervention::Add(delta));
        }

        for layer in 0..n_layers {
            hooks.capture(HookPoint::ResidPost(layer));
        }

        let cache = model.forward(&clean_input, &hooks)?;
        let s_logits = last_token_logits(cache.output(), seq_len)?;
        let s_probs = softmax_1d(&s_logits)?;
        let p_target = extract_prob(&s_probs, target_id)?;
        let kl = kl_divergence(&baseline_logits, &s_logits)?;

        // Compute convergence row for this strength
        let mut conv_row = Vec::with_capacity(n_layers);
        // EXPLICIT: the loop variable is a LAYER index, not an incidental container index --
        // it selects `HookPoint::ResidPost(..)` as well as the parallel per-layer vectors, so
        // iterating one of those slices directly would not remove it (CONVENTIONS Rule 9).
        // INDEX: `obs_layer` is in `0..n_layers` and `baseline_resid` holds one
        // entry per layer.
        #[allow(clippy::needless_range_loop, clippy::indexing_slicing)]
        for obs_layer in 0..n_layers {
            let resid = cache.require(&HookPoint::ResidPost(obs_layer))?;
            let steered_last = resid.get(0)?.get(seq_len - 1)?;
            conv_row.push(cosine_similarity(
                &baseline_resid[obs_layer],
                &steered_last,
            )?);
        }
        let absorption = find_absorption_boundary(&conv_row, best_layer, args.threshold);

        strength_sweep.push(JsonStrengthPoint {
            strength,
            p_target,
            kl_divergence: kl,
            absorption_layer: absorption,
        });
    }

    let total_time = t1.elapsed();
    println!("\n  Total experiment time: {total_time:.2?}");

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------

    print_convergence_matrix(&convergence_matrix, n_layers);
    print_layer_summary(&layer_summaries, &target_text);
    print_strength_sweep(&strength_sweep, best_layer, &target_text);
    print_interpretation(&convergence_matrix, &layer_summaries, args.threshold);

    // -----------------------------------------------------------------------
    // JSON output
    // -----------------------------------------------------------------------
    if let Some(ref path) = args.output {
        let output = JsonOutput {
            model_id: args.model.clone(),
            prompt: clean_prompt.to_owned(),
            contrastive_prompt: contrastive_prompt.to_owned(),
            n_layers,
            hidden_size: hidden,
            target_token: target_text,
            target_token_id: target_id,
            baseline_p_target,
            baseline_top5,
            convergence_matrix,
            layer_summaries,
            best_injection_layer: best_layer,
            strength_sweep,
            threshold: args.threshold,
        };
        write_json(path, &output)?;
        println!("\n  JSON output written to {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Steering vector computation (CLT or contrastive)
// ---------------------------------------------------------------------------

/// Compute steering vectors: either from a CLT decoder vector or via contrastive
/// residual subtraction. Returns one `[hidden]` vector per layer.
///
/// In CLT mode, the same decoder vector is used for all injection layers.
/// In contrastive mode, each layer gets its own (baseline − contrastive) vector.
#[allow(clippy::too_many_arguments)]
fn compute_steering_vectors(
    model: &MIModel,
    args: &Args,
    capture_hooks: &HookSpec,
    contrastive_input: &Tensor,
    baseline_at_inject: &[Tensor],
    n_layers: usize,
    inject_pos: usize,
    device: &Device,
) -> candle_mi::Result<Vec<Tensor>> {
    #[cfg(feature = "clt")]
    if let Some(ref clt_repo) = args.clt {
        return compute_clt_steering(clt_repo, args, n_layers, device);
    }

    let _ = (args, device); // used only in clt mode

    // Contrastive mode (default)
    println!("  Step 2: Contrastive run → extracting steering vectors...");
    let contrastive_cache = model.forward(contrastive_input, capture_hooks)?;

    let mut steering_vectors: Vec<Tensor> = Vec::with_capacity(n_layers);
    // EXPLICIT: the loop variable is a LAYER index, not an incidental container index --
    // it selects `HookPoint::ResidPost(..)` as well as the parallel per-layer vectors, so
    // iterating one of those slices directly would not remove it (CONVENTIONS Rule 9).
    // INDEX: `layer` is in `0..n_layers` and `baseline_at_inject` holds one
    // entry per layer.
    #[allow(clippy::needless_range_loop, clippy::indexing_slicing)]
    for layer in 0..n_layers {
        let contrastive_resid = contrastive_cache.require(&HookPoint::ResidPost(layer))?;
        // contrastive_resid: [1, seq_len, hidden] → [hidden] at inject position
        let contrastive_at_pos = contrastive_resid.get(0)?.get(inject_pos)?;
        // steering = baseline − contrastive at inject position
        steering_vectors.push((&baseline_at_inject[layer] - &contrastive_at_pos)?);
    }
    Ok(steering_vectors)
}

/// Load a CLT and extract per-layer decoder vectors to use as steering directions.
///
/// Each injection layer L gets the decoder vector for target layer L, so the
/// steering direction matches the layer's internal geometry. For layers below
/// the feature's source layer, a zero vector is used (the feature doesn't
/// write to those layers).
///
/// If `--decoder-layer` is set, that single decoder vector is used for all
/// layers (legacy single-vector mode).
#[cfg(feature = "clt")]
fn compute_clt_steering(
    clt_repo: &str,
    args: &Args,
    n_layers: usize,
    device: &Device,
) -> candle_mi::Result<Vec<Tensor>> {
    let feature_str = args.feature.as_deref().ok_or_else(|| {
        candle_mi::MIError::Config("--clt requires --feature (e.g., --feature L22:10243)".into())
    })?;
    let feature = parse_clt_feature(feature_str)?;

    println!("  Step 2: CLT decoder vector extraction...");
    println!("    CLT: {clt_repo}");

    let mut clt = CrossLayerTranscoder::open(clt_repo)?;
    let d_model = clt.config().d_model;

    if let Some(decoder_layer) = args.decoder_layer {
        // Single-vector mode (legacy): same vector for all layers
        println!(
            "    Feature: L{}:{}, single decoder target layer: {decoder_layer}",
            feature.layer, feature.index
        );
        // decoder_vector: [d_model]
        let decoder_vec = clt.decoder_vector(&feature, decoder_layer, device)?;
        // PROMOTE: CLT decoder weights are BF16; steering needs F32
        let decoder_vec = decoder_vec.to_dtype(DType::F32)?;
        let norm: f32 = decoder_vec.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        println!("    Decoder vector norm: {norm:.4}");
        let steering_vectors: Vec<Tensor> = (0..n_layers).map(|_| decoder_vec.clone()).collect();
        return Ok(steering_vectors);
    }

    // Per-layer mode: each injection layer L gets the decoder vector for target layer L
    println!(
        "    Feature: L{}:{}, per-layer decoder vectors (layers {}..{})",
        feature.layer,
        feature.index,
        feature.layer,
        n_layers - 1
    );

    let mut steering_vectors: Vec<Tensor> = Vec::with_capacity(n_layers);
    let zero = Tensor::zeros(d_model, DType::F32, device)?;

    for target_layer in 0..n_layers {
        if target_layer < feature.layer {
            // Feature doesn't write to layers below its source
            steering_vectors.push(zero.clone());
        } else {
            // decoder_vector: [d_model]
            let vec = clt.decoder_vector(&feature, target_layer, device)?;
            // PROMOTE: CLT decoder weights are BF16; steering needs F32
            let vec = vec.to_dtype(DType::F32)?;
            steering_vectors.push(vec);
        }
    }

    // Report norm range for the active layers
    let mut min_norm = f32::MAX;
    let mut max_norm = 0.0_f32;
    for (i, v) in steering_vectors.iter().enumerate() {
        if i >= feature.layer {
            let norm: f32 = v.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            if norm < min_norm {
                min_norm = norm;
            }
            if norm > max_norm {
                max_norm = norm;
            }
        }
    }
    println!("    Decoder vector norms: {min_norm:.4}..{max_norm:.4}");

    Ok(steering_vectors)
}

// ---------------------------------------------------------------------------
// Helpers: tensor operations
// ---------------------------------------------------------------------------

/// Cosine similarity between two `[hidden]` tensors.
fn cosine_similarity(a: &Tensor, b: &Tensor) -> candle_mi::Result<f32> {
    // PROMOTE: F32 for dot product precision
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let dot: f32 = (&a * &b)?.sum_all()?.to_scalar()?;
    let norm_a: f32 = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let norm_b: f32 = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// Build a `[1, seq_len, hidden]` delta tensor with `vector` at `position`,
/// zeros elsewhere. Uses the narrow+cat pattern from CLT injection.
///
/// # Shapes
/// - `vector`: `[hidden]`
/// - returns: `[1, seq_len, hidden]`
fn build_position_delta(
    vector: &Tensor,
    seq_len: usize,
    hidden: usize,
    position: usize,
    device: &Device,
) -> candle_mi::Result<Tensor> {
    let zeros = Tensor::zeros((1, seq_len, hidden), DType::F32, device)?;
    let scaled_3d = vector.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, hidden]

    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if position > 0 {
        parts.push(zeros.narrow(1, 0, position)?);
    }
    parts.push(scaled_3d);
    if position + 1 < seq_len {
        parts.push(zeros.narrow(1, position + 1, seq_len - position - 1)?);
    }

    Ok(Tensor::cat(&parts, 1)?)
}

/// Extract logits for the last token position.
///
/// # Shapes
/// - `output`: `[1, seq_len, vocab]`
/// - returns: `[vocab]`
fn last_token_logits(output: &Tensor, seq_len: usize) -> candle_mi::Result<Tensor> {
    Ok(output.get(0)?.get(seq_len - 1)?)
}

/// Softmax over a 1D logits tensor.
///
/// # Shapes
/// - `logits`: `[vocab]`
/// - returns: `[vocab]` (probabilities)
fn softmax_1d(logits: &Tensor) -> candle_mi::Result<Tensor> {
    // PROMOTE: F32 for softmax numerical stability
    let logits = logits.to_dtype(DType::F32)?;
    let max_val: f64 = logits.max(0)?.to_scalar::<f32>()?.into();
    let shifted = (logits - max_val)?;
    let exp = shifted.exp()?;
    let sum: f64 = exp.sum_all()?.to_scalar::<f32>()?.into();
    Ok((exp / sum)?)
}

/// Extract probability for a specific token ID from a probability tensor.
fn extract_prob(probs: &Tensor, token_id: u32) -> candle_mi::Result<f32> {
    // CAST: u32 → usize, token_id fits in usize
    #[allow(clippy::as_conversions)]
    let idx = token_id as usize;
    Ok(probs.get(idx)?.to_scalar()?)
}

/// Resolve `--inject-position`: "auto" finds the first differing token,
/// a number uses that position, `None` defaults to last token.
fn resolve_inject_position(
    arg: Option<&str>,
    clean_tokens: &[u32],
    contrastive_tokens: &[u32],
    seq_len: usize,
) -> candle_mi::Result<usize> {
    match arg {
        None => Ok(seq_len - 1),
        Some("auto") => {
            for (i, (a, b)) in clean_tokens
                .iter()
                .zip(contrastive_tokens.iter())
                .enumerate()
            {
                if a != b {
                    println!("  Auto-detected inject position: {i} (first differing token)");
                    return Ok(i);
                }
            }
            Err(candle_mi::MIError::Intervention(
                "auto: clean and contrastive prompts have identical tokens".into(),
            ))
        }
        Some(s) => {
            let pos: usize = s.parse().map_err(|_| {
                candle_mi::MIError::Intervention(format!(
                    "--inject-position must be \"auto\" or a number, got \"{s}\""
                ))
            })?;
            if pos >= seq_len {
                return Err(candle_mi::MIError::Intervention(format!(
                    "--inject-position {pos} is out of bounds (seq_len={seq_len})"
                )));
            }
            Ok(pos)
        }
    }
}

/// Get top-k predictions from a probability tensor.
fn top_k_predictions(
    probs: &Tensor,
    tokenizer: &MITokenizer,
    k: usize,
) -> candle_mi::Result<Vec<JsonPrediction>> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = Vec::with_capacity(k);
    for &(idx, prob) in indexed.iter().take(k) {
        // CAST: usize → u32, vocab indices fit in u32
        #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
        let token_id = idx as u32;
        let token = tokenizer.decode(&[token_id])?;
        results.push(JsonPrediction {
            token,
            token_id,
            probability: prob,
        });
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers: analysis
// ---------------------------------------------------------------------------

/// Find the absorption boundary: the earliest observation layer strictly after
/// `injection_layer` where cosine similarity >= threshold.
fn find_absorption_boundary(
    convergence_row: &[f32],
    injection_layer: usize,
    threshold: f32,
) -> Option<usize> {
    convergence_row
        .iter()
        .enumerate()
        .skip(injection_layer + 1)
        .find(|&(_, &sim)| sim >= threshold)
        .map(|(obs, _)| obs)
}

/// Find a contrastive prompt whose token count matches the clean prompt.
fn find_contrastive_prompt<'a>(
    tokenizer: &MITokenizer,
    clean_tokens: &[u32],
) -> candle_mi::Result<(&'a str, Vec<u32>)> {
    let target_len = clean_tokens.len();
    for &candidate in CONTRASTIVE_CANDIDATES {
        let tokens = tokenizer.encode(candidate)?;
        if tokens.len() == target_len {
            return Ok((candidate, tokens));
        }
    }
    Err(candle_mi::MIError::Tokenizer(format!(
        "no contrastive candidate matches clean prompt length ({target_len} tokens)"
    )))
}

// ---------------------------------------------------------------------------
// Helpers: output
// ---------------------------------------------------------------------------

fn print_convergence_matrix(matrix: &[Vec<f32>], n_layers: usize) {
    println!("\n=== Convergence Matrix (cosine similarity: steered vs natural) ===");
    println!("Rows = injection layer, Cols = observation layer\n");

    // Header
    print!("  Inj\\Obs");
    for obs in 0..n_layers {
        print!("  {obs:>5}");
    }
    println!();
    print!("  -------");
    for _ in 0..n_layers {
        print!("  -----");
    }
    println!();

    // Rows
    for (inj, row) in matrix.iter().enumerate() {
        print!("  {inj:>5}  ");
        for &sim in row {
            if sim >= 0.999 {
                print!("  1.000");
            } else {
                print!("  {sim:.3}");
            }
        }
        println!();
    }
}

fn print_layer_summary(summaries: &[JsonLayerSummary], target_text: &str) {
    println!("\n=== Injection Layer Summary ===");
    println!(
        "  {:>5}  {:>10}  {:>8}  {:>12}",
        "Layer",
        format!("P({target_text})"),
        "KL Div",
        "Absorption"
    );
    println!(
        "  {:>5}  {:>10}  {:>8}  {:>12}",
        "-----", "----------", "--------", "----------"
    );

    for s in summaries {
        // BORROW: owned String needed for format alignment
        let absorption = s
            .absorption_layer
            .map_or_else(|| "--".to_owned(), |l| format!("Layer {l}"));
        println!(
            "  {:>5}  {:>10}  {:>8.4}  {:>12}",
            s.injection_layer,
            format_probability(s.p_target),
            s.kl_divergence,
            absorption,
        );
    }
}

fn print_strength_sweep(sweep: &[JsonStrengthPoint], best_layer: usize, target_text: &str) {
    println!("\n=== Strength Sweep at Layer {best_layer} ===");
    println!(
        "  {:>8}  {:>10}  {:>8}  {:>12}",
        "Strength",
        format!("P({target_text})"),
        "KL Div",
        "Absorption"
    );
    println!(
        "  {:>8}  {:>10}  {:>8}  {:>12}",
        "--------", "----------", "--------", "----------"
    );

    for pt in sweep {
        // BORROW: owned String needed for format alignment
        let absorption = pt
            .absorption_layer
            .map_or_else(|| "--".to_owned(), |l| format!("Layer {l}"));
        println!(
            "  {:>8.2}  {:>10}  {:>8.4}  {:>12}",
            pt.strength,
            format_probability(pt.p_target),
            pt.kl_divergence,
            absorption,
        );
    }
}

fn print_interpretation(matrix: &[Vec<f32>], summaries: &[JsonLayerSummary], threshold: f32) {
    println!("\n=== Interpretation ===");

    // Count how many injection layers achieve absorption
    let absorbed: Vec<&JsonLayerSummary> = summaries
        .iter()
        .filter(|s| s.absorption_layer.is_some())
        .collect();

    // CAST: usize → f32, n_layers is small
    #[allow(clippy::as_conversions)]
    let frac = absorbed.len() as f32 / summaries.len() as f32 * 100.0;

    println!(
        "  {}/{} injection layers achieve absorption (threshold {threshold:.2})",
        absorbed.len(),
        summaries.len()
    );
    println!("  ({frac:.0}% of layers converge back to natural computation)\n");

    if !absorbed.is_empty() {
        // Average absorption depth (layers after injection)
        // CAST: usize → f32 throughout; layer indices and the `absorbed` count are
        // both bounded by n_layers, far inside f32's exact-integer range.
        #[allow(clippy::as_conversions)]
        let avg_depth: f32 = absorbed
            .iter()
            .map(|s| s.absorption_layer.unwrap_or(0) as f32 - s.injection_layer as f32)
            .sum::<f32>()
            / absorbed.len() as f32;

        println!("  Average absorption depth: {avg_depth:.1} layers after injection");
        println!(
            "  → The model absorbs external perturbations within ~{:.0} layers on average.",
            avg_depth.ceil()
        );
    }

    // Check diagonal pattern
    let n = matrix.len();
    if n >= 4 {
        // `matrix` is square when built by `run_convergence`, but this function
        // takes a plain slice, so read the three probe cells through `.get()`
        // rather than assume it.
        let cell = |row: usize, col: usize| matrix.get(row).and_then(|r| r.get(col)).copied();
        let (Some(early_sim), Some(mid_sim), Some(late_sim)) = (
            cell(0, n / 2),     // inject early, observe middle
            cell(n / 2, n - 1), // inject middle, observe late
            cell(n - 2, n - 1), // inject late, observe last
        ) else {
            return;
        };

        println!();
        if early_sim > threshold && mid_sim > threshold {
            println!("  Pattern: ATTRACTOR — the model converges to its natural state");
            println!("  regardless of where the steering is injected.");
        } else if early_sim > threshold && late_sim < 0.9 {
            println!("  Pattern: DEPTH-DEPENDENT — early injections converge (the model");
            println!("  has enough layers to course-correct), but late injections diverge.");
        } else if early_sim < 0.9 && mid_sim < 0.9 {
            println!("  Pattern: MULTIPLE PATHS — the model reaches the same output");
            println!("  through different internal trajectories.");
        } else {
            println!("  Pattern: MIXED — convergence depends on the injection site.");
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: utilities
// ---------------------------------------------------------------------------

/// Rough estimate of F32 weight size in MB.
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    // CAST: usize → f64, values are small
    #[allow(clippy::as_conversions)]
    let params =
        (hidden as f64).mul_add(128_000.0, 12.0 * n_layers as f64 * (hidden as f64).powi(2));
    params * 4.0 / 1_048_576.0
}

fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Io(std::io::Error::other(e)))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(candle_mi::MIError::Io)?;
    }
    std::fs::write(path, json).map_err(candle_mi::MIError::Io)?;
    Ok(())
}
