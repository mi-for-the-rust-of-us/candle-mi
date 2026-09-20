// SPDX-License-Identifier: MIT OR Apache-2.0

//! Test whether downstream layers can reverse a prolepsis commitment.
//!
//! Establishes a commitment (suppress + inject at the planning site), then
//! injects a contradictory feature at late layers only, sweeping correction
//! strength to test whether the output can be redirected.
//!
//! ```bash
//! cargo run --release --features clt,transformer,mmap --example correction_test \
//!     -- --suppress L16:13725 --suppress L25:9385 \
//!        --output examples/results/correction_test/gemma-426k.json
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **baseline** forward pass (no intervention).
//! 2. Runs a **commitment-only** pass: suppress natural rhyme group +
//!    inject commit feature at the planning site. Validates P(commit_word).
//! 3. Runs a **correction sweep**: for each correction strength (0 to max),
//!    adds a contradictory feature injection at late layers only
//!    (`correct_from_layer..n_layers`) and measures:
//!    - P(commit_word): does the commitment hold?
//!    - P(correct_word): does the correction succeed?
//!    - L21:H5 attention delta: does routing change?
//!    - Total routing shift across all heads.
//!
//! Paper reference:
//! > "What Is the Minimum Architecture for Prolepsis?" (COLM 2026 submission)

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::indexing_slicing)]

use candle_core::{DType, Device, Tensor};
use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel, extract_token_prob};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "correction_test")]
#[command(about = "Test whether downstream layers can reverse a prolepsis commitment")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// CLT repository
    #[arg(long, default_value = "mntss/clt-gemma-2-2b-426k")]
    clt_repo: String,

    /// Prompt text
    #[arg(long)]
    prompt: Option<String>,

    // -- Commitment phase --
    /// CLT feature to inject for initial commitment, "L<layer>:<index>"
    #[arg(long, default_value = "L22:10243")]
    commit_feature: String,

    /// Word produced by the commit feature (for probability extraction)
    #[arg(long, default_value = "around")]
    commit_word: String,

    /// CLT features to suppress (repeatable), "L<layer>:<index>"
    #[arg(long)]
    suppress: Vec<String>,

    /// Commitment steering strength
    #[arg(long, default_value_t = 10.0)]
    commit_strength: f32,

    // -- Correction phase --
    /// CLT feature to inject for correction attempt, "L<layer>:<index>"
    #[arg(long, default_value = "L20:12386")]
    correct_feature: String,

    /// Word produced by the correction feature
    #[arg(long, default_value = "back")]
    correct_word: String,

    /// First layer at which to inject the correction (inclusive)
    #[arg(long, default_value_t = 23)]
    correct_from_layer: usize,

    /// Maximum correction strength for sweep
    #[arg(long, default_value_t = 20.0)]
    max_correct_strength: f32,

    /// Number of correction strength steps
    #[arg(long, default_value_t = 12)]
    correct_steps: usize,

    // -- Shared --
    /// Planning site position (auto-detected from "about" if not set)
    #[arg(long)]
    planning_site: Option<usize>,

    /// Output JSON file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Suppress per-step runtime reporting
    #[arg(long)]
    no_runtime: bool,
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    clt_repo: String,
    prompt: String,
    planning_site: usize,
    output_position: usize,
    n_layers: usize,
    n_heads: usize,

    // Commitment parameters
    commit_feature: String,
    commit_word: String,
    suppress_features: Vec<String>,
    commit_strength: f32,

    // Correction parameters
    correct_feature: String,
    correct_word: String,
    correct_from_layer: usize,

    // Results
    baseline: CorrectionPoint,
    commitment_only: CorrectionPoint,
    correction_sweep: Vec<CorrectionPoint>,

    total_time_secs: f64,
}

#[derive(Serialize)]
struct CorrectionPoint {
    correct_strength: f32,
    p_commit_word: f32,
    p_correct_word: f32,
    top1_token: String,
    top1_prob: f32,
    routing_head_delta: f32,
    total_routing_shift: f32,
    time_secs: f64,
}

// ---------------------------------------------------------------------------
// Default prompt (same as Figure 13 Gemma preset)
// ---------------------------------------------------------------------------

const DEFAULT_PROMPT: &str = "The stars were twinkling in the night,\n\
     The lanterns cast a golden light.\n\
     She wandered in the dark about,\n\
     And found a hidden passage";

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
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let t_total = Instant::now();

    // --- Parse features ---
    let commit_feature = parse_clt_feature(&args.commit_feature)?;
    let correct_feature = parse_clt_feature(&args.correct_feature)?;
    let suppress_features: Vec<CltFeatureId> = args
        .suppress
        .iter()
        .map(|s| parse_clt_feature(s))
        .collect::<candle_mi::Result<Vec<_>>>()?;

    // --- Validate ---
    if args.correct_from_layer < correct_feature.layer {
        return Err(candle_mi::MIError::Config(format!(
            "correct_from_layer ({}) must be >= correct_feature source layer ({})",
            args.correct_from_layer, correct_feature.layer
        )));
    }

    eprintln!("=== Correction Test: Can Downstream Layers Reverse a Commitment? ===\n");
    eprintln!("Model:          {}", args.model);
    eprintln!("CLT:            {}", args.clt_repo);
    eprintln!(
        "Commit:         {} (\"{}\")",
        commit_feature, args.commit_word
    );
    eprintln!(
        "Correct:        {} (\"{}\")",
        correct_feature, args.correct_word
    );
    eprintln!("Correct from:   L{}", args.correct_from_layer);
    eprintln!("Commit str:     {}", args.commit_strength);
    eprintln!("Max correct str:{}", args.max_correct_strength);
    eprintln!("Steps:          {}", args.correct_steps);
    if !suppress_features.is_empty() {
        eprintln!("Suppress:       {suppress_features:?}");
    }

    // --- Load model ---
    eprintln!("\nLoading model...");
    #[cfg(feature = "memory")]
    let mem_before =
        MemorySnapshot::now(&candle_core::Device::cuda_if_available(0).unwrap_or(Device::Cpu))?;

    let t_load = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    let load_time = t_load.elapsed();

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;

    if !args.no_runtime {
        eprintln!("  Load time: {load_time:.2?}");
    }
    eprintln!("  {n_layers} layers, {n_heads} heads/layer, device={device:?}");

    // --- Open CLT + cache steering vectors ---
    eprintln!("Opening CLT: {}...", args.clt_repo);
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let mut all_features: Vec<CltFeatureId> = suppress_features.clone();
    all_features.push(commit_feature);
    all_features.push(correct_feature);
    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // --- Tokenize + find positions ---
    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
    let prompt_with_space = format!("{prompt} ");
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let output_pos = seq_len - 1;

    let planning_site = match args.planning_site {
        Some(pos) => pos,
        None => find_planning_site(tokenizer, &token_ids, "about")?,
    };

    let commit_token_id = tokenizer.find_token_id(&args.commit_word)?;
    let correct_token_id = tokenizer.find_token_id(&args.correct_word)?;
    eprintln!(
        "  Commit token: \"{}\" (id={commit_token_id})",
        args.commit_word
    );
    eprintln!(
        "  Correct token: \"{}\" (id={correct_token_id})",
        args.correct_word
    );
    eprintln!("  Planning site: position {planning_site}");
    eprintln!("  Output position: {output_pos}");

    // --- Build feature entry vectors ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let commit_entries: Vec<(CltFeatureId, usize)> = (commit_feature.layer..n_layers)
        .map(|l| (commit_feature, l))
        .collect();
    let correct_entries: Vec<(CltFeatureId, usize)> = (args.correct_from_layer..n_layers)
        .map(|l| (correct_feature, l))
        .collect();

    eprintln!(
        "  Suppress: {} entries, Commit: {} entries, Correct: {} entries (L{}--L{})",
        suppress_entries.len(),
        commit_entries.len(),
        correct_entries.len(),
        args.correct_from_layer,
        n_layers - 1
    );

    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;

    // --- Step 1: Baseline (no intervention) ---
    eprintln!("\n--- Baseline (no intervention) ---");
    let t_step = Instant::now();
    let mut baseline_hooks = HookSpec::new();
    for layer in 0..n_layers {
        baseline_hooks.capture(HookPoint::AttnPattern(layer));
    }
    let baseline_cache = model.forward(&input, &baseline_hooks)?;
    let baseline_attn = extract_attention_weights(
        &baseline_cache,
        n_layers,
        n_heads,
        output_pos,
        planning_site,
    )?;
    let baseline_p_commit = extract_token_prob(baseline_cache.output(), commit_token_id)?;
    let baseline_p_correct = extract_token_prob(baseline_cache.output(), correct_token_id)?;
    let baseline_top1 = extract_top1(&baseline_cache, output_pos, tokenizer)?;
    let baseline_time = t_step.elapsed();

    eprintln!(
        "  P(\"{}\") = {baseline_p_commit:.6e}, P(\"{}\") = {baseline_p_correct:.6e}, top1 = \"{}\" ({:.4})",
        args.commit_word, args.correct_word, baseline_top1.0, baseline_top1.1
    );

    let baseline_point = CorrectionPoint {
        correct_strength: 0.0,
        p_commit_word: baseline_p_commit,
        p_correct_word: baseline_p_correct,
        top1_token: baseline_top1.0.clone(),
        top1_prob: baseline_top1.1,
        routing_head_delta: 0.0,
        total_routing_shift: 0.0,
        time_secs: baseline_time.as_secs_f64(),
    };

    // --- Step 2: Commitment only (correction strength = 0) ---
    eprintln!("\n--- Commitment only (suppress + inject, no correction) ---");
    let commitment_point = run_correction_pass(
        &model,
        &input,
        &clt,
        &suppress_entries,
        &commit_entries,
        &correct_entries,
        n_layers,
        n_heads,
        seq_len,
        planning_site,
        output_pos,
        args.commit_strength,
        0.0, // no correction
        commit_token_id,
        correct_token_id,
        &baseline_attn,
        tokenizer,
        &args.commit_word,
        &args.correct_word,
        &device,
        !args.no_runtime,
    )?;

    // --- Step 3: Correction strength sweep ---
    eprintln!(
        "\n--- Correction sweep (commit_str={}, correct L{}+) ---",
        args.commit_strength, args.correct_from_layer
    );
    eprintln!(
        "  {:>6} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "str", "P(commit)", "P(correct)", "top1", "H5 delta", "total"
    );

    let mut sweep: Vec<CorrectionPoint> = Vec::with_capacity(args.correct_steps + 1);
    // CAST: usize -> f32, correct_steps fits in f32 (small integer)
    #[allow(clippy::as_conversions)]
    let step_size = args.max_correct_strength / args.correct_steps as f32;

    for step in 0..=args.correct_steps {
        // CAST: usize -> f32, step fits in f32 (small integer)
        #[allow(clippy::as_conversions)]
        let correct_strength = step as f32 * step_size;

        let point = run_correction_pass(
            &model,
            &input,
            &clt,
            &suppress_entries,
            &commit_entries,
            &correct_entries,
            n_layers,
            n_heads,
            seq_len,
            planning_site,
            output_pos,
            args.commit_strength,
            correct_strength,
            commit_token_id,
            correct_token_id,
            &baseline_attn,
            tokenizer,
            &args.commit_word,
            &args.correct_word,
            &device,
            false, // suppress verbose per-step output
        )?;

        eprintln!(
            "  {:>6.1} {:>12.6e} {:>12.6e} {:>10} {:>+10.4} {:>10.4}",
            correct_strength,
            point.p_commit_word,
            point.p_correct_word,
            point.top1_token,
            point.routing_head_delta,
            point.total_routing_shift
        );

        sweep.push(point);
    }

    // --- Summary ---
    let total_time = t_total.elapsed();
    eprintln!("\n=== Summary ===");
    eprintln!(
        "  Baseline P(\"{}\") = {baseline_p_commit:.6e}",
        args.commit_word
    );
    eprintln!(
        "  Committed P(\"{}\") = {:.6e}",
        args.commit_word, commitment_point.p_commit_word
    );
    eprintln!(
        "  At max correction (str={}): P(\"{}\") = {:.6e}, P(\"{}\") = {:.6e}",
        args.max_correct_strength,
        args.commit_word,
        // INDEX: sweep is non-empty (correct_steps >= 0, so sweep has >= 1 entry)
        sweep[sweep.len() - 1].p_commit_word,
        args.correct_word,
        sweep[sweep.len() - 1].p_correct_word
    );
    if !args.no_runtime {
        eprintln!("  Total time: {total_time:.2?}");
    }

    // --- JSON output ---
    let output = JsonOutput {
        model_id: args.model,
        clt_repo: args.clt_repo,
        prompt: prompt.to_owned(),
        planning_site,
        output_position: output_pos,
        n_layers,
        n_heads,
        commit_feature: args.commit_feature,
        // BORROW: .to_owned() for owned String storage
        commit_word: args.commit_word.clone(),
        suppress_features: args.suppress,
        commit_strength: args.commit_strength,
        correct_feature: args.correct_feature,
        // BORROW: .to_owned() for owned String storage
        correct_word: args.correct_word.clone(),
        correct_from_layer: args.correct_from_layer,
        baseline: baseline_point,
        commitment_only: commitment_point,
        correction_sweep: sweep,
        total_time_secs: total_time.as_secs_f64(),
    };

    if let Some(ref path) = args.output {
        write_json(path, &output)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Core: correction pass (commit + optional correction)
// ---------------------------------------------------------------------------

/// Run a forward pass with commitment (suppress + inject) and optional
/// correction (inject at late layers only). Returns measurements.
#[allow(clippy::too_many_arguments)]
fn run_correction_pass(
    model: &MIModel,
    input: &Tensor,
    clt: &CrossLayerTranscoder,
    suppress_entries: &[(CltFeatureId, usize)],
    commit_entries: &[(CltFeatureId, usize)],
    correct_entries: &[(CltFeatureId, usize)],
    n_layers: usize,
    n_heads: usize,
    seq_len: usize,
    planning_site: usize,
    output_pos: usize,
    commit_strength: f32,
    correct_strength: f32,
    commit_token_id: u32,
    correct_token_id: u32,
    baseline_attn: &[Vec<f32>],
    tokenizer: &candle_mi::MITokenizer,
    commit_word: &str,
    correct_word: &str,
    device: &Device,
    verbose: bool,
) -> candle_mi::Result<CorrectionPoint> {
    let t_step = Instant::now();

    // Build commitment hooks: suppress (negative) + inject (positive)
    let mut hooks = if suppress_entries.is_empty() {
        HookSpec::new()
    } else {
        clt.prepare_hook_injection(
            suppress_entries,
            planning_site,
            seq_len,
            -commit_strength,
            device,
        )?
    };
    let commit_hooks = clt.prepare_hook_injection(
        commit_entries,
        planning_site,
        seq_len,
        commit_strength,
        device,
    )?;
    hooks.extend(&commit_hooks);

    // Add correction hooks (positive, late layers only)
    if correct_strength > 0.0 && !correct_entries.is_empty() {
        let correct_hooks = clt.prepare_hook_injection(
            correct_entries,
            planning_site,
            seq_len,
            correct_strength,
            device,
        )?;
        hooks.extend(&correct_hooks);
    }

    // Capture attention patterns
    for layer in 0..n_layers {
        hooks.capture(HookPoint::AttnPattern(layer));
    }

    let cache = model.forward(input, &hooks)?;

    // Extract probabilities
    let p_commit = extract_token_prob(cache.output(), commit_token_id)?;
    let p_correct = extract_token_prob(cache.output(), correct_token_id)?;
    let (top1_token, top1_prob) = extract_top1(&cache, output_pos, tokenizer)?;

    // Extract attention weights and compute deltas
    let attn = extract_attention_weights(&cache, n_layers, n_heads, output_pos, planning_site)?;

    // Routing head delta (L21:H5 for Gemma)
    // INDEX: layer 21 and head 5 exist in Gemma 2 2B (26 layers, 8 heads)
    let routing_head_delta = if n_layers > 21 && n_heads > 5 {
        attn[21][5] - baseline_attn[21][5]
    } else {
        0.0
    };

    // Total routing shift
    let total_routing_shift: f32 = (0..n_layers)
        .flat_map(|l| (0..n_heads).map(move |h| (l, h)))
        .map(|(l, h)| (attn[l][h] - baseline_attn[l][h]).abs())
        .sum();

    let step_time = t_step.elapsed();

    if verbose {
        eprintln!(
            "  P(\"{commit_word}\") = {p_commit:.6e}, P(\"{correct_word}\") = {p_correct:.6e}, \
             top1 = \"{top1_token}\" ({top1_prob:.4})"
        );
    }

    Ok(CorrectionPoint {
        correct_strength,
        p_commit_word: p_commit,
        p_correct_word: p_correct,
        top1_token,
        top1_prob,
        routing_head_delta,
        total_routing_shift,
        time_secs: step_time.as_secs_f64(),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract attention weights from `query_pos -> key_pos` for all heads.
///
/// # Shapes
/// - `AttnPattern(layer)`: `[batch, n_heads, seq_len, seq_len]`
/// - returns: `[n_layers][n_heads]` f32 values
fn extract_attention_weights(
    cache: &candle_mi::HookCache,
    n_layers: usize,
    n_heads: usize,
    query_pos: usize,
    key_pos: usize,
) -> candle_mi::Result<Vec<Vec<f32>>> {
    let mut result: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let pattern = cache.require(&HookPoint::AttnPattern(layer))?;
        let slice = pattern
            .get(0)? // [n_heads, seq_len, seq_len]
            .narrow(1, query_pos, 1)? // [n_heads, 1, seq_len]
            .squeeze(1)? // [n_heads, seq_len]
            .narrow(1, key_pos, 1)? // [n_heads, 1]
            .squeeze(1)? // [n_heads]
            // PROMOTE: attention pattern may be BF16; extraction needs F32
            .to_dtype(DType::F32)?;
        let weights: Vec<f32> = slice.to_vec1()?;
        assert!(
            weights.len() == n_heads,
            "expected {n_heads} heads, got {}",
            weights.len()
        );
        result.push(weights);
    }
    Ok(result)
}

/// Extract top-1 token and its probability from model output.
fn extract_top1(
    cache: &candle_mi::HookCache,
    output_pos: usize,
    tokenizer: &candle_mi::MITokenizer,
) -> candle_mi::Result<(String, f32)> {
    let logits = cache
        .output()
        .get(0)? // [seq_len, vocab]
        .narrow(0, output_pos, 1)? // [1, vocab]
        .squeeze(0)? // [vocab]
        // PROMOTE: logits may be BF16; softmax needs F32
        .to_dtype(DType::F32)?;
    // PROMOTE: softmax requires F32 for numerical stability
    let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let (top_idx, &top_prob) = probs_vec
        .iter()
        .enumerate()
        .max_by(|(_, a): &(usize, &f32), (_, b): &(usize, &f32)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| candle_mi::MIError::Config("empty probability vector".into()))?;
    // CAST: usize -> u32, vocab index fits in u32
    #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    let token_str = tokenizer.decode(&[top_idx as u32])?;
    Ok((token_str, top_prob))
}

/// Find the token position of a specific word in the prompt.
fn find_planning_site(
    tokenizer: &candle_mi::MITokenizer,
    tokens: &[u32],
    word: &str,
) -> candle_mi::Result<usize> {
    for (i, &tid) in tokens.iter().enumerate() {
        let decoded = tokenizer.decode(&[tid])?;
        // BORROW: explicit .trim() on owned String
        let trimmed = decoded.trim();
        if trimmed == word {
            eprintln!("  Auto-detected planning site: position {i} (token \"{trimmed}\")");
            return Ok(i);
        }
    }
    Err(candle_mi::MIError::Tokenizer(format!(
        "could not find \"{word}\" in tokenized prompt"
    )))
}

/// Parse a CLT feature ID from "L<layer>:<index>" format.
fn parse_clt_feature(s: &str) -> candle_mi::Result<CltFeatureId> {
    let s = s.trim();
    if !s.starts_with('L') {
        return Err(candle_mi::MIError::Config(format!(
            "CLT feature must start with 'L', got \"{s}\""
        )));
    }
    // INDEX: s[1..] is safe because we just checked s starts with 'L' (1 byte)
    let rest = &s[1..];
    let parts: Vec<&str> = rest.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(candle_mi::MIError::Config(format!(
            "CLT feature must be \"L<layer>:<index>\", got \"{s}\""
        )));
    }
    // INDEX: parts[0] and parts[1] are safe because we checked parts.len() == 2
    let layer: usize = parts[0]
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid layer number in \"{s}\"")))?;
    let index: usize = parts[1]
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid feature index in \"{s}\"")))?;
    Ok(CltFeatureId { layer, index })
}

/// Write JSON output to a file.
fn write_json(path: &std::path::Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    eprintln!("\nOutput written to {}", path.display());
    Ok(())
}
