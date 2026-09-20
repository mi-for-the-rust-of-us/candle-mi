// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention routing analysis: how does CLT suppress+inject at the planning
//! site change attention patterns at the output position?
//!
//! ```bash
//! # 426K CLT with suppress+inject (full Figure 13 paradigm, recommended)
//! cargo run --release --features clt,transformer,mmap --example attention_routing -- --suppress L16:13725 --suppress L25:9385
//!
//! # 2.5M CLT with suppress+inject
//! cargo run --release --features clt,transformer,mmap --example attention_routing -- --clt-repo mntss/clt-gemma-2-2b-2.5m --feature L25:82839 --suppress L25:57092 --suppress L23:49923 --suppress L20:77102
//!
//! # Inject only (13x weaker signal, for comparison)
//! cargo run --release --features clt,transformer,mmap --example attention_routing
//!
//! # With JSON output
//! cargo run --release --features clt,transformer,mmap --example attention_routing -- --suppress L16:13725 --suppress L25:9385 --output routing.json
//! ```
//!
//! **What it does:**
//!
//! 1. Runs a **baseline** forward pass capturing [`AttnPattern`] at all layers.
//! 2. Runs a **CLT-steered** forward pass using the Figure 13 API
//!    ([`CrossLayerTranscoder::prepare_hook_injection`]): suppress competing
//!    features + inject alternative, all at the planning site position.
//! 3. For each attention head at each layer, extracts the attention weight
//!    from the **last token → planning site** and computes the delta.
//! 4. Reports which heads re-route attention toward the planning site.
//! 5. Runs a **strength sweep** to measure how attention re-routing scales,
//!    revealing the planning attractor's basin in attention space.
//!
//! This measures "regime 3" — the attention-mediated routing that propagates
//! planning decisions from the planning site to the output position. It fills
//! a specific gap identified by Anthropic: *"attention head routing is
//! invisible to our current approach"*
//! ([Lindsey et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems)).

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::indexing_slicing)]

use candle_core::{DType, Device, Tensor};
use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, MIModel};
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
#[command(name = "attention_routing")]
#[command(
    about = "Attention routing: how does CLT injection redirect attention from output to planning site?"
)]
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

    /// CLT feature to inject, format "L<layer>:<index>"
    #[arg(long, default_value = "L22:10243")]
    feature: String,

    /// CLT features to suppress (negative steering), repeatable.
    /// Format "L<layer>:<index>". Use with inject for full Figure 13 paradigm.
    #[arg(long)]
    suppress: Vec<String>,

    /// Steering strength (matches Figure 13 default)
    #[arg(long, default_value_t = 10.0)]
    strength: f32,

    /// Maximum strength for sweep
    #[arg(long, default_value_t = 20.0)]
    max_strength: f32,

    /// Number of strength steps
    #[arg(long, default_value_t = 12)]
    strength_steps: usize,

    /// Planning site position (auto-detected if not set)
    #[arg(long)]
    planning_site: Option<usize>,

    /// Output JSON file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    clt_repo: String,
    prompt: String,
    feature: String,
    strength: f32,
    planning_site: usize,
    output_position: usize,
    n_layers: usize,
    n_heads: usize,
    /// Per-head attention delta: `[layer][head]`
    head_deltas: Vec<Vec<HeadDelta>>,
    /// Heads sorted by absolute delta (largest first)
    top_heads: Vec<HeadDelta>,
    /// Strength sweep for the top routing head
    strength_sweep: Vec<StrengthPoint>,
}

#[derive(Serialize, Clone)]
struct HeadDelta {
    layer: usize,
    head: usize,
    baseline_attn: f32,
    steered_attn: f32,
    delta: f32,
}

#[derive(Serialize)]
struct StrengthPoint {
    strength: f32,
    /// Attention delta for the top routing head at this strength
    top_head_delta: f32,
    /// Attention delta for all heads (flattened)
    total_routing_shift: f32,
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
    let args = Args::parse();
    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);

    // Parse feature
    let feature = parse_clt_feature(&args.feature)?;

    println!("=== Attention Routing Analysis ===");
    println!("  Model: {}", args.model);
    println!("  CLT: {}", args.clt_repo);
    println!(
        "  Feature: L{}:{} (inject strength {})",
        feature.layer, feature.index, args.strength
    );

    // Load model
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(&Device::cuda_if_available(0).unwrap_or(Device::Cpu))?;

    let t0 = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let n_heads = model.num_heads();
    let device = model.device().clone();

    println!("  Layers: {n_layers}, heads: {n_heads}, device: {device:?}");
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(&device)?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let tokenizer = model.tokenizer().ok_or(candle_mi::MIError::Tokenizer(
        "model has no embedded tokenizer".into(),
    ))?;

    let tokens = tokenizer.encode(prompt)?;
    let seq_len = tokens.len();
    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;

    println!("  Prompt: \"{prompt}\" ({seq_len} tokens)");

    // Find planning site: position of "about" (or user-specified)
    let planning_site = if let Some(pos) = args.planning_site {
        pos
    } else {
        find_planning_site(tokenizer, &tokens, "about")?
    };
    let output_pos = seq_len - 1;
    println!("  Planning site: position {planning_site}");
    println!("  Output position: {output_pos}");

    // Parse suppress features
    let suppress_features: Vec<CltFeatureId> = args
        .suppress
        .iter()
        .map(|s| parse_clt_feature(s))
        .collect::<candle_mi::Result<Vec<_>>>()?;
    if !suppress_features.is_empty() {
        println!("  Suppress features: {suppress_features:?}");
    }

    // Load CLT and cache decoder vectors (Figure 13 API)
    println!("\n  Loading CLT and caching decoder vectors...");
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let mut all_features: Vec<CltFeatureId> = suppress_features.clone();
    all_features.push(feature);
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // Build feature entries for all downstream layers (same as Figure 13)
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> =
        (feature.layer..n_layers).map(|l| (feature, l)).collect();
    println!(
        "  Suppress: {} entries, Inject: {} entries (layers {}–{})",
        suppress_entries.len(),
        inject_entries.len(),
        feature.layer,
        n_layers - 1
    );

    let t1 = Instant::now();

    // -----------------------------------------------------------------------
    // Step 1: Baseline forward pass — capture attention patterns + residuals
    // -----------------------------------------------------------------------
    println!("\n  Step 1: Baseline forward pass...");
    let mut baseline_hooks = HookSpec::new();
    for layer in 0..n_layers {
        baseline_hooks.capture(HookPoint::AttnPattern(layer));
        baseline_hooks.capture(HookPoint::ResidPost(layer));
    }
    let baseline_cache = model.forward(&input, &baseline_hooks)?;

    // Extract baseline attention: last_token → planning_site for all heads
    let baseline_attn = extract_attention_weights(
        &baseline_cache,
        n_layers,
        n_heads,
        output_pos,
        planning_site,
    )?;

    // Diagnostic: baseline residual norms at planning site
    // PROMOTE: residual may be BF16; norm computation needs F32
    let baseline_resid_at_site = baseline_cache
        .require(&HookPoint::ResidPost(n_layers - 1))?
        .get(0)?
        .get(planning_site)?
        .to_dtype(DType::F32)?;
    let baseline_norm: f32 = baseline_resid_at_site
        .sqr()?
        .sum_all()?
        .to_scalar::<f32>()?
        .sqrt();
    println!("    Baseline residual norm at planning site: {baseline_norm:.1}");

    // -----------------------------------------------------------------------
    // Step 2: CLT-steered forward pass — suppress + inject (Figure 13 API)
    // -----------------------------------------------------------------------
    println!(
        "  Step 2: CLT-steered forward pass (strength {})...",
        args.strength
    );
    let steered_attn = run_steered_pass_fig13(
        &model,
        &input,
        &clt,
        &suppress_entries,
        &inject_entries,
        n_layers,
        n_heads,
        seq_len,
        planning_site,
        output_pos,
        args.strength,
        &device,
    )?;

    // -----------------------------------------------------------------------
    // Step 3: Compute per-head deltas
    // -----------------------------------------------------------------------
    println!("  Step 3: Computing attention deltas...");
    let mut head_deltas: Vec<Vec<HeadDelta>> = Vec::with_capacity(n_layers);
    let mut all_deltas: Vec<HeadDelta> = Vec::new();

    for layer in 0..n_layers {
        let mut layer_deltas = Vec::with_capacity(n_heads);
        for head in 0..n_heads {
            let delta = HeadDelta {
                layer,
                head,
                baseline_attn: baseline_attn[layer][head],
                steered_attn: steered_attn[layer][head],
                delta: steered_attn[layer][head] - baseline_attn[layer][head],
            };
            all_deltas.push(delta.clone());
            layer_deltas.push(delta);
        }
        head_deltas.push(layer_deltas);
    }

    // Sort by absolute delta
    all_deltas.sort_by(|a, b| {
        b.delta
            .abs()
            .partial_cmp(&a.delta.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_heads: Vec<HeadDelta> = all_deltas.iter().take(20).cloned().collect();

    // -----------------------------------------------------------------------
    // Step 4: Strength sweep on attention routing
    // -----------------------------------------------------------------------
    println!(
        "  Step 4: Strength sweep ({} steps)...",
        args.strength_steps
    );

    let top_layer = top_heads[0].layer;
    let top_head = top_heads[0].head;
    // CAST: usize → f32, strength_steps is small (≤12)
    #[allow(clippy::as_conversions)]
    let step_size = args.max_strength / args.strength_steps as f32;

    let mut strength_sweep: Vec<StrengthPoint> = Vec::with_capacity(args.strength_steps);

    for step in 1..=args.strength_steps {
        // CAST: usize → f32, step is small
        #[allow(clippy::as_conversions)]
        let strength = step as f32 * step_size;

        let attn = run_steered_pass_fig13(
            &model,
            &input,
            &clt,
            &suppress_entries,
            &inject_entries,
            n_layers,
            n_heads,
            seq_len,
            planning_site,
            output_pos,
            strength,
            &device,
        )?;

        let top_delta = attn[top_layer][top_head] - baseline_attn[top_layer][top_head];

        // Total routing shift: sum of absolute deltas across all heads
        let mut total_shift = 0.0_f32;
        for layer in 0..n_layers {
            for head in 0..n_heads {
                total_shift += (attn[layer][head] - baseline_attn[layer][head]).abs();
            }
        }

        strength_sweep.push(StrengthPoint {
            strength,
            top_head_delta: top_delta,
            total_routing_shift: total_shift,
        });
    }

    let total_time = t1.elapsed();
    println!("\n  Total experiment time: {total_time:.2?}");

    // -----------------------------------------------------------------------
    // Print results
    // -----------------------------------------------------------------------
    println!(
        "\n=== Attention Routing: last token (pos {output_pos}) → planning site (pos {planning_site}) ===\n"
    );

    println!(
        "  {:>5}  {:>4}  {:>12}  {:>12}  {:>10}",
        "Layer", "Head", "Baseline", "Steered", "Delta"
    );
    println!(
        "  {:>5}  {:>4}  {:>12}  {:>12}  {:>10}",
        "-----", "----", "----------", "---------", "--------"
    );

    for d in top_heads.iter().take(20) {
        let marker = if d.delta.abs() > 0.01 { " ***" } else { "" };
        println!(
            "  {:>5}  {:>4}  {:>12.6}  {:>12.6}  {:>+10.6}{}",
            d.layer, d.head, d.baseline_attn, d.steered_attn, d.delta, marker
        );
    }

    println!("\n=== Strength Sweep (top head: L{top_layer}:H{top_head}) ===\n");
    println!(
        "  {:>8}  {:>12}  {:>14}",
        "Strength", "TopHeadDelta", "TotalRouting"
    );
    println!(
        "  {:>8}  {:>12}  {:>14}",
        "--------", "------------", "--------------"
    );
    for pt in &strength_sweep {
        println!(
            "  {:>8.2}  {:>+12.6}  {:>14.6}",
            pt.strength, pt.top_head_delta, pt.total_routing_shift
        );
    }

    // -----------------------------------------------------------------------
    // JSON output
    // -----------------------------------------------------------------------
    if let Some(ref path) = args.output {
        let output = JsonOutput {
            model_id: args.model.clone(),
            clt_repo: args.clt_repo.clone(),
            // BORROW: owned String needed for JSON serialization
            prompt: prompt.to_owned(),
            feature: args.feature.clone(),
            strength: args.strength,
            planning_site,
            output_position: output_pos,
            n_layers,
            n_heads,
            head_deltas,
            top_heads,
            strength_sweep,
        };
        let json = serde_json::to_string_pretty(&output)
            .map_err(|e| candle_mi::MIError::Io(std::io::Error::other(e)))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(candle_mi::MIError::Io)?;
        }
        std::fs::write(path, json).map_err(candle_mi::MIError::Io)?;
        println!("\n  JSON output written to {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Core: steered forward pass with multi-layer CLT injection
// ---------------------------------------------------------------------------

/// Run a forward pass with CLT suppress + inject at the planning site
/// (using the same `prepare_hook_injection` API as Figure 13),
/// capturing attention patterns. Returns `[n_layers][n_heads]` attention weights
/// from `output_pos → planning_site`.
#[allow(clippy::too_many_arguments)]
fn run_steered_pass_fig13(
    model: &MIModel,
    input: &Tensor,
    clt: &CrossLayerTranscoder,
    suppress_entries: &[(CltFeatureId, usize)],
    inject_entries: &[(CltFeatureId, usize)],
    n_layers: usize,
    n_heads: usize,
    seq_len: usize,
    planning_site: usize,
    output_pos: usize,
    strength: f32,
    device: &Device,
) -> candle_mi::Result<Vec<Vec<f32>>> {
    // Build hooks using the Figure 13 API: suppress (negative) + inject (positive)
    let mut hooks = if suppress_entries.is_empty() {
        HookSpec::new()
    } else {
        clt.prepare_hook_injection(suppress_entries, planning_site, seq_len, -strength, device)?
    };
    let inject_hooks =
        clt.prepare_hook_injection(inject_entries, planning_site, seq_len, strength, device)?;
    hooks.extend(&inject_hooks);

    // Capture attention patterns
    for layer in 0..n_layers {
        hooks.capture(HookPoint::AttnPattern(layer));
    }

    let cache = model.forward(input, &hooks)?;
    extract_attention_weights(&cache, n_layers, n_heads, output_pos, planning_site)
}

/// Extract attention weights from `query_pos → key_pos` for all heads at all layers.
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
        // pattern: [1, n_heads, seq_len, seq_len]
        // Extract [n_heads] at [0, :, query_pos, key_pos]
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the token position of a specific word in the prompt.
fn find_planning_site(
    tokenizer: &candle_mi::MITokenizer,
    tokens: &[u32],
    word: &str,
) -> candle_mi::Result<usize> {
    for (i, &tid) in tokens.iter().enumerate() {
        let decoded = tokenizer.decode(&[tid])?;
        let trimmed = decoded.trim();
        if trimmed == word {
            println!("  Auto-detected planning site: position {i} (token \"{trimmed}\")");
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
    let rest = &s[1..];
    let parts: Vec<&str> = rest.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(candle_mi::MIError::Config(format!(
            "CLT feature must be \"L<layer>:<index>\", got \"{s}\""
        )));
    }
    let layer: usize = parts[0]
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid layer number in \"{s}\"")))?;
    let index: usize = parts[1]
        .parse()
        .map_err(|_| candle_mi::MIError::Config(format!("invalid feature index in \"{s}\"")))?;
    Ok(CltFeatureId { layer, index })
}
