// SPDX-License-Identifier: MIT OR Apache-2.0

//! CounterFact activation patching: replicate the Transluce protocol
//! (Li et al., 2025) on consumer hardware.
//!
//! ```bash
//! # Default: Llama 3.2 1B, 4-layer blocks
//! cargo run --release --features transformer --example counterfact_patching
//!
//! # Quick test with 3 prompts
//! cargo run --release --features transformer --example counterfact_patching -- --limit 3
//!
//! # With JSON output
//! cargo run --release --features transformer --example counterfact_patching -- --output examples/results/counterfact_patching/llama-3.2-1b.json
//!
//! # Finer layer granularity (2-layer blocks)
//! cargo run --release --features transformer --example counterfact_patching -- --block-size 2
//! ```
//!
//! **What it does:**
//!
//! 1. For each `CounterFact` prompt pair (original fact + counterfactual
//!    subject), runs both through the model via
//!    [`MIModel::forward_text`](candle_mi::MIModel::forward_text), capturing
//!    [`HookPoint::ResidPost`](candle_mi::HookPoint) at every layer of the
//!    counterfactual pass only. That pass is the donor; the original is the
//!    recipient and is edited in flight, so it needs no captures.
//! 2. Classifies tokens by role (subject, relation, other) using
//!    [`EncodingWithOffsets::label_spans`](candle_mi::EncodingWithOffsets::label_spans).
//! 3. For each (layer-block, token-position) combination, replaces the
//!    original residual stream at that position with the counterfactual's
//!    residual across all layers in the block via
//!    [`Intervention::PatchAt`](candle_mi::Intervention).
//! 4. Checks if the greedy next token changed (`is_different`), matching
//!    the Transluce protocol exactly.
//! 5. Prints per-prompt tables and aggregate statistics by layer block
//!    and token type.
//!
//! This replicates the data-generation protocol from:
//!
//! > Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt,
//! > and Jacob Andreas. "Training Language Models to Explain Their Own
//! > Computations." arXiv:2511.08579, 2025.
//! > <https://arxiv.org/abs/2511.08579>

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]

use candle_mi::interp::intervention::kl_divergence;
use candle_mi::{FullActivationCache, HookPoint, HookSpec, Intervention, MIModel, sample_token};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "counterfact_patching")]
#[command(about = "CounterFact activation patching (Li et al., 2025 / Transluce protocol)")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Number of contiguous layers per patching block
    #[arg(long, default_value = "4")]
    block_size: usize,

    /// Run only the first N prompt pairs (for quick testing)
    #[arg(long)]
    limit: Option<usize>,

    /// Path to `CounterFact` prompt pairs JSON file
    #[arg(long, default_value = "data/counterfact_transluce_test.json")]
    data: PathBuf,

    /// Suppress per-pair and total runtime reporting
    #[arg(long)]
    no_runtime: bool,
}

// ---------------------------------------------------------------------------
// CounterFact prompt pairs (loaded from JSON)
// ---------------------------------------------------------------------------

/// A prompt pair from the Transluce `CounterFact` dataset.
#[derive(serde::Deserialize)]
struct CounterFactPair {
    /// The original prompt (full text including forced-choice template).
    original_prompt: String,
    /// The counterfactual prompt (different subject, same template).
    counterfactual_prompt: String,
    /// Ground-truth answer for the original prompt.
    gt_original_target: String,
    /// Ground-truth answer for the counterfactual prompt.
    gt_counterfactual_target: String,
}

impl CounterFactPair {
    /// Extract the "fact" portion (everything before the first `\n\n`).
    fn fact_span_len(&self) -> usize {
        self.original_prompt
            .find("\n\n")
            .unwrap_or(self.original_prompt.len())
    }

    /// Short label for display (first 30 chars of the fact part).
    fn label(&self) -> String {
        let end = self.fact_span_len().min(30);
        // BORROW: chars — safe Unicode truncation
        self.original_prompt.chars().take(end).collect()
    }
}

/// Load prompt pairs from a JSON file.
fn load_pairs(path: &Path) -> candle_mi::Result<Vec<CounterFactPair>> {
    let data = std::fs::read_to_string(path).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
    })?;
    let pairs: Vec<CounterFactPair> = serde_json::from_str(&data).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to parse {}: {e}", path.display()))
    })?;
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    block_size: usize,
    n_layers: usize,
    n_blocks: usize,
    total_time_secs: f64,
    results: Vec<PromptResult>,
    summary: Summary,
}

#[derive(Serialize)]
struct PromptResult {
    label: String,
    original_prompt: String,
    counterfactual_prompt: String,
    gt_original_target: String,
    gt_counterfactual_target: String,
    input_tokens: Vec<String>,
    original_continuation: String,
    cf_continuation: String,
    time_secs: f64,
    patches: Vec<PatchResult>,
}

#[derive(Serialize)]
struct PatchResult {
    layer: Vec<usize>,
    orig_pos: usize,
    cf_pos: usize,
    ablated_continuation: String,
    is_different: bool,
    kl_from_original: f32,
    token_type: String,
}

#[derive(Serialize)]
struct Summary {
    pct_different_by_block: Vec<BlockSummary>,
    pct_different_by_token_type: Vec<TypeSummary>,
    total_patches: usize,
    total_different: usize,
}

#[derive(Serialize)]
struct BlockSummary {
    block: usize,
    layers: Vec<usize>,
    pct_different: f32,
    count_different: usize,
    count_total: usize,
}

#[derive(Serialize)]
struct TypeSummary {
    token_type: String,
    pct_different: f32,
    count_different: usize,
    count_total: usize,
}

// ---------------------------------------------------------------------------
// Core patching logic
// ---------------------------------------------------------------------------

/// Build a `HookSpec` that overwrites `ResidPost` at every layer in
/// `block_layers` with the counterfactual activation at `cf_pos`, inserted
/// at `orig_pos` in the original sequence.
///
/// Only the one position is touched; every other position of the original
/// forward pass reaches the hook point and passes through unchanged, which is
/// why the original run's own activations are not needed here.
///
/// # Shapes
///
/// - `cf_acts`: `FullActivationCache` with `[seq_len, hidden]` per layer
/// - Result: `HookSpec` with `len(block_layers)` interventions
fn build_block_patch(
    cf_acts: &FullActivationCache,
    block_layers: &[usize],
    orig_pos: usize,
    cf_pos: usize,
) -> candle_mi::Result<HookSpec> {
    let mut hooks = HookSpec::new();

    for &layer in block_layers {
        let cf_vector = cf_acts.get_position(layer, cf_pos)?; // [hidden]

        hooks.intervene(
            HookPoint::ResidPost(layer),
            Intervention::PatchAt {
                position: orig_pos,
                value: cf_vector,
            },
        );
    }

    Ok(hooks)
}

/// Compute layer blocks for a model with `n_layers` layers and given `block_size`.
fn compute_blocks(n_layers: usize, block_size: usize) -> Vec<Vec<usize>> {
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < n_layers {
        let end = (start + block_size).min(n_layers);
        blocks.push((start..end).collect());
        start = end;
    }
    blocks
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();

    // Load model
    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    println!("Loading {}...", args.model);
    let t0 = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    let load_time = t0.elapsed();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Config("model has no tokenizer".into()))?;

    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    let blocks = compute_blocks(n_layers, args.block_size);
    let n_blocks = blocks.len();

    println!(
        "  Layers: {n_layers}, hidden: {hidden}, device: {:?}",
        model.device()
    );
    if !args.no_runtime {
        println!("  Load time: {load_time:.2?}");
    }
    println!("  Block size: {}, blocks: {n_blocks}", args.block_size);

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(model.device())?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    let pairs = load_pairs(&args.data)?;
    println!("  Data: {} pairs from {}", pairs.len(), args.data.display());
    let n_pairs = args.limit.unwrap_or(pairs.len()).min(pairs.len());

    // Aggregate counters for summary
    let mut block_different: Vec<usize> = vec![0; n_blocks];
    let mut block_total: Vec<usize> = vec![0; n_blocks];
    let mut type_different: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut type_total: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut total_patches: usize = 0;
    let mut total_different: usize = 0;

    let mut prompt_results: Vec<PromptResult> = Vec::with_capacity(n_pairs);

    let total_start = Instant::now();

    for (pair_idx, pair) in pairs.iter().take(n_pairs).enumerate() {
        let pair_start = Instant::now();
        let label = pair.label();
        println!("\n  --- [{}/{}] {} ---", pair_idx + 1, n_pairs, label);

        // ── Capture passes via forward_text ─────────────────────────────
        let capture_start = Instant::now();
        let mut capture_hooks = HookSpec::new();
        for layer in 0..n_layers {
            capture_hooks.capture(HookPoint::ResidPost(layer));
        }

        // Only the counterfactual pass is captured: it is the donor. The
        // original pass is the recipient, and `Intervention::PatchAt` edits its
        // residual stream in flight rather than from a stored copy.
        let orig_result = model.forward_text(&pair.original_prompt, &HookSpec::new())?;
        let cf_result = model.forward_text(&pair.counterfactual_prompt, &capture_hooks)?;
        let capture_time = capture_start.elapsed();

        let orig_seq_len = orig_result.seq_len();
        let cf_seq_len = cf_result.seq_len();

        // Classify tokens: "fact" = everything before first \n\n, rest = "other"
        let fact_len = pair.fact_span_len();
        let token_labels = orig_result.encoding().label_spans(&[("fact", 0..fact_len)]);

        println!(
            "  GT: {} -> {}",
            pair.gt_original_target, pair.gt_counterfactual_target
        );
        println!("  Tokens: {orig_seq_len} (original), {cf_seq_len} (counterfactual)");

        // Extract greedy predictions
        let orig_logits = orig_result.output().get(0)?.get(orig_seq_len - 1)?; // [vocab]
        let orig_token_id = sample_token(&orig_logits, 0.0)?;
        let orig_continuation = tokenizer.decode(&[orig_token_id])?;

        let cf_logits = cf_result.output().get(0)?.get(cf_seq_len - 1)?;
        let cf_token_id = sample_token(&cf_logits, 0.0)?;
        let cf_continuation = tokenizer.decode(&[cf_token_id])?;

        // Build the donor FullActivationCache from the counterfactual residuals
        let mut cf_acts = FullActivationCache::with_capacity(n_layers);
        for layer in 0..n_layers {
            let resid = cf_result.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
            cf_acts.push(resid.get(0)?); // [seq, hidden]
        }

        let orig_matches_gt = orig_continuation.trim() == pair.gt_original_target;
        let cf_matches_gt = cf_continuation.trim() == pair.gt_counterfactual_target;
        println!(
            "  Original prediction: {orig_continuation} {}  |  CF prediction: {cf_continuation} {}",
            if orig_matches_gt { "correct" } else { "WRONG" },
            if cf_matches_gt { "correct" } else { "WRONG" },
        );

        // ── Patching sweep ──────────────────────────────────────────────
        let sweep_start = Instant::now();
        // Compute offset for position mapping between different-length prompts
        // CAST: isize arithmetic for offset, values are small token counts
        // CAST: usize -> isize, sequence lengths are far below isize::MAX
        #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
        let offset = cf_seq_len as isize - orig_seq_len as isize;

        // We need the original input tensor for patching forward passes
        let orig_input = candle_core::Tensor::new(&orig_result.encoding().ids[..], model.device())?
            .unsqueeze(0)?;

        // Print header
        print!("\n  {:>3} {:>14}", "Pos", "Token");
        for block in &blocks {
            // INDEX: block is non-empty by construction
            #[allow(clippy::indexing_slicing)]
            let first = block[0];
            #[allow(clippy::indexing_slicing)]
            let last = block[block.len() - 1];
            print!("  {:>12}", format!("[{first}-{last}]"));
        }
        println!("  Type");
        print!("  {:->3} {:->14}", "", "");
        for _ in &blocks {
            print!("  {:->12}", "");
        }
        println!("  {:->14}", "");

        let mut patches: Vec<PatchResult> = Vec::new();

        // EXPLICIT: `orig_pos` is a sequence position mapped onto the counterfactual
        // sequence via `offset`, not an incidental container index (CONVENTIONS Rule 9).
        #[allow(clippy::needless_range_loop)]
        for orig_pos in 0..orig_seq_len {
            // INDEX: orig_pos bounded by orig_seq_len from the loop
            #[allow(clippy::indexing_slicing)]
            let ttype = &token_labels[orig_pos];

            // Position mapping: fact tokens clamp, template tokens apply offset
            // CAST: isize arithmetic for position mapping; the tail cast back to usize
            // is safe because `mapped` is clamped non-negative by `max(0)` below.
            #[allow(clippy::as_conversions, clippy::cast_sign_loss)]
            let cf_pos = if ttype.starts_with("fact") {
                orig_pos.min(cf_seq_len.saturating_sub(1))
            } else {
                // CAST: usize -> isize then back; clamped non-negative by max(0)
                #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
                let mapped = (orig_pos as isize + offset).max(0);
                // CAST: isize → usize, value is non-negative after max(0)
                (mapped as usize).min(cf_seq_len.saturating_sub(1))
            };

            // Token display from encoding (no manual decode needed)
            // INDEX: orig_pos bounded by loop
            #[allow(clippy::indexing_slicing)]
            let token_str = &orig_result.tokens()[orig_pos];
            // BORROW: chars().take() — safe Unicode truncation for display
            let short_label: String = token_str.chars().take(12).collect();
            print!("  {orig_pos:>3} {short_label:>14}");

            for (bi, block) in blocks.iter().enumerate() {
                let patch_hooks = build_block_patch(&cf_acts, block, orig_pos, cf_pos)?;

                let patched_cache = model.forward(&orig_input, &patch_hooks)?;
                let patched_logits = patched_cache.output().get(0)?.get(orig_seq_len - 1)?;
                let patched_token_id = sample_token(&patched_logits, 0.0)?;
                let ablated_continuation = tokenizer.decode(&[patched_token_id])?;
                let is_different = ablated_continuation != orig_continuation;
                let kl = kl_divergence(&orig_logits, &patched_logits)?;

                let marker = if is_different { "CHANGED" } else { "same" };
                print!("  {marker:>12}");

                // Update counters
                // INDEX: bi bounded by blocks.len() = n_blocks
                #[allow(clippy::indexing_slicing)]
                {
                    block_total[bi] += 1;
                    if is_different {
                        block_different[bi] += 1;
                    }
                }
                *type_total.entry(ttype.clone()).or_insert(0) += 1;
                if is_different {
                    *type_different.entry(ttype.clone()).or_insert(0) += 1;
                    total_different += 1;
                }
                total_patches += 1;

                patches.push(PatchResult {
                    layer: block.clone(),
                    orig_pos,
                    cf_pos,
                    ablated_continuation,
                    is_different,
                    kl_from_original: kl,
                    token_type: ttype.clone(),
                });
            }
            println!("  {ttype}");
        }

        let sweep_time = sweep_start.elapsed();
        let pair_time = pair_start.elapsed();
        let pair_secs = pair_time.as_secs_f64();
        let n_patches = patches.len();
        let n_changed: usize = patches.iter().filter(|p| p.is_different).count();
        if !args.no_runtime {
            println!(
                "\n  Pair time: {pair_time:.2?} (capture: {capture_time:.2?}, sweep: {sweep_time:.2?}, {n_patches} patches, {n_changed} changed)"
            );
        }

        prompt_results.push(PromptResult {
            label,
            // BORROW: clone — String for JSON serialization
            original_prompt: pair.original_prompt.clone(),
            counterfactual_prompt: pair.counterfactual_prompt.clone(),
            gt_original_target: pair.gt_original_target.clone(),
            gt_counterfactual_target: pair.gt_counterfactual_target.clone(),
            input_tokens: orig_result.tokens().to_vec(),
            original_continuation: orig_continuation,
            cf_continuation,
            time_secs: pair_secs,
            patches,
        });
    }

    let total_time = total_start.elapsed();

    // ── Summary ─────────────────────────────────────────────────────────
    if args.no_runtime {
        println!("\n  === Summary ({n_pairs} prompts, {total_patches} patches) ===\n");
    } else {
        println!(
            "\n  === Summary ({n_pairs} prompts, {total_patches} patches, {total_time:.1?}) ===\n"
        );
    }

    let mut block_summaries = Vec::with_capacity(n_blocks);
    for (bi, block) in blocks.iter().enumerate() {
        // INDEX: bi bounded by blocks.len()
        #[allow(clippy::indexing_slicing)]
        let diff = block_different[bi];
        #[allow(clippy::indexing_slicing)]
        let total = block_total[bi];
        // CAST: usize → f32 for percentage, values are small counts
        #[allow(clippy::as_conversions)]
        let pct = if total > 0 {
            diff as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        // INDEX: block is non-empty by construction
        #[allow(clippy::indexing_slicing)]
        let first = block[0];
        #[allow(clippy::indexing_slicing)]
        let last = block[block.len() - 1];
        println!("  Block [{first:>2}-{last:>2}]:  {diff:>4}/{total:<4} changed ({pct:.1}%)");
        block_summaries.push(BlockSummary {
            block: bi,
            layers: block.clone(),
            pct_different: pct,
            count_different: diff,
            count_total: total,
        });
    }

    println!();
    let mut type_summaries = Vec::new();
    let mut type_keys: Vec<String> = type_total.keys().cloned().collect();
    type_keys.sort();
    for key in &type_keys {
        let total = type_total.get(key).copied().unwrap_or(0);
        let diff = type_different.get(key).copied().unwrap_or(0);
        // CAST: usize → f32 for percentage, values are small counts
        #[allow(clippy::as_conversions)]
        let pct = if total > 0 {
            diff as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        println!("  {key:<16} {diff:>4}/{total:<4} changed ({pct:.1}%)");
        type_summaries.push(TypeSummary {
            token_type: key.clone(),
            pct_different: pct,
            count_different: diff,
            count_total: total,
        });
    }

    // ── JSON output ─────────────────────────────────────────────────────
    if let Some(path) = &args.output {
        let output = JsonOutput {
            // BORROW: clone for JSON ownership
            model_id: args.model.clone(),
            block_size: args.block_size,
            n_layers,
            n_blocks,
            total_time_secs: total_time.as_secs_f64(),
            results: prompt_results,
            summary: Summary {
                pct_different_by_block: block_summaries,
                pct_different_by_token_type: type_summaries,
                total_patches,
                total_different,
            },
        };
        write_json(path, &output)?;
        println!("\n  JSON written to {}", path.display());
    }

    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Serialize `output` as pretty JSON and write to `path`.
fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
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
    Ok(())
}
