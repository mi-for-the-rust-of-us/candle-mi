// SPDX-License-Identifier: MIT OR Apache-2.0

//! # CLT Feature Probe
//!
//! Two modes:
//!
//! **Encoder probe** — what CLT features fire at a given token position:
//! ```bash
//! cargo run --release --features clt,transformer,mmap --example clt_probe -- \
//!   --prompt "A little mouse ran through the house, ..." --word cat
//! ```
//!
//! **Decoder search** — which features' decoders project toward a target word
//! (for finding suppress candidates):
//! ```bash
//! cargo run --release --features clt,transformer,mmap --example clt_probe -- \
//!   --decoder-search cat
//! ```

#![allow(
    clippy::doc_markdown,
    clippy::missing_docs_in_private_items,
    clippy::cast_precision_loss,
    clippy::indexing_slicing
)]

use std::path::PathBuf;
use std::time::Instant;

use candle_mi::{MIModel, Result, clt::CrossLayerTranscoder, hooks::HookPoint, hooks::HookSpec};
use clap::Parser;
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Probe CLT feature activations at a specific token position")]
struct Args {
    /// HuggingFace model ID
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// CLT repository on HuggingFace
    #[arg(long, default_value = "mntss/clt-llama-3.2-1b-524k")]
    clt_repo: String,

    /// Prompt text (required for encoder probe, not needed for --decoder-search)
    #[arg(long)]
    prompt: Option<String>,

    /// Token position to probe (0-indexed, including BOS)
    #[arg(long)]
    position: Option<usize>,

    /// Find position of this word in the tokenized prompt
    #[arg(long)]
    word: Option<String>,

    /// Top features per layer
    #[arg(long, default_value_t = 10)]
    top_k: usize,

    /// Top features in the global aggregate
    #[arg(long, default_value_t = 20)]
    global_top_k: usize,

    /// JSON output file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Suppress runtime display
    #[arg(long)]
    no_runtime: bool,

    /// Decoder search: find features whose decoders project toward this word.
    /// Runs independently of encoder probe (no --prompt/--position needed).
    #[arg(long)]
    decoder_search: Option<String>,
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ProbeOutput {
    model_id: String,
    clt_repo: String,
    prompt: String,
    tokens: Vec<TokenEntry>,
    probed_position: usize,
    probed_token: String,
    n_layers: usize,
    n_features_per_layer: usize,
    top_k: usize,
    layers: Vec<LayerResult>,
    aggregate: Vec<FeatureEntry>,
    total_time_secs: f64,
}

#[derive(Serialize)]
struct TokenEntry {
    position: usize,
    text: String,
}

#[derive(Serialize, Clone)]
struct FeatureEntry {
    layer: usize,
    index: usize,
    activation: f32,
}

#[derive(Serialize)]
struct LayerResult {
    layer: usize,
    features: Vec<FeatureEntry>,
    time_secs: f64,
}

#[derive(Serialize)]
struct DecoderSearchOutput {
    model_id: String,
    clt_repo: String,
    search_word: String,
    token_id: u32,
    n_layers: usize,
    top_k: usize,
    features: Vec<DecoderHit>,
    total_time_secs: f64,
}

#[derive(Serialize)]
struct DecoderHit {
    layer: usize,
    index: usize,
    cosine: f32,
}

// ---------------------------------------------------------------------------
// Word matching
// ---------------------------------------------------------------------------

/// Find the token position whose text matches `word` (case-insensitive,
/// ignoring the Ġ BPE prefix).
fn find_word_position(tokens: &[String], word: &str) -> Option<usize> {
    tokens.iter().position(|t| {
        // BORROW: replace produces a new String for comparison
        let clean = t.replace('\u{0120}', "");
        clean.eq_ignore_ascii_case(word)
    })
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

fn run() -> Result<()> {
    let args = Args::parse();

    // Branch: decoder search mode vs encoder probe mode.
    if let Some(ref search_word) = args.decoder_search {
        return run_decoder_search(&args, search_word);
    }

    let total_start = Instant::now();

    // --- Load model ---
    let load_start = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    let device = model.device().clone();
    // CAST: Instant → f64 for display
    let load_secs = load_start.elapsed().as_secs_f64();

    // --- Open CLT ---
    let clt_start = Instant::now();
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let n_features_per_layer = clt.config().n_features_per_layer;
    // CAST: Instant → f64 for display
    let clt_secs = clt_start.elapsed().as_secs_f64();

    println!("=== CLT Feature Probe ===");
    println!("  Model: {}", args.model);
    println!(
        "  CLT: {} ({n_layers} layers, {n_features_per_layer} features/layer)",
        args.clt_repo
    );
    if !args.no_runtime {
        println!("  Load time: {load_secs:.2}s, CLT open: {clt_secs:.2}s");
    }

    // --- Encoder probe requires a prompt ---
    let prompt = args
        .prompt
        .as_deref()
        .ok_or_else(|| candle_mi::MIError::Config("encoder probe requires --prompt".into()))?;

    // --- Forward pass capturing ResidMid at all layers ---
    let mut hooks = HookSpec::new();
    for layer in 0..n_layers {
        hooks.capture(HookPoint::ResidMid(layer));
    }
    let fwd_start = Instant::now();
    let result = model.forward_text(prompt, &hooks)?;
    // CAST: Instant → f64 for display
    let fwd_secs = fwd_start.elapsed().as_secs_f64();

    let encoding = result.encoding();
    let tokens: Vec<String> = encoding.tokens.clone();
    let seq_len = tokens.len();

    // --- Print token table ---
    println!();
    println!("  === Tokens ({seq_len} total) ===");
    println!("  {:>4}  Token", "Pos");

    // --- Resolve target position ---
    let position = if let Some(pos) = args.position {
        pos
    } else if let Some(ref word) = args.word {
        find_word_position(&tokens, word).ok_or_else(|| {
            candle_mi::MIError::Tokenizer(format!("word \"{word}\" not found in tokens"))
        })?
    } else {
        return Err(candle_mi::MIError::Config(
            "provide --position or --word".into(),
        ));
    };

    if position >= seq_len {
        return Err(candle_mi::MIError::Config(format!(
            "position {position} out of range (seq_len = {seq_len})"
        )));
    }

    for (i, tok) in tokens.iter().enumerate() {
        let marker = if i == position { "  ← probing" } else { "" };
        println!("  {i:>4}  {tok}{marker}");
    }

    if !args.no_runtime {
        println!();
        println!("  Forward time: {fwd_secs:.2}s");
    }

    // INDEX: position < seq_len checked above (line 239)
    let probed_token = tokens[position].clone();
    println!();
    println!("  === Features at position {position} (\"{probed_token}\") ===");
    println!(
        "  {:>5}  {:>4}  {:>14}  {:>10}",
        "Layer", "Rank", "Feature", "Activation"
    );

    // --- Probe each layer ---
    let probe_start = Instant::now();
    let mut all_layers: Vec<LayerResult> = Vec::with_capacity(n_layers);
    let mut all_features: Vec<FeatureEntry> = Vec::new();

    for layer in 0..n_layers {
        let layer_start = Instant::now();
        clt.load_encoder(layer, &device)?;

        let resid = result
            .cache()
            .require(&HookPoint::ResidMid(layer))?
            .get(0)?
            .get(position)?;

        let sparse = clt.top_k(&resid, layer, args.top_k)?;

        let features: Vec<FeatureEntry> = sparse
            .features
            .iter()
            .map(|(fid, act)| FeatureEntry {
                layer: fid.layer,
                index: fid.index,
                activation: *act,
            })
            .collect();

        for (rank, feat) in features.iter().enumerate() {
            println!(
                "  {:>5}  {:>4}  L{:<2}:{:<9}  {:>10.3}",
                feat.layer,
                rank + 1,
                feat.layer,
                feat.index,
                feat.activation,
            );
        }

        all_features.extend_from_slice(&features);

        // CAST: Instant → f64 for JSON
        let layer_secs = layer_start.elapsed().as_secs_f64();
        all_layers.push(LayerResult {
            layer,
            features,
            time_secs: layer_secs,
        });
    }

    // CAST: Instant → f64 for display
    let probe_secs = probe_start.elapsed().as_secs_f64();
    if !args.no_runtime {
        println!();
        println!("  Probe time: {probe_secs:.2}s ({n_layers} layers)");
    }

    // --- Global aggregate ---
    all_features.sort_by(|a, b| {
        b.activation
            .partial_cmp(&a.activation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_features.truncate(args.global_top_k);

    println!();
    println!("  === Top {} across all layers ===", args.global_top_k);
    for (rank, feat) in all_features.iter().enumerate() {
        println!(
            "  {:>4}. L{}:{}    act={:.3}",
            rank + 1,
            feat.layer,
            feat.index,
            feat.activation,
        );
    }

    // --- Suppress candidates ---
    println!();
    println!("  === Suppress candidates (paste into figure13/attention_routing) ===");
    // Take top 3 as candidates (different layers preferred)
    let mut seen_layers = std::collections::HashSet::new();
    let mut candidates: Vec<&FeatureEntry> = Vec::new();
    for feat in &all_features {
        if seen_layers.insert(feat.layer) {
            candidates.push(feat);
            if candidates.len() >= 3 {
                break;
            }
        }
    }
    if candidates.is_empty() && !all_features.is_empty() {
        // INDEX: guarded by !is_empty() check
        candidates.push(&all_features[0]);
    }
    let suppress_flags: Vec<String> = candidates
        .iter()
        .map(|f| format!("--suppress L{}:{}", f.layer, f.index))
        .collect();
    println!("  {}", suppress_flags.join(" "));

    // --- Timing ---
    // CAST: Instant → f64 for display
    let total_secs = total_start.elapsed().as_secs_f64();
    println!();
    println!("  Total time: {total_secs:.2}s");

    // --- JSON output ---
    if let Some(ref path) = args.output {
        let output = ProbeOutput {
            model_id: args.model.clone(),
            clt_repo: args.clt_repo.clone(),
            // BORROW: clone prompt for JSON ownership
            prompt: prompt.to_owned(),
            tokens: tokens
                .iter()
                .enumerate()
                .map(|(i, t)| TokenEntry {
                    position: i,
                    // BORROW: clone token string for JSON ownership
                    text: t.clone(),
                })
                .collect(),
            probed_position: position,
            probed_token,
            n_layers,
            n_features_per_layer,
            top_k: args.top_k,
            layers: all_layers,
            aggregate: all_features,
            total_time_secs: total_secs,
        };

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| candle_mi::MIError::Config(format!("create dir: {e}")))?;
        }

        let json = serde_json::to_string_pretty(&output)
            .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| candle_mi::MIError::Config(format!("write {}: {e}", path.display())))?;
        println!("  JSON output written to {}", path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Decoder search mode
// ---------------------------------------------------------------------------

/// Find CLT features whose decoder vectors project toward a target word.
///
/// This answers: "which features, when suppressed, would remove the signal
/// for this word?" — the key question for finding suppress candidates.
fn run_decoder_search(args: &Args, search_word: &str) -> Result<()> {
    let total_start = Instant::now();

    // --- Load model (for tokenizer + embedding) ---
    let load_start = Instant::now();
    let model = MIModel::from_pretrained(&args.model)?;
    // CAST: Instant → f64 for display
    let load_secs = load_start.elapsed().as_secs_f64();

    // --- Open CLT ---
    let clt_start = Instant::now();
    let mut clt = CrossLayerTranscoder::open(&args.clt_repo)?;
    let n_layers = clt.config().n_layers;
    let n_features_per_layer = clt.config().n_features_per_layer;
    // CAST: Instant → f64 for display
    let clt_secs = clt_start.elapsed().as_secs_f64();

    println!("=== CLT Decoder Search ===");
    println!("  Model: {}", args.model);
    println!(
        "  CLT: {} ({n_layers} layers, {n_features_per_layer} features/layer)",
        args.clt_repo
    );
    if !args.no_runtime {
        println!("  Load time: {load_secs:.2}s, CLT open: {clt_secs:.2}s");
    }

    // --- Tokenize the search word and get its embedding ---
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Config("decoder search requires a tokenizer".into()))?;
    let word_tokens = tokenizer.encode_raw(search_word)?;
    if word_tokens.len() != 1 {
        return Err(candle_mi::MIError::Config(format!(
            "\"{search_word}\" tokenizes to {} tokens (need exactly 1): {:?}",
            word_tokens.len(),
            word_tokens
        )));
    }
    // INDEX: length == 1 checked above
    let token_id = word_tokens[0];

    let direction = model.backend().embedding_vector(token_id)?;
    println!("  Search word: \"{search_word}\" (token ID {token_id})");
    println!(
        "  Direction: embedding vector [{}]",
        direction
            .dims()
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Score against ALL target layers — a feature may decode to the target
    // word at an intermediate layer, not just the last one.  This matches
    // the plip-rs vocabulary scan approach (explore-vocabulary mode).
    println!("  Scanning all {n_layers} target layers...");

    let search_start = Instant::now();
    let mut all_hits: Vec<(candle_mi::clt::CltFeatureId, f32)> = Vec::new();
    for target_layer in 0..n_layers {
        let hits = clt.score_features_by_decoder_projection(
            &direction,
            target_layer,
            args.top_k,
            true, // cosine similarity
        )?;
        for (fid, cosine) in hits {
            // Keep the best cosine per feature across target layers.
            if let Some(existing) = all_hits.iter_mut().find(|(f, _)| *f == fid) {
                if cosine > existing.1 {
                    existing.1 = cosine;
                }
            } else {
                all_hits.push((fid, cosine));
            }
        }
    }
    all_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    all_hits.truncate(args.global_top_k);
    // CAST: Instant → f64 for display
    let search_secs = search_start.elapsed().as_secs_f64();
    let hits = all_hits;

    println!();
    println!(
        "  === Top {} features by cosine to \"{search_word}\" (best across all target layers) ===",
        args.global_top_k
    );
    println!("  {:>4}  {:>14}  {:>8}", "Rank", "Feature", "Cosine");
    for (rank, (fid, cosine)) in hits.iter().enumerate() {
        println!(
            "  {:>4}  L{:<2}:{:<9}  {:>8.4}",
            rank + 1,
            fid.layer,
            fid.index,
            *cosine,
        );
    }

    // --- Suppress candidates ---
    println!();
    println!("  === Suppress candidates ===");
    let mut seen_layers = std::collections::HashSet::new();
    let mut candidates: Vec<String> = Vec::new();
    for (fid, _) in &hits {
        if seen_layers.insert(fid.layer) {
            candidates.push(format!("--suppress L{}:{}", fid.layer, fid.index));
            if candidates.len() >= 3 {
                break;
            }
        }
    }
    println!("  {}", candidates.join(" "));

    if !args.no_runtime {
        println!();
        println!("  Search time: {search_secs:.2}s");
    }

    // CAST: Instant → f64 for display
    let total_secs = total_start.elapsed().as_secs_f64();
    println!("  Total time: {total_secs:.2}s");

    // --- JSON output ---
    if let Some(ref path) = args.output {
        let output = DecoderSearchOutput {
            model_id: args.model.clone(),
            clt_repo: args.clt_repo.clone(),
            // BORROW: clone for JSON ownership
            search_word: search_word.to_owned(),
            token_id,
            n_layers,
            top_k: args.global_top_k,
            features: hits
                .iter()
                .map(|(fid, cosine)| DecoderHit {
                    layer: fid.layer,
                    index: fid.index,
                    cosine: *cosine,
                })
                .collect(),
            total_time_secs: total_secs,
        };

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| candle_mi::MIError::Config(format!("create dir: {e}")))?;
        }

        let json = serde_json::to_string_pretty(&output)
            .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| candle_mi::MIError::Config(format!("write {}: {e}", path.display())))?;
        println!("  JSON output written to {}", path.display());
    }

    Ok(())
}
