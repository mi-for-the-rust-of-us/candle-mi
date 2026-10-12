// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recurrent feedback (anacrousis) for rhyme completion.
//!
//! Demonstrates `RecurrentPassSpec`, `RecurrentFeedbackEntry`,
//! `forward_recurrent()`, and `generate_recurrent()` on a rhyme completion
//! task.  The recurrence re-runs commitment layers with directional feedback
//! derived from token embeddings, giving the model extra depth to sustain
//! planning signals through generation.
//!
//! Inspired by the structural correspondence between DRC planning
//! (Taufeeque et al., "Planning in a recurrent neural network that plays
//! Sokoban", arXiv:2407.15421, 2024; mechanistic follow-up: Taufeeque et al.,
//! "Path Channels and Plan Extension Kernels", arXiv:2506.10138, 2025) and
//! transformer planning (Lindsey et al., "On the Biology of a Large Language
//! Model", Anthropic, 2025).
//!
//! See also: Eric Jacopin, "Replicating 'Planning in Poems' with Open Tools"
//! (plip-rs, anacrousis branch) for the full 28-condition experiment.
//!
//! ```bash
//! # Default: Llama 3.2 1B, layers 8-15, strength 2.0, depth 2, prefill-only
//! cargo run --release --features transformer --example recurrent_feedback
//!
//! # Deeper recurrence (3 passes through the loop layers)
//! cargo run --release --features transformer --example recurrent_feedback -- --depth 3
//!
//! # Sustained feedback (applied at every generation step)
//! cargo run --release --features transformer --example recurrent_feedback -- --sustained
//!
//! # Custom layer range, strength, and depth
//! cargo run --release --features transformer --example recurrent_feedback -- \
//!     --loop-start 14 --loop-end 15 --strength 1.0 --depth 4 --sustained
//!
//! # Rhyme-token rank diagnostic at the planning position
//! cargo run --release --features transformer --example recurrent_feedback -- --diagnose
//!
//! # With JSON output
//! cargo run --release --features transformer --example recurrent_feedback -- \
//!     --output examples/results/recurrent_feedback/llama-3.2-1b-prefill.json
//! ```

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_docs_in_private_items)]

use std::path::{Path, PathBuf};

use candle_core::{DType, IndexOp, Tensor};
use clap::Parser;
use serde::Serialize;

use candle_mi::{
    GenericTransformer, HookSpec, MIBackend, MITokenizer, RecurrentPassSpec, TransformerConfig,
    sample_token,
};
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "recurrent_feedback")]
#[command(about = "Recurrent feedback (anacrousis) for rhyme completion")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(long, default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// First layer of the recurrent block (inclusive)
    #[arg(long, default_value_t = 8)]
    loop_start: usize,

    /// Last layer of the recurrent block (inclusive)
    #[arg(long, default_value_t = 15)]
    loop_end: usize,

    /// Feedback strength (amplification factor)
    #[arg(long, default_value_t = 2.0)]
    strength: f32,

    /// Recurrence depth: total number of passes through the loop layers.
    /// 1 = no recurrence (single pass), 2 = default (one recurrent pass),
    /// 3+ = deeper recurrence.
    #[arg(long, default_value_t = 2)]
    depth: usize,

    /// Apply feedback at every generation step (sustained mode)
    #[arg(long)]
    sustained: bool,

    /// Maximum number of couplets to test (default: all 15)
    #[arg(long)]
    max_couplets: Option<usize>,

    /// Show rhyme-token rank analysis at the planning position
    #[arg(long)]
    diagnose: bool,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    n_layers: usize,
    hidden_size: usize,
    loop_start: usize,
    loop_end: usize,
    depth: usize,
    strength: f32,
    mode: String,
    baseline_rhymes: usize,
    recurrent_rhymes: usize,
    total_couplets: usize,
    couplets: Vec<JsonCouplet>,
}

#[derive(Serialize)]
struct JsonCouplet {
    id: u32,
    target: String,
    line1: String,
    baseline_word: String,
    baseline_rhymes: bool,
    recurrent_word: String,
    recurrent_rhymes: bool,
    recurrent_line: String,
    result: String,
}

// ---------------------------------------------------------------------------
// Couplet definitions (canonical 15 from plip-rs anacrousis experiments)
// ---------------------------------------------------------------------------

struct CoupletDef {
    id: u32,
    target_word: &'static str,
    line1: &'static str,
    rhyme_family: &'static [&'static str],
}

fn couplet_defs() -> Vec<CoupletDef> {
    vec![
        CoupletDef {
            id: 1,
            target_word: "light",
            line1: "The moon casts silver light,",
            rhyme_family: &[
                "light", "night", "bright", "sight", "might", "flight", "right", "tight", "white",
                "bite", "kite", "quite", "knight", "delight", "blight", "plight", "slight",
                "fright", "height",
            ],
        },
        CoupletDef {
            id: 2,
            target_word: "play",
            line1: "The children laugh and play,",
            rhyme_family: &[
                "play", "day", "way", "say", "stay", "sway", "ray", "bay", "may", "lay", "pay",
                "gray", "away", "display", "pray", "stray", "clay", "hay", "decay", "delay",
            ],
        },
        CoupletDef {
            id: 3,
            target_word: "sound",
            line1: "The thunder makes a sound,",
            rhyme_family: &[
                "sound", "ground", "found", "round", "bound", "around", "mound", "pound", "hound",
                "wound", "profound", "abound", "astound",
            ],
        },
        CoupletDef {
            id: 4,
            target_word: "rain",
            line1: "The clouds bring heavy rain,",
            rhyme_family: &[
                "rain", "pain", "gain", "main", "vain", "plain", "chain", "train", "brain",
                "strain", "remain", "again", "drain", "lane", "crane", "bane", "wane", "reign",
                "feign",
            ],
        },
        CoupletDef {
            id: 5,
            target_word: "time",
            line1: "The old clock measures time,",
            rhyme_family: &[
                "time", "rhyme", "climb", "crime", "dime", "lime", "mime", "prime", "chime",
                "sublime", "paradigm", "thyme",
            ],
        },
        CoupletDef {
            id: 6,
            target_word: "air",
            line1: "The geese fly through the air,",
            rhyme_family: &[
                "air", "there", "fair", "care", "bare", "dare", "rare", "share", "stare", "where",
                "pair", "aware", "compare", "despair", "prayer", "hair", "chair", "bear", "wear",
                "spare", "snare", "glare",
            ],
        },
        CoupletDef {
            id: 7,
            target_word: "gold",
            line1: "The sunset gleams like gold,",
            rhyme_family: &[
                "gold",
                "old",
                "bold",
                "cold",
                "fold",
                "hold",
                "told",
                "sold",
                "mold",
                "behold",
                "unfold",
                "rolled",
                "controlled",
            ],
        },
        CoupletDef {
            id: 8,
            target_word: "fire",
            line1: "The embers feed the fire,",
            rhyme_family: &[
                "fire", "hire", "wire", "desire", "tire", "inspire", "acquire", "higher", "entire",
                "admire", "liar", "dire", "sire", "pyre", "mire", "conspire", "expire",
            ],
        },
        CoupletDef {
            id: 9,
            target_word: "stone",
            line1: "The castle walls of stone,",
            rhyme_family: &[
                "stone", "bone", "tone", "lone", "zone", "throne", "phone", "own", "known",
                "blown", "grown", "shown", "moan", "groan", "clone", "drone",
            ],
        },
        CoupletDef {
            id: 10,
            target_word: "dream",
            line1: "I wandered through a dream,",
            rhyme_family: &[
                "dream", "stream", "seem", "team", "beam", "cream", "gleam", "scheme", "theme",
                "extreme", "esteem", "scream", "steam",
            ],
        },
        CoupletDef {
            id: 11,
            target_word: "strange",
            line1: "The silence felt so strange,",
            rhyme_family: &[
                "strange", "change", "range", "arrange", "exchange", "grange",
            ],
        },
        CoupletDef {
            id: 12,
            target_word: "love",
            line1: "I never knew such love,",
            rhyme_family: &["love", "above", "dove", "of", "shove", "glove", "thereof"],
        },
        CoupletDef {
            id: 13,
            target_word: "truth",
            line1: "She spoke the honest truth,",
            rhyme_family: &[
                "truth", "youth", "tooth", "booth", "smooth", "sleuth", "ruth", "uncouth",
            ],
        },
        CoupletDef {
            id: 14,
            target_word: "world",
            line1: "He traveled all the world,",
            rhyme_family: &[
                "world", "curled", "unfurled", "whirled", "hurled", "swirled", "pearled", "furled",
                "twirled",
            ],
        },
        CoupletDef {
            id: 15,
            target_word: "earth",
            line1: "The seeds lay in the earth,",
            rhyme_family: &[
                "earth", "birth", "worth", "mirth", "berth", "girth", "dearth", "rebirth", "hearth",
            ],
        },
    ]
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
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mode = if args.sustained {
        "sustained"
    } else {
        "prefill-only"
    };
    println!("=== {} ===", args.model);

    // --- Load model ---
    let t0 = std::time::Instant::now();

    #[cfg(feature = "memory")]
    let mem_before = MemorySnapshot::now(
        &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
    )?;

    let (model, tokenizer, _config, eos_tokens) = load_transformer(&args.model)?;
    let load_time = t0.elapsed();

    let n_layers = model.num_layers();
    let hidden = model.hidden_size();
    let device = model.embedding_vector(0)?.device().clone();
    // CAST: usize → f64, values are small enough for exact representation
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let weight_mb = estimate_weight_mb(n_layers, hidden);
    println!("  Layers: {n_layers}, hidden: {hidden}, device: {device:?}");
    println!("  Estimated F32 weight size: {weight_mb:.0} MB");
    println!("  Load time: {load_time:.2?}");

    #[cfg(feature = "memory")]
    {
        let mem_after = MemorySnapshot::now(&device)?;
        MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
    }

    println!();
    println!("  Recurrent layers: {}–{}", args.loop_start, args.loop_end);
    println!("  Depth:     {}", args.depth);
    println!("  Strength:  {:.1}", args.strength);
    println!("  Mode:      {mode}");
    println!();

    // --- Prepare couplets ---
    let couplets = couplet_defs();
    let max_couplets = args.max_couplets.unwrap_or(couplets.len());
    let limit = max_couplets.min(couplets.len());
    let couplets = couplets.get(..limit).unwrap_or(&couplets);

    let max_tokens: usize = 30;
    let stop_tokens = if eos_tokens.is_empty() {
        eprintln!("  Warning: no eos_token_id in config.json; using LLaMA defaults");
        vec![128_001_u32, 128_009]
    } else {
        eos_tokens
    };

    // --- Run experiment ---
    println!(
        "  {:>3}  {:>10}  {:>10}  {:>10}  {:>10}  Line 2 (recurrent)",
        "ID", "Target", "Baseline", "Recurrent", "Result"
    );
    println!("  {}", "-".repeat(80));

    let mut baseline_rhymes = 0_usize;
    let mut recurrent_rhymes = 0_usize;
    let mut json_couplets: Vec<JsonCouplet> = Vec::new();

    for couplet in couplets {
        let prompt = format!("{}\n", couplet.line1);
        let prompt_ids = tokenizer.encode(&prompt)?;
        let planning_pos = prompt_ids.len() - 1;

        let rhyme_set: Vec<String> = couplet
            .rhyme_family
            .iter()
            .map(|w| (*w).to_lowercase())
            .collect();

        // --- Baseline generation ---
        let baseline_tokens =
            generate_baseline(&model, &prompt_ids, max_tokens, &stop_tokens, &device)?;
        // INDEX: prompt_ids.len() is always <= baseline_tokens.len()
        // because generate_baseline starts with prompt_tokens.to_vec()
        let baseline_gen = baseline_tokens.get(prompt_ids.len()..).unwrap_or(&[]);
        let baseline_text = tokenizer.decode(baseline_gen)?;
        let baseline_line = baseline_text.lines().next().unwrap_or("");
        let baseline_word = extract_last_word(baseline_line);
        let baseline_ok = word_rhymes(&baseline_word, &rhyme_set);
        if baseline_ok {
            baseline_rhymes += 1;
        }

        // --- Recurrent generation ---
        let rhyme_dir = averaged_rhyme_direction(&model, &tokenizer, couplet.rhyme_family)?;
        let mut spec = RecurrentPassSpec::no_feedback(args.loop_start, args.loop_end)
            .with_depth(args.depth)
            .with_sustained(args.sustained);
        spec.add_feedback(planning_pos, rhyme_dir, args.strength);

        let recurrent_tokens =
            model.generate_recurrent(&prompt_ids, max_tokens, 0.0, &stop_tokens, &spec)?;
        // INDEX: prompt_ids.len() is always <= recurrent_tokens.len()
        // because generate_recurrent starts with prompt_tokens.to_vec()
        let recurrent_gen = recurrent_tokens.get(prompt_ids.len()..).unwrap_or(&[]);
        let recurrent_text = tokenizer.decode(recurrent_gen)?;
        let recurrent_line = recurrent_text.lines().next().unwrap_or("");
        let recurrent_word = extract_last_word(recurrent_line);
        let recurrent_ok = word_rhymes(&recurrent_word, &rhyme_set);
        if recurrent_ok {
            recurrent_rhymes += 1;
        }

        let result = match (baseline_ok, recurrent_ok) {
            (false, true) => "RESCUED",
            (true, false) => "REGRESS",
            (true, true) => "OK",
            (false, false) => "-",
        };

        println!(
            "  {:>3}  {:>10}  {:>10}  {:>10}  {:>10}  {}",
            couplet.id,
            couplet.target_word,
            baseline_word,
            recurrent_word,
            result,
            recurrent_line.trim()
        );

        // --- Diagnostic: rhyme-token rank analysis at the planning position ---
        if args.diagnose {
            let rhyme_ids = rhyme_word_token_ids(&tokenizer, couplet.rhyme_family)?;
            let input_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;

            // Baseline logits at planning position
            let base_cache = model.forward(&input_tensor, &HookSpec::new())?;
            let base_logits = base_cache.output();
            let base_last = base_logits
                .i((.., planning_pos, ..))?
                .squeeze(0)?
                .squeeze(0)?;
            let base_ranks = rank_rhyme_tokens(&base_last, &rhyme_ids)?;

            // Recurrent logits at planning position
            let rec_cache = model.forward_recurrent(&input_tensor, &HookSpec::new(), &spec)?;
            let rec_logits = rec_cache.output();
            let rec_last = rec_logits
                .i((.., planning_pos, ..))?
                .squeeze(0)?
                .squeeze(0)?;
            let rec_ranks = rank_rhyme_tokens(&rec_last, &rhyme_ids)?;

            println!(
                "       baseline: best rhyme \"{}\", rank {:>5}, prob {:.6}, top-100: {}",
                base_ranks.best_word,
                base_ranks.best_rank + 1,
                base_ranks.best_prob,
                base_ranks.in_top_100
            );
            println!(
                "       recurrent: best rhyme \"{}\", rank {:>5}, prob {:.6}, top-100: {}",
                rec_ranks.best_word,
                rec_ranks.best_rank + 1,
                rec_ranks.best_prob,
                rec_ranks.in_top_100
            );
            // CAST: usize → isize, ranks are small enough
            #[allow(clippy::as_conversions, clippy::cast_possible_wrap)]
            let rank_delta = base_ranks.best_rank as isize - rec_ranks.best_rank as isize;
            match rank_delta.cmp(&0) {
                std::cmp::Ordering::Greater => {
                    println!("       → nudge improved best rhyme rank by {rank_delta} positions");
                }
                std::cmp::Ordering::Less => {
                    println!(
                        "       → nudge worsened best rhyme rank by {} positions",
                        rank_delta.unsigned_abs()
                    );
                }
                std::cmp::Ordering::Equal => {
                    println!("       → no rank change");
                }
            }
        }

        json_couplets.push(JsonCouplet {
            id: couplet.id,
            target: couplet.target_word.into(),
            line1: couplet.line1.into(),
            baseline_word: baseline_word.clone(),
            baseline_rhymes: baseline_ok,
            recurrent_word: recurrent_word.clone(),
            recurrent_rhymes: recurrent_ok,
            recurrent_line: recurrent_line.trim().into(),
            result: result.into(),
        });
    }

    // --- Summary ---
    println!("\n  Baseline:  {baseline_rhymes}/{}", couplets.len());
    println!("  Recurrent: {recurrent_rhymes}/{}", couplets.len());
    match recurrent_rhymes.cmp(&baseline_rhymes) {
        std::cmp::Ordering::Greater => {
            println!(
                "  Improvement: +{} couplet(s) rescued",
                recurrent_rhymes - baseline_rhymes
            );
        }
        std::cmp::Ordering::Less => {
            println!(
                "  Degradation: -{} couplet(s) lost",
                baseline_rhymes - recurrent_rhymes
            );
        }
        std::cmp::Ordering::Equal => {
            println!("  No change in rhyme success rate");
        }
    }

    // --- JSON output ---
    if let Some(ref path) = args.output {
        write_json_output(
            path,
            &args.model,
            n_layers,
            model.hidden_size(),
            args.loop_start,
            args.loop_end,
            args.depth,
            args.strength,
            mode,
            baseline_rhymes,
            recurrent_rhymes,
            couplets.len(),
            json_couplets,
        )?;
    }

    println!("\n  Total elapsed: {:.2}s", t0.elapsed().as_secs_f64());

    Ok(())
}

// ---------------------------------------------------------------------------
// Model loading (direct GenericTransformer — needed for generate_recurrent)
// ---------------------------------------------------------------------------

fn load_transformer(
    model_id: &str,
) -> candle_mi::Result<(GenericTransformer, MITokenizer, TransformerConfig, Vec<u32>)> {
    // Go through the shared builder; the token of a gated model resolves
    // inside hf-fetch-model.
    // BORROW: explicit .to_owned() — &str → String for download API
    let fetch_config = candle_mi::fetch_config_builder()
        .build()
        .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;
    let files =
        hf_fetch_model::download_files_with_config_blocking(model_id.to_owned(), &fetch_config)
            .map(hf_fetch_model::DownloadOutcome::into_inner)
            .map_err(|e| candle_mi::MIError::Download(e.to_string()))?;

    let config_path = files
        .get("config.json")
        .ok_or_else(|| candle_mi::MIError::Config("config.json not found".into()))?;
    let config_str = std::fs::read_to_string(config_path)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to read config.json: {e}")))?;
    let json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to parse config.json: {e}")))?;

    let config = TransformerConfig::from_hf_config(&json)?;

    // Extract eos_token_id(s) from config.json — can be a single int or an array
    let eos_tokens = parse_eos_token_ids(&json);

    let device = candle_core::Device::cuda_if_available(0).map_err(candle_mi::MIError::Model)?;
    let dtype = DType::F32;

    let weights_paths = resolve_safetensors_paths(&files)?;
    let vb = create_var_builder(&weights_paths, dtype, &device)?;

    let model = GenericTransformer::load(config.clone(), &device, dtype, vb)?;

    let tokenizer_path = files
        .get("tokenizer.json")
        .ok_or_else(|| candle_mi::MIError::Tokenizer("tokenizer.json not found".into()))?;
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    Ok((model, tokenizer, config, eos_tokens))
}

/// Parse `eos_token_id` from a HuggingFace `config.json` value.
///
/// Handles both single-integer and array-of-integers formats.
/// Returns an empty `Vec` if the field is absent or unparseable.
fn parse_eos_token_ids(json: &serde_json::Value) -> Vec<u32> {
    match json.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => {
            // CAST: u64 → u32, token IDs fit in u32 (vocab sizes < 2^32)
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            n.as_u64().map_or_else(Vec::new, |id| vec![id as u32])
        }
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                // CAST: u64 → u32, token IDs fit in u32 (vocab sizes < 2^32)
                #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
                v.as_u64().map(|id| id as u32)
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Resolve safetensors file paths from a downloaded file map.
fn resolve_safetensors_paths(
    files: &std::collections::HashMap<String, PathBuf>,
) -> candle_mi::Result<Vec<PathBuf>> {
    // Try sharded first
    if let Some(index_path) = files.get("model.safetensors.index.json") {
        let index_str = std::fs::read_to_string(index_path)
            .map_err(|e| candle_mi::MIError::Config(format!("failed to read index: {e}")))?;
        let index: serde_json::Value = serde_json::from_str(&index_str)
            .map_err(|e| candle_mi::MIError::Config(format!("failed to parse index: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| candle_mi::MIError::Config("missing weight_map in index".into()))?;

        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(String::from)
            .collect();
        shard_names.sort();
        shard_names.dedup();

        let mut paths = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            // BORROW: explicit .as_str() — String → &str for HashMap lookup
            let path = files
                .get(name.as_str())
                .ok_or_else(|| candle_mi::MIError::Config(format!("shard {name} not found")))?;
            paths.push(path.clone());
        }
        return Ok(paths);
    }

    // Single file
    let path = files
        .get("model.safetensors")
        .ok_or_else(|| candle_mi::MIError::Config("model.safetensors not found".into()))?;
    Ok(vec![path.clone()])
}

/// Create a `VarBuilder` from safetensors file paths.
fn create_var_builder(
    paths: &[PathBuf],
    dtype: DType,
    device: &candle_core::Device,
) -> candle_mi::Result<candle_nn::VarBuilder<'static>> {
    #[cfg(feature = "mmap")]
    {
        // SAFETY: safetensors files must not be modified while loaded.
        #[allow(unsafe_code)]
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        Ok(vb)
    }
    #[cfg(not(feature = "mmap"))]
    {
        if paths.len() > 1 {
            return Err(candle_mi::MIError::Config(
                "sharded models require the `mmap` feature".into(),
            ));
        }
        let path = paths
            .first()
            .ok_or_else(|| candle_mi::MIError::Config("no safetensors files".into()))?;
        let data = std::fs::read(path).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to read {}: {e}", path.display()))
        })?;
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, dtype, device)?;
        Ok(vb)
    }
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn write_json_output(
    path: &Path,
    model_id: &str,
    n_layers: usize,
    hidden_size: usize,
    loop_start: usize,
    loop_end: usize,
    depth: usize,
    strength: f32,
    mode: &str,
    baseline_rhymes: usize,
    recurrent_rhymes: usize,
    total_couplets: usize,
    couplets: Vec<JsonCouplet>,
) -> candle_mi::Result<()> {
    let output = JsonOutput {
        model_id: model_id.into(),
        n_layers,
        hidden_size,
        loop_start,
        loop_end,
        depth,
        strength,
        mode: mode.into(),
        baseline_rhymes,
        recurrent_rhymes,
        total_couplets,
        couplets,
    };

    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }

    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;

    println!("  JSON written to {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Generation helpers
// ---------------------------------------------------------------------------

/// Greedy baseline generation (standard forward, no recurrence).
fn generate_baseline(
    model: &GenericTransformer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    stop_tokens: &[u32],
    device: &candle_core::Device,
) -> candle_mi::Result<Vec<u32>> {
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..max_tokens {
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let cache = model.forward(&input, &HookSpec::new())?;
        let logits = cache.output();
        let seq_len = logits.dim(1)?;
        let last_logits = logits.i((.., seq_len - 1, ..))?.squeeze(1)?.flatten_all()?;
        let next_token = sample_token(&last_logits, 0.0)?;
        if stop_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }
    Ok(tokens)
}

/// Compute L2-normalised average of embedding vectors for rhyme family words.
fn averaged_rhyme_direction(
    model: &GenericTransformer,
    tokenizer: &MITokenizer,
    rhyme_words: &[&str],
) -> candle_mi::Result<Tensor> {
    let mut embeddings: Vec<Tensor> = Vec::new();

    for word in rhyme_words {
        let with_space = format!(" {word}");
        let ids = tokenizer.encode_raw(&with_space)?;
        let Some(&token_id) = ids.last() else {
            continue;
        };
        let emb = model.embedding_vector(token_id)?;
        embeddings.push(emb);
    }

    if embeddings.is_empty() {
        return Err(candle_mi::MIError::Tokenizer(
            "no valid embeddings for rhyme family".into(),
        ));
    }

    let stacked = Tensor::stack(&embeddings, 0)?;
    let avg = stacked.mean(0)?;
    // PROMOTE: ensure F32 for norm computation
    let avg_f32 = avg.to_dtype(DType::F32)?;
    let norm: f32 = avg_f32.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
    if norm > 1e-8 {
        Ok(avg_f32.affine(1.0 / f64::from(norm), 0.0)?)
    } else {
        Ok(avg_f32)
    }
}

/// Extract the last word-like token from generated text.
fn extract_last_word(text: &str) -> String {
    text.split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Rough estimate of F32 weight memory in MB.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn estimate_weight_mb(n_layers: usize, hidden: usize) -> f64 {
    // CAST: usize → f64, values are small enough for exact representation
    let params_per_layer = 12.0 * (hidden as f64) * (hidden as f64);
    // CAST: usize → f64, values are small enough for exact representation
    let total_params = (n_layers as f64) * params_per_layer;
    total_params * 4.0 / 1_000_000.0
}

// ---------------------------------------------------------------------------
// Diagnostic: rhyme-token rank analysis at the planning position
// ---------------------------------------------------------------------------

/// Result of checking rhyme token ranks in the logit distribution.
struct RhymeRankResult {
    /// Best (lowest) rank among all rhyme family tokens.
    best_rank: usize,
    /// The rhyme word with the best rank.
    best_word: String,
    /// Probability of the best-ranked rhyme token (softmax).
    best_prob: f32,
    /// Number of rhyme tokens found in top 100.
    in_top_100: usize,
}

/// Map rhyme family words to their token IDs.
///
/// Returns `(token_id, word)` pairs for words that tokenize to a single
/// token (with a leading space, as they'd appear mid-sentence).
fn rhyme_word_token_ids(
    tokenizer: &MITokenizer,
    rhyme_words: &[&str],
) -> candle_mi::Result<Vec<(u32, String)>> {
    let mut pairs = Vec::new();
    for word in rhyme_words {
        let with_space = format!(" {word}");
        let ids = tokenizer.encode_raw(&with_space)?;
        // Only use words that tokenize to a single token (the space + word)
        if let [single] = ids.as_slice() {
            pairs.push((*single, (*word).to_lowercase()));
        } else if let Some(&last) = ids.last() {
            // Multi-token: use the last subword as a proxy
            pairs.push((last, (*word).to_lowercase()));
        }
    }
    Ok(pairs)
}

/// Compute ranks of rhyme tokens in a logit distribution.
///
/// `logits` should be a flat `[vocab_size]` tensor.
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
fn rank_rhyme_tokens(
    logits: &Tensor,
    rhyme_token_ids: &[(u32, String)],
) -> candle_mi::Result<RhymeRankResult> {
    // PROMOTE: ensure F32 for softmax
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let probs = candle_nn::ops::softmax(&logits_f32, 0)?;
    let probs_vec: Vec<f32> = probs.to_vec1()?;

    // Build (index, prob) and sort descending by probability
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build rank lookup: token_id → rank (0-based)
    let mut rank_of = vec![0_usize; probs_vec.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        if let Some(slot) = rank_of.get_mut(idx) {
            *slot = rank;
        }
    }

    let mut best_rank = usize::MAX;
    let mut best_word = String::new();
    let mut best_prob = 0.0_f32;
    let mut in_top_100 = 0_usize;

    for (token_id, word) in rhyme_token_ids {
        // CAST: u32 → usize, token IDs are always valid vocab indices
        let tid = *token_id as usize;
        if let (Some(&r), Some(&p)) = (rank_of.get(tid), probs_vec.get(tid)) {
            if r < best_rank {
                best_rank = r;
                best_word.clone_from(word);
                best_prob = p;
            }
            if r < 100 {
                in_top_100 += 1;
            }
        }
    }

    if best_rank == usize::MAX {
        best_rank = probs_vec.len();
    }

    Ok(RhymeRankResult {
        best_rank,
        best_word,
        best_prob,
        in_top_100,
    })
}

/// Check if a word matches any member of the rhyme family.
fn word_rhymes(word: &str, rhyme_family: &[String]) -> bool {
    let clean = word
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();
    rhyme_family.contains(&clean)
}
