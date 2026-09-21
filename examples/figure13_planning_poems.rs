// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replication of Anthropic's "Planning in Poems" Figure 13: suppress + inject
//! position sweep.
//!
//! Suppresses natural rhyme group CLT features while injecting an alternative
//! feature, sweeping the injection position across the prompt to locate
//! planning sites.
//!
//! Eight built-in presets select model, CLT, prompt, features, and strength:
//!
//! | Preset | Model | CLT | Suppress | Inject |
//! |--------|-------|-----|----------|--------|
//! | `llama3.2-1b-524k` | Llama 3.2 1B | mntss 524K | -ee group: L13:30985 + L9:5488 + L14:27874 + L13:32049 | L14:13043 (`that`) |
//! | `gemma2-2b-426k` | Gemma 2 2B | mntss 426K | L16:13725 + L25:9385 (-out) | L22:10243 (`around`) |
//! | `gemma2-2b-2.5m` | Gemma 2 2B | mntss 2.5M | L25:57092 + L23:49923 + L20:77102 (-out) | L25:82839 (`can`) |
//! | `qwen3-1.7b-20k-ation` | `Qwen3-1.7B-Base` | `BlueLightAI` 20K | L15:263 + L18:3801 + L18:4404 (-ation) | L21:3908 (-self, `cos→" myself"` = 0.39) |
//! | `qwen3-1.7b-20k-teen`  | `Qwen3-1.7B-Base` | `BlueLightAI` 20K | L27:16975 + L20:3668 + L18:10986 (-teen) | L15:263 (-ation cluster-broad) |
//! | `qwen3-0.6b-20k-ation` | `Qwen3-0.6B-Base` | `BlueLightAI` 20K | L19:9578 + L0:8867 + L25:4979 (-ation) | L22:4081 (-self, `cos→" myself"` = 0.42) |
//! | `qwen3-0.6b-20k-teen`  | `Qwen3-0.6B-Base` | `BlueLightAI` 20K | L27:16425 + L23:15839 + L26:6308 (-teen) | L19:9578 (-ation cluster-broad) |
//! | `qwen3-0.6b-16k-ation` | `Qwen3-0.6B-Base` | `BlueLightAI`-dev 16K | L23:11154 + L20:10987 + L14:10719 (-ation) | L22:8011 (-self, `cos→" myself"` = 0.30) |
//!
//! The five `qwen3-*` presets are populated from `JumpReLU` `CltSplit`
//! vocabulary scans against `BlueLightAI`'s `Qwen3` `CLT`s
//! (`bluelightai/clt-qwen3-{1.7b,0.6b}-base-20k`,
//! `bluelightai-dev/clt-Qwen3-0.6B-Base-16k-test`).  Per pairing, suppress
//! is the top-3 features (by `max_cosine`) of the prompt's natural rhyme
//! group.  Inject is one of two complementary picks: for `-ation` poems the
//! inject feature targets `" myself"`-cosine specifically (a narrow,
//! word-level inject works because the prompt has no natural `-self`
//! prior to displace); for `-teen` poems the inject feature is the
//! cluster-broad top-1 `EY1 SH AH0 N` feature (which empirically beat the
//! word-narrow `" duration"` pick by 3–14× because the suppress side
//! already clears the natural `-teen` prior).  Regenerate picks with
//! `examples/vocab_scan` + `scripts/vocab_scan_cmudict_filter.py` +
//! `scripts/pick_features.py` + `scripts/pick_inject_feature.py`.
//!
//! ```bash
//! # Llama 3.2 1B (default)
//! cargo run --release --features clt,transformer --example figure13_planning_poems
//!
//! # Gemma 2 2B, 426K CLT
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-426k
//!
//! # Gemma 2 2B, 2.5M CLT (word-level features)
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-2.5m
//!
//! # Qwen3 1.7B Base, BlueLightAI 20K CLT, -ation suppress + -self inject
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-1.7b-20k-ation
//!
//! # Qwen3 0.6B Base, BlueLightAI-dev 16K CLT, -ation suppress + -self inject
//! cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-16k-ation
//! ```
//!
//! Outputs JSON suitable for direct import into Mathematica via
//! `Import["output.json"]`.

#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::SafeTensors;
use serde::Serialize;

use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::{HookPoint, HookSpec, Intervention, MIModel, extract_token_prob};

// Shared Figure-13 cell presets (see module docs).  `#[path]`-included rather
// than declared as its own example: the file lives in a `main.rs`-free
// subdirectory of `examples/`, so Cargo's example auto-discovery skips it.
#[path = "figure13_common/presets.rs"]
mod presets;

use presets::{feature_id, parse_feature, select_preset};

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "figure13_planning_poems")]
#[command(about = "Anthropic Figure 13 replication: suppress + inject position sweep")]
struct Args {
    /// Preset name: one of `llama3.2-1b-524k`, `gemma2-2b-426k`,
    /// `gemma2-2b-2.5m`, `qwen3-1.7b-20k-ation`, `qwen3-1.7b-20k-teen`,
    /// `qwen3-0.6b-20k-ation`, `qwen3-0.6b-20k-teen`, or
    /// `qwen3-0.6b-16k-ation`.
    #[arg(long, default_value = "llama3.2-1b-524k")]
    preset: String,

    /// `HuggingFace` model ID (overrides preset)
    #[arg(long)]
    model: Option<String>,

    /// `HuggingFace` CLT repository (overrides preset)
    #[arg(long)]
    clt_repo: Option<String>,

    /// Prompt text (overrides preset)
    #[arg(long)]
    prompt: Option<String>,

    /// Word to suppress (overrides preset)
    #[arg(long)]
    suppress_word: Option<String>,

    /// Word to inject (overrides preset)
    #[arg(long)]
    inject_word: Option<String>,

    /// Suppress features in "layer:index" format; repeatable (overrides preset)
    #[arg(long)]
    suppress_feature: Vec<String>,

    /// Inject feature in "layer:index" format (overrides preset)
    #[arg(long)]
    inject_feature: Option<String>,

    /// Steering strength (overrides preset).  Ignored when `--strength-grid`
    /// is set.
    #[arg(long)]
    strength: Option<f32>,

    /// Strength grid (comma-separated `f32`s) for a 2D position × strength
    /// sweep, e.g. `--strength-grid 0.5,1,2.5,5,10,25,50,100`.  When set,
    /// `--strength` is ignored, the position sweep is rerun for each
    /// strength, and the output gains `sweep_grid` + `best_*` fields; the
    /// top-level `sweep` is populated from the best (strength, position)
    /// row in the grid (for `Mathematica` `Import` backward compat).
    #[arg(long, value_delimiter = ',')]
    strength_grid: Vec<f32>,

    /// Skip the suppress side of the intervention; inject the inject feature
    /// alone at each position.  Tests whether the redirect requires both
    /// halves (suppress + inject) or whether the inject decoder vector
    /// suffices as a pure additive steering direction.  Recorded in the
    /// output JSON as `no_suppress: true`.
    #[arg(long, default_value_t = false)]
    no_suppress: bool,

    /// Output file path (defaults to stdout)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Exp 3a random-feature control: draw `N` random CLT features from the
    /// inject feature's source layer (or `N:LAYER` to override the layer) and
    /// run the suppress + random-inject sweep for each, measuring both
    /// `P(target)` and `P(the drawn feature's own top decoder token)`. Setting
    /// this (or `--random-direction`) switches the example into random-control
    /// mode: the normal grid sweep is skipped and a `random_inject`-shaped JSON
    /// is written instead. Draws are seeded by `--seed` (logged in the output).
    #[arg(long)]
    random_inject: Option<String>,

    /// Exp 3a random-direction control: draw `N` Gaussian residual-stream
    /// directions (or `N:SEED` to override the base seed), each norm-matched
    /// per downstream layer to the real inject feature's decoder-vector norm,
    /// and run the suppress + random-direction sweep for each, measuring
    /// `P(target)`. Switches the example into random-control mode.
    #[arg(long)]
    random_direction: Option<String>,

    /// Base RNG seed for the Exp 3a random controls (Date-independent;
    /// per-draw seeds are derived deterministically and logged). Also seeds the
    /// Exp 3b `--random-init` model.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Exp 3b random-model control ("dead salmon"): build the model from config
    /// with seeded Gaussian-random weights (no trained weight values read)
    /// instead of the trained checkpoint, then run the standard suppress+inject
    /// sweep with the REAL CLT. Seeded by `--seed`. Registered prediction: no
    /// target spike at any position, unstable across seeds.
    #[arg(long, default_value_t = false)]
    random_init: bool,

    /// Exp 3b random-model control, stricter variant: permute the elements of
    /// every trained weight tensor (seeded by `--seed`), preserving each
    /// tensor's norm/scale statistics while destroying learned structure. Rules
    /// out "the effect is just the weight scales". Mutually exclusive with
    /// `--random-init`.
    #[arg(long, default_value_t = false)]
    shuffle_weights: bool,
}

// ── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SweepOutput {
    model: String,
    clt_repo: String,
    prompt: String,
    tokens: Vec<String>,
    suppress_word: String,
    inject_word: String,
    suppress_features: Vec<CltFeatureId>,
    inject_feature: CltFeatureId,
    /// `true` when `--no-suppress` was passed; suppression skipped, inject
    /// applied alone.
    #[serde(default)]
    no_suppress: bool,
    /// In single-strength mode this is the chosen strength; in `sweep_grid`
    /// mode it is the *best* strength (the one yielding the highest
    /// `prob / baseline_prob` across all positions).
    strength: f32,
    baseline_prob: f32,
    /// Per-position results at `strength` above (i.e. the best strength's
    /// row when `sweep_grid` is `Some`).  Kept at the top level for
    /// backwards-compatible `Mathematica` `Import` of legacy single-strength
    /// outputs.
    sweep: Vec<PositionResult>,
    /// Full 2D grid present only when `--strength-grid` was passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    sweep_grid: Option<Vec<StrengthRow>>,
    /// Top-position probability ratio (`prob / baseline_prob`) at the best
    /// (strength, position) cell.  Present only in grid mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_ratio: Option<f32>,
    /// Position index of the best cell in the grid.  Present only in grid
    /// mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    best_position: Option<usize>,
    /// The Exp 3b dead-salmon weight source (e.g. `random_init(seed=0, std=0.02)`
    /// or `shuffled(seed=0)`), absent for a normal trained-weight run.
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_source: Option<String>,
}

#[derive(Serialize, Clone)]
struct PositionResult {
    position: usize,
    token: String,
    prob: f32,
}

#[derive(Serialize, Clone)]
struct StrengthRow {
    strength: f32,
    sweep: Vec<PositionResult>,
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // --- Select preset ---
    let preset = select_preset(&args.preset)?;

    // --- Resolve experiment parameters (CLI overrides preset) ---
    // BORROW: .to_owned() — convert &'static str to String for owned storage
    let model_id = args.model.unwrap_or_else(|| preset.model.to_owned());
    let clt_repo = args.clt_repo.unwrap_or_else(|| preset.clt_repo.to_owned());
    let prompt = args.prompt.unwrap_or_else(|| preset.prompt.to_owned());
    let suppress_word = args
        .suppress_word
        .unwrap_or_else(|| preset.suppress_word.to_owned());
    let inject_word = args
        .inject_word
        .unwrap_or_else(|| preset.inject_word.to_owned());
    let strength = args.strength.unwrap_or(preset.strength);

    let suppress_features: Vec<CltFeatureId> = if args.no_suppress {
        // EXPLICIT: empty Vec disables the suppress half of the intervention.
        Vec::new()
    } else if args.suppress_feature.is_empty() {
        preset
            .suppress_features
            .iter()
            .copied()
            .map(feature_id)
            .collect()
    } else {
        args.suppress_feature
            .iter()
            .map(|s| parse_feature(s))
            .collect::<candle_mi::Result<Vec<_>>>()?
    };

    let inject_feature = match &args.inject_feature {
        Some(s) => parse_feature(s)?,
        None => feature_id(preset.inject_feature),
    };

    eprintln!("=== Figure 13: Suppress + Inject Position Sweep ===\n");
    eprintln!("Preset:   {}", args.preset);
    eprintln!("Model:    {model_id}");
    eprintln!("CLT:      {clt_repo}");
    if args.no_suppress {
        eprintln!("Suppress: SKIPPED (--no-suppress; inject-only mode)");
    } else {
        eprintln!("Suppress: \"{suppress_word}\" features {suppress_features:?}");
    }
    eprintln!("Inject:   \"{inject_word}\" feature {inject_feature}");

    // --- Exp 3a random-control mode (skips the normal grid sweep) ---
    if args.random_inject.is_some() || args.random_direction.is_some() {
        let ri = args
            .random_inject
            .as_deref()
            .map(|s| parse_n_opt(s, inject_feature.layer))
            .transpose()?;
        let rd = args
            .random_direction
            .as_deref()
            .map(|s| parse_n_seed(s, args.seed))
            .transpose()?;
        eprintln!(
            "Mode:     Exp 3a random controls (base seed {})\n",
            args.seed
        );
        return run_random_controls(&RandomControlCfg {
            model_id: &model_id,
            clt_repo: &clt_repo,
            prompt: &prompt,
            inject_word: &inject_word,
            suppress_features: &suppress_features,
            real_inject: inject_feature,
            strength,
            seed: args.seed,
            random_inject: ri,
            random_direction: rd,
            output_path: args.output.as_deref(),
        });
    }

    if args.strength_grid.is_empty() {
        eprintln!("Strength: {strength} (single-strength mode)\n");
    } else {
        eprintln!("Strength: grid {:?} (2D sweep mode)\n", args.strength_grid);
    }

    let weight_mode = match (args.random_init, args.shuffle_weights) {
        (true, true) => {
            return Err(candle_mi::MIError::Config(
                "--random-init and --shuffle-weights are mutually exclusive".into(),
            ));
        }
        (true, false) => WeightMode::RandomInit(args.seed),
        (false, true) => WeightMode::Shuffled(args.seed),
        (false, false) => WeightMode::Trained,
    };

    run_experiment(
        &model_id,
        &clt_repo,
        &prompt,
        &suppress_word,
        &inject_word,
        &suppress_features,
        inject_feature,
        strength,
        &args.strength_grid,
        args.no_suppress,
        weight_mode,
        args.output.as_deref(),
    )
}

/// Which weights to run the sweep on (Exp 3b random-model controls select a
/// non-trained variant).
#[derive(Clone, Copy)]
enum WeightMode {
    /// The real trained checkpoint.
    Trained,
    /// Fresh seeded `N(0, 0.02)` weights (dead-salmon random init).
    RandomInit(u64),
    /// Seeded per-tensor element permutation of the trained weights.
    Shuffled(u64),
}

/// Load model + CLT, run the position sweep, print summary, and write output.
///
/// `strength` is the single-strength choice; `strength_grid` selects 2D mode
/// when non-empty (and `strength` is ignored).
#[allow(clippy::too_many_arguments)]
fn run_experiment(
    model_id: &str,
    clt_repo_name: &str,
    prompt: &str,
    suppress_word: &str,
    inject_word: &str,
    suppress_features: &[CltFeatureId],
    inject_feature: CltFeatureId,
    strength: f32,
    strength_grid: &[f32],
    no_suppress: bool,
    weight_mode: WeightMode,
    output_path: Option<&Path>,
) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    // --- Load model (real weights, or an Exp 3b dead-salmon variant) ---
    let model = match weight_mode {
        WeightMode::RandomInit(seed) => {
            eprintln!(
                "Building RANDOM-INIT model (Exp 3b dead-salmon control, seed {seed}, std 0.02)..."
            );
            // 0.02 = the usual transformer weight-init scale.
            MIModel::from_pretrained_random_init(model_id, seed, 0.02)?
        }
        WeightMode::Shuffled(seed) => {
            eprintln!("Building WEIGHT-SHUFFLED model (Exp 3b strict control, seed {seed})...");
            MIModel::from_pretrained_shuffled(model_id, seed)?
        }
        WeightMode::Trained => {
            eprintln!("Loading model...");
            MIModel::from_pretrained(model_id)?
        }
    };
    let weight_source = match weight_mode {
        WeightMode::Trained => None,
        WeightMode::RandomInit(seed) => Some(format!("random_init(seed={seed}, std=0.02)")),
        WeightMode::Shuffled(seed) => Some(format!("shuffled(seed={seed})")),
    };
    let n_layers = model.num_layers();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!(
        "Model: {n_layers} layers, {} hidden, device={device:?}",
        model.hidden_size()
    );

    // --- Open CLT + cache steering vectors ---
    eprintln!("Opening CLT: {clt_repo_name}...");
    let mut clt = CrossLayerTranscoder::open(clt_repo_name)?;
    let mut all_features: Vec<CltFeatureId> = suppress_features.to_vec();
    all_features.push(inject_feature);
    eprintln!("Caching decoder vectors for all downstream layers...");
    clt.cache_steering_vectors_all_downstream(&all_features, &device)?;

    // --- Tokenize ---
    let prompt_with_space = format!("{prompt} ");
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();
    eprintln!("Tokens ({seq_len}): {token_strs:?}");

    let inject_token_id = tokenizer.find_token_id(inject_word)?;
    let inject_token_str = tokenizer.decode_token(inject_token_id)?;
    eprintln!("Inject token: \"{inject_token_str}\" (id={inject_token_id})");

    // --- Build feature entries for all downstream layers ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = suppress_features
        .iter()
        .flat_map(|feat| (feat.layer..n_layers).map(move |l| (*feat, l)))
        .collect();
    let inject_entries: Vec<(CltFeatureId, usize)> = (inject_feature.layer..n_layers)
        .map(|l| (inject_feature, l))
        .collect();
    eprintln!(
        "Suppress: {} entries across {} features",
        suppress_entries.len(),
        suppress_features.len()
    );
    eprintln!(
        "Inject: {} entries (layers {}–{})",
        inject_entries.len(),
        inject_feature.layer,
        n_layers - 1
    );

    // --- Baseline (no intervention) ---
    eprintln!("\nRunning baseline...");
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let result = model.forward(&input, &HookSpec::new())?;
    let baseline_prob = extract_token_prob(result.output(), inject_token_id)?;
    eprintln!("Baseline P(\"{inject_token_str}\") = {baseline_prob:.6e}");

    // --- Position sweep (single-strength) or 2D grid (position × strength) ---
    let (final_strength, best_positions, sweep_grid_opt, best_meta) = if strength_grid.is_empty() {
        let positions = sweep_positions(
            &model,
            &clt,
            &input,
            seq_len,
            &token_strs,
            &suppress_entries,
            &inject_entries,
            strength,
            inject_token_id,
            baseline_prob,
            &device,
        )?;
        (strength, positions, None, None)
    } else {
        let mut rows: Vec<StrengthRow> = Vec::with_capacity(strength_grid.len());
        // (strength, position, ratio) of the best cell across the grid.
        let mut best: (f32, usize, f32) = (0.0, 0, 0.0);
        for &s in strength_grid {
            eprintln!("\n--- Strength {s} ---");
            let positions = sweep_positions(
                &model,
                &clt,
                &input,
                seq_len,
                &token_strs,
                &suppress_entries,
                &inject_entries,
                s,
                inject_token_id,
                baseline_prob,
                &device,
            )?;
            for p in &positions {
                let ratio = if baseline_prob > 0.0 {
                    p.prob / baseline_prob
                } else {
                    0.0
                };
                if ratio > best.2 {
                    best = (s, p.position, ratio);
                }
            }
            // BORROW: positions moves into the row; no clone.
            rows.push(StrengthRow {
                strength: s,
                sweep: positions,
            });
        }
        // EXPLICIT: linear search across at most ~10 rows; building an index
        // would obscure intent.
        // BORROW: r.sweep.clone() — `rows` is still needed below for the
        // `sweep_grid` field; we need the best row's positions duplicated at
        // the top level for Mathematica `Import` backward compat.
        let best_row_sweep = rows
            .iter()
            .find(|r| (r.strength - best.0).abs() < f32::EPSILON)
            .map_or_else(Vec::new, |r| r.sweep.clone());
        (best.0, best_row_sweep, Some(rows), Some((best.2, best.1)))
    };

    // --- Summary (best row in grid mode; the only row in single mode) ---
    if let Some((ratio, pos)) = best_meta {
        eprintln!(
            "\n=== Best cell across the strength grid: strength={final_strength}, \
             position={pos}, ratio={ratio:.2}x ===",
        );
    }
    print_sweep_summary(&best_positions, baseline_prob, &token_strs);

    // --- JSON output ---
    let output = SweepOutput {
        model: model_id.into(),
        clt_repo: clt_repo_name.into(),
        prompt: prompt.into(),
        tokens: token_strs,
        suppress_word: suppress_word.into(),
        inject_word: inject_word.into(),
        suppress_features: suppress_features.to_vec(),
        inject_feature,
        no_suppress,
        strength: final_strength,
        baseline_prob,
        sweep: best_positions,
        sweep_grid: sweep_grid_opt,
        best_ratio: best_meta.map(|(r, _)| r),
        best_position: best_meta.map(|(_, p)| p),
        weight_source,
    };
    write_sweep_output(&output, output_path)?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

/// Run the position sweep and print progress.
#[allow(clippy::too_many_arguments)]
fn sweep_positions(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    inject_entries: &[(CltFeatureId, usize)],
    strength: f32,
    inject_token_id: u32,
    baseline_prob: f32,
    device: &candle_core::Device,
) -> candle_mi::Result<Vec<PositionResult>> {
    eprintln!("\nSweeping {seq_len} positions (strength={strength})...");
    let mut positions: Vec<PositionResult> = Vec::with_capacity(seq_len);

    for pos in 0..seq_len {
        let mut combined =
            clt.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        let inject_hooks =
            clt.prepare_hook_injection(inject_entries, pos, seq_len, strength, device)?;
        combined.extend(&inject_hooks);

        let result = model.forward(input, &combined)?;
        let p_inject = extract_token_prob(result.output(), inject_token_id)?;

        positions.push(PositionResult {
            position: pos,
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            prob: p_inject,
        });

        let delta = p_inject - baseline_prob;
        let marker = if delta > baseline_prob * 10.0 && delta > 1e-12 {
            " ***"
        } else if delta > baseline_prob && delta > 1e-12 {
            " *"
        } else {
            ""
        };
        // BORROW: explicit .as_str() — String to &str for display
        let display = token_strs
            .get(pos)
            .map_or("?", String::as_str)
            .replace('\n', "\\n");
        eprintln!("  pos {pos:>3}  {display:<20}  P={p_inject:.6e}  delta={delta:+.6e}{marker}");
    }

    Ok(positions)
}

/// Print the sweep summary to stderr.
fn print_sweep_summary(positions: &[PositionResult], baseline_prob: f32, token_strs: &[String]) {
    let (max_pos, max_p) = positions
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.prob
                .partial_cmp(&b.prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or((0, 0.0), |(i, p)| (i, p.prob));

    let ratio = if baseline_prob > 0.0 {
        max_p / baseline_prob
    } else {
        0.0
    };

    eprintln!("\n=== Results ===");
    eprintln!("Baseline:   {baseline_prob:.6e}");
    // BORROW: explicit .as_str() — String to &str for display
    eprintln!(
        "Max P:      {max_p:.6e} at position {max_pos} (\"{}\")  ratio={ratio:.1}x",
        token_strs
            .get(max_pos)
            .map_or("?", String::as_str)
            .replace('\n', "\\n")
    );
}

/// Serialize sweep results to JSON; write to file or stdout.
fn write_sweep_output(output: &SweepOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;

    if let Some(p) = path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
            })?;
        }
        fs::write(p, &json)
            .map_err(|e| candle_mi::MIError::Config(format!("write output: {e}")))?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

// ── Exp 3a: random-feature / random-direction inject controls ────────────────
//
// The paper's positive result is a suppress + inject sweep whose target-word
// probability spikes at the final token. Two "dead-salmon" questions the
// reproducibility track will ask: (2) is the spike specific to the *chosen*
// feature, or does any strong write-direction at the final token move any
// target? and (secondary) does a random feature spike its *own* decoder token?
// This mode holds the suppress side and strength fixed and replaces the inject
// with N random draws:
//   * `--random-inject N[:LAYER]` — N random CLT features from the inject
//     feature's source layer (LAYER overrides), measuring P(target) and
//     P(the drawn feature's own top decoder token).
//   * `--random-direction N[:SEED]` — N Gaussian residual directions,
//     norm-matched per downstream layer to the real decoder-vector norm.
// Registered prediction: P(target) stays flat within 10x of baseline for every
// draw (against 1e5-1e7x for the real feature).

/// Parse a `"N"` or `"N:SECOND"` argument. `default_second` fills the second
/// field when `:SECOND` is omitted.
fn parse_n_opt(s: &str, default_second: usize) -> candle_mi::Result<(usize, usize)> {
    let cfg_err = |m: String| candle_mi::MIError::Config(m);
    match s.split_once(':') {
        Some((n, second)) => Ok((
            n.parse()
                .map_err(|e| cfg_err(format!("invalid N '{n}': {e}")))?,
            second
                .parse()
                .map_err(|e| cfg_err(format!("invalid ':' field '{second}': {e}")))?,
        )),
        None => Ok((
            s.parse()
                .map_err(|e| cfg_err(format!("invalid N '{s}': {e}")))?,
            default_second,
        )),
    }
}

/// Parse a `"N"` or `"N:SEED"` argument (the `--random-direction` form), with a
/// `u64` seed field. `default_seed` fills the seed when `:SEED` is omitted.
fn parse_n_seed(s: &str, default_seed: u64) -> candle_mi::Result<(usize, u64)> {
    let cfg_err = |m: String| candle_mi::MIError::Config(m);
    match s.split_once(':') {
        Some((n, sd)) => Ok((
            n.parse()
                .map_err(|e| cfg_err(format!("invalid N '{n}': {e}")))?,
            sd.parse()
                .map_err(|e| cfg_err(format!("invalid seed '{sd}': {e}")))?,
        )),
        None => Ok((
            s.parse()
                .map_err(|e| cfg_err(format!("invalid N '{s}': {e}")))?,
            default_seed,
        )),
    }
}

/// Parameters for [`run_random_controls`] (grouped to avoid a many-argument fn).
struct RandomControlCfg<'a> {
    model_id: &'a str,
    clt_repo: &'a str,
    prompt: &'a str,
    inject_word: &'a str,
    suppress_features: &'a [CltFeatureId],
    real_inject: CltFeatureId,
    strength: f32,
    seed: u64,
    /// `(n_draws, source_layer)` for the random-feature control.
    random_inject: Option<(usize, usize)>,
    /// `(n_draws, seed)` for the random-direction control.
    random_direction: Option<(usize, u64)>,
    output_path: Option<&'a Path>,
}

#[derive(Serialize, Clone)]
struct RandPositionResult {
    position: usize,
    token: String,
    p_target: f32,
    /// P(the drawn feature's own top decoder token) — random-inject draws only.
    #[serde(skip_serializing_if = "Option::is_none")]
    p_own: Option<f32>,
}

#[derive(Serialize)]
struct RealInjectRow {
    feature: CltFeatureId,
    max_p_target: f32,
    max_ratio_target: f32,
    max_position: usize,
    per_position: Vec<RandPositionResult>,
}

#[derive(Serialize)]
struct RandomInjectDraw {
    draw: usize,
    feature: CltFeatureId,
    own_top_token: String,
    own_top_token_id: u32,
    max_p_target: f32,
    max_ratio_target: f32,
    max_position_target: usize,
    max_p_own: f32,
    max_position_own: usize,
    per_position: Vec<RandPositionResult>,
}

#[derive(Serialize)]
struct RandomDirectionDraw {
    draw: usize,
    seed: u64,
    max_p_target: f32,
    max_ratio_target: f32,
    max_position_target: usize,
    per_position: Vec<RandPositionResult>,
}

#[derive(Serialize)]
struct RandomControlOutput {
    model: String,
    clt_repo: String,
    prompt: String,
    tokens: Vec<String>,
    target_word: String,
    target_token_id: u32,
    baseline_prob: f32,
    strength: f32,
    seed: u64,
    real_inject: RealInjectRow,
    #[serde(skip_serializing_if = "Option::is_none")]
    random_inject_layer: Option<usize>,
    random_inject: Vec<RandomInjectDraw>,
    random_direction: Vec<RandomDirectionDraw>,
    /// Registered decision-criterion summary: the real feature's max ratio and
    /// the worst-case (max over draws) ratio under each random control.
    real_max_ratio: f32,
    random_inject_max_ratio: f32,
    random_direction_max_ratio: f32,
}

/// Max `p_target` over positions and its position.
fn max_target(positions: &[RandPositionResult]) -> (f32, usize) {
    positions
        .iter()
        .max_by(|a, b| {
            a.p_target
                .partial_cmp(&b.p_target)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or((0.0, 0), |p| (p.p_target, p.position))
}

/// Max `p_own` over positions and its position (random-inject draws).
fn max_own(positions: &[RandPositionResult]) -> (f32, usize) {
    positions
        .iter()
        .filter_map(|p| p.p_own.map(|v| (v, p.position)))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0.0, 0))
}

fn ratio_of(p: f32, baseline: f32) -> f32 {
    if baseline > 0.0 { p / baseline } else { 0.0 }
}

/// A unit-norm Gaussian vector of length `d` via Box-Muller (no `rand_distr`
/// dependency). Deterministic given `rng`'s seed.
// The `f64 → f32` narrowings below store each standard-normal sample at model
// (`F32`) precision; the lost mantissa bits are far below steering noise.
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
fn gaussian_unit_vec(rng: &mut StdRng, d: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(d);
    while v.len() < d {
        // Box-Muller: two uniforms give two independent standard normals.
        let u1 = rng.r#gen::<f64>().max(1e-12);
        let u2 = rng.r#gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let ang = 2.0 * std::f64::consts::PI * u2;
        // CAST: f64 → f32, sample stored at model precision.
        v.push((r * ang.cos()) as f32);
        if v.len() < d {
            // CAST: f64 → f32, as above (Box-Muller's second sample).
            v.push((r * ang.sin()) as f32);
        }
    }
    v.truncate(d);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Append an `Add(vector)` intervention at `ResidPost(target_layer)`, placing
/// `strength * vector` at `position` and zeros elsewhere — the same
/// construction as [`CrossLayerTranscoder::prepare_hook_injection`], but for a
/// raw (non-CLT) direction.
fn add_vector_at_position(
    hooks: &mut HookSpec,
    target_layer: usize,
    vector: &Tensor,
    position: usize,
    seq_len: usize,
    strength: f32,
    device: &Device,
) -> candle_mi::Result<()> {
    let scaled = (vector * f64::from(strength))?; // [d_model]
    // Delegates to the crate helper, which bounds-checks `position`. The inline
    // narrow/cat this replaced did not.
    let _ = device;
    let injection = candle_mi::steering::position_delta(&scaled, position, seq_len)?;
    hooks.intervene(
        HookPoint::ResidPost(target_layer),
        Intervention::Add(injection),
    );
    Ok(())
}

/// One suppress + CLT-inject sweep, measuring `P(target)` (and optionally
/// `P(own_token)`) at every position.
#[allow(clippy::too_many_arguments)]
fn sweep_clt_inject(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    inject_entries: &[(CltFeatureId, usize)],
    strength: f32,
    target_id: u32,
    own_id: Option<u32>,
    device: &Device,
) -> candle_mi::Result<Vec<RandPositionResult>> {
    let mut out: Vec<RandPositionResult> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut combined =
            clt.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        let inj = clt.prepare_hook_injection(inject_entries, pos, seq_len, strength, device)?;
        combined.extend(&inj);
        let res = model.forward(input, &combined)?;
        let p_target = extract_token_prob(res.output(), target_id)?;
        let p_own = match own_id {
            Some(id) => Some(extract_token_prob(res.output(), id)?),
            None => None,
        };
        out.push(RandPositionResult {
            position: pos,
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            p_target,
            p_own,
        });
    }
    Ok(out)
}

/// One suppress + raw-direction sweep, measuring `P(target)` at every position.
/// `vecs` are `(target_layer, direction)` pairs (already norm-matched).
#[allow(clippy::too_many_arguments)]
fn sweep_direction(
    model: &MIModel,
    clt: &CrossLayerTranscoder,
    input: &Tensor,
    seq_len: usize,
    token_strs: &[String],
    suppress_entries: &[(CltFeatureId, usize)],
    vecs: &[(usize, Tensor)],
    strength: f32,
    target_id: u32,
    device: &Device,
) -> candle_mi::Result<Vec<RandPositionResult>> {
    let mut out: Vec<RandPositionResult> = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let mut combined =
            clt.prepare_hook_injection(suppress_entries, pos, seq_len, -strength, device)?;
        for (t, v) in vecs {
            add_vector_at_position(&mut combined, *t, v, pos, seq_len, strength, device)?;
        }
        let res = model.forward(input, &combined)?;
        let p_target = extract_token_prob(res.output(), target_id)?;
        out.push(RandPositionResult {
            position: pos,
            token: token_strs.get(pos).cloned().unwrap_or_default(),
            p_target,
            p_own: None,
        });
    }
    Ok(out)
}

/// The drawn feature's top decoder token, by cosine of its decoder vector
/// (projected to `final_layer`) against the model's input embedding rows — the
/// same decoder-map convention used by `vocab_scan` / plip-rs to pick the real
/// inject features. Returns `(token_id, token_str)`.
fn feature_top_token(
    clt: &mut CrossLayerTranscoder,
    fid: &CltFeatureId,
    final_layer: usize,
    embed: &Tensor,
    tokenizer: &candle_mi::MITokenizer,
    device: &Device,
) -> candle_mi::Result<(u32, String)> {
    // The cosine runs on CPU against a CPU-resident `embed` (see
    // `load_embedding_tensor`): the 256K×d_model embedding is already in the
    // model's GPU weights, so a second GPU copy would waste ~2.4 GB of VRAM and
    // spill Gemma 2 2B (F32) into shared memory. This one-shot matmul is cheap.
    let dec = clt
        .decoder_vector(fid, final_layer, device)?
        .to_device(&Device::Cpu)?;
    let dec_col = dec.unsqueeze(1)?; // [d_model, 1]
    // embed [vocab, d_model] @ dec [d_model, 1] -> [vocab]
    let dots = embed.matmul(&dec_col)?.squeeze(1)?;
    let row_norms = embed.sqr()?.sum(1)?.sqrt()?; // [vocab]
    // cos = dots / (|row| * |dec|); |dec| is constant, so argmax(cos) =
    // argmax(dots / |row|).
    let cos = dots.broadcast_div(&row_norms)?;
    let top = cos.argmax(0)?.to_scalar::<u32>()?;
    let tok = tokenizer
        .decode_token(top)
        .unwrap_or_else(|_| format!("[{top}]"));
    Ok((top, tok))
}

/// Load `model.embed_tokens.weight` as an `F32` `[vocab, d_model]` tensor on
/// `device`, from the model's local HF snapshot (must be cached).
///
/// Callers pass `Device::Cpu`: the matrix already lives in the model's GPU
/// weights, so a GPU copy would only duplicate ~2.4 GB of VRAM.
fn load_embedding_tensor(model_id: &str, device: &Device) -> candle_mi::Result<Tensor> {
    let snapshot = find_snapshot(model_id).ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "model '{model_id}' not found in local HF cache (needed for the \
             random-inject own-token readout)"
        ))
    })?;
    let (values, vocab_size, d_model) = load_embedding_matrix(&snapshot)
        .map_err(|e| candle_mi::MIError::Config(format!("embedding load: {e}")))?;
    Ok(Tensor::from_vec(values, (vocab_size, d_model), device)?)
}

fn hf_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
}

fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Extract `model.embed_tokens.weight` as `(values, vocab_size, d_model)` from
/// either a single-file or sharded safetensors layout.
fn load_embedding_matrix(snapshot: &Path) -> Result<(Vec<f32>, usize, usize), String> {
    let tensor_name = "model.embed_tokens.weight";
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        let data = fs::read(&single).map_err(|e| format!("read {}: {e}", single.display()))?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| format!("parse {}: {e}", single.display()))?;
        if st.tensor(tensor_name).is_ok() {
            return extract_embedding(&st, tensor_name);
        }
    }
    let idx_path = snapshot.join("model.safetensors.index.json");
    let idx_str =
        fs::read_to_string(&idx_path).map_err(|e| format!("read {}: {e}", idx_path.display()))?;
    let idx: serde_json::Value =
        serde_json::from_str(&idx_str).map_err(|e| format!("parse {}: {e}", idx_path.display()))?;
    let shard = idx
        .get("weight_map")
        .and_then(|m| m.get(tensor_name))
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("tensor '{tensor_name}' missing from weight_map"))?;
    let shard_path = snapshot.join(shard);
    let data = fs::read(&shard_path).map_err(|e| format!("read {}: {e}", shard_path.display()))?;
    let st = SafeTensors::deserialize(&data)
        .map_err(|e| format!("parse {}: {e}", shard_path.display()))?;
    extract_embedding(&st, tensor_name)
}

fn extract_embedding(st: &SafeTensors<'_>, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
    let view = st
        .tensor(name)
        .map_err(|e| format!("tensor '{name}': {e}"))?;
    let shape = view.shape();
    if shape.len() != 2 {
        return Err(format!("expected 2D embedding, got shape {shape:?}"));
    }
    // INDEX: shape.len() == 2 verified just above.
    #[allow(clippy::indexing_slicing)]
    let (vocab_size, d_model) = (shape[0], shape[1]);
    let bytes = view.data();
    // EXPLICIT: safetensors exposes many dtypes; embeddings in the wild are
    // BF16 (Qwen3/Gemma/Llama BF16 checkpoints) or F32. Others -> error.
    #[allow(clippy::wildcard_enum_match_arm)]
    let values: Vec<f32> = match view.dtype() {
        safetensors::Dtype::BF16 => bf16_bytes_to_f32(bytes),
        safetensors::Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        other => return Err(format!("unsupported embedding dtype {other:?}")),
    };
    Ok((values, vocab_size, d_model))
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // INDEX: i*2 + 1 < n*2 == bytes.len() by construction of n.
        #[allow(clippy::indexing_slicing)]
        let bf16_bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        // `u16 → u32` (via `From`) then shifted into the upper half of `f32`:
        // the canonical `BF16 → F32` widening (shared exponent layout).
        let f32_bits = u32::from(bf16_bits) << 16;
        out.push(f32::from_bits(f32_bits));
    }
    out
}

/// Run the Exp 3a random-feature / random-direction controls and write the
/// `random_inject`-shaped JSON.
#[allow(clippy::too_many_lines)]
fn run_random_controls(cfg: &RandomControlCfg<'_>) -> candle_mi::Result<()> {
    let t_start = std::time::Instant::now();

    // --- Load model + CLT + tokenizer ---
    eprintln!("Loading model...");
    let model = MIModel::from_pretrained(cfg.model_id)?;
    let n_layers = model.num_layers();
    let device = model.device().clone();
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    eprintln!("Opening CLT: {}...", cfg.clt_repo);
    let mut clt = CrossLayerTranscoder::open(cfg.clt_repo)?;

    // --- Tokenize ---
    let prompt_with_space = format!("{} ", cfg.prompt);
    let token_ids = tokenizer.encode(&prompt_with_space)?;
    let seq_len = token_ids.len();
    let token_strs: Vec<String> = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .decode_token(id)
                .unwrap_or_else(|_| format!("[{id}]"))
        })
        .collect();
    let input = Tensor::new(&token_ids[..], &device)?.unsqueeze(0)?;
    let target_id = tokenizer.find_token_id(cfg.inject_word)?;
    eprintln!(
        "Tokens ({seq_len}), target \"{}\" (id={target_id})",
        cfg.inject_word
    );

    // --- Baseline ---
    let baseline_prob =
        extract_token_prob(model.forward(&input, &HookSpec::new())?.output(), target_id)?;
    eprintln!("Baseline P(\"{}\") = {baseline_prob:.6e}", cfg.inject_word);

    // --- Suppress entries (unchanged across all conditions) ---
    let suppress_entries: Vec<(CltFeatureId, usize)> = cfg
        .suppress_features
        .iter()
        .flat_map(|f| (f.layer..n_layers).map(move |l| (*f, l)))
        .collect();

    // --- Draw random-inject features (seeded) ---
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let n_feat = clt.n_features_per_layer();
    let ri_features: Vec<CltFeatureId> = match cfg.random_inject {
        Some((n, layer)) => {
            if layer >= clt.n_layers() {
                return Err(candle_mi::MIError::Config(format!(
                    "random-inject layer {layer} out of range (CLT has {} layers)",
                    clt.n_layers()
                )));
            }
            (0..n)
                .map(|_| CltFeatureId {
                    layer,
                    index: rng.gen_range(0..n_feat),
                })
                .collect()
        }
        None => Vec::new(),
    };

    // --- Cache all decoder vectors we will inject (batched by source layer) ---
    let mut to_cache: Vec<CltFeatureId> = cfg.suppress_features.to_vec();
    to_cache.push(cfg.real_inject);
    to_cache.extend_from_slice(&ri_features);
    clt.cache_steering_vectors_all_downstream(&to_cache, &device)?;

    // --- Real-inject reference row ---
    eprintln!("Real inject {} sweep...", cfg.real_inject);
    let real_entries: Vec<(CltFeatureId, usize)> = (cfg.real_inject.layer..n_layers)
        .map(|l| (cfg.real_inject, l))
        .collect();
    let real_positions = sweep_clt_inject(
        &model,
        &clt,
        &input,
        seq_len,
        &token_strs,
        &suppress_entries,
        &real_entries,
        cfg.strength,
        target_id,
        None,
        &device,
    )?;
    let (real_max_p, real_max_pos) = max_target(&real_positions);
    let real_max_ratio = ratio_of(real_max_p, baseline_prob);
    eprintln!("  real: max P={real_max_p:.4e} at pos {real_max_pos}  ratio={real_max_ratio:.1}x");

    // --- Random-inject draws ---
    let embed = if ri_features.is_empty() {
        None
    } else {
        eprintln!("Loading embedding matrix (CPU) for own-token readout...");
        // CPU-resident on purpose — avoids duplicating the model's embedding on
        // the GPU (would spill Gemma 2 2B F32 past 16 GB into shared memory).
        Some(load_embedding_tensor(cfg.model_id, &Device::Cpu)?)
    };
    let mut random_inject: Vec<RandomInjectDraw> = Vec::new();
    for (i, fid) in ri_features.iter().enumerate() {
        let (own_id, own_tok) = feature_top_token(
            &mut clt,
            fid,
            n_layers - 1,
            embed
                .as_ref()
                .ok_or_else(|| candle_mi::MIError::Config("embedding not loaded".into()))?,
            tokenizer,
            &device,
        )?;
        let entries: Vec<(CltFeatureId, usize)> =
            (fid.layer..n_layers).map(|l| (*fid, l)).collect();
        let positions = sweep_clt_inject(
            &model,
            &clt,
            &input,
            seq_len,
            &token_strs,
            &suppress_entries,
            &entries,
            cfg.strength,
            target_id,
            Some(own_id),
            &device,
        )?;
        let (mp_t, pos_t) = max_target(&positions);
        let (mp_o, pos_o) = max_own(&positions);
        eprintln!(
            "  random-inject {}/{}: {fid} own=\"{own_tok}\"  max P(target)={mp_t:.4e} \
             (ratio {:.2}x) max P(own)={mp_o:.4e}",
            i + 1,
            ri_features.len(),
            ratio_of(mp_t, baseline_prob)
        );
        random_inject.push(RandomInjectDraw {
            draw: i,
            feature: *fid,
            own_top_token: own_tok,
            own_top_token_id: own_id,
            max_p_target: mp_t,
            max_ratio_target: ratio_of(mp_t, baseline_prob),
            max_position_target: pos_t,
            max_p_own: mp_o,
            max_position_own: pos_o,
            per_position: positions,
        });
    }

    // --- Random-direction draws ---
    let mut random_direction: Vec<RandomDirectionDraw> = Vec::new();
    if let Some((n, dir_seed)) = cfg.random_direction {
        let d_model = clt.d_model();
        let layers: Vec<usize> = (cfg.real_inject.layer..n_layers).collect();
        // Per-layer real decoder-vector norms (the norm we match).
        let mut norms: Vec<f32> = Vec::with_capacity(layers.len());
        for &t in &layers {
            let dv = clt.decoder_vector(&cfg.real_inject, t, &device)?;
            let nrm = dv.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
            norms.push(nrm);
        }
        let mut drng = StdRng::seed_from_u64(dir_seed);
        for i in 0..n {
            let vecs: Vec<(usize, Tensor)> = layers
                .iter()
                .zip(&norms)
                .map(|(&t, &nrm)| {
                    let g = gaussian_unit_vec(&mut drng, d_model);
                    let unit = Tensor::from_vec(g, d_model, &device)?; // [d_model], unit norm
                    let scaled = (unit * f64::from(nrm))?; // norm-matched to real decoder
                    Ok((t, scaled))
                })
                .collect::<candle_mi::Result<Vec<_>>>()?;
            let positions = sweep_direction(
                &model,
                &clt,
                &input,
                seq_len,
                &token_strs,
                &suppress_entries,
                &vecs,
                cfg.strength,
                target_id,
                &device,
            )?;
            let (mp_t, pos_t) = max_target(&positions);
            eprintln!(
                "  random-direction {}/{}: max P(target)={mp_t:.4e} (ratio {:.2}x)",
                i + 1,
                n,
                ratio_of(mp_t, baseline_prob)
            );
            random_direction.push(RandomDirectionDraw {
                draw: i,
                seed: dir_seed,
                max_p_target: mp_t,
                max_ratio_target: ratio_of(mp_t, baseline_prob),
                max_position_target: pos_t,
                per_position: positions,
            });
        }
    }

    // --- Summary ratios (registered decision criterion) ---
    let worst_inject_ratio = random_inject
        .iter()
        .map(|d| d.max_ratio_target)
        .fold(0.0_f32, f32::max);
    let worst_direction_ratio = random_direction
        .iter()
        .map(|d| d.max_ratio_target)
        .fold(0.0_f32, f32::max);

    eprintln!("\n=== Exp 3a summary ===");
    eprintln!("real feature max ratio:      {real_max_ratio:.1}x");
    eprintln!(
        "random-inject max ratio:     {worst_inject_ratio:.2}x (over {} draws)",
        random_inject.len()
    );
    eprintln!(
        "random-direction max ratio:  {worst_direction_ratio:.2}x (over {} draws)",
        random_direction.len()
    );

    let output = RandomControlOutput {
        model: cfg.model_id.into(),
        clt_repo: cfg.clt_repo.into(),
        prompt: cfg.prompt.into(),
        tokens: token_strs,
        target_word: cfg.inject_word.into(),
        target_token_id: target_id,
        baseline_prob,
        strength: cfg.strength,
        seed: cfg.seed,
        real_inject: RealInjectRow {
            feature: cfg.real_inject,
            max_p_target: real_max_p,
            max_ratio_target: real_max_ratio,
            max_position: real_max_pos,
            per_position: real_positions,
        },
        random_inject_layer: cfg.random_inject.map(|(_, l)| l),
        random_inject,
        random_direction,
        real_max_ratio,
        random_inject_max_ratio: worst_inject_ratio,
        random_direction_max_ratio: worst_direction_ratio,
    };

    // Reuse the JSON writer's file/stdout logic.
    let json = serde_json::to_string_pretty(&output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(p) = cfg.output_path {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
            })?;
        }
        fs::write(p, &json)
            .map_err(|e| candle_mi::MIError::Config(format!("write output: {e}")))?;
        eprintln!("\nOutput written to {}", p.display());
    } else {
        println!("{json}");
    }

    eprintln!("Total elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}
