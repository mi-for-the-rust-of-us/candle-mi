// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diagnostic micro-bench for hook-architecture overhead.
//!
//! Decomposes the full-capture overhead into three independently measured
//! components so a `Vec<LayerPlan>` refactor can be evaluated *before*
//! it is written:
//!
//!   A. Pure spec-lookup cost — `is_captured` + `interventions_at` calls
//!      walked in the same order the forward path walks them, with no
//!      tensor work.
//!   B. Real forward overhead — empty `HookSpec` vs full-capture spec
//!      on `Llama-3.2-1B`, mirroring `bench_hook_overhead.rs`.
//!   C. Capture-machinery cost — `Tensor::clone` plus `HashMap::insert`
//!      for one tensor per hook point, with no model in the loop.
//!
//! If A is small relative to (B's delta), then a `LayerPlan` refactor
//! (which only reduces lookup cost) is not worth doing. If C dominates,
//! the hot path is the capture machinery itself and must be addressed
//! differently (e.g., flat `Vec<(HookPoint, Tensor)>`, `SmallVec`-keyed
//! cache, or fewer captures).
//!
//! Run:
//!   `cargo test --test bench_hook_diagnostic --features transformer,mmap --release -- --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    unsafe_code,
    missing_docs
)]

use std::hint::black_box;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{
    GenericTransformer, HookCache, HookPoint, HookSpec, MIBackend, MITokenizer, TransformerConfig,
};

// ---------------------------------------------------------------------------
// Helpers (mirrored from bench_hook_overhead.rs to keep this self-contained)
// ---------------------------------------------------------------------------

fn hf_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory");
}

fn find_snapshot(model_id: &str) -> Option<std::path::PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

fn load_model_on(
    model_id: &str,
    device: &Device,
) -> (GenericTransformer, MITokenizer, TransformerConfig) {
    let snapshot =
        find_snapshot(model_id).unwrap_or_else(|| panic!("{model_id} not found in HF cache"));

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();
    let dtype = DType::F32;

    let single = snapshot.join("model.safetensors");
    let paths = if single.exists() {
        vec![single]
    } else {
        let index_path = snapshot.join("model.safetensors.index.json");
        let index_str = std::fs::read_to_string(&index_path).unwrap();
        let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
        let weight_map = index["weight_map"].as_object().unwrap();
        let mut shard_names: Vec<String> = weight_map
            .values()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        shard_names.sort();
        shard_names.dedup();
        shard_names.iter().map(|name| snapshot.join(name)).collect()
    };

    // SAFETY: mmaped safetensors are read-only and the mapping outlives the
    // `VarBuilder` for the duration of this test; same pattern used across
    // the validate/bench suite.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };

    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();

    (model, tokenizer, config)
}

/// Build a `HookSpec` capturing every per-layer hook point plus globals.
fn full_capture_spec(num_layers: usize) -> HookSpec {
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::Embed);
    for i in 0..num_layers {
        hooks.capture(HookPoint::ResidPre(i));
        hooks.capture(HookPoint::AttnQ(i));
        hooks.capture(HookPoint::AttnK(i));
        hooks.capture(HookPoint::AttnV(i));
        hooks.capture(HookPoint::AttnScores(i));
        hooks.capture(HookPoint::AttnPattern(i));
        hooks.capture(HookPoint::AttnOut(i));
        hooks.capture(HookPoint::ResidMid(i));
        hooks.capture(HookPoint::MlpPre(i));
        hooks.capture(HookPoint::MlpPost(i));
        hooks.capture(HookPoint::MlpOut(i));
        hooks.capture(HookPoint::ResidPost(i));
    }
    hooks.capture(HookPoint::FinalNorm);
    hooks
}

/// Capture `ResidPost` at evenly spaced layers. Single shape per layer
/// (residual stream `[B, T, hidden]`). Used to vary capture count while
/// holding hook *type* constant.
fn resid_post_every_n(num_layers: usize, stride: usize) -> HookSpec {
    let mut hooks = HookSpec::new();
    let mut i = 0;
    while i < num_layers {
        hooks.capture(HookPoint::ResidPost(i));
        i += stride;
    }
    hooks
}

/// Capture a single hook *type* at every layer — used to compare
/// equal-count specs that hold tensors of different shapes / different
/// fusion implications. `selector` chooses which per-layer hook to use.
fn one_hook_per_layer<F: Fn(usize) -> HookPoint>(num_layers: usize, selector: F) -> HookSpec {
    let mut hooks = HookSpec::new();
    for i in 0..num_layers {
        hooks.capture(selector(i));
    }
    hooks
}

/// Walk the spec the same way the transformer forward path walks it —
/// at every per-layer hook point, query both `is_captured` and
/// `interventions_at`. No tensor work, just spec lookups.
#[inline(never)]
fn walk_spec(spec: &HookSpec, num_layers: usize) {
    black_box(spec.is_captured(&HookPoint::Embed));
    black_box(spec.interventions_at(&HookPoint::Embed).next());
    for i in 0..num_layers {
        for hp in [
            HookPoint::ResidPre(i),
            HookPoint::AttnQ(i),
            HookPoint::AttnK(i),
            HookPoint::AttnV(i),
            HookPoint::AttnScores(i),
            HookPoint::AttnPattern(i),
            HookPoint::AttnOut(i),
            HookPoint::ResidMid(i),
            HookPoint::MlpPre(i),
            HookPoint::MlpPost(i),
            HookPoint::MlpOut(i),
            HookPoint::ResidPost(i),
        ] {
            black_box(spec.is_captured(&hp));
            black_box(spec.interventions_at(&hp).next());
        }
    }
    black_box(spec.is_captured(&HookPoint::FinalNorm));
    black_box(spec.interventions_at(&HookPoint::FinalNorm).next());
}

/// Mimic the capture-machinery cost: build a `HookCache` and store one
/// (Arc-cloned) tensor at every hook point that full capture would
/// touch. Same number of inserts as `walk_spec` has lookups.
#[inline(never)]
fn store_all(num_layers: usize, dummy: &Tensor) -> HookCache {
    let mut cache = HookCache::new(dummy.clone());
    cache.store(HookPoint::Embed, dummy.clone());
    for i in 0..num_layers {
        cache.store(HookPoint::ResidPre(i), dummy.clone());
        cache.store(HookPoint::AttnQ(i), dummy.clone());
        cache.store(HookPoint::AttnK(i), dummy.clone());
        cache.store(HookPoint::AttnV(i), dummy.clone());
        cache.store(HookPoint::AttnScores(i), dummy.clone());
        cache.store(HookPoint::AttnPattern(i), dummy.clone());
        cache.store(HookPoint::AttnOut(i), dummy.clone());
        cache.store(HookPoint::ResidMid(i), dummy.clone());
        cache.store(HookPoint::MlpPre(i), dummy.clone());
        cache.store(HookPoint::MlpPost(i), dummy.clone());
        cache.store(HookPoint::MlpOut(i), dummy.clone());
        cache.store(HookPoint::ResidPost(i), dummy.clone());
    }
    cache.store(HookPoint::FinalNorm, dummy.clone());
    cache
}

const MODEL_ID: &str = "meta-llama/Llama-3.2-1B";
const PROMPT: &str = "The capital of France is";
const FWD_WARMUP: usize = 5;
const FWD_RUNS: usize = 100;
const LOOKUP_ITERS: usize = 100_000;
const STORE_ITERS: usize = 10_000;

/// Time `FWD_RUNS` forwards under `spec` and return the per-forward average.
fn time_forward(
    model: &GenericTransformer,
    input: &Tensor,
    spec: &HookSpec,
) -> std::time::Duration {
    let start = Instant::now();
    for _ in 0..FWD_RUNS {
        let _ = model.forward(input, spec).unwrap();
    }
    // CAST: usize → u32, FWD_RUNS = 10 fits comfortably.
    start.elapsed() / FWD_RUNS as u32
}

// EXPLICIT: one linear diagnostic script per device; the sequence of timed
// phases is the artefact, and splitting it would scatter the reported table.
#[allow(clippy::too_many_lines)]
fn run_diagnostic(label: &str, device: &Device) {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in cache");
        return;
    }

    let (model, tokenizer, _) = load_model_on(MODEL_ID, device);
    let token_ids = tokenizer.encode(PROMPT).unwrap();
    let input = Tensor::new(&token_ids[..], device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let num_layers = model.num_layers();
    let empty_hooks = HookSpec::new();
    let full_hooks = full_capture_spec(num_layers);
    let n_caps = full_hooks.num_captures();
    // Deterministic capture count: 12 per-layer points + Embed + FinalNorm.
    assert_eq!(n_caps, num_layers * 12 + 2, "unexpected capture count");

    println!("\n=== Hook diagnostic micro-bench: {MODEL_ID} ({label}) ===");
    println!("  Layers: {num_layers}, total captures (full spec): {n_caps}");
    println!("  Prompt tokens: {}", token_ids.len());

    // ------------------------------------------------------------------
    // Sanity check: confirm the forward returns the expected next token
    // ("Paris") before we trust any of the timings below.
    // ------------------------------------------------------------------
    let cache = model.forward(&input, &empty_hooks).unwrap();
    let logits_cpu = cache
        .output()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let seq_len = token_ids.len();
    let last: Vec<f32> = logits_cpu.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();
    let mut ranked: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top3: Vec<String> = ranked
        .iter()
        .take(3)
        .map(|(i, _)| {
            // CAST: usize → u32, vocab indices fit in u32 for all models we use.
            tokenizer.decode(&[*i as u32]).unwrap()
        })
        .collect();
    println!("  Sanity check — top-3 next tokens for \"{PROMPT}\": {top3:?}");
    // Bind the end-to-end forward correctness: "Paris" must be a top-3 next token
    // for this prompt. Previously this block only printed — a broken forward would
    // have produced no test failure.
    assert!(
        top3.iter().any(|t| t.contains("Paris")),
        "expected 'Paris' among top-3 next tokens for \"{PROMPT}\", got {top3:?}"
    );

    // ------------------------------------------------------------------
    // B. Real forward overhead (empty vs full)
    // ------------------------------------------------------------------
    for _ in 0..FWD_WARMUP {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }
    let start = Instant::now();
    for _ in 0..FWD_RUNS {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }
    // CAST: usize → u32, FWD_RUNS = 10 fits comfortably.
    let empty_avg = start.elapsed() / FWD_RUNS as u32;

    let start = Instant::now();
    for _ in 0..FWD_RUNS {
        let _ = model.forward(&input, &full_hooks).unwrap();
    }
    let full_avg = start.elapsed() / FWD_RUNS as u32;
    let fwd_delta = full_avg.saturating_sub(empty_avg);

    println!("\nB. Real forward overhead ({FWD_RUNS} runs each):");
    println!("   Empty:  {empty_avg:>12.2?}");
    println!("   Full:   {full_avg:>12.2?}");
    println!("   Delta:  {fwd_delta:>12.2?}");

    // ------------------------------------------------------------------
    // A. Pure spec-lookup cost (no tensors)
    // ------------------------------------------------------------------
    for _ in 0..1_000 {
        walk_spec(&full_hooks, num_layers);
    }
    let start = Instant::now();
    for _ in 0..LOOKUP_ITERS {
        walk_spec(&full_hooks, num_layers);
    }
    let walk_total = start.elapsed();
    // CAST: usize → u32, LOOKUP_ITERS = 100_000 fits.
    let walk_per_fwd = walk_total / LOOKUP_ITERS as u32;

    println!("\nA. Pure spec-lookup cost ({n_caps} hooks per synthetic forward):");
    println!("   Total ({LOOKUP_ITERS} iters): {walk_total:>12.2?}");
    println!("   Per synthetic forward:        {walk_per_fwd:>12.2?}");

    // ------------------------------------------------------------------
    // C. Capture machinery (Tensor::clone + HashMap::insert)
    // ------------------------------------------------------------------
    let dummy = Tensor::zeros((1, 5, 2048), DType::F32, device).unwrap();
    for _ in 0..100 {
        let _ = store_all(num_layers, &dummy);
    }
    let start = Instant::now();
    for _ in 0..STORE_ITERS {
        let cache = store_all(num_layers, &dummy);
        black_box(cache);
    }
    let store_total = start.elapsed();
    // CAST: usize → u32, STORE_ITERS = 10_000 fits.
    let store_per_fwd = store_total / STORE_ITERS as u32;

    println!("\nC. Capture-machinery cost ({n_caps} clone+insert per synthetic forward):");
    println!("   Total ({STORE_ITERS} iters): {store_total:>12.2?}");
    println!("   Per synthetic forward:       {store_per_fwd:>12.2?}");

    // ------------------------------------------------------------------
    // Attribution
    // ------------------------------------------------------------------
    // CAST: u128 → f64, durations small enough that f64 holds them losslessly.
    let delta_ns = fwd_delta.as_nanos() as f64;
    let walk_ns = walk_per_fwd.as_nanos() as f64;
    let store_ns = store_per_fwd.as_nanos() as f64;
    let lookup_pct = if delta_ns > 0.0 {
        walk_ns / delta_ns * 100.0
    } else {
        0.0
    };
    let store_pct = if delta_ns > 0.0 {
        store_ns / delta_ns * 100.0
    } else {
        0.0
    };
    let remainder = (100.0 - lookup_pct - store_pct).max(0.0);

    println!("\nAttribution of the forward delta ({fwd_delta:.2?}):");
    println!("   Pure-lookup share (A / delta):       {lookup_pct:>6.1}%");
    println!("   Capture-machinery share (C / delta): {store_pct:>6.1}%");
    println!("   Remainder (forward-internal):        {remainder:>6.1}%");

    // ------------------------------------------------------------------
    // D. Capture-density sweep — vary count, hold hook type constant.
    // If overhead scales linearly with count and is similar per-capture,
    // held references are the dominant effect. If non-linear or saturates,
    // it is something else.
    // ------------------------------------------------------------------
    println!("\nD. Capture-density sweep (`ResidPost` only — single shape):");
    println!(
        "   {:>6}  {:>10}  {:>10}  {:>12}",
        "count", "avg", "delta", "delta/cap"
    );
    let strides = [num_layers, 8, 4, 2, 1]; // -> 1, 2, 4, 8, 16 captures (16-layer model)
    for stride in strides {
        let spec = resid_post_every_n(num_layers, stride);
        let count = spec.num_captures();
        if count == 0 {
            continue;
        }
        let avg = time_forward(&model, &input, &spec);
        let delta = avg.saturating_sub(empty_avg);
        // CAST: u128 → f64 / u32 → f64, both small enough.
        let per_cap_ns = delta.as_nanos() as f64 / count as f64;
        println!("   {count:>6}  {avg:>10.2?}  {delta:>10.2?}  {per_cap_ns:>10.0} ns");
    }

    // ------------------------------------------------------------------
    // E. Equal-count, varying hook type — distinguishes "per-capture"
    // (refs/clone/insert) from "per-hook-type" (extra computation, e.g.
    // softmax-fusion blocking on `AttnScores` / `AttnPattern`).
    // ------------------------------------------------------------------
    println!("\nE. Equal-count shape comparison ({num_layers} captures, one per layer):");
    println!("   {:>16}  {:>10}  {:>10}", "hook type", "avg", "delta");
    let cases: [(&str, HookSpec); 4] = [
        (
            "ResidPost",
            one_hook_per_layer(num_layers, HookPoint::ResidPost),
        ),
        (
            "AttnOut",
            one_hook_per_layer(num_layers, HookPoint::AttnOut),
        ),
        (
            "AttnScores",
            one_hook_per_layer(num_layers, HookPoint::AttnScores),
        ),
        (
            "AttnPattern",
            one_hook_per_layer(num_layers, HookPoint::AttnPattern),
        ),
    ];
    for (name, spec) in cases {
        let avg = time_forward(&model, &input, &spec);
        let delta = avg.saturating_sub(empty_avg);
        println!("   {name:>16}  {avg:>10.2?}  {delta:>10.2?}");
    }
    println!();
}

#[test]
fn bench_hook_diagnostic_cpu() {
    run_diagnostic("CPU F32", &Device::Cpu);
}

#[test]
fn bench_hook_diagnostic_gpu() {
    let Some(device) = Device::cuda_if_available(0)
        .ok()
        .filter(candle_core::Device::is_cuda)
    else {
        // A skip reports `ok`, indistinguishable from a pass in the summary
        // line. That is correct when the crate was built without `cuda` (there
        // is no GPU path compiled in to exercise), but with the feature on it
        // would hide a broken environment behind a green run, so fail loudly.
        // Two clippy lints disagree about this guard: `assertions_on_constants`
        // rejects `assert!(!cfg!(..))` because the condition folds at compile
        // time, and `manual_assert` rejects the `if`/`panic!` form below because
        // it wants the assert back. The condition really is a compile-time
        // constant -- checking the build configuration is the whole point -- so
        // one of the two has to be waived. The `if` is waived here because it
        // keeps the panic message at the site that produces it, and the
        // exemption names a single lint rather than a whole file.
        #[allow(clippy::manual_assert)]
        if cfg!(feature = "cuda") {
            panic!(
                "built with the `cuda` feature but no CUDA device is available;                  this test would otherwise report `ok` without exercising the GPU"
            );
        }
        eprintln!("SKIP: built without the `cuda` feature, so there is no GPU path to exercise");
        return;
    };
    run_diagnostic("CUDA F32", &device);
}
