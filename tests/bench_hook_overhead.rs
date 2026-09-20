// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hook overhead benchmark: measures forward pass time with hooks inactive
//! vs. full capture on `LLaMA` 3.2 1B.
//!
//! Run:
//!   `cargo test --test bench_hook_overhead --features transformer,mmap --release -- --nocapture`

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

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_mi::{
    GenericTransformer, HookPoint, HookSpec, MIBackend, MITokenizer, TransformerConfig,
};

// ---------------------------------------------------------------------------
// Helpers (duplicated from validate_models.rs to keep this self-contained)
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

    // F32 everywhere: research-grade precision matching Python/PyTorch.
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

    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };

    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();

    (model, tokenizer, config)
}

/// Build a `HookSpec` that captures every hook point at every layer.
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

/// Non-flaky correctness checks for the capture path (independent of any
/// wall-clock timing): the capture count is exactly `12*num_layers + 2`, every
/// requested hook point is present, the capture-only forward is numerically
/// identical to the no-hook forward (capture is pure observation), and captured
/// activations are finite. A hook-path regression that silently corrupts the
/// forward — invisible to the timing loop — fails here.
fn verify_capture_correctness(model: &GenericTransformer, input: &Tensor, num_layers: usize) {
    let empty = HookSpec::new();
    let full = full_capture_spec(num_layers);

    assert_eq!(
        full.num_captures(),
        num_layers * 12 + 2,
        "capture count should be 12/layer + Embed + FinalNorm"
    );

    let captured = model.forward(input, &full).unwrap();
    let baseline = model.forward(input, &empty).unwrap();

    // Every requested per-layer hook point (plus Embed / FinalNorm) is present.
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
            assert!(captured.get(&hp).is_some(), "missing capture: {hp:?}");
        }
    }
    assert!(captured.get(&HookPoint::Embed).is_some());
    assert!(captured.get(&HookPoint::FinalNorm).is_some());

    // Capture-only hooks must not change the forward: logits must match the
    // no-hook run (both F32 here). A capture path that mutates the residual
    // stream would diverge and fail this.
    let cap = captured
        .output()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let base = baseline
        .output()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(cap.len(), base.len(), "output length mismatch");
    let max_abs = cap
        .iter()
        .zip(&base)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs < 1e-4,
        "capture changed the forward output (max abs diff {max_abs})"
    );

    // Captured activations are finite (sample the residual stream at layer 0).
    let sample = captured
        .get(&HookPoint::ResidPost(0))
        .expect("ResidPost(0) captured")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert!(!sample.is_empty(), "captured activation is empty");
    assert!(
        sample.iter().all(|v| v.is_finite()),
        "ResidPost(0) has non-finite values"
    );
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

const MODEL_ID: &str = "meta-llama/Llama-3.2-1B";
const PROMPT: &str = "The capital of France is";
const WARMUP_RUNS: usize = 2;
const BENCH_RUNS: usize = 10;

#[test]
fn bench_hook_overhead_cpu() {
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on(MODEL_ID, &device);

    let token_ids = tokenizer.encode(PROMPT).unwrap();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let num_layers = model.num_layers();
    let empty_hooks = HookSpec::new();
    let full_hooks = full_capture_spec(num_layers);

    // Correctness (non-flaky) checks before the timing loops.
    verify_capture_correctness(&model, &input, num_layers);

    println!("\n=== Hook Overhead Benchmark: {MODEL_ID} (CPU F32) ===");
    println!(
        "  Layers: {}, Hooks per layer: 12, Total hook points: {}",
        num_layers,
        full_hooks.num_captures()
    );
    println!("  Prompt: \"{}\" ({} tokens)", PROMPT, token_ids.len());
    println!("  Warmup: {WARMUP_RUNS} runs, Bench: {BENCH_RUNS} runs\n");

    // --- Warmup ---
    for _ in 0..WARMUP_RUNS {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }

    // --- No hooks ---
    let start = Instant::now();
    for _ in 0..BENCH_RUNS {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }
    let no_hooks_total = start.elapsed();
    let no_hooks_avg = no_hooks_total / BENCH_RUNS as u32;

    // --- Full capture ---
    let start = Instant::now();
    for _ in 0..BENCH_RUNS {
        let result = model.forward(&input, &full_hooks).unwrap();
        // Verify captures are present
        assert!(result.get(&HookPoint::Embed).is_some());
        assert!(result.get(&HookPoint::FinalNorm).is_some());
        assert!(result.get(&HookPoint::AttnPattern(0)).is_some());
    }
    let full_capture_total = start.elapsed();
    let full_capture_avg = full_capture_total / BENCH_RUNS as u32;

    let overhead_pct = if no_hooks_avg.as_nanos() > 0 {
        ((full_capture_avg.as_nanos() as f64 / no_hooks_avg.as_nanos() as f64) - 1.0) * 100.0
    } else {
        0.0
    };

    println!("  No hooks:     {no_hooks_avg:>8.2?} avg ({BENCH_RUNS} runs)");
    println!(
        "  Full capture: {:>8.2?} avg ({BENCH_RUNS} runs, {} captures)",
        full_capture_avg,
        full_hooks.num_captures()
    );
    println!("  Overhead:     {overhead_pct:>+.1}%\n");
}

#[test]
fn bench_hook_overhead_gpu() {
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
    if find_snapshot(MODEL_ID).is_none() {
        eprintln!("SKIP: {MODEL_ID} not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on(MODEL_ID, &device);

    let token_ids = tokenizer.encode(PROMPT).unwrap();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let num_layers = model.num_layers();
    let empty_hooks = HookSpec::new();
    let full_hooks = full_capture_spec(num_layers);

    // Correctness (non-flaky) checks before the timing loops.
    verify_capture_correctness(&model, &input, num_layers);

    println!("\n=== Hook Overhead Benchmark: {MODEL_ID} (CUDA BF16) ===");
    println!(
        "  Layers: {}, Hooks per layer: 12, Total hook points: {}",
        num_layers,
        full_hooks.num_captures()
    );
    println!("  Prompt: \"{}\" ({} tokens)", PROMPT, token_ids.len());
    println!("  Warmup: {WARMUP_RUNS} runs, Bench: {BENCH_RUNS} runs\n");

    // --- Warmup ---
    for _ in 0..WARMUP_RUNS {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }

    // --- No hooks ---
    let start = Instant::now();
    for _ in 0..BENCH_RUNS {
        let _ = model.forward(&input, &empty_hooks).unwrap();
    }
    let no_hooks_total = start.elapsed();
    let no_hooks_avg = no_hooks_total / BENCH_RUNS as u32;

    // --- Full capture ---
    let start = Instant::now();
    for _ in 0..BENCH_RUNS {
        let result = model.forward(&input, &full_hooks).unwrap();
        assert!(result.get(&HookPoint::Embed).is_some());
        assert!(result.get(&HookPoint::FinalNorm).is_some());
        assert!(result.get(&HookPoint::AttnPattern(0)).is_some());
    }
    let full_capture_total = start.elapsed();
    let full_capture_avg = full_capture_total / BENCH_RUNS as u32;

    let overhead_pct = if no_hooks_avg.as_nanos() > 0 {
        ((full_capture_avg.as_nanos() as f64 / no_hooks_avg.as_nanos() as f64) - 1.0) * 100.0
    } else {
        0.0
    };

    println!("  No hooks:     {no_hooks_avg:>8.2?} avg ({BENCH_RUNS} runs)");
    println!(
        "  Full capture: {:>8.2?} avg ({BENCH_RUNS} runs, {} captures)",
        full_capture_avg,
        full_hooks.num_captures()
    );
    println!("  Overhead:     {overhead_pct:>+.1}%\n");
}
