// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase B integration test: full MI analysis pipeline on M₂,₂ fixture.
//!
//! Exercises all six Phase B modules (fast, standardize, piecewise, ablation,
//! probing, surprise) on the M₂,₂ fixture committed in Phase A.

// Test/example target: these lints are denied crate-wide for library code,
// where a panic is a bug. Here a failed unwrap IS the failure signal, and
// indexing a fixture whose shape the test itself fixes cannot go out of
// bounds. Same allowance as the other 30+ files under tests/ and examples/.
#![allow(clippy::unwrap_used)]

use candle_core::Device;
use candle_mi::stoicheia::StoicheiaRnn;
use candle_mi::stoicheia::ablation;
use candle_mi::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
use candle_mi::stoicheia::fast::{self, RnnWeights};
use candle_mi::stoicheia::piecewise;
use candle_mi::stoicheia::probing;
use candle_mi::stoicheia::standardize;
use candle_mi::stoicheia::surprise;

/// Path to the M₂,₂ fixture (10 parameters, committed in Phase A).
const FIXTURE_PATH: &str = "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors";

/// Generate deterministic test inputs for M₂,₂.
fn generate_inputs(n: usize) -> Vec<f32> {
    // CAST: usize → f32, small test indices
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    (0..n * 2)
        .map(|i| ((i as f32) * 0.618_034).sin() * 3.0)
        .collect()
}

/// Compute second-argmax targets from flat inputs.
fn compute_targets(inputs: &[f32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            // INDEX: i * 2 and i * 2 + 1 bounded by n * 2
            #[allow(clippy::indexing_slicing)]
            let (a, b) = (inputs[i * 2], inputs[i * 2 + 1]);
            // Second argmax of 2 elements: position of the smaller one
            u32::from(a > b)
        })
        .collect()
}

#[test]
fn fast_path_matches_candle() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let inputs = generate_inputs(100);
    let n = 100;

    // Run candle-based forward
    let input_tensor = candle_core::Tensor::from_slice(&inputs, (n, 2), &Device::Cpu).unwrap();
    let hooks = candle_mi::HookSpec::new();
    let cache = candle_mi::MIBackend::forward(&model, &input_tensor, &hooks).unwrap();
    let candle_output = cache.output().squeeze(1).unwrap();
    let candle_vec: Vec<f32> = candle_output.flatten_all().unwrap().to_vec1().unwrap();

    // Run fast path
    let mut fast_output = vec![0.0_f32; n * 2];
    fast::forward_fast(&weights, &inputs, &mut fast_output, n, &config).unwrap();

    // Compare outputs (1e-4 tolerance: fast path uses mul_add / FMA which
    // produces slightly different rounding than candle's separate ops)
    for (i, (c, f)) in candle_vec.iter().zip(&fast_output).enumerate() {
        assert!(
            (c - f).abs() < 1e-4,
            "mismatch at {i}: candle={c}, fast={f}"
        );
    }
}

#[test]
fn standardize_preserves_output() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let std_rnn = standardize::standardize_rnn(&model).unwrap();
    let quality = standardize::standardization_quality(&std_rnn);
    assert!(quality < 1e-5, "quality = {quality}");

    let std_weights = std_rnn.to_rnn_weights();

    let inputs = generate_inputs(50);
    let n = 50;
    let mut orig_out = vec![0.0_f32; n * 2];
    let mut std_out = vec![0.0_f32; n * 2];

    fast::forward_fast(&weights, &inputs, &mut orig_out, n, &config).unwrap();
    fast::forward_fast(&std_weights, &inputs, &mut std_out, n, &config).unwrap();

    for (i, (a, b)) in orig_out.iter().zip(&std_out).enumerate() {
        assert!((a - b).abs() < 1e-4, "mismatch at {i}: orig={a}, std={b}");
    }
}

#[test]
fn region_count_reasonable() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let inputs = generate_inputs(1000);
    let n = 1000;

    let region_map = piecewise::classify_regions(&weights, &inputs, n, &config).unwrap();

    // M₂,₂ has at most 2^(2*2) = 16 regions
    assert!(
        region_map.regions.len() <= 16,
        "found {} regions",
        region_map.regions.len()
    );
    assert_eq!(region_map.total_inputs, n);

    // All inputs accounted for
    let total: usize = region_map.regions.iter().map(|r| r.count).sum();
    assert_eq!(total, n);
}

#[test]
fn ablation_baseline_correct() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let inputs = generate_inputs(200);
    let n = 200;
    let targets = compute_targets(&inputs, n);

    let sweep = ablation::ablate_neurons(&weights, &inputs, &targets, n, &config).unwrap();

    // Baseline should be > chance (50% for 2-class)
    assert!(
        sweep.baseline_accuracy > 0.5,
        "baseline = {}",
        sweep.baseline_accuracy
    );
    assert_eq!(sweep.results.len(), 2);
}

#[test]
fn probing_identifies_roles() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let report = probing::probe_neurons(&weights, &config, 500).unwrap();

    assert_eq!(report.neurons.len(), 2);
    // At least one neuron should have non-trivial correlation
    let max_corr = report
        .neurons
        .iter()
        .map(|n| n.correlation)
        .fold(0.0_f32, f32::max);
    assert!(
        max_corr > 0.3,
        "max correlation = {max_corr} (expected > 0.3)"
    );
}

#[test]
fn surprise_oracle_agreement() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(config.clone(), FIXTURE_PATH, &Device::Cpu).unwrap();
    let weights = RnnWeights::from_model(&model).unwrap();

    let oracle = surprise::OracleEstimator::new(StoicheiaTask::SecondArgmax, 2);
    let report = surprise::surprise_accounting(&weights, &oracle, &config, 500).unwrap();

    assert_eq!(report.n_samples, 500);
    // Model accuracy should be > chance
    assert!(
        report.model_accuracy > report.chance_accuracy,
        "model accuracy {} <= chance {}",
        report.model_accuracy,
        report.chance_accuracy
    );
    // Oracle estimate should also be > chance
    assert!(
        report.estimate_accuracy > report.chance_accuracy,
        "estimate accuracy {} <= chance {}",
        report.estimate_accuracy,
        report.chance_accuracy
    );
}
