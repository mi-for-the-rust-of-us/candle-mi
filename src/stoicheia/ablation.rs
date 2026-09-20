// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exhaustive neuron ablation for `AlgZoo` `ReLU` RNNs.
//!
//! Zero each neuron's hidden state at all timesteps, measure the accuracy
//! change. Identifies which neurons are critical for the task and which
//! are functionally redundant.

use crate::error::Result;
use crate::stoicheia::config::StoicheiaConfig;
use crate::stoicheia::fast::{self, RnnWeights};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of ablating a single neuron.
#[non_exhaustive]
pub struct NeuronAblationResult {
    /// Neuron index (0-indexed).
    pub neuron: usize,
    /// Accuracy with this neuron zeroed (0.0 to 1.0).
    pub ablated_accuracy: f32,
    /// Accuracy change relative to baseline (negative = important).
    pub accuracy_delta: f32,
}

/// Result of a full single-neuron ablation sweep.
#[non_exhaustive]
pub struct AblationSweep {
    /// Baseline accuracy (no ablation).
    pub baseline_accuracy: f32,
    /// Per-neuron results, sorted by `accuracy_delta` ascending
    /// (most damaging ablation first).
    pub results: Vec<NeuronAblationResult>,
    /// Number of test inputs used.
    pub n_inputs: usize,
}

/// Result of ablating a pair of neurons simultaneously.
#[non_exhaustive]
pub struct PairAblationResult {
    /// First neuron index.
    pub neuron_a: usize,
    /// Second neuron index.
    pub neuron_b: usize,
    /// Accuracy with both neurons zeroed.
    pub ablated_accuracy: f32,
    /// Accuracy change relative to baseline.
    pub accuracy_delta: f32,
    /// Interaction score: `pair_delta - (delta_a + delta_b)`.
    /// Negative = super-additive damage (redundancy — each neuron
    /// compensates for the other, but removing both is catastrophic).
    /// Near zero = independent (additive damage).
    /// Positive = sub-additive damage (ceiling effect — each neuron
    /// is independently important, but combined damage saturates).
    pub interaction_score: f32,
}

// ---------------------------------------------------------------------------
// Ablation functions
// ---------------------------------------------------------------------------

/// Run exhaustive single-neuron ablation on an RNN.
///
/// For each of the H neurons, zeros that neuron's hidden state at
/// every timestep and measures accuracy on `inputs`.
///
/// Uses [`fast::forward_fast_ablated`] internally.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
/// - `targets`: `[n_inputs]`, ground-truth output class indices
///
/// # Errors
///
/// Returns [`MIError::Config`](crate::MIError::Config) if slice dimensions
/// do not match.
pub fn ablate_neurons(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> Result<AblationSweep> {
    let h = weights.hidden_size;

    // Baseline accuracy (no ablation)
    let baseline_accuracy = fast::accuracy(weights, inputs, targets, n_inputs, config)?;

    let mut results = Vec::with_capacity(h);
    let out_size = weights.output_size;
    let mut outputs = vec![0.0_f32; n_inputs * out_size];

    for neuron in 0..h {
        // Build ablation mask: only this neuron is zeroed
        let mut ablated = vec![false; h];
        // INDEX: neuron bounded by h
        #[allow(clippy::indexing_slicing)]
        {
            ablated[neuron] = true;
        }

        fast::forward_fast_ablated(weights, inputs, &mut outputs, n_inputs, config, &ablated)?;

        // Compute accuracy on ablated outputs
        let ablated_accuracy = fast::accuracy_from_outputs(&outputs, targets, out_size, n_inputs);

        results.push(NeuronAblationResult {
            neuron,
            ablated_accuracy,
            accuracy_delta: ablated_accuracy - baseline_accuracy,
        });
    }

    // Sort by accuracy_delta ascending (most damaging first)
    results.sort_by(|a, b| {
        a.accuracy_delta
            .partial_cmp(&b.accuracy_delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(AblationSweep {
        baseline_accuracy,
        results,
        n_inputs,
    })
}

/// Run pairwise neuron ablation.
///
/// For each pair `(i, j)` with `i < j`, zeros both neurons and measures
/// accuracy. The interaction score reveals functional redundancy or synergy.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
/// - `targets`: `[n_inputs]`, ground-truth output class indices
///
/// # Errors
///
/// Returns [`MIError::Config`](crate::MIError::Config) if slice dimensions
/// do not match.
pub fn ablate_neuron_pairs(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
    single_results: &AblationSweep,
) -> Result<Vec<PairAblationResult>> {
    let h = weights.hidden_size;
    let out_size = weights.output_size;
    let mut outputs = vec![0.0_f32; n_inputs * out_size];

    // Build a lookup: neuron → accuracy_delta
    let mut single_deltas = vec![0.0_f32; h];
    for r in &single_results.results {
        // INDEX: r.neuron bounded by h
        #[allow(clippy::indexing_slicing)]
        {
            single_deltas[r.neuron] = r.accuracy_delta;
        }
    }

    let baseline = single_results.baseline_accuracy;
    let mut pair_results = Vec::new();

    for a in 0..h {
        for b in (a + 1)..h {
            let mut ablated = vec![false; h];
            // INDEX: a, b bounded by h
            #[allow(clippy::indexing_slicing)]
            {
                ablated[a] = true;
                ablated[b] = true;
            }

            fast::forward_fast_ablated(weights, inputs, &mut outputs, n_inputs, config, &ablated)?;

            let ablated_accuracy =
                fast::accuracy_from_outputs(&outputs, targets, out_size, n_inputs);
            let pair_delta = ablated_accuracy - baseline;

            // INDEX: a, b bounded by h
            #[allow(clippy::indexing_slicing)]
            let interaction_score = pair_delta - (single_deltas[a] + single_deltas[b]);

            pair_results.push(PairAblationResult {
                neuron_a: a,
                neuron_b: b,
                ablated_accuracy,
                accuracy_delta: pair_delta,
                interaction_score,
            });
        }
    }

    Ok(pair_results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // Used by the expected-value helpers below; the module's own code now goes
    // through `fast::accuracy_from_outputs` instead.
    use crate::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
    use crate::stoicheia::fast::argmax_f32;

    fn test_weights() -> RnnWeights {
        RnnWeights::new(
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
        )
    }

    fn test_config() -> StoicheiaConfig {
        StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2)
    }

    #[test]
    fn no_ablation_matches_baseline() {
        let weights = test_weights();
        let config = test_config();
        let inputs = vec![0.5_f32, -0.3, 1.0, 2.0, -1.0, 0.7];
        let n = 3;

        // Compute targets using forward_fast (argmax of output)
        let mut outputs = vec![0.0_f32; n * 2];
        fast::forward_fast(&weights, &inputs, &mut outputs, n, &config).unwrap();
        let targets: Vec<u32> = (0..n)
            .map(|i| argmax_f32(&outputs[i * 2..(i + 1) * 2]))
            .collect();

        let sweep = ablate_neurons(&weights, &inputs, &targets, n, &config).unwrap();

        // Baseline should be 1.0 since targets were computed from the model
        assert!(
            (sweep.baseline_accuracy - 1.0).abs() < 1e-6,
            "baseline = {}",
            sweep.baseline_accuracy
        );
    }

    #[test]
    fn full_ablation_near_chance() {
        let weights = test_weights();
        let config = test_config();

        // Generate inputs where the model makes varied predictions
        let inputs: Vec<f32> = (0..200)
            .map(|i| {
                // CAST: usize → f32, small test indices
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let v = ((i as f32) * 0.618_034).sin() * 3.0;
                v
            })
            .collect();
        let n = 100;

        // Compute true targets
        let mut outputs = vec![0.0_f32; n * 2];
        fast::forward_fast(&weights, &inputs, &mut outputs, n, &config).unwrap();
        let targets: Vec<u32> = (0..n)
            .map(|i| argmax_f32(&outputs[i * 2..(i + 1) * 2]))
            .collect();

        let sweep = ablate_neurons(&weights, &inputs, &targets, n, &config).unwrap();

        // Ablating any single neuron should change accuracy
        assert_eq!(sweep.results.len(), 2);

        // Targets are the model's own argmax, so baseline accuracy is exactly 1.0
        // and ablation can only preserve or DAMAGE accuracy — every
        // `accuracy_delta` must be <= 0. A sign flip in
        // `accuracy_delta = ablated_accuracy - baseline_accuracy` would make these
        // positive and fail here (the defect this guards against).
        assert!(
            (sweep.baseline_accuracy - 1.0).abs() < 1e-6,
            "baseline = {}",
            sweep.baseline_accuracy
        );
        for r in &sweep.results {
            assert!(
                r.accuracy_delta <= 1e-6,
                "neuron {} delta {} should be <= 0 (ablation cannot improve a \
                 model-derived target)",
                r.neuron,
                r.accuracy_delta
            );
        }
    }

    #[test]
    fn pairwise_ablation_runs() {
        let weights = test_weights();
        let config = test_config();
        let inputs = vec![0.5_f32, -0.3, 1.0, 2.0];
        let n = 2;

        let mut outputs = vec![0.0_f32; n * 2];
        fast::forward_fast(&weights, &inputs, &mut outputs, n, &config).unwrap();
        let targets: Vec<u32> = (0..n)
            .map(|i| argmax_f32(&outputs[i * 2..(i + 1) * 2]))
            .collect();

        let sweep = ablate_neurons(&weights, &inputs, &targets, n, &config).unwrap();
        let pairs = ablate_neuron_pairs(&weights, &inputs, &targets, n, &config, &sweep).unwrap();

        // C(2, 2) = 1 pair
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].neuron_a, 0);
        assert_eq!(pairs[0].neuron_b, 1);

        // The interaction score must equal `pair_delta - (delta_a + delta_b)`,
        // re-derived here from the single-ablation sweep. This pins the arithmetic
        // in `ablate_neuron_pairs` (a sign flip or wrong-term bug fails here).
        let delta_a = sweep
            .results
            .iter()
            .find(|r| r.neuron == 0)
            .map(|r| r.accuracy_delta)
            .expect("neuron 0 in single sweep");
        let delta_b = sweep
            .results
            .iter()
            .find(|r| r.neuron == 1)
            .map(|r| r.accuracy_delta)
            .expect("neuron 1 in single sweep");
        let expected = pairs[0].accuracy_delta - (delta_a + delta_b);
        assert!(
            (pairs[0].interaction_score - expected).abs() < 1e-6,
            "interaction_score {} != pair_delta - (delta_a + delta_b) = {}",
            pairs[0].interaction_score,
            expected
        );
    }
}
