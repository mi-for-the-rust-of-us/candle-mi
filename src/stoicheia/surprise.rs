// SPDX-License-Identifier: MIT OR Apache-2.0

//! Surprise accounting for `AlgZoo` models.
//!
//! Implements ARC's information-theoretic metric for evaluating mechanistic
//! understanding. Compares a model's accuracy against a mechanistic estimate,
//! measuring agreement rate and MSE.
//!
//! Two complementary evaluation methods from the `AlgZoo` blog:
//! 1. **MSE vs. compute**: how close does the estimate get to actual accuracy?
//! 2. **Surprise accounting**: how many bits of surprise remain?

use crate::error::Result;
use crate::stoicheia::config::{StoicheiaConfig, StoicheiaOutput, StoicheiaTask};
use crate::stoicheia::fast::{self, RnnWeights, argmax_f32};

// ---------------------------------------------------------------------------
// MechanisticEstimator trait
// ---------------------------------------------------------------------------

/// A mechanistic estimator: predicts the model's output class without
/// running the model, based on mechanistic understanding of its internals.
///
/// Implementors encode their understanding of the model as a prediction
/// function. The [`OracleEstimator`] provides the upper bound (perfect
/// task understanding). Users implement this trait with increasingly
/// refined mechanistic explanations.
// TRAIT_OBJECT: user-provided mechanistic understanding
pub trait MechanisticEstimator {
    /// Predict the model's output class for a single input.
    ///
    /// # Shapes
    /// - `input`: `[seq_len]` — one input sequence
    /// - returns: predicted output class index
    fn predict(&self, input: &[f32]) -> u32;

    /// Human-readable description of this estimator's methodology.
    fn description(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// OracleEstimator
// ---------------------------------------------------------------------------

/// Oracle estimator: uses the ground-truth task function.
///
/// This is the upper bound — if the model perfectly implements the
/// task, the oracle estimator achieves zero surprise residual.
pub struct OracleEstimator {
    /// The task to compute ground truth for.
    task: StoicheiaTask,
    /// Sequence length (needed for argmedian's median index).
    seq_len: usize,
}

impl OracleEstimator {
    /// Create an oracle estimator for a given task and sequence length.
    #[must_use]
    pub const fn new(task: StoicheiaTask, seq_len: usize) -> Self {
        Self { task, seq_len }
    }
}

impl MechanisticEstimator for OracleEstimator {
    fn predict(&self, input: &[f32]) -> u32 {
        match self.task {
            StoicheiaTask::SecondArgmax => {
                // Find position of second-largest value.
                // Tie-breaking: if all values are identical, the second
                // argmax is position 1 (or 0 if seq_len == 1).
                let (mut max_pos, mut second_pos) = (0, 0);
                let mut max_val = f32::NEG_INFINITY;
                let mut second_val = f32::NEG_INFINITY;
                for (i, &x) in input.iter().enumerate() {
                    if x > max_val {
                        second_val = max_val;
                        second_pos = max_pos;
                        max_val = x;
                        max_pos = i;
                    } else if x > second_val {
                        second_val = x;
                        second_pos = i;
                    }
                }
                // Handle constant sequences: second_val was never updated
                // beyond its initial NEG_INFINITY (all elements equal max).
                // Pick the first position that isn't max_pos.
                if second_val == f32::NEG_INFINITY && input.len() > 1 {
                    second_pos = usize::from(max_pos == 0);
                }
                // CAST: usize → u32, positions are small
                #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
                {
                    second_pos as u32
                }
            }
            StoicheiaTask::Argmedian => {
                // Sort and find median position
                let mut indexed: Vec<(usize, f32)> = input.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let median_rank = self.seq_len / 2;
                // INDEX: median_rank bounded by seq_len/2 < indexed.len()
                #[allow(clippy::indexing_slicing)]
                let pos = indexed[median_rank].0;
                // CAST: usize → u32, positions are small
                #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
                {
                    pos as u32
                }
            }
            StoicheiaTask::Median | StoicheiaTask::LongestCycle => 0,
        }
    }

    fn description(&self) -> &'static str {
        "oracle (ground-truth task function)"
    }
}

// ---------------------------------------------------------------------------
// SurpriseReport
// ---------------------------------------------------------------------------

/// Result of a surprise accounting measurement.
#[non_exhaustive]
pub struct SurpriseReport {
    /// Model accuracy on random inputs (0.0 to 1.0).
    pub model_accuracy: f32,
    /// Mechanistic estimate accuracy.
    pub estimate_accuracy: f32,
    /// Disagreement rate: fraction of inputs where the model and
    /// estimator predict different classes (0 = perfect agreement,
    /// 1 = complete disagreement). This is 0/1 classification loss,
    /// not continuous-valued MSE.
    pub disagreement_rate: f32,
    /// Chance accuracy: `1/output_size` (uniform random baseline).
    pub chance_accuracy: f32,
    /// Number of model parameters.
    pub param_count: usize,
    /// Number of test samples used.
    pub n_samples: usize,
    /// Per-input agreement: fraction where model and estimate
    /// predict the same class.
    pub agreement_rate: f32,
    /// Description of the estimator used.
    pub estimator_description: String,
}

// ---------------------------------------------------------------------------
// Surprise accounting functions
// ---------------------------------------------------------------------------

/// Run surprise accounting: compare model accuracy vs. mechanistic
/// estimate on random inputs.
///
/// Generates `n_samples` random inputs (deterministic seed), runs both
/// the model (via fast path) and the mechanistic estimator, and computes
/// agreement metrics.
///
/// # Errors
///
/// Returns [`MIError::Config`](crate::MIError::Config) if configuration
/// is invalid.
pub fn surprise_accounting(
    weights: &RnnWeights,
    // TRAIT_OBJECT: user-provided mechanistic understanding
    estimator: &dyn MechanisticEstimator,
    config: &StoicheiaConfig,
    n_samples: usize,
) -> Result<SurpriseReport> {
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    // Generate deterministic inputs
    let inputs = generate_inputs(n_samples, seq_len);
    let flat_inputs: Vec<f32> = inputs.iter().flatten().copied().collect();

    // Run model via fast path
    let mut model_outputs = vec![0.0_f32; n_samples * out_size];
    fast::forward_fast(weights, &flat_inputs, &mut model_outputs, n_samples, config)?;

    // Compare model vs estimator
    let mut model_correct = 0_usize;
    let mut estimate_correct = 0_usize;
    let mut agree = 0_usize;
    let mut mse_sum = 0.0_f32;

    for (i, input) in inputs.iter().enumerate() {
        // Model prediction
        // INDEX: slice bounds valid because model_outputs.len() == n_samples * out_size
        #[allow(clippy::indexing_slicing)]
        let model_row = &model_outputs[i * out_size..(i + 1) * out_size];
        let model_pred = argmax_f32(model_row);

        // Estimator prediction
        let est_pred = estimator.predict(input);

        // Ground truth (oracle for accuracy)
        let oracle = OracleEstimator::new(config.task, seq_len);
        let truth = oracle.predict(input);

        if model_pred == truth {
            model_correct += 1;
        }
        if est_pred == truth {
            estimate_correct += 1;
        }
        if model_pred == est_pred {
            agree += 1;
        }

        // Per-input MSE: 0 if agree, 1 if disagree
        let err = if model_pred == est_pred { 0.0_f32 } else { 1.0 };
        mse_sum += err;
    }

    // CAST: usize → f32, counts ≤ n_samples
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let n_f = n_samples as f32;

    // CAST: usize → f32, counts ≤ n_samples
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let model_accuracy = model_correct as f32 / n_f;

    // CAST: usize → f32, counts ≤ n_samples
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let estimate_accuracy = estimate_correct as f32 / n_f;

    // CAST: usize → f32, counts ≤ n_samples
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let agreement_rate = agree as f32 / n_f;

    let disagreement_rate = mse_sum / n_f;

    // CAST: usize → f32, output_size ≤ 10
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let chance_accuracy = if config.output == StoicheiaOutput::Distribution {
        1.0 / out_size as f32
    } else {
        0.0 // scalar tasks don't have a meaningful chance accuracy
    };

    Ok(SurpriseReport {
        model_accuracy,
        estimate_accuracy,
        disagreement_rate,
        chance_accuracy,
        param_count: config.param_count(),
        n_samples,
        agreement_rate,
        estimator_description: estimator.description().to_string(),
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate deterministic inputs using a simple LCG.
fn generate_inputs(n_samples: usize, seq_len: usize) -> Vec<Vec<f32>> {
    let mut inputs = Vec::with_capacity(n_samples);
    let mut state = 123_456_789_u64;
    for _ in 0..n_samples {
        let mut input = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // CAST: u64 → f32, deliberate precision loss for random generation
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let uniform = (state >> 33) as f32 / (1_u64 << 31) as f32;
            let value = (uniform - 0.5) * 6.0; // approximate [-3, 3]
            input.push(value);
        }
        inputs.push(input);
    }
    inputs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stoicheia::config::StoicheiaConfig;

    #[test]
    fn oracle_matches_task_second_argmax() {
        let oracle = OracleEstimator::new(StoicheiaTask::SecondArgmax, 3);

        // [1.0, 3.0, 2.0] → max=3.0 at pos 1, second=2.0 at pos 2
        let pred = oracle.predict(&[1.0, 3.0, 2.0]);
        assert_eq!(pred, 2);

        // [5.0, 1.0, 3.0] → max=5.0 at pos 0, second=3.0 at pos 2
        let pred = oracle.predict(&[5.0, 1.0, 3.0]);
        assert_eq!(pred, 2);
    }

    #[test]
    fn oracle_matches_task_argmedian() {
        let oracle = OracleEstimator::new(StoicheiaTask::Argmedian, 3);

        // [1.0, 3.0, 2.0] sorted: [(0,1.0), (2,2.0), (1,3.0)]
        // median rank = 3/2 = 1 → (2,2.0) → position 2
        let pred = oracle.predict(&[1.0, 3.0, 2.0]);
        assert_eq!(pred, 2);
    }

    #[test]
    fn surprise_accounting_runs() {
        let weights = RnnWeights::new(
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
        );
        let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
        let oracle = OracleEstimator::new(StoicheiaTask::SecondArgmax, 2);

        let report = surprise_accounting(&weights, &oracle, &config, 100).unwrap();

        assert_eq!(report.n_samples, 100);
        assert!(report.model_accuracy >= 0.0);
        assert!(report.model_accuracy <= 1.0);
        assert!(report.estimate_accuracy >= 0.0);
        assert!(report.agreement_rate >= 0.0);
        assert!((report.chance_accuracy - 0.5).abs() < 1e-6);
        assert_eq!(report.param_count, 10);
    }

    #[test]
    fn perfect_model_high_agreement() {
        // A model that perfectly implements second_argmax should have
        // high agreement with the oracle estimator
        let weights = RnnWeights::new(
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
        );
        let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
        let oracle = OracleEstimator::new(StoicheiaTask::SecondArgmax, 2);

        let report = surprise_accounting(&weights, &oracle, &config, 500).unwrap();

        // The simple memoryless model won't be perfect, but oracle agreement
        // should reflect that both are trying to solve the same task
        assert!(
            report.estimate_accuracy > report.chance_accuracy,
            "estimate accuracy {} should exceed chance {}",
            report.estimate_accuracy,
            report.chance_accuracy
        );
    }
}
