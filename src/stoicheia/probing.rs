// SPDX-License-Identifier: MIT OR Apache-2.0

//! Neuron functional classification for `AlgZoo` `ReLU` RNNs.
//!
//! Runs structured inputs through the RNN and classifies each neuron's
//! response pattern by correlating activations with known reference signals.
//! For example, a neuron that tracks the running maximum will have high
//! Pearson correlation with `cummax(x[0..t])`.

use crate::error::Result;
use crate::stoicheia::config::StoicheiaConfig;
use crate::stoicheia::fast::{self, RnnWeights};

// ---------------------------------------------------------------------------
// NeuronRole
// ---------------------------------------------------------------------------

/// Functional role identified for a neuron.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronRole {
    /// Tracks the running maximum: `h_t ≈ max(0, x_0, ..., x_{t-1})`.
    RunningMax,
    /// Tracks the running minimum: `h_t ≈ min(x_0, ..., x_{t-1})`.
    RunningMin,
    /// Tracks max increment: `h_t ≈ max(...,x_{t-1}) − max(...,x_{t-2})`.
    MaxIncrement,
    /// Tracks a leave-one-out maximum: `h_T ≈ max(x \ x_i)` for some `i`.
    LeaveOneOutMax,
    /// Tracks recent input: `h_t ≈ x_{t-1}` or `h_t ≈ -x_{t-1}`.
    RecentInput,
    /// Compares two specific positions: `h_T ≈ sign(x_i - x_j)`.
    Comparator,
    /// No clear functional role identified (correlation below threshold).
    Unknown,
}

// ---------------------------------------------------------------------------
// Probe results
// ---------------------------------------------------------------------------

/// Result of probing a single neuron.
#[non_exhaustive]
pub struct NeuronProbeResult {
    /// Neuron index.
    pub neuron: usize,
    /// Best-matching functional role.
    pub role: NeuronRole,
    /// Pearson correlation between neuron activation and the best
    /// reference signal (at the final timestep, absolute value).
    pub correlation: f32,
}

/// Probe results for all neurons in a model.
#[non_exhaustive]
pub struct ProbeReport {
    /// Per-neuron probe results, ordered by neuron index.
    pub neurons: Vec<NeuronProbeResult>,
    /// Number of probe inputs used.
    pub n_probes: usize,
}

// ---------------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------------

/// Minimum absolute correlation to assign a role (below this → `Unknown`).
const CORRELATION_THRESHOLD: f32 = 0.8;

/// Run functional probes on all neurons of an RNN.
///
/// For each probe type, generates `n_probes` random inputs (deterministic
/// seed), captures per-timestep activations via [`fast::forward_fast_traced`],
/// and correlates each neuron's final-timestep activation with each probe's
/// reference signal. The best-matching probe (highest absolute correlation)
/// determines the neuron's role.
///
/// # Errors
///
/// Returns [`MIError::Config`](crate::MIError::Config) if the model
/// configuration is invalid.
#[allow(clippy::needless_range_loop)]
pub fn probe_neurons(
    weights: &RnnWeights,
    config: &StoicheiaConfig,
    n_probes: usize,
) -> Result<ProbeReport> {
    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    // Generate deterministic inputs
    let inputs = generate_probe_inputs(n_probes, seq_len);

    // Collect final-timestep activations for each neuron
    let mut neuron_activations = vec![vec![0.0_f32; n_probes]; h];
    let mut pre_acts = vec![0.0_f32; seq_len * h];
    let mut output = vec![0.0_f32; out_size];

    for (idx, input) in inputs.iter().enumerate() {
        fast::forward_fast_traced(weights, input, &mut pre_acts, &mut output, config)?;

        // Final-timestep hidden state = relu(pre_acts[last_timestep])
        for j in 0..h {
            // INDEX: (seq_len - 1) * h + j bounded by seq_len * h
            #[allow(clippy::indexing_slicing)]
            let pre = pre_acts[(seq_len - 1) * h + j];
            // INDEX: j bounded by h, idx bounded by n_probes
            #[allow(clippy::indexing_slicing)]
            {
                neuron_activations[j][idx] = pre.max(0.0);
            }
        }
    }

    // Compute reference signals and correlate
    let mut results = Vec::with_capacity(h);
    for j in 0..h {
        // INDEX: j bounded by h
        #[allow(clippy::indexing_slicing)]
        let activations = &neuron_activations[j];

        let (role, corr) = best_probe_match(activations, &inputs, seq_len);
        results.push(NeuronProbeResult {
            neuron: j,
            role,
            correlation: corr,
        });
    }

    Ok(ProbeReport {
        neurons: results,
        n_probes,
    })
}

// ---------------------------------------------------------------------------
// Internal: input generation
// ---------------------------------------------------------------------------

/// Generate deterministic probe inputs using a simple LCG.
fn generate_probe_inputs(n_probes: usize, seq_len: usize) -> Vec<Vec<f32>> {
    let mut inputs = Vec::with_capacity(n_probes);
    let mut state = 42_u64;
    for _ in 0..n_probes {
        let mut input = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            // Simple LCG: xₙ₊₁ = (a·xₙ + c) mod m
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Map to approximate Gaussian via simple transform
            // CAST: u64 → f32, deliberate precision loss for random generation
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let uniform = (state >> 33) as f32 / (1_u64 << 31) as f32; // [0, 1)
            let value = (uniform - 0.5) * 6.0; // approximate [-3, 3]
            input.push(value);
        }
        inputs.push(input);
    }
    inputs
}

// ---------------------------------------------------------------------------
// Internal: probe matching
// ---------------------------------------------------------------------------

/// Try all probe types, return the best match.
#[allow(clippy::too_many_lines)]
fn best_probe_match(activations: &[f32], inputs: &[Vec<f32>], seq_len: usize) -> (NeuronRole, f32) {
    let mut best_role = NeuronRole::Unknown;
    let mut best_corr = 0.0_f32;

    // Probe: running max at final timestep
    let ref_running_max: Vec<f32> = inputs
        .iter()
        .map(|inp| {
            let mut mx = f32::NEG_INFINITY;
            for &x in inp {
                if x > mx {
                    mx = x;
                }
            }
            mx.max(0.0) // ReLU
        })
        .collect();
    let c = pearson_abs(activations, &ref_running_max);
    if c > best_corr {
        best_corr = c;
        best_role = NeuronRole::RunningMax;
    }

    // Probe: running min at final timestep
    let ref_running_min: Vec<f32> = inputs
        .iter()
        .map(|inp| {
            let mut mn = f32::INFINITY;
            for &x in inp {
                if x < mn {
                    mn = x;
                }
            }
            mn.max(0.0) // ReLU: will be 0 for negative mins
        })
        .collect();
    let c = pearson_abs(activations, &ref_running_min);
    if c > best_corr {
        best_corr = c;
        best_role = NeuronRole::RunningMin;
    }

    // Probe: max increment at final timestep
    if seq_len >= 2 {
        let ref_max_inc: Vec<f32> = inputs
            .iter()
            .map(|inp| {
                let mut prev_max = f32::NEG_INFINITY;
                let mut curr_max = f32::NEG_INFINITY;
                for (t, &x) in inp.iter().enumerate() {
                    if t < seq_len - 1 && x > prev_max {
                        prev_max = x;
                    }
                    if x > curr_max {
                        curr_max = x;
                    }
                }
                (curr_max.max(0.0) - prev_max.max(0.0)).max(0.0)
            })
            .collect();
        let c = pearson_abs(activations, &ref_max_inc);
        if c > best_corr {
            best_corr = c;
            best_role = NeuronRole::MaxIncrement;
        }
    }

    // Probe: leave-one-out max (max of all minus last element)
    let ref_loo_max: Vec<f32> = inputs
        .iter()
        .map(|inp| {
            let overall_max = inp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // INDEX: seq_len - 1 valid because seq_len >= 2 for all AlgZoo tasks
            #[allow(clippy::indexing_slicing)]
            let last = inp[seq_len - 1];
            (overall_max.max(0.0) - last).max(0.0)
        })
        .collect();
    let c = pearson_abs(activations, &ref_loo_max);
    if c > best_corr {
        best_corr = c;
        best_role = NeuronRole::LeaveOneOutMax;
    }

    // Probe: recent input (last element)
    // INDEX: seq_len - 1 valid because seq_len >= 2
    #[allow(clippy::indexing_slicing)]
    let ref_recent: Vec<f32> = inputs.iter().map(|inp| inp[seq_len - 1].max(0.0)).collect();
    let c = pearson_abs(activations, &ref_recent);
    if c > best_corr {
        best_corr = c;
        best_role = NeuronRole::RecentInput;
    }

    // Probe: comparator — sign(x_a - x_b) for each pair.
    // O(T² × n_probes) — acceptable for T ≤ 10; for T > 20, precompute
    // a correlation matrix in one O(n_probes × T × H) pass instead.
    if seq_len >= 2 {
        for a in 0..seq_len {
            for b in 0..seq_len {
                if a == b {
                    continue;
                }
                // INDEX: a, b bounded by seq_len
                #[allow(clippy::indexing_slicing)]
                let ref_comp: Vec<f32> = inputs
                    .iter()
                    .map(|inp| (inp[a] - inp[b]).max(0.0))
                    .collect();
                let c = pearson_abs(activations, &ref_comp);
                if c > best_corr {
                    best_corr = c;
                    best_role = NeuronRole::Comparator;
                }
            }
        }
    }

    if best_corr < CORRELATION_THRESHOLD {
        best_role = NeuronRole::Unknown;
    }

    (best_role, best_corr)
}

/// Absolute Pearson correlation between two `f32` slices.
///
/// Returns 0.0 if either slice has zero variance.
fn pearson_abs(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    // CAST: usize → f32, n is small (n_probes typically 1000)
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let n_f = n as f32;

    let sum_a: f32 = a.iter().take(n).sum();
    let sum_b: f32 = b.iter().take(n).sum();
    let mean_a = sum_a / n_f;
    let mean_b = sum_b / n_f;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;

    for i in 0..n {
        // INDEX: i bounded by n = min(a.len(), b.len())
        #[allow(clippy::indexing_slicing)]
        {
            let da = a[i] - mean_a;
            let db = b[i] - mean_b;
            cov = da.mul_add(db, cov);
            var_a = da.mul_add(da, var_a);
            var_b = db.mul_add(db, var_b);
        }
    }

    let denom = (var_a * var_b).sqrt();
    // Guard handles both zero-variance and NaN: partial_cmp returns
    // None for NaN, so unwrap_or(Less) treats NaN as below threshold.
    if denom.partial_cmp(&1e-12) != Some(std::cmp::Ordering::Greater) {
        return 0.0;
    }
    (cov / denom).abs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_abs_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let c = pearson_abs(&a, &b);
        assert!((c - 1.0).abs() < 1e-5, "corr = {c}");
    }

    #[test]
    fn pearson_abs_negative_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let c = pearson_abs(&a, &b);
        // abs correlation should be ~1.0
        assert!((c - 1.0).abs() < 1e-5, "corr = {c}");
    }

    #[test]
    fn pearson_abs_zero_variance() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let c = pearson_abs(&a, &b);
        assert!(c < 1e-6, "corr = {c}");
    }

    #[test]
    fn probe_runs_on_tiny_model() {
        let weights = RnnWeights::new(
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
        );
        let config = crate::stoicheia::config::StoicheiaConfig::from_task(
            crate::stoicheia::config::StoicheiaTask::SecondArgmax,
            2,
            2,
        );

        let report = probe_neurons(&weights, &config, 100).unwrap();
        assert_eq!(report.neurons.len(), 2);
        assert_eq!(report.n_probes, 100);

        // Each neuron should have a role assigned (or Unknown)
        for n in &report.neurons {
            assert!(n.correlation >= 0.0);
        }
    }

    /// Eight synthetic probe inputs (`seq_len == 4`), all positive so the `ReLU` in
    /// the reference signals is a no-op.
    fn synthetic_probe_inputs() -> Vec<Vec<f32>> {
        vec![
            vec![3.0, 1.0, 2.0, 0.5],
            vec![1.0, 5.0, 2.0, 1.0],
            vec![0.2, 0.1, 4.0, 1.0],
            vec![2.0, 2.0, 2.0, 6.0],
            vec![1.0, 1.5, 1.0, 1.2],
            vec![7.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.5, 8.0],
            vec![2.5, 3.5, 1.0, 0.0],
        ]
    }

    #[test]
    fn best_probe_match_selects_running_max() {
        let inputs = synthetic_probe_inputs();
        // Per-input running max (== column-wise max, all positive).
        let activations = vec![3.0_f32, 5.0, 4.0, 6.0, 1.5, 7.0, 8.0, 3.5];
        let (role, corr) = best_probe_match(&activations, &inputs, 4);
        assert!(
            matches!(role, NeuronRole::RunningMax),
            "expected RunningMax, got {role:?} (corr {corr})"
        );
        assert!(corr > 0.99, "corr = {corr}");
    }

    #[test]
    fn best_probe_match_selects_recent_input() {
        let inputs = synthetic_probe_inputs();
        // Per-input last element (the RecentInput reference signal).
        let activations = vec![0.5_f32, 1.0, 1.0, 6.0, 1.2, 0.0, 8.0, 0.0];
        let (role, corr) = best_probe_match(&activations, &inputs, 4);
        assert!(
            matches!(role, NeuronRole::RecentInput),
            "expected RecentInput, got {role:?} (corr {corr})"
        );
        assert!(corr > 0.99, "corr = {corr}");
    }

    #[test]
    fn best_probe_match_constant_activation_is_unknown() {
        let inputs = synthetic_probe_inputs();
        // Zero-variance activations correlate with nothing (pearson_abs → 0), so
        // the best correlation stays below CORRELATION_THRESHOLD → Unknown.
        let activations = vec![4.0_f32; 8];
        let (role, corr) = best_probe_match(&activations, &inputs, 4);
        assert!(
            matches!(role, NeuronRole::Unknown),
            "expected Unknown, got {role:?} (corr {corr})"
        );
        assert!(corr < 1e-6, "corr = {corr}");
    }
}
