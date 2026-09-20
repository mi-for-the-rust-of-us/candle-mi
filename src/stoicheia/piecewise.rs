// SPDX-License-Identifier: MIT OR Apache-2.0

//! Piecewise-linear region enumeration for `ReLU` RNNs.
//!
//! A single-layer `ReLU` RNN with H neurons over T timesteps has at most
//! 2^(H·T) linear regions in input space. Each region is defined by an
//! activation pattern (which neurons fire at which timesteps). Within each
//! region, the RNN reduces to a single linear map.
//!
//! For tiny `AlgZoo` models (M₂,₂: 16 regions, M₄,₃: 4096), exhaustive
//! enumeration is tractable. For larger models, sampling reveals the
//! most-populated regions.

use std::collections::HashMap;

use crate::error::{MIError, Result};
use crate::stoicheia::config::StoicheiaConfig;
use crate::stoicheia::fast::{self, RnnWeights};

// ---------------------------------------------------------------------------
// ActivationPattern
// ---------------------------------------------------------------------------

/// The activation pattern of an RNN over a sequence.
///
/// For each timestep `t` and neuron `j`, records whether
/// `pre_activation[t][j] >= 0` (active) or `< 0` (inactive).
/// Bit `t * H + j` in the internal representation corresponds
/// to neuron `j` at timestep `t`.
///
/// Uses `[u64; 5]` for up to 320 bits, covering all `AlgZoo` RNN
/// configurations (max H=32, T=10 = 320).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActivationPattern {
    /// Compact bit storage (320 bits).
    bits: [u64; 5],
    /// Number of neurons (H).
    hidden_size: usize,
    /// Number of timesteps (T).
    seq_len: usize,
}

impl ActivationPattern {
    /// Create from a pre-activation trace.
    ///
    /// # Shapes
    /// - `pre_acts`: row-major `[seq_len, hidden_size]`
    /// # Panics
    ///
    /// Panics if `hidden_size * seq_len > 320` (exceeds bit-vector capacity).
    #[must_use]
    pub fn from_pre_activations(pre_acts: &[f32], hidden_size: usize, seq_len: usize) -> Self {
        assert!(
            hidden_size * seq_len <= 320,
            "hidden_size ({hidden_size}) * seq_len ({seq_len}) = {} exceeds 320-bit capacity",
            hidden_size * seq_len
        );
        let mut bits = [0_u64; 5];
        for t in 0..seq_len {
            for j in 0..hidden_size {
                let idx = t * hidden_size + j;
                // INDEX: idx bounded by seq_len * hidden_size = pre_acts.len()
                #[allow(clippy::indexing_slicing)]
                if pre_acts[idx] >= 0.0 {
                    // INDEX: idx / 64 < 320 / 64 = 5
                    #[allow(clippy::indexing_slicing)]
                    {
                        bits[idx / 64] |= 1_u64 << (idx % 64);
                    }
                }
            }
        }
        Self {
            bits,
            hidden_size,
            seq_len,
        }
    }

    /// Whether neuron `j` is active (pre-act ≥ 0) at timestep `t`.
    ///
    /// # Panics
    ///
    /// Panics if `t * hidden_size + j >= 320`.
    #[must_use]
    // INDEX: idx / 64 bounded by 320 / 64 = 4 < 5 for valid AlgZoo configs
    #[allow(clippy::indexing_slicing)]
    pub const fn is_active(&self, t: usize, j: usize) -> bool {
        let idx = t * self.hidden_size + j;
        (self.bits[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// Number of active neurons across all timesteps.
    #[must_use]
    pub fn count_active(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Total neuron-timestep slots (H · T).
    #[must_use]
    pub const fn total_slots(&self) -> usize {
        self.hidden_size * self.seq_len
    }

    /// Per-timestep active neuron count.
    ///
    /// Returns a vector of length `seq_len`, where entry `t` is the
    /// number of active neurons at timestep `t`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    pub fn active_per_timestep(&self) -> Vec<u32> {
        (0..self.seq_len)
            .map(|t| {
                // CAST: usize → u32, count ≤ hidden_size ≤ 32
                (0..self.hidden_size)
                    .filter(|&j| self.is_active(t, j))
                    .count() as u32
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// RegionInfo / RegionMap
// ---------------------------------------------------------------------------

/// Information about a single linear region.
#[non_exhaustive]
pub struct RegionInfo {
    /// Number of inputs that fell into this region.
    pub count: usize,
    /// A representative input from this region.
    pub representative: Vec<f32>,
    /// The activation pattern defining this region.
    pub pattern: ActivationPattern,
}

/// Result of classifying a batch of inputs into linear regions.
#[non_exhaustive]
pub struct RegionMap {
    /// Distinct activation patterns observed, with counts and
    /// representatives. Sorted by count descending (most populated
    /// region first).
    pub regions: Vec<RegionInfo>,
    /// Total number of inputs classified.
    pub total_inputs: usize,
}

// ---------------------------------------------------------------------------
// Region classification
// ---------------------------------------------------------------------------

/// Classify inputs into their piecewise-linear regions.
///
/// Runs [`fast::forward_fast_traced`] on each input, records the activation
/// pattern, and groups inputs by pattern.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
///
/// # Errors
///
/// Returns [`MIError::Config`] if slice dimensions
/// do not match.
pub fn classify_regions(
    weights: &RnnWeights,
    inputs: &[f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> Result<RegionMap> {
    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    // Temporary buffers for single-input traced forward
    let mut pre_acts = vec![0.0_f32; seq_len * h];
    let mut output = vec![0.0_f32; out_size];

    // Group by activation pattern
    let mut by_pattern: HashMap<ActivationPattern, (usize, Vec<f32>)> = HashMap::new();

    for i in 0..n_inputs {
        // INDEX: slice bounds valid because inputs.len() == n_inputs * seq_len
        #[allow(clippy::indexing_slicing)]
        let input_slice = &inputs[i * seq_len..(i + 1) * seq_len];

        fast::forward_fast_traced(weights, input_slice, &mut pre_acts, &mut output, config)?;

        let pattern = ActivationPattern::from_pre_activations(&pre_acts, h, seq_len);

        let entry = by_pattern
            .entry(pattern)
            .or_insert_with(|| (0, input_slice.to_vec()));
        entry.0 += 1;
    }

    // Convert to sorted Vec<RegionInfo>
    let mut regions: Vec<RegionInfo> = by_pattern
        .into_iter()
        .map(|(pattern, (count, representative))| RegionInfo {
            count,
            representative,
            pattern,
        })
        .collect();

    // Sort by count descending (most populated first)
    regions.sort_by_key(|r| std::cmp::Reverse(r.count));

    Ok(RegionMap {
        regions,
        total_inputs: n_inputs,
    })
}

/// Compute the linear map for a given activation pattern.
///
/// For a fixed activation pattern, the RNN (with `bias=False`) reduces to
/// a pure linear map: `output = A · input` where `A` has shape
/// `[output_size, seq_len]`.
///
/// Column `t` of `A` is `W_oh · M(T-1, t+1) · G_t · W_ih`, where
/// `M(b, a) = G_b · W_hh · G_{b-1} · W_hh · ... · G_a · W_hh`
/// (product in decreasing index order), `M(b, a) = I` when `a > b`,
/// and `G_t` is the diagonal gate matrix for timestep `t`.
///
/// # Returns
///
/// Matrix `A` of shape `[output_size, seq_len]`, row-major.
///
/// # Errors
///
/// Returns [`MIError::Model`] if the iterated
/// matrix-vector product overflows (non-contractive `W_hh`).
// EXPLICIT: index arithmetic matches the row-major matrix layout.
#[allow(clippy::needless_range_loop)]
pub fn region_linear_map(
    weights: &RnnWeights,
    pattern: &ActivationPattern,
    config: &StoicheiaConfig,
) -> Result<Vec<f32>> {
    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    // Result matrix A: [output_size, seq_len], row-major
    let mut a_mat = vec![0.0_f32; out_size * seq_len];

    // For each input timestep t, compute column t of A
    for t in 0..seq_len {
        // Start with G_t · W_ih (a vector of length H)
        // G_t is diagonal: G_t[j,j] = 1 if active, else 0
        // W_ih is [H, 1], so G_t · W_ih is just gated W_ih
        let mut vec_h = [0.0_f32; 32];
        for j in 0..h {
            // INDEX: j bounded by h ≤ 32
            #[allow(clippy::indexing_slicing)]
            if pattern.is_active(t, j) {
                vec_h[j] = weights.weight_ih[j];
            }
        }

        // Multiply by M(T-1, t+1) = G_{T-1}·W_hh · G_{T-2}·W_hh · ... · G_{t+1}·W_hh
        // Applied right-to-left: for s from t+1 to T-1, apply G_s · W_hh
        for s in (t + 1)..seq_len {
            // new_vec = G_s · W_hh · vec_h
            let mut new_vec = [0.0_f32; 32];
            for j in 0..h {
                if pattern.is_active(s, j) {
                    let mut acc = 0.0_f32;
                    for k in 0..h {
                        // INDEX: j * h + k bounded by h * h
                        #[allow(clippy::indexing_slicing)]
                        {
                            acc = weights.weight_hh[j * h + k].mul_add(vec_h[k], acc);
                        }
                    }
                    // INDEX: j bounded by h ≤ 32
                    #[allow(clippy::indexing_slicing)]
                    {
                        new_vec[j] = acc;
                    }
                }
                // else: G_s[j,j] = 0, so new_vec[j] stays 0
            }
            vec_h = new_vec;

            // Guard: detect non-contractive W_hh causing exponential growth
            // INDEX: h ≤ 32 = vec_h.len()
            #[allow(clippy::indexing_slicing)]
            if vec_h[..h].iter().any(|v| v.is_infinite()) {
                return Err(MIError::Model(candle_core::Error::Msg(format!(
                    "region_linear_map: overflow at timestep {s} \
                     (W_hh likely has spectral radius > 1)"
                ))));
            }
        }

        // Multiply by W_oh: column t of A = W_oh · vec_h
        for o in 0..out_size {
            let mut acc = 0.0_f32;
            for j in 0..h {
                // INDEX: o * h + j bounded by out_size * h
                #[allow(clippy::indexing_slicing)]
                {
                    acc = weights.weight_oh[o * h + j].mul_add(vec_h[j], acc);
                }
            }
            // INDEX: o * seq_len + t bounded by out_size * seq_len
            #[allow(clippy::indexing_slicing)]
            {
                a_mat[o * seq_len + t] = acc;
            }
        }
    }

    Ok(a_mat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
    use crate::stoicheia::fast::RnnWeights;

    fn test_weights_2_2() -> RnnWeights {
        RnnWeights::new(
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, -1.0, -1.0, 1.0],
            2,
            2,
        )
    }

    fn test_config_2_2() -> StoicheiaConfig {
        StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2)
    }

    #[test]
    fn activation_pattern_roundtrip() {
        // pre_acts: [seq_len=2, H=2] = [0.5, -0.3, 1.0, 0.0]
        let pre_acts = vec![0.5_f32, -0.3, 1.0, 0.0];
        let pattern = ActivationPattern::from_pre_activations(&pre_acts, 2, 2);

        // t=0: neuron 0 active (0.5 >= 0), neuron 1 inactive (-0.3 < 0)
        assert!(pattern.is_active(0, 0));
        assert!(!pattern.is_active(0, 1));
        // t=1: neuron 0 active (1.0 >= 0), neuron 1 active (0.0 >= 0)
        assert!(pattern.is_active(1, 0));
        assert!(pattern.is_active(1, 1));

        assert_eq!(pattern.count_active(), 3);
        assert_eq!(pattern.total_slots(), 4);
        assert_eq!(pattern.active_per_timestep(), vec![1, 2]);
    }

    #[test]
    fn m2_2_has_few_regions() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        // Generate 1000 random-ish inputs (deterministic via simple formula)
        let n = 1000;
        let inputs: Vec<f32> = (0..n * 2)
            .map(|i| {
                // CAST: usize → f32, small test indices
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let v = ((i as f32) * 0.618_034).sin() * 3.0;
                v
            })
            .collect();

        let region_map = classify_regions(&weights, &inputs, n, &config).unwrap();

        // M₂,₂ has at most 2^(2*2) = 16 regions
        assert!(
            region_map.regions.len() <= 16,
            "found {} regions, expected ≤ 16",
            region_map.regions.len()
        );
        assert_eq!(region_map.total_inputs, n);

        // All inputs accounted for
        let total: usize = region_map.regions.iter().map(|r| r.count).sum();
        assert_eq!(total, n);
    }

    #[test]
    fn linear_map_matches_forward() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        // Pick a specific input and compute its region
        let input = vec![0.5_f32, -0.3];
        let mut pre_acts = vec![0.0_f32; 4];
        let mut traced_out = vec![0.0_f32; 2];
        fast::forward_fast_traced(&weights, &input, &mut pre_acts, &mut traced_out, &config)
            .unwrap();

        let pattern = ActivationPattern::from_pre_activations(&pre_acts, 2, 2);
        let a_mat = region_linear_map(&weights, &pattern, &config).unwrap();

        // A is [output_size=2, seq_len=2], row-major
        // output = A · input
        let mut linear_out = [0.0_f32; 2];
        for o in 0..2 {
            for t in 0..2 {
                // INDEX: o * 2 + t bounded by 4 = a_mat.len()
                #[allow(clippy::indexing_slicing)]
                {
                    linear_out[o] = a_mat[o * 2 + t].mul_add(input[t], linear_out[o]);
                }
            }
        }

        // The linear map output should match the forward output exactly
        // (since the input is in the same activation region)
        for (a, b) in linear_out.iter().zip(&traced_out) {
            assert!((a - b).abs() < 1e-5, "linear={a}, forward={b}");
        }
    }
}
