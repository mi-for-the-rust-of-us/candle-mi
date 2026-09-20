// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight standardization for `AlgZoo` `ReLU` RNNs.
//!
//! Rescales each neuron so that `|W_ih[j]| = 1`, following the methodology
//! from the `AlgZoo` blog post. After standardization, the sign of `W_ih[j]`
//! directly indicates whether neuron `j` is excited (+1) or inhibited (−1)
//! by its input.
//!
//! The transformation is exact: the standardized model produces identical
//! outputs to the original for all inputs.

use crate::error::{MIError, Result};
use crate::stoicheia::StoicheiaRnn;
use crate::stoicheia::fast::RnnWeights;

// ---------------------------------------------------------------------------
// StandardizedRnn
// ---------------------------------------------------------------------------

/// Result of standardizing an RNN's weights.
///
/// The standardized model is input-output equivalent to the original:
/// `forward(standardized) == forward(original)` for all inputs.
///
/// Standardization uses scale factor `s_j = |W_ih_orig[j]|` per neuron:
/// - `W_ih[j] = W_ih_orig[j] / s_j` → ±1
/// - `W_hh[j,k] = W_hh_orig[j,k] * s_k / s_j`
/// - `W_oh[o,j] = W_oh_orig[o,j] * s_j`
#[non_exhaustive]
#[derive(Debug, Clone)]
#[allow(clippy::similar_names)]
pub struct StandardizedRnn {
    /// Per-neuron scale factors (`scales[j] = |W_ih_orig[j]|`).
    pub scales: Vec<f32>,
    /// Standardized input-to-hidden weights, row-major `[H, 1]`.
    /// Each entry is ±1 (to floating-point precision).
    pub weight_ih: Vec<f32>,
    /// Standardized hidden-to-hidden weights, row-major `[H, H]`.
    pub weight_hh: Vec<f32>,
    /// Standardized output weights, row-major `[output_size, H]`.
    pub weight_oh: Vec<f32>,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Output dimension.
    pub output_size: usize,
}

impl StandardizedRnn {
    /// Convert to [`RnnWeights`] for use with the fast-path kernel.
    #[must_use]
    pub fn to_rnn_weights(&self) -> RnnWeights {
        RnnWeights::new(
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.weight_oh.clone(),
            self.hidden_size,
            self.output_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Standardization functions
// ---------------------------------------------------------------------------

/// Standardize an RNN so that each neuron's input weight satisfies
/// `|W_ih[j]| = 1`.
///
/// This preserves the input-output map exactly. After standardization,
/// the sign of `W_ih[j]` directly indicates whether neuron `j` is
/// excited (+1) or inhibited (−1) by its input.
///
/// # Errors
///
/// Returns [`MIError::Model`] if weight extraction
/// fails.
/// Returns [`MIError::Config`] if any `W_ih[j]`
/// is exactly zero (degenerate neuron — no input sensitivity).
pub fn standardize_rnn(model: &StoicheiaRnn) -> Result<StandardizedRnn> {
    let weights = RnnWeights::from_model(model)?;
    standardize_weights(&weights)
}

/// Standardize from raw weights (avoids re-extracting from `Tensor`).
///
/// # Errors
///
/// Returns [`MIError::Config`] if any `W_ih[j]`
/// is exactly zero.
// EXPLICIT: index arithmetic matches the row-major weight layout.
#[allow(clippy::needless_range_loop, clippy::similar_names)]
pub fn standardize_weights(weights: &RnnWeights) -> Result<StandardizedRnn> {
    let h = weights.hidden_size;
    let out_size = weights.output_size;

    // Compute scale factors: s_j = |W_ih[j]|
    let mut scales = Vec::with_capacity(h);
    for j in 0..h {
        // INDEX: j bounded by h = weight_ih.len()
        #[allow(clippy::indexing_slicing)]
        let s = weights.weight_ih[j].abs();
        if s < f32::EPSILON {
            return Err(MIError::Config(format!(
                "W_ih[{j}] magnitude {s} is below f32::EPSILON \
                 (degenerate neuron, standardization would amplify by 1/{s})"
            )));
        }
        scales.push(s);
    }

    // W_ih[j] / s_j → ±1
    let mut std_ih = Vec::with_capacity(h);
    for j in 0..h {
        // INDEX: j bounded by h
        #[allow(clippy::indexing_slicing)]
        {
            std_ih.push(weights.weight_ih[j] / scales[j]);
        }
    }

    // W_hh[j,k] * s_k / s_j
    let mut std_hh = vec![0.0_f32; h * h];
    for j in 0..h {
        for k in 0..h {
            // INDEX: j * h + k bounded by h * h
            #[allow(clippy::indexing_slicing)]
            {
                std_hh[j * h + k] = weights.weight_hh[j * h + k] * scales[k] / scales[j];
            }
        }
    }

    // W_oh[o,j] * s_j
    let mut std_oh = vec![0.0_f32; out_size * h];
    for o in 0..out_size {
        for j in 0..h {
            // INDEX: o * h + j bounded by out_size * h
            #[allow(clippy::indexing_slicing)]
            {
                std_oh[o * h + j] = weights.weight_oh[o * h + j] * scales[j];
            }
        }
    }

    Ok(StandardizedRnn {
        scales,
        weight_ih: std_ih,
        weight_hh: std_hh,
        weight_oh: std_oh,
        hidden_size: h,
        output_size: out_size,
    })
}

/// Maximum deviation of standardized `W_ih` from {+1, −1}.
///
/// A value of 0.0 exactly means perfect standardization (always the
/// case unless limited by `f32` precision).
#[must_use]
pub fn standardization_quality(std_rnn: &StandardizedRnn) -> f32 {
    let mut max_dev = 0.0_f32;
    for &w in &std_rnn.weight_ih {
        let dev = (w.abs() - 1.0).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    max_dev
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_weights() -> RnnWeights {
        // M₂,₂ with non-unit W_ih to test rescaling
        RnnWeights::new(
            vec![2.0, -0.5],            // W_ih [2, 1]
            vec![1.0, 0.0, 0.0, 1.0],   // W_hh [2, 2] = identity
            vec![1.0, -1.0, -1.0, 1.0], // W_oh [2, 2]
            2,
            2,
        )
    }

    #[test]
    fn standardized_wih_near_one() {
        let weights = test_weights();
        let std_rnn = standardize_weights(&weights).unwrap();

        // W_ih should be [+1, -1] after standardization
        assert!((std_rnn.weight_ih[0] - 1.0).abs() < 1e-6);
        assert!((std_rnn.weight_ih[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn standardization_quality_is_zero() {
        let weights = test_weights();
        let std_rnn = standardize_weights(&weights).unwrap();
        let quality = standardization_quality(&std_rnn);
        assert!(quality < 1e-6, "quality = {quality}");
    }

    #[test]
    fn standardize_preserves_output() {
        let weights = test_weights();
        let config = crate::stoicheia::config::StoicheiaConfig::from_task(
            crate::stoicheia::config::StoicheiaTask::SecondArgmax,
            2,
            2,
        );

        let std_rnn = standardize_weights(&weights).unwrap();
        let std_weights = std_rnn.to_rnn_weights();

        // Run both on the same input
        let inputs = vec![0.5_f32, -0.3, 1.0, 2.0, -1.0, 0.7];
        let n = 3;
        let mut orig_out = vec![0.0_f32; n * 2];
        let mut std_out = vec![0.0_f32; n * 2];

        crate::stoicheia::fast::forward_fast(&weights, &inputs, &mut orig_out, n, &config).unwrap();
        crate::stoicheia::fast::forward_fast(&std_weights, &inputs, &mut std_out, n, &config)
            .unwrap();

        for (a, b) in orig_out.iter().zip(&std_out) {
            assert!((a - b).abs() < 1e-5, "output mismatch: orig={a}, std={b}");
        }
    }

    #[test]
    fn degenerate_neuron_errors() {
        let weights = RnnWeights::new(
            vec![0.0, 1.0], // W_ih[0] = 0 → degenerate
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            2,
        );
        let result = standardize_weights(&weights);
        assert!(result.is_err());
    }

    #[test]
    fn to_rnn_weights_roundtrip() {
        let weights = test_weights();
        let std_rnn = standardize_weights(&weights).unwrap();
        let rnn_weights = std_rnn.to_rnn_weights();

        assert_eq!(rnn_weights.hidden_size, std_rnn.hidden_size);
        assert_eq!(rnn_weights.output_size, std_rnn.output_size);
        assert_eq!(rnn_weights.weight_ih, std_rnn.weight_ih);
    }
}
