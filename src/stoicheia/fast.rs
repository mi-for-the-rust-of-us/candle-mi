// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fast-path RNN kernel — raw `f32` forward pass without `Tensor` overhead.
//!
//! Phase A benchmarks showed candle's per-tensor-operation overhead dominates
//! on tiny `AlgZoo` models (18–25× slower than `PyTorch`). This module
//! provides a raw `f32` forward pass that operates on `&[f32]` slices with
//! no heap allocation per timestep.
//!
//! All Phase B analysis modules (`ablation`, `piecewise`, `probing`,
//! `surprise`) use these functions internally for bulk forward passes.

use crate::error::{MIError, Result};
use crate::stoicheia::StoicheiaRnn;
use crate::stoicheia::config::StoicheiaConfig;

/// Maximum hidden size supported by the stack-allocated buffer.
///
/// Covers all `AlgZoo` hidden sizes (2, 4, 6, 8, 10, 12, 16, 20, 24, 32).
const MAX_H: usize = 32;

// ---------------------------------------------------------------------------
// RnnWeights
// ---------------------------------------------------------------------------

/// Raw `f32` weight storage for an `AlgZoo` RNN.
///
/// Extracted from [`Tensor`](candle_core::Tensor) once at load time, then
/// used by the fast-path kernel and all analysis modules without further
/// candle overhead.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[allow(clippy::similar_names)]
pub struct RnnWeights {
    /// Input-to-hidden weights, row-major `[hidden_size, 1]`.
    pub weight_ih: Vec<f32>,
    /// Hidden-to-hidden weights, row-major `[hidden_size, hidden_size]`.
    pub weight_hh: Vec<f32>,
    /// Output projection weights, row-major `[output_size, hidden_size]`.
    pub weight_oh: Vec<f32>,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Output dimension (`seq_len` for distribution tasks, 1 for scalar).
    pub output_size: usize,
}

impl RnnWeights {
    /// Extract raw `f32` weights from a loaded [`StoicheiaRnn`].
    ///
    /// Copies each weight tensor to a contiguous `Vec<f32>` on CPU.
    /// This is a one-time cost; all subsequent fast-path operations
    /// use these slices directly.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor
    /// extraction fails.
    #[allow(clippy::similar_names)]
    pub fn from_model(model: &StoicheiaRnn) -> Result<Self> {
        let config = model.config();
        let weight_ih: Vec<f32> = model.weight_ih().flatten_all()?.to_vec1()?;
        let weight_hh: Vec<f32> = model.weight_hh().flatten_all()?.to_vec1()?;
        let weight_oh: Vec<f32> = model.weight_oh().flatten_all()?.to_vec1()?;

        Ok(Self {
            weight_ih,
            weight_hh,
            weight_oh,
            hidden_size: config.hidden_size,
            output_size: config.output_size(),
        })
    }

    /// Create from explicit weight vectors (for standardized or handcrafted
    /// models).
    ///
    /// # Panics
    ///
    /// Panics if `hidden_size > 32` (exceeds stack buffer) or if slice
    /// lengths do not match the declared dimensions.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn new(
        weight_ih: Vec<f32>,
        weight_hh: Vec<f32>,
        weight_oh: Vec<f32>,
        hidden_size: usize,
        output_size: usize,
    ) -> Self {
        assert!(
            hidden_size <= MAX_H,
            "hidden_size {hidden_size} exceeds MAX_H {MAX_H}"
        );
        assert_eq!(weight_ih.len(), hidden_size);
        assert_eq!(weight_hh.len(), hidden_size * hidden_size);
        assert_eq!(weight_oh.len(), output_size * hidden_size);
        Self {
            weight_ih,
            weight_hh,
            weight_oh,
            hidden_size,
            output_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Forward pass functions
// ---------------------------------------------------------------------------

/// Run a batch of RNN forward passes on raw `f32` slices.
///
/// No [`Tensor`](candle_core::Tensor) objects, no heap allocation per
/// timestep. The inner loop is structured for auto-vectorization:
/// contiguous slices, no branches in the hot path, hoisted invariants.
///
/// # Shapes
/// - `inputs`: row-major `[n_inputs, seq_len]`
/// - `outputs`: row-major `[n_inputs, output_size]`
///
/// # Errors
///
/// Returns [`MIError::Config`] if slice lengths
/// do not match the declared dimensions, or if `hidden_size` exceeds the
/// stack buffer.
pub fn forward_fast(
    weights: &RnnWeights,
    inputs: &[f32],
    outputs: &mut [f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> Result<()> {
    validate_fast_args(weights, inputs, outputs, n_inputs, config)?;

    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    for i in 0..n_inputs {
        let mut hidden = [0.0_f32; MAX_H];
        let mut pre_act = [0.0_f32; MAX_H];

        // RNN timestep loop
        for t in 0..seq_len {
            // INDEX: i * seq_len + t bounded by n_inputs * seq_len = inputs.len()
            #[allow(clippy::indexing_slicing)]
            let x_t = inputs[i * seq_len + t];

            rnn_cell(weights, x_t, &hidden, &mut pre_act, h);

            // ReLU
            for j in 0..h {
                // INDEX: j bounded by h <= MAX_H
                #[allow(clippy::indexing_slicing)]
                {
                    hidden[j] = pre_act[j].max(0.0);
                }
            }
        }

        // Output projection: W_oh @ hidden
        output_projection(weights, &hidden, outputs, i, h, out_size);
    }

    Ok(())
}

/// Run forward passes with neuron ablation.
///
/// Same as [`forward_fast`], but neurons marked `true` in `ablated`
/// have their hidden state forced to zero after each `ReLU` step.
///
/// # Shapes
/// - `ablated`: `[hidden_size]`, `true` = zero this neuron
///
/// # Errors
///
/// Returns [`MIError::Config`] if slice lengths
/// do not match the declared dimensions.
pub fn forward_fast_ablated(
    weights: &RnnWeights,
    inputs: &[f32],
    outputs: &mut [f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
    ablated: &[bool],
) -> Result<()> {
    validate_fast_args(weights, inputs, outputs, n_inputs, config)?;
    if ablated.len() != weights.hidden_size {
        return Err(MIError::Config(format!(
            "ablated length {} != hidden_size {}",
            ablated.len(),
            weights.hidden_size
        )));
    }

    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    for i in 0..n_inputs {
        let mut hidden = [0.0_f32; MAX_H];
        let mut pre_act = [0.0_f32; MAX_H];

        for t in 0..seq_len {
            // INDEX: i * seq_len + t bounded by n_inputs * seq_len = inputs.len()
            #[allow(clippy::indexing_slicing)]
            let x_t = inputs[i * seq_len + t];

            rnn_cell(weights, x_t, &hidden, &mut pre_act, h);

            // ReLU + ablation
            for j in 0..h {
                // INDEX: j bounded by h <= MAX_H
                #[allow(clippy::indexing_slicing)]
                {
                    hidden[j] = if ablated[j] { 0.0 } else { pre_act[j].max(0.0) };
                }
            }
        }

        output_projection(weights, &hidden, outputs, i, h, out_size);
    }

    Ok(())
}

/// Run a single forward pass, returning the full activation trace.
///
/// Records pre-activation values at every (timestep, neuron) pair.
/// Used by piecewise-linear analysis to determine which neurons fire.
///
/// # Shapes
/// - `input`: `[seq_len]` (single input)
/// - `pre_activations`: row-major `[seq_len, hidden_size]` (output)
/// - `output`: `[output_size]` (output)
///
/// # Errors
///
/// Returns [`MIError::Config`] if slice lengths
/// do not match the declared dimensions.
// EXPLICIT: range loops with index arithmetic for SIMD-friendly layout.
#[allow(clippy::needless_range_loop)]
pub fn forward_fast_traced(
    weights: &RnnWeights,
    input: &[f32],
    pre_activations: &mut [f32],
    output: &mut [f32],
    config: &StoicheiaConfig,
) -> Result<()> {
    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    if h > MAX_H {
        return Err(MIError::Config(format!(
            "hidden_size {h} exceeds MAX_H {MAX_H}"
        )));
    }
    if input.len() != seq_len {
        return Err(MIError::Config(format!(
            "input length {} != seq_len {seq_len}",
            input.len()
        )));
    }
    if pre_activations.len() != seq_len * h {
        return Err(MIError::Config(format!(
            "pre_activations length {} != seq_len * hidden_size {}",
            pre_activations.len(),
            seq_len * h
        )));
    }
    if output.len() != out_size {
        return Err(MIError::Config(format!(
            "output length {} != output_size {out_size}",
            output.len()
        )));
    }

    let mut hidden = [0.0_f32; MAX_H];

    for t in 0..seq_len {
        // INDEX: t bounded by seq_len = input.len()
        #[allow(clippy::indexing_slicing)]
        let x_t = input[t];

        // Compute pre-activation into the output trace buffer directly
        for j in 0..h {
            // INDEX: j bounded by h, array indices bounded by weight slice lengths
            #[allow(clippy::indexing_slicing)]
            {
                let mut acc = x_t * weights.weight_ih[j];
                for k in 0..h {
                    acc = weights.weight_hh[j * h + k].mul_add(hidden[k], acc);
                }
                pre_activations[t * h + j] = acc;
            }
        }

        // ReLU → hidden
        for j in 0..h {
            // INDEX: j bounded by h <= MAX_H
            #[allow(clippy::indexing_slicing)]
            {
                hidden[j] = pre_activations[t * h + j].max(0.0);
            }
        }
    }

    // Output projection
    for o in 0..out_size {
        let mut acc = 0.0_f32;
        for j in 0..h {
            // INDEX: o * h + j bounded by out_size * h = weight_oh.len()
            #[allow(clippy::indexing_slicing)]
            {
                acc = weights.weight_oh[o * h + j].mul_add(hidden[j], acc);
            }
        }
        // INDEX: o bounded by out_size = output.len()
        #[allow(clippy::indexing_slicing)]
        {
            output[o] = acc;
        }
    }

    Ok(())
}

/// Fraction of rows whose `argmax` matches the target.
///
/// `outputs` is the flat `[n_inputs * out_size]` buffer the fast forward fills,
/// read one `out_size` row per target. Extracted because `accuracy` here and
/// both ablation sweeps in [`ablation`](super::ablation) computed it with
/// identical code.
///
/// Returns `0.0` when `n_inputs` is zero rather than dividing by it.
pub(crate) fn accuracy_from_outputs(
    outputs: &[f32],
    targets: &[u32],
    out_size: usize,
    n_inputs: usize,
) -> f32 {
    if n_inputs == 0 {
        return 0.0;
    }
    let mut correct = 0_usize;
    for (i, target) in targets.iter().enumerate() {
        // INDEX: slice bounds valid because outputs.len() == n_inputs * out_size
        #[allow(clippy::indexing_slicing)]
        let row = &outputs[i * out_size..(i + 1) * out_size];
        if *target == argmax_f32(row) {
            correct += 1;
        }
    }
    // CAST: usize → f32, counts are small (<= n_inputs)
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    {
        correct as f32 / n_inputs as f32
    }
}

/// Compute model accuracy on a batch of inputs.
///
/// Runs [`forward_fast`], takes argmax of each output row, compares
/// with targets.
///
/// # Returns
///
/// Fraction of correct predictions (0.0 to 1.0).
///
/// # Errors
///
/// Returns [`MIError::Config`] if slice lengths
/// do not match.
pub fn accuracy(
    weights: &RnnWeights,
    inputs: &[f32],
    targets: &[u32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> Result<f32> {
    if targets.len() != n_inputs {
        return Err(MIError::Config(format!(
            "targets length {} != n_inputs {n_inputs}",
            targets.len()
        )));
    }

    let out_size = weights.output_size;
    let mut outputs = vec![0.0_f32; n_inputs * out_size];
    forward_fast(weights, inputs, &mut outputs, n_inputs, config)?;

    Ok(accuracy_from_outputs(&outputs, targets, out_size, n_inputs))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Single RNN cell: `pre_act = x_t * W_ih + W_hh @ hidden`.
// EXPLICIT: range loops with index arithmetic are SIMD-friendly;
// iterators would obscure the memory-access pattern.
#[inline]
#[allow(clippy::needless_range_loop)]
fn rnn_cell(
    weights: &RnnWeights,
    x_t: f32,
    hidden: &[f32; MAX_H],
    pre_act: &mut [f32; MAX_H],
    h: usize,
) {
    for j in 0..h {
        // INDEX: j bounded by h <= MAX_H; j * h + k bounded by h * h = weight_hh.len()
        #[allow(clippy::indexing_slicing)]
        {
            let mut acc = x_t * weights.weight_ih[j];
            for k in 0..h {
                acc = weights.weight_hh[j * h + k].mul_add(hidden[k], acc);
            }
            pre_act[j] = acc;
        }
    }
}

/// Output projection: `outputs[i] = W_oh @ hidden`.
// EXPLICIT: range loops with index arithmetic for SIMD-friendly layout.
#[inline]
#[allow(clippy::needless_range_loop)]
fn output_projection(
    weights: &RnnWeights,
    hidden: &[f32; MAX_H],
    outputs: &mut [f32],
    i: usize,
    h: usize,
    out_size: usize,
) {
    for o in 0..out_size {
        let mut acc = 0.0_f32;
        for j in 0..h {
            // INDEX: o * h + j bounded by out_size * h = weight_oh.len()
            #[allow(clippy::indexing_slicing)]
            {
                acc = weights.weight_oh[o * h + j].mul_add(hidden[j], acc);
            }
        }
        // INDEX: i * out_size + o bounded by n_inputs * out_size = outputs.len()
        #[allow(clippy::indexing_slicing)]
        {
            outputs[i * out_size + o] = acc;
        }
    }
}

/// Argmax over a `f32` slice. Returns 0 for empty slices.
///
/// Used by [`accuracy`] and by the `ablation` and `surprise` modules.
#[must_use]
pub fn argmax_f32(slice: &[f32]) -> u32 {
    debug_assert!(
        !slice.iter().any(|v| v.is_nan()),
        "argmax_f32: input contains NaN"
    );
    let mut best_idx = 0_u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            // CAST: usize → u32, output_size fits in u32 (AlgZoo max = 10)
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            {
                best_idx = i as u32;
            }
        }
    }
    best_idx
}

/// Validate arguments shared by [`forward_fast`] and [`forward_fast_ablated`].
fn validate_fast_args(
    weights: &RnnWeights,
    inputs: &[f32],
    outputs: &[f32],
    n_inputs: usize,
    config: &StoicheiaConfig,
) -> Result<()> {
    let h = weights.hidden_size;
    let seq_len = config.seq_len;
    let out_size = weights.output_size;

    if h > MAX_H {
        return Err(MIError::Config(format!(
            "hidden_size {h} exceeds MAX_H {MAX_H}"
        )));
    }
    if inputs.len() != n_inputs * seq_len {
        return Err(MIError::Config(format!(
            "inputs length {} != n_inputs * seq_len {}",
            inputs.len(),
            n_inputs * seq_len
        )));
    }
    if outputs.len() != n_inputs * out_size {
        return Err(MIError::Config(format!(
            "outputs length {} != n_inputs * output_size {}",
            outputs.len(),
            n_inputs * out_size
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a tiny M₂,₂ model with known weights for testing.
    ///
    /// Weights chosen so the model is a simple comparator:
    /// - `W_ih = [1, -1]` — neuron 0 excited by input, neuron 1 inhibited
    /// - `W_hh = [[0, 0], [0, 0]]` — no recurrence (memoryless)
    /// - `W_oh = [[1, -1], [-1, 1]]` — output differentiates the two neurons
    fn test_weights_2_2() -> RnnWeights {
        RnnWeights::new(
            vec![1.0, -1.0],            // W_ih [2, 1]
            vec![0.0, 0.0, 0.0, 0.0],   // W_hh [2, 2]
            vec![1.0, -1.0, -1.0, 1.0], // W_oh [2, 2]
            2,
            2,
        )
    }

    fn test_config_2_2() -> StoicheiaConfig {
        StoicheiaConfig::from_task(crate::stoicheia::config::StoicheiaTask::SecondArgmax, 2, 2)
    }

    #[test]
    fn forward_fast_shape_and_output() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        // Single input: [0.5, -0.3]
        let inputs = vec![0.5_f32, -0.3];
        let mut outputs = vec![0.0_f32; 2];

        forward_fast(&weights, &inputs, &mut outputs, 1, &config).unwrap();

        // With memoryless RNN (W_hh = 0):
        // t=0: pre = [0.5, -0.5], h = [0.5, 0.0] (ReLU)
        // t=1: pre = [-0.3, 0.3], h = [0.0, 0.3] (ReLU)
        // output = W_oh @ [0, 0.3] = [-0.3, 0.3]
        assert!((outputs[0] - (-0.3)).abs() < 1e-6);
        assert!((outputs[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn forward_fast_batch() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let inputs = vec![0.5, -0.3, 1.0, 2.0];
        let mut outputs = vec![0.0_f32; 4];

        forward_fast(&weights, &inputs, &mut outputs, 2, &config).unwrap();

        // First input same as above
        assert!((outputs[0] - (-0.3)).abs() < 1e-6);
        assert!((outputs[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn forward_fast_traced_matches_forward() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let input = vec![0.5_f32, -0.3];
        let mut pre_acts = vec![0.0_f32; 4]; // [2, 2]
        let mut traced_out = vec![0.0_f32; 2];

        forward_fast_traced(&weights, &input, &mut pre_acts, &mut traced_out, &config).unwrap();

        let mut fast_out = vec![0.0_f32; 2];
        forward_fast(&weights, &input, &mut fast_out, 1, &config).unwrap();

        // Outputs must match
        for (a, b) in traced_out.iter().zip(&fast_out) {
            assert!((a - b).abs() < 1e-6, "traced={a}, fast={b}");
        }

        // Pre-activations at t=0: [0.5, -0.5]
        assert!((pre_acts[0] - 0.5).abs() < 1e-6);
        assert!((pre_acts[1] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn ablated_all_zeros_gives_zero_output() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let inputs = vec![0.5_f32, -0.3];
        let mut outputs = vec![0.0_f32; 2];
        let ablated = vec![true, true]; // ablate all neurons

        forward_fast_ablated(&weights, &inputs, &mut outputs, 1, &config, &ablated).unwrap();

        // All neurons zeroed → output = W_oh @ [0, 0] = [0, 0]
        assert!((outputs[0]).abs() < 1e-6);
        assert!((outputs[1]).abs() < 1e-6);
    }

    #[test]
    fn ablated_none_matches_normal() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let inputs = vec![0.5_f32, -0.3];
        let mut normal_out = vec![0.0_f32; 2];
        let mut ablated_out = vec![0.0_f32; 2];
        let ablated = vec![false, false]; // no ablation

        forward_fast(&weights, &inputs, &mut normal_out, 1, &config).unwrap();
        forward_fast_ablated(&weights, &inputs, &mut ablated_out, 1, &config, &ablated).unwrap();

        for (a, b) in normal_out.iter().zip(&ablated_out) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn accuracy_perfect() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        // Input where output[1] > output[0], so argmax = 1
        let inputs = vec![0.5_f32, -0.3];
        let targets = vec![1_u32]; // correct prediction

        let acc = accuracy(&weights, &inputs, &targets, 1, &config).unwrap();
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_wrong() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let inputs = vec![0.5_f32, -0.3];
        let targets = vec![0_u32]; // wrong prediction (model predicts 1)

        let acc = accuracy(&weights, &inputs, &targets, 1, &config).unwrap();
        assert!(acc.abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_wrong_input_length() {
        let weights = test_weights_2_2();
        let config = test_config_2_2();

        let inputs = vec![0.5_f32]; // too short
        let mut outputs = vec![0.0_f32; 2];

        let result = forward_fast(&weights, &inputs, &mut outputs, 1, &config);
        assert!(result.is_err());
    }
}
