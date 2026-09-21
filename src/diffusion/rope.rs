// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotary position embeddings for the `MDLM` backend.
//!
//! A minimal, self-contained `RoPE` cache (no `rope_scaling` — `MDLM` uses
//! plain rotary at base `10000`).  It pre-computes `cos`/`sin` of shape
//! `[max_position, head_dim / 2]` and applies them with
//! `candle_nn::rotary_emb::rope`, the non-interleaved (`rotate_half`)
//! convention that matches the upstream `apply_rotary`.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

/// Pre-computed cosine and sine tensors for `MDLM` rotary embeddings.
pub struct MdlmRope {
    /// Cosine values: `[max_position, head_dim / 2]`.
    cos: Tensor,
    /// Sine values: `[max_position, head_dim / 2]`.
    sin: Tensor,
}

impl MdlmRope {
    /// Pre-compute the rotary cache.
    ///
    /// Inverse frequencies are `theta^(-2i/head_dim)` for `i` in
    /// `0..head_dim/2`, computed in `f64` for parity with the `PyTorch`
    /// reference, then cast to `dtype`.
    ///
    /// # Shapes
    /// - `cos`: `[max_position, head_dim / 2]`
    /// - `sin`: `[max_position, head_dim / 2]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation
    /// failures.
    pub fn new(
        head_dim: usize,
        max_position: usize,
        theta: f64,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| {
                // CAST: usize → f64, loop index and head_dim fit in f64 mantissa
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let exponent = 2.0 * i as f64 / head_dim as f64;
                // CAST: f64 → f32, precision loss acceptable for RoPE frequencies
                #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
                let freq = (1.0 / theta.powf(exponent)) as f32;
                freq
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?.to_dtype(dtype)?;

        let positions: Vec<f32> = (0..max_position)
            .map(|p| {
                // CAST: usize → f32, max_position (<= context length) is exact in f32
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let pos = p as f32;
                pos
            })
            .collect();
        let positions = Tensor::from_vec(positions, (max_position, 1), device)?.to_dtype(dtype)?;

        // Outer product: [max_position, half_dim]
        let freqs = positions.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// Apply rotary embeddings to a query or key tensor.
    ///
    /// Uses `candle_nn::rotary_emb::rope` (non-interleaved `rotate_half`).
    ///
    /// # Shapes
    /// - `x`: `[batch, n_heads, seq_len, head_dim]`
    /// - returns: `[batch, n_heads, seq_len, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on tensor operation or
    /// shape errors (e.g. `seq_len` exceeds `max_position`).
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        // CONTIGUOUS: the rope kernel requires a contiguous input
        crate::nn_ops::rope(&x.contiguous()?, &cos, &sin)
    }
}
