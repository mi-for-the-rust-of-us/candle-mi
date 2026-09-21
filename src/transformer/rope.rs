// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rotary position embeddings (`RoPE`).
//!
//! Pre-computes `cos` and `sin` tensors at model load time and applies
//! them to query and key tensors during the forward pass.
//!
//! Uses `candle_nn::rotary_emb::rope()` for the actual rotation, matching
//! the reference implementation in plip-rs (frozen predecessor project, v1.4.0).

use candle_core::{DType, Device, Tensor};

use crate::config::RopeScaling;
use crate::error::Result;

/// Apply Llama 3 frequency-band rescaling to inverse frequencies in place.
///
/// Mirrors `HuggingFace`'s `_compute_llama3_parameters`: inverse frequencies
/// whose wavelength exceeds `low_freq_wavelen` are divided by `factor`
/// (low-frequency band), those below `high_freq_wavelen` are left intact
/// (high-frequency band), and the band between is smoothly interpolated.
///
/// Operates on `f64` for parity with the `PyTorch` reference (which computes
/// the rescaled frequencies in double precision before casting).
fn apply_llama3_scaling(
    inv_freq: &mut [f64],
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: usize,
) {
    use std::f64::consts::PI;

    // CAST: usize → f64, context length fits in f64 mantissa (<= 2^52).
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let orig_max = original_max_position_embeddings as f64;
    let low_freq_wavelen = orig_max / low_freq_factor;
    let high_freq_wavelen = orig_max / high_freq_factor;

    for freq in inv_freq.iter_mut() {
        let wavelen = 2.0 * PI / *freq;
        if wavelen > low_freq_wavelen {
            // Low-frequency band: divide the inverse frequency by `factor`.
            *freq /= factor;
        } else if wavelen >= high_freq_wavelen {
            // Medium band (`high_freq_wavelen <= wavelen <= low_freq_wavelen`):
            // smooth interpolation between scaled and unscaled.  Kept as two
            // separate products plus an add (NOT a fused `mul_add`): PyTorch
            // computes this non-fused, so fusing would diverge — and splitting
            // also sidesteps `clippy::suboptimal_flops`, which fires on the
            // MSRV (1.91) toolchain but not on current stable.
            let smooth =
                (orig_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            let scaled = (1.0 - smooth) * *freq / factor;
            let unscaled = smooth * *freq;
            *freq = scaled + unscaled;
        }
        // else high-frequency band: inverse frequency unchanged.
    }
}

/// Divide base inverse frequencies element-wise by a per-dimension factor
/// array (`longrope` `short_factor`/`long_factor`).
///
/// # Errors
///
/// Returns [`MIError::Config`] if the factor length does not match `head_dim/2`.
fn divide_per_dim(base_inv_freq: &[f64], factor: &[f64]) -> Result<Vec<f64>> {
    if base_inv_freq.len() != factor.len() {
        return Err(crate::error::MIError::Config(format!(
            "longrope factor length {} does not match head_dim/2 = {}",
            factor.len(),
            base_inv_freq.len()
        )));
    }
    Ok(base_inv_freq
        .iter()
        .zip(factor)
        .map(|(b, f)| b / f)
        .collect())
}

/// Build `(cos, sin)` tensors of shape `[max_position, inv_freq.len()]` from
/// inverse frequencies and integer positions, optionally scaled by `mscale`
/// (the `longrope` attention factor; `1.0` is a no-op and skips the op).
fn build_cos_sin(
    inv_freq: &[f64],
    max_position: usize,
    mscale: f64,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = inv_freq.len();
    // CAST: f64 → f32, precision loss acceptable for RoPE frequencies
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|&f| f as f32).collect();
    let inv_freq_tensor = Tensor::from_vec(inv_freq_f32, (1, half_dim), device)?.to_dtype(dtype)?;

    let positions: Vec<f32> = (0..max_position)
        .map(|p| {
            // CAST: usize → f32; max_position <= ~128K is exact in f32.
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let pos = p as f32;
            pos
        })
        .collect();
    let pos_tensor = Tensor::from_vec(positions, (max_position, 1), device)?.to_dtype(dtype)?;

    // Outer product: [max_position, half_dim]
    let freqs = pos_tensor.matmul(&inv_freq_tensor)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Attention scaling (mscale). Non-longrope schemes pass 1.0; skip the op so
    // their cos/sin stay bit-identical to before this refactor.
    if (mscale - 1.0).abs() > f64::EPSILON {
        Ok((cos.affine(mscale, 0.0)?, sin.affine(mscale, 0.0)?))
    } else {
        Ok((cos, sin))
    }
}

// ---------------------------------------------------------------------------
// RoPE cache — pre-computed cos/sin
// ---------------------------------------------------------------------------

/// `longrope` long-regime cache: used when a forward's sequence length exceeds
/// the pretraining context.
struct LongRegime {
    /// Cosine values (`long_factor`): `[max_position, head_dim / 2]`.
    cos: Tensor,
    /// Sine values (`long_factor`): `[max_position, head_dim / 2]`.
    sin: Tensor,
    /// Sequence-length boundary: sequences longer than this use this cache.
    original_max_position: usize,
}

/// Pre-computed cosine and sine tensors for rotary position embeddings.
pub struct RopeCache {
    /// Cosine values (default / `longrope` short regime): `[rows, head_dim / 2]`.
    cos: Tensor,
    /// Sine values (default / `longrope` short regime): `[rows, head_dim / 2]`.
    sin: Tensor,
    /// `longrope` only: the long-regime cache + boundary.  `None` otherwise.
    long: Option<LongRegime>,
}

impl RopeCache {
    /// Pre-compute the `RoPE` cache.
    ///
    /// `scaling` applies a `rope_scaling` interpolation scheme (see
    /// [`RopeScaling`]):
    /// - [`RopeScaling::Linear`] divides every position index by `factor`
    ///   (uniform across frequencies).
    /// - [`RopeScaling::Llama3`] rescales the inverse frequencies by
    ///   frequency band (position-independent).
    /// - [`RopeScaling::Longrope`] divides the inverse frequencies by a
    ///   per-dimension factor array (`short_factor` for sequences
    ///   `<= original_max_position_embeddings`, `long_factor` beyond) and
    ///   scales `cos`/`sin` by `attention_factor`.  Two caches are built; the
    ///   short one is capped at `original_max_position_embeddings`.
    /// - `None` is standard, unscaled `RoPE`.
    ///
    /// # Shapes
    /// - `cos`: `[rows, head_dim / 2]` (`rows <= max_position`)
    /// - `sin`: `[rows, head_dim / 2]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures, or
    /// [`MIError::Config`] if a `longrope` factor array is mis-sized.
    // EXPLICIT: takes `scaling` by value for call-site ergonomics (the caller
    // owns it); only borrowed internally, hence the allow.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        head_dim: usize,
        max_position: usize,
        theta: f64,
        scaling: Option<RopeScaling>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Base inverse frequencies in f64 (theta^(-2i/d)); rescaled below.
        // f64 throughout matches the PyTorch reference.
        let base_inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| {
                // CAST: usize → f64, loop index and head_dim fit in f64 mantissa
                #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();

        // --- longrope: two caches (short capped at original_max, long full) ---
        if let Some(RopeScaling::Longrope {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            attention_factor,
        }) = &scaling
        {
            let short_inv = divide_per_dim(&base_inv_freq, short_factor)?;
            let short_max = max_position.min(*original_max_position_embeddings);
            let (cos, sin) =
                build_cos_sin(&short_inv, short_max, *attention_factor, device, dtype)?;

            let long_inv = divide_per_dim(&base_inv_freq, long_factor)?;
            let (long_cos, long_sin) =
                build_cos_sin(&long_inv, max_position, *attention_factor, device, dtype)?;

            return Ok(Self {
                cos,
                sin,
                long: Some(LongRegime {
                    cos: long_cos,
                    sin: long_sin,
                    original_max_position: *original_max_position_embeddings,
                }),
            });
        }

        // --- non-longrope: llama3 acts on frequencies, linear on positions ---
        let mut inv_freq = base_inv_freq;
        if let Some(RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) = &scaling
        {
            apply_llama3_scaling(
                &mut inv_freq,
                *factor,
                *low_freq_factor,
                *high_freq_factor,
                *original_max_position_embeddings,
            );
        }

        // CAST: f64 → f32, precision loss acceptable for RoPE frequencies
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let inv_freq_f32: Vec<f32> = inv_freq.iter().map(|&f| f as f32).collect();
        let inv_freq_tensor =
            Tensor::from_vec(inv_freq_f32, (1, half_dim), device)?.to_dtype(dtype)?;

        // Linear scaling divides each position by `factor` (fractional); every
        // other scheme uses raw integer positions (`longrope` is handled above).
        let positions: Vec<f32> = if let Some(RopeScaling::Linear { factor }) = &scaling {
            (0..max_position)
                .map(|p| {
                    // CAST: usize → f64 then f32; max_position <= ~128K is exact in f64.
                    #[allow(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        clippy::as_conversions
                    )]
                    let scaled = (p as f64 / factor) as f32;
                    scaled
                })
                .collect()
        } else {
            (0..max_position)
                .map(|p| {
                    // CAST: usize → f32; max_position <= ~128K is exact in f32.
                    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
                    let pos = p as f32;
                    pos
                })
                .collect()
        };
        let pos_tensor = Tensor::from_vec(positions, (max_position, 1), device)?.to_dtype(dtype)?;

        // Outer product: [max_position, half_dim]
        let freqs = pos_tensor.matmul(&inv_freq_tensor)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self {
            cos,
            sin,
            long: None,
        })
    }

    /// Apply rotary embeddings to a query or key tensor.
    ///
    /// Uses `candle_nn::rotary_emb::rope()` for the rotation.  For `longrope`,
    /// a sequence longer than `original_max_position_embeddings` selects the
    /// long-factor cache (matching `HuggingFace`'s seq-length branch).
    ///
    /// # Shapes
    /// - `x`: `[batch, n_heads, seq_len, head_dim]`
    /// - returns: `[batch, n_heads, seq_len, head_dim]`
    ///
    /// The `start_pos` parameter offsets positions for incremental generation
    /// (KV-cache) so cached keys keep their original positional encoding. The
    /// transformer backend currently recomputes the full sequence each step and
    /// always passes `start_pos = 0`; the offset is reserved for cached generation.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation or shape errors.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;

        // longrope: pick the long-factor cache once the (total) sequence length
        // exceeds the pretraining context; otherwise the default/short cache.
        let (cos_full, sin_full) = self.long.as_ref().map_or((&self.cos, &self.sin), |lr| {
            if start_pos + seq_len > lr.original_max_position {
                (&lr.cos, &lr.sin)
            } else {
                (&self.cos, &self.sin)
            }
        });

        // Slice cos/sin for the relevant positions: [seq_len, half_dim]
        let cos = cos_full.narrow(0, start_pos, seq_len)?;
        let sin = sin_full.narrow(0, start_pos, seq_len)?;

        // CONTIGUOUS: the rope kernel requires a contiguous input
        crate::nn_ops::rope(&x.contiguous()?, &cos, &sin)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn longrope_short_inv_freq_matches_phi35_ground_truth() {
        // Phi-3.5-mini: head_dim 96, theta 10000. The first two short_factor
        // entries are 1.0 and 1.02; the loaded model reports inv_freq[0]=1.0,
        // inv_freq[1]=0.80921978 — reproduce that exactly from the formula
        // inv_freq[i] = 1 / (short_factor[i] * theta^(2i/head_dim)).
        let head_dim = 96usize;
        let theta = 10_000.0f64;
        let half = head_dim / 2;
        // CAST: usize → f64, i < 48 and head_dim = 96 fit exactly in the f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let base: Vec<f64> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let short_factor = [1.0_f64, 1.02];
        // inv_freq[0] = 1/(1.0 * 1) = 1.0
        assert!((base[0] / short_factor[0] - 1.0).abs() < 1e-12);
        // inv_freq[1] = (1/theta^(2/96)) / 1.02
        let inv1 = base[1] / short_factor[1];
        assert!(
            (inv1 - 0.809_219_78).abs() < 1e-6,
            "inv_freq[1] = {inv1}, expected ~0.80921978"
        );
    }

    #[test]
    fn divide_per_dim_rejects_length_mismatch() {
        let base = vec![1.0, 0.5, 0.25];
        assert!(divide_per_dim(&base, &[2.0, 2.0]).is_err());
        assert_eq!(
            divide_per_dim(&base, &[2.0, 1.0, 0.5]).unwrap(),
            vec![0.5, 0.5, 0.5]
        );
    }
}
