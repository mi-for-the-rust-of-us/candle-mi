// SPDX-License-Identifier: MIT OR Apache-2.0

//! Masked-diffusion ancestral sampler (`SUBS` parameterization).
//!
//! A faithful port of the noflash `sample.py` (Sahoo et al., `NeurIPS` 2024):
//! absorbing/masked diffusion on a linear schedule `t: 1 → 0`.  Starting from a
//! fully-masked sequence (with an optional carried-over prompt prefix), each of
//! `num_steps` steps predicts the clean token at every position, forbids the
//! `[MASK]` token (`SUBS`), and reveals each currently-masked position to a
//! sampled token with probability `(t - s) / t`.  Once revealed, a position is
//! carried over (never re-masked); the final step reveals everything remaining.
//!
//! The sampler is **backend-agnostic** — it drives any
//! [`MIBackend`] whose forward returns `[batch, seq, vocab]`
//! logits — so it serves both the `MDLM` `DiT` and (later) decoder-style
//! diffusion models.  Determinism is by seed: the same `seed` reproduces the
//! same unmasking schedule and tokens.
//!
//! Note this deliberately keeps `rand`'s `StdRng`, rather than the frozen
//! `ChaCha8` generator that weight creation uses.  The frozen one exists so a
//! *model* stays reproducible across `rand` releases; sampling carries the same
//! latent hazard but a narrower promise, and switching it would change the
//! tokens every published sampling run produced.  Revisit deliberately, as its
//! own decision, not as a consistency sweep.

use candle_core::{DType, Device, IndexOp, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::backend::MIBackend;
use crate::error::Result;
use crate::hooks::HookSpec;

/// Configuration for masked-diffusion ancestral sampling.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DiffusionSamplingConfig {
    /// Total sequence length, including any prompt prefix.
    pub seq_len: usize,
    /// Number of denoising steps `K` (the time grid has `K + 1` knots).
    pub num_steps: usize,
    /// Softmax temperature (clamped to `>= 1e-6`).
    pub temperature: f32,
    /// Optional top-k truncation of the per-position distribution.
    pub top_k: Option<usize>,
    /// RNG seed (fixes the unmasking schedule and sampled tokens).
    pub seed: u64,
}

impl Default for DiffusionSamplingConfig {
    fn default() -> Self {
        Self {
            seq_len: 64,
            num_steps: 128,
            temperature: 1.0,
            top_k: None,
            seed: 0,
        }
    }
}

impl DiffusionSamplingConfig {
    /// Set the total sequence length, consuming `self`.
    ///
    /// These chainable setters exist because the struct is
    /// `#[non_exhaustive]`: outside this crate a struct expression is rejected,
    /// **including** the `..Default::default()` form, so
    /// [`default`](Default::default) plus setters is the construction path.
    /// Assigning to the public fields of an existing value also works.
    ///
    /// ```
    /// use candle_mi::DiffusionSamplingConfig;
    /// let cfg = DiffusionSamplingConfig::default()
    ///     .with_seq_len(16)
    ///     .with_num_steps(16)
    ///     .with_top_k(Some(50));
    /// assert_eq!(cfg.seq_len, 16);
    /// ```
    #[must_use]
    pub const fn with_seq_len(mut self, seq_len: usize) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Set the number of denoising steps, consuming `self`.
    #[must_use]
    pub const fn with_num_steps(mut self, num_steps: usize) -> Self {
        self.num_steps = num_steps;
        self
    }

    /// Set the softmax temperature, consuming `self`.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the optional top-k truncation, consuming `self`.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: Option<usize>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the RNG seed, consuming `self`.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Generate one sequence by masked-diffusion ancestral sampling.
///
/// Returns the final token ids (length `config.seq_len`, no `[MASK]` left).
/// `prompt_ids` is clamped to the prefix and carried over unchanged.
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on a forward-pass or tensor
/// failure.
// TRAIT_OBJECT: the sampler is backend-agnostic (any diffusion `MIBackend`).
pub fn generate(
    model: &dyn MIBackend,
    device: &Device,
    mask_token_id: u32,
    prompt_ids: &[u32],
    config: &DiffusionSamplingConfig,
) -> Result<Vec<u32>> {
    let mut trajectory = generate_trajectory(model, device, mask_token_id, prompt_ids, config)?;
    Ok(trajectory.pop().unwrap_or_default())
}

/// Run ancestral sampling and return the token state after **every** step.
///
/// The returned vector has length `config.num_steps + 1`: index `0` is the
/// initial (prompt + `[MASK]`) state, index `k` is the state after denoising
/// step `k`, and the last entry is fully revealed.  This is the artifact the
/// diffusion-time MI examples consume: re-running the model's forward on each
/// state (with hooks) yields the per-`(k, layer, position)` activations.
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on a forward-pass or tensor
/// failure.
// TRAIT_OBJECT: the sampler is backend-agnostic (any diffusion `MIBackend`).
pub fn generate_trajectory(
    model: &dyn MIBackend,
    device: &Device,
    mask_token_id: u32,
    prompt_ids: &[u32],
    config: &DiffusionSamplingConfig,
) -> Result<Vec<Vec<u32>>> {
    let seq_len = config.seq_len;
    let mut rng = StdRng::seed_from_u64(config.seed);

    // Initial state: fully masked, with the prompt carried into the prefix.
    let mut x = vec![mask_token_id; seq_len];
    for (i, &token) in prompt_ids.iter().take(seq_len).enumerate() {
        if let Some(slot) = x.get_mut(i) {
            *slot = token;
        }
    }

    let mut trajectory = Vec::with_capacity(config.num_steps + 1);
    trajectory.push(x.clone());

    let hooks = HookSpec::new();
    for step in 0..config.num_steps {
        // Linear time grid 1 -> 0: t = times[step], s = times[step + 1].
        // CAST: usize -> f64, step counts fit in f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let (t, s) = {
            let n = config.num_steps as f64;
            (1.0 - step as f64 / n, 1.0 - (step + 1) as f64 / n)
        };
        let unmask_prob = if t > 0.0 { (t - s) / t } else { 1.0 };

        let input = Tensor::new(x.as_slice(), device)?.unsqueeze(0)?; // [1, seq]
        let cache = model.forward(&input, &hooks)?;
        // Raw logits at every position: [seq, vocab] on the host in F32.
        let logits = cache
            .output()
            .i(0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        for (pos, row) in logits.iter().enumerate() {
            // Carry-over: only currently-masked positions can be revealed.
            if x.get(pos).copied() != Some(mask_token_id) {
                continue;
            }
            if rng.r#gen::<f64>() < unmask_prob {
                let token = sample_token_from_logits(
                    row,
                    mask_token_id,
                    config.temperature,
                    config.top_k,
                    &mut rng,
                );
                if let Some(slot) = x.get_mut(pos) {
                    *slot = token;
                }
            }
        }
        trajectory.push(x.clone());
    }

    Ok(trajectory)
}

/// Sample a token from one position's logits: forbid `[MASK]` (`SUBS`), apply
/// temperature, optional top-k truncation, then multinomial sampling.
fn sample_token_from_logits(
    row: &[f32],
    mask_token_id: u32,
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut StdRng,
) -> u32 {
    // CAST: u32 -> usize, vocab index used to forbid the [MASK] logit
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    let mask_idx = mask_token_id as usize;
    let inv_temp = 1.0 / temperature.max(1e-6);

    let mut logits: Vec<f32> = row
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            if i == mask_idx {
                f32::NEG_INFINITY
            } else {
                l * inv_temp
            }
        })
        .collect();

    if let Some(k) = top_k.filter(|&k| k >= 1 && k < logits.len()) {
        let mut ranked = logits.clone();
        // Partition so the (k-1)-th element is the k-th largest logit.
        ranked.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = ranked.get(k - 1).copied().unwrap_or(f32::NEG_INFINITY);
        for l in &mut logits {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // Numerically stable softmax.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();

    // Multinomial draw over the (unnormalized) weights.
    let target = rng.r#gen::<f32>() * sum;
    let mut cumulative = 0.0_f32;
    for (i, &e) in exps.iter().enumerate() {
        cumulative += e;
        if target < cumulative {
            // CAST: usize -> u32, vocab index fits in u32
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            return i as u32;
        }
    }
    // Floating-point edge case: fall back to the last index.
    // CAST: usize -> u32, vocab index fits in u32
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    {
        exps.len().saturating_sub(1) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SUBS` must forbid the `[MASK]` token even when it has the largest logit;
    /// sampling then concentrates on the dominant *non-mask* token.
    #[test]
    fn subs_never_samples_mask() {
        let mask = 7u32;
        let mut row = vec![0.0f32; 8];
        row[2] = 50.0; // dominant non-mask token
        row[7] = 100.0; // [MASK] — higher, but must be forbidden
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..32 {
            assert_eq!(sample_token_from_logits(&row, mask, 1.0, None, &mut rng), 2);
        }
    }

    /// `top_k = 1` collapses the support to the single largest *non-mask* logit.
    #[test]
    fn top_k_one_is_greedy_over_non_mask() {
        let mask = 7u32;
        let mut row = vec![0.0f32; 8];
        row[5] = 10.0; // argmax among non-mask
        row[3] = 9.0;
        row[7] = 20.0; // [MASK] — highest overall, forbidden
        let mut rng = StdRng::seed_from_u64(123);
        for _ in 0..16 {
            assert_eq!(
                sample_token_from_logits(&row, mask, 1.0, Some(1), &mut rng),
                5
            );
        }
    }

    /// Same seed ⇒ same draw (reproducible schedules).
    #[test]
    fn deterministic_given_seed() {
        let row = vec![0.1f32, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6];
        let mask = 7u32;
        let mut r1 = StdRng::seed_from_u64(7);
        let mut r2 = StdRng::seed_from_u64(7);
        assert_eq!(
            sample_token_from_logits(&row, mask, 1.0, None, &mut r1),
            sample_token_from_logits(&row, mask, 1.0, None, &mut r2)
        );
    }
}
