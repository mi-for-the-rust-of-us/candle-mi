// SPDX-License-Identifier: MIT OR Apache-2.0

//! Maar et al. (2026) "What's the plan?" contrastive activation steering.
//!
//! Construct a residual-stream direction vector by averaging captured
//! activations from a set of *positive* prompts (those that should elicit a
//! target behaviour, e.g. a rhyme family) and subtracting the average from a
//! set of *negative* prompts (those that should not).  The resulting vector,
//! scaled by a signed strength and added to the residual stream during a
//! forward pass, steers the model toward the positive-set behaviour.
//!
//! ## Formula
//!
//! For positive prompts `P` and negative prompts `N`, layer `l`, and
//! position selector `pos`:
//!
//! ```text
//! direction = mean(x_l(p)[pos]  for p in P)
//!           − mean(x_l(n)[pos]  for n in N)
//! ```
//!
//! Where `x_l(prompt)[pos]` is the residual stream of `prompt` at layer `l`
//! and token position `pos`.  If `normalise = true` (the default in
//! [`build_contrastive_direction`]'s callers), the direction is
//! L2-normalised so it is a unit vector, matching Maar et al.'s reported
//! `m = 1.5` magnitude convention (the `m` only makes physical sense as a
//! magnitude when the direction is unit).
//!
//! ## Intervention
//!
//! The direction is applied as an [`Intervention::Add`] at
//! [`HookPoint::ResidPost`] of the same layer.  By default the addition
//! broadcasts over every sequence position (because the `[hidden]` direction
//! broadcasts against `[batch, seq, hidden]`); use
//! [`position_delta`] to mask the injection to a single position when
//! Maar's "first newline" or our "last token" protocol calls for it.
//!
//! ## Reference
//!
//! Maar, Paperno, `McDougall`, Nanda. *What's the plan? Metrics for implicit
//! planning in LLMs and their application to rhyme generation and question
//! answering*.  ICLR 2026 (poster).  arXiv 2601.20164.  `OpenReview`
//! `Z10pxu0Q7X`.

use candle_core::{DType, Device, Tensor};

use crate::backend::MIModel;
use crate::error::{MIError, Result};
use crate::hooks::{HookPoint, HookSpec, Intervention};
use crate::tokenizer::MITokenizer;

// ---------------------------------------------------------------------------
// PositionStrategy
// ---------------------------------------------------------------------------

/// Strategy for selecting which token position contributes to the contrastive
/// mean from each prompt's residual stream.
///
/// Maar et al.'s paper text documents two strategies explicitly: "last word
/// of first line" (most consistent across the 23 models in their wide sweep)
/// and "newline token" (effective for the `Gemma 2 9B` documented case).
/// We expose three so the example can dispatch by `--position-strategy`:
///
/// - [`Self::Last`]: the last token of the encoded prompt.  When the prompt
///   is `"A rhyming couplet:\n{line}"` (no trailing newline), the last
///   token IS the last word of the first line — equivalent to Maar's
///   "first line last word" position.  This is also the `plip-rs` / COLM
///   2026 paper convention for the planning-site spike.
/// - [`Self::FirstNewline`]: the index of the first `\n` token in the
///   encoded prompt.  Matches Maar's documented `Gemma 2 9B` choice when the
///   prompt has a trailing newline after the first line.
/// - [`Self::Explicit`]: an absolute position index.  Used when the Maar
///   supplementary code specifies a per-model position that doesn't match
///   either of the above.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionStrategy {
    /// Last token of the encoded prompt.
    Last,
    /// First `\n` token in the encoded prompt.
    FirstNewline,
    /// Absolute position index; must be `< seq_len` for every prompt.
    Explicit(usize),
}

impl PositionStrategy {
    /// Resolve the strategy to an absolute position index for a specific
    /// tokenised prompt.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the strategy cannot be resolved:
    /// - [`Self::FirstNewline`] but no `\n` token is present in `tokens`.
    /// - [`Self::Explicit`] with an index `>= tokens.len()`.
    /// - `tokens` is empty (no position to select).
    pub fn resolve(self, tokens: &[u32], newline_token_id: u32) -> Result<usize> {
        if tokens.is_empty() {
            return Err(MIError::Config(
                "PositionStrategy::resolve called on empty token sequence".into(),
            ));
        }
        match self {
            Self::Last => Ok(tokens.len() - 1),
            Self::FirstNewline => tokens
                .iter()
                .position(|&id| id == newline_token_id)
                .ok_or_else(|| {
                    MIError::Config(format!(
                        "PositionStrategy::FirstNewline: no newline token (id={newline_token_id}) \
                         found in {len}-token prompt",
                        len = tokens.len()
                    ))
                }),
            Self::Explicit(pos) => {
                if pos < tokens.len() {
                    Ok(pos)
                } else {
                    Err(MIError::Config(format!(
                        "PositionStrategy::Explicit({pos}) out of range for {len}-token prompt",
                        len = tokens.len()
                    )))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ContrastiveDirection
// ---------------------------------------------------------------------------

/// A residual-stream contrastive direction together with provenance metadata.
///
/// Returned by [`build_contrastive_direction`].  Pass to
/// [`contrastive_intervention`] (with a signed strength) to obtain an
/// [`Intervention`] ready to register on a [`HookSpec`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ContrastiveDirection {
    /// Residual-stream layer at which the direction was computed.  The
    /// direction is meaningful only when applied at the same layer.
    pub layer: usize,
    /// The direction vector.  Shape `[hidden_size]`, dtype `F32`.
    pub vector: Tensor,
    /// `true` when `vector` has been L2-normalised to a unit vector.
    pub is_normalised: bool,
    /// Number of positive prompts averaged.
    pub n_positive: usize,
    /// Number of negative prompts averaged.
    pub n_negative: usize,
    /// Position-selection strategy used to extract per-prompt residuals.
    pub position_strategy: PositionStrategy,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a contrastive activation-steering direction from positive and
/// negative prompt sets at a single residual-stream layer.
///
/// Runs one forward pass per prompt with
/// [`HookPoint::ResidPost(layer)`](HookPoint::ResidPost) captured, slices the
/// per-prompt residual at the position chosen by `strategy`, averages each
/// set, takes the difference, and optionally L2-normalises.
///
/// ## Shapes
///
/// - per-prompt captured residual: `[1, seq, hidden]`, sliced at `pos` →
///   `[hidden]`.
/// - returned [`ContrastiveDirection::vector`]: `[hidden]`, `F32`.
///
/// ## Memory
///
/// Holds `n_positive + n_negative` `[hidden]` accumulator tensors on the
/// model's device at the same time (e.g. 170 prompts × 2304 dims × 4 B ≈
/// 1.5 MB for `Gemma 2 2B`).  Per-prompt captured residuals are dropped
/// before the next forward pass.
///
/// # Errors
///
/// - [`MIError::Config`] if `positive` or `negative` is empty.
/// - [`MIError::Config`] if `layer >= model.num_layers()`.
/// - [`MIError::Tokenizer`] if any prompt fails to encode.
/// - [`MIError::Config`] if `strategy.resolve()` fails for any prompt
///   (e.g. [`PositionStrategy::FirstNewline`] with no newline in the prompt).
/// - [`MIError::Model`] on forward-pass or tensor operation failure.
pub fn build_contrastive_direction(
    model: &MIModel,
    tokenizer: &MITokenizer,
    positive: &[&str],
    negative: &[&str],
    layer: usize,
    strategy: PositionStrategy,
    normalise: bool,
) -> Result<ContrastiveDirection> {
    if positive.is_empty() {
        return Err(MIError::Config(
            "build_contrastive_direction: positive prompt set is empty".into(),
        ));
    }
    if negative.is_empty() {
        return Err(MIError::Config(
            "build_contrastive_direction: negative prompt set is empty".into(),
        ));
    }
    let n_layers = model.num_layers();
    if layer >= n_layers {
        return Err(MIError::Config(format!(
            "build_contrastive_direction: layer {layer} >= num_layers {n_layers}"
        )));
    }

    // Resolve the newline token id once (only used for FirstNewline; cheap
    // even when not needed because encode_raw is fast).
    let newline_token_id = resolve_newline_token_id(tokenizer)?;

    let hook = HookPoint::ResidPost(layer);
    let pos_residuals = capture_per_prompt(
        model,
        tokenizer,
        positive,
        &hook,
        strategy,
        newline_token_id,
    )?;
    let neg_residuals = capture_per_prompt(
        model,
        tokenizer,
        negative,
        &hook,
        strategy,
        newline_token_id,
    )?;

    let vector = compute_direction(&pos_residuals, &neg_residuals, normalise, model.device())?;

    Ok(ContrastiveDirection {
        layer,
        vector,
        is_normalised: normalise,
        n_positive: positive.len(),
        n_negative: negative.len(),
        position_strategy: strategy,
    })
}

/// Build a [`HookSpec`]-ready [`Intervention`] from a
/// [`ContrastiveDirection`] and a signed strength scalar.
///
/// The returned [`Intervention::Add`] carries a `[hidden]` payload that
/// `broadcast_add`s against the residual-stream tensor of shape
/// `[batch, seq, hidden]` — so by default it adds the same vector at *every*
/// sequence position.  To inject at a single position only, build the
/// `[1, seq_len, hidden]` payload with [`position_delta`] and pass it
/// directly to [`Intervention::Add`].
///
/// ## Shapes
///
/// - `direction.vector`: `[hidden]`
/// - returned intervention's payload: `[hidden]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor scaling failure.
#[must_use = "the intervention must be registered on a HookSpec to take effect"]
pub fn contrastive_intervention(
    direction: &ContrastiveDirection,
    strength: f32,
) -> Result<Intervention> {
    // PROMOTE: cast strength to f64 for Tensor::affine / scalar mul
    //          (candle's scalar-mul takes f64 internally).
    // CAST: f32 → f64, lossless widening cast for the scalar multiplier.
    let scaled = (&direction.vector * f64::from(strength))?;
    Ok(Intervention::Add(scaled))
}

/// Build a `[1, seq_len, hidden]` injection payload that carries `direction`
/// at sequence position `position` and zeros elsewhere.
///
/// Use this to construct an [`Intervention::Add`] that only fires at one
/// position (instead of the default broadcast over every position).
///
/// To *overwrite* a position rather than add to it, no payload builder is
/// needed: `Intervention::PatchAt` takes the position directly. This function
/// exists because `Add` has no position field of its own.
///
/// ## Shapes
///
/// - `direction`: `[hidden]`
/// - returns: `[1, seq_len, hidden]`, same dtype as `direction`
///
/// # Errors
///
/// - [`MIError::Config`] if `position >= seq_len`.
/// - [`MIError::Config`] if `direction` is not 1-D `[hidden]`.
/// - [`MIError::Model`] on tensor construction failure.
pub fn position_delta(direction: &Tensor, position: usize, seq_len: usize) -> Result<Tensor> {
    if position >= seq_len {
        return Err(MIError::Config(format!(
            "position_delta: position {position} >= seq_len {seq_len}"
        )));
    }
    let dims = direction.dims();
    if dims.len() != 1 {
        return Err(MIError::Config(format!(
            "position_delta: direction must be 1-D [hidden]; got shape {dims:?}"
        )));
    }

    // INDEX: dims has length 1, just confirmed above.
    let hidden = dims.first().copied().unwrap_or(0);

    // Build a [seq_len, hidden] tensor by stacking per-position rows: the
    // chosen position holds `direction`, all others hold zeros_like(direction).
    let zero_row = direction.zeros_like()?;
    // BORROW: rows is a Vec<&Tensor> for Tensor::stack; entries borrow either
    // `direction` (for the chosen position) or `zero_row` (for all others).
    let rows: Vec<&Tensor> = (0..seq_len)
        .map(|i| if i == position { direction } else { &zero_row })
        .collect();
    let stacked = Tensor::stack(&rows, 0)?;
    // Add batch dim -> [1, seq_len, hidden].
    let with_batch = stacked.unsqueeze(0)?;
    // EXPLICIT: discard `hidden` after the shape check; tensors carry their
    // own shape, the local binding existed only to validate dims.
    let _ = hidden;
    Ok(with_batch)
}

// ---------------------------------------------------------------------------
// Internal helpers (private)
// ---------------------------------------------------------------------------

/// Resolve the newline-character token id by encoding `"\n"` alone and
/// checking that it produces a single token.
///
/// Most BPE tokenisers (Llama, Gemma, Qwen) tokenise `"\n"` as a single
/// dedicated token.  When that's not the case we surface an error rather
/// than silently using a wrong id (mirroring the `find_token_id` v0.1.11
/// fix for the same class of bug).
fn resolve_newline_token_id(tokenizer: &MITokenizer) -> Result<u32> {
    let ids = tokenizer.encode_raw("\n")?;
    match ids.len() {
        1 => {
            // INDEX: len just confirmed == 1; .first() cannot fail.
            ids.first().copied().ok_or_else(|| {
                MIError::Tokenizer("resolve_newline_token_id: unexpected empty Vec".into())
            })
        }
        n => Err(MIError::Tokenizer(format!(
            "resolve_newline_token_id: '\\n' encodes to {n} tokens, not 1; \
             PositionStrategy::FirstNewline is not usable with this tokenizer"
        ))),
    }
}

/// Run one forward pass per prompt, capture the residual at `hook`, slice
/// at the position chosen by `strategy`, and return the per-prompt `[hidden]`
/// tensors.
fn capture_per_prompt(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prompts: &[&str],
    hook: &HookPoint,
    strategy: PositionStrategy,
    newline_token_id: u32,
) -> Result<Vec<Tensor>> {
    let mut residuals: Vec<Tensor> = Vec::with_capacity(prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        let tokens = tokenizer.encode(prompt).map_err(|e| {
            MIError::Tokenizer(format!(
                "capture_per_prompt: prompt #{i} encode failed: {e}"
            ))
        })?;
        if tokens.is_empty() {
            return Err(MIError::Config(format!(
                "capture_per_prompt: prompt #{i} encoded to zero tokens"
            )));
        }
        let position = strategy
            .resolve(&tokens, newline_token_id)
            .map_err(|e| MIError::Config(format!("capture_per_prompt: prompt #{i}: {e}")))?;

        let input = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
        let mut hooks = HookSpec::new();
        // BORROW: `hook` is &HookPoint (Clone, not Copy due to Custom(String)).
        // Capture takes by value; we clone once per prompt and reuse `hook`
        // for the subsequent `require` call.  Single clone per prompt;
        // negligible alongside the forward pass cost.
        hooks.capture(hook.clone());
        let cache = model.forward(&input, &hooks)?;

        // Shape: [1, seq, hidden]; squeeze batch -> [seq, hidden]; index
        // position -> [hidden].  PROMOTE to F32 for stable accumulation
        // across prompts (residuals may be BF16/F16 on the model's native
        // dtype path).
        // INDEX (via Tensor::get): position bounded by strategy.resolve.
        let residual_3d = cache.require(hook)?;
        let residual_2d = residual_3d.squeeze(0)?;
        let residual_1d = residual_2d.get(position)?;
        // PROMOTE: residuals may be BF16/F16; accumulate in F32 for
        // numerical stability across 85+ prompts.
        let residual_f32 = residual_1d.to_dtype(DType::F32)?;
        residuals.push(residual_f32);
    }
    Ok(residuals)
}

/// Pure-tensor-math helper: compute `mean(pos) − mean(neg)`, optionally
/// L2-normalise, on `device`.  Exposed only to crate-internal tests.
fn compute_direction(
    positive: &[Tensor],
    negative: &[Tensor],
    normalise: bool,
    _device: &Device,
) -> Result<Tensor> {
    if positive.is_empty() || negative.is_empty() {
        return Err(MIError::Config(
            "compute_direction: positive and negative sets must both be non-empty".into(),
        ));
    }
    let pos_mean = stack_and_mean(positive)?;
    let neg_mean = stack_and_mean(negative)?;
    let diff = (&pos_mean - &neg_mean)?;
    if normalise {
        l2_normalise(&diff)
    } else {
        Ok(diff)
    }
}

/// Stack `[hidden]` tensors into `[N, hidden]` then take mean over dim 0 →
/// `[hidden]`.
fn stack_and_mean(tensors: &[Tensor]) -> Result<Tensor> {
    let stacked = Tensor::stack(tensors, 0)?;
    let mean = stacked.mean(0)?;
    Ok(mean)
}

/// L2-normalise a 1-D tensor to unit norm.  No-op (returns the input) when
/// the norm is below `1e-12` (avoids dividing by zero on degenerate
/// directions; the caller's `n_positive == n_negative` with identical sets
/// produces such a near-zero direction).
fn l2_normalise(v: &Tensor) -> Result<Tensor> {
    let norm_sq = (v * v)?.sum_all()?;
    // PROMOTE: norm comes out as a scalar tensor; extract as f64 for the
    // threshold check.
    let norm_sq_scalar = norm_sq.to_dtype(DType::F64)?.to_scalar::<f64>()?;
    let norm = norm_sq_scalar.sqrt();
    if norm < 1e-12_f64 {
        // EXPLICIT: degenerate direction (e.g. pos == neg in unit tests).
        // Return as-is to avoid NaN; the caller's is_normalised flag still
        // reads true, but the vector itself is effectively zero.
        return Ok(v.clone());
    }
    // CAST: f64 → f64 (no-op here, kept for clarity); affine takes f64.
    let scaled = (v / norm)?;
    Ok(scaled)
}

// ---------------------------------------------------------------------------
// Unit tests (no GPU, synthetic tensors only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn position_strategy_last_returns_seq_len_minus_one() {
        let tokens = vec![1_u32, 2, 3, 4, 5];
        let pos = PositionStrategy::Last.resolve(&tokens, 0).unwrap();
        assert_eq!(pos, 4);
    }

    #[test]
    fn position_strategy_first_newline_finds_correct_index() {
        let tokens = vec![1_u32, 2, 198, 3, 198];
        let pos = PositionStrategy::FirstNewline
            .resolve(&tokens, 198)
            .unwrap();
        assert_eq!(pos, 2);
    }

    #[test]
    fn position_strategy_first_newline_errors_when_absent() {
        let tokens = vec![1_u32, 2, 3, 4, 5];
        let err = PositionStrategy::FirstNewline.resolve(&tokens, 198);
        assert!(err.is_err());
    }

    #[test]
    fn position_strategy_explicit_errors_on_out_of_range() {
        let tokens = vec![1_u32, 2, 3];
        let err = PositionStrategy::Explicit(5).resolve(&tokens, 0);
        assert!(err.is_err());
    }

    #[test]
    fn position_strategy_resolve_errors_on_empty_tokens() {
        let tokens: Vec<u32> = vec![];
        let err = PositionStrategy::Last.resolve(&tokens, 0);
        assert!(err.is_err());
    }

    #[test]
    fn compute_direction_with_identical_sets_is_near_zero() {
        let device = cpu();
        let row = Tensor::new(&[1.0_f32, 2.0, 3.0, 4.0], &device).unwrap();
        let positive = vec![row.clone(), row.clone()];
        let negative = vec![row.clone(), row.clone()];
        let direction = compute_direction(&positive, &negative, false, &device).unwrap();
        let norm_sq = (&direction * &direction).unwrap().sum_all().unwrap();
        let norm_sq_scalar = norm_sq.to_scalar::<f32>().unwrap();
        assert!(norm_sq_scalar < 1e-10, "norm_sq = {norm_sq_scalar}");
    }

    #[test]
    fn compute_direction_with_disjoint_means() {
        let device = cpu();
        let pos_row = Tensor::new(&[2.0_f32, 4.0, 6.0], &device).unwrap();
        let neg_row = Tensor::new(&[1.0_f32, 2.0, 3.0], &device).unwrap();
        let positive = vec![pos_row.clone(), pos_row.clone()];
        let negative = vec![neg_row.clone(), neg_row.clone()];
        let direction = compute_direction(&positive, &negative, false, &device).unwrap();
        let values: Vec<f32> = direction.to_vec1().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
        assert!((values[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn normalised_direction_has_unit_l2_norm() {
        let device = cpu();
        let pos_row = Tensor::new(&[3.0_f32, 0.0, 4.0], &device).unwrap();
        let neg_row = Tensor::new(&[0.0_f32, 0.0, 0.0], &device).unwrap();
        let positive = vec![pos_row];
        let negative = vec![neg_row];
        let direction = compute_direction(&positive, &negative, true, &device).unwrap();
        let norm_sq = (&direction * &direction).unwrap().sum_all().unwrap();
        let norm = norm_sq.to_scalar::<f32>().unwrap().sqrt();
        assert!((norm - 1.0_f32).abs() < 1e-6, "norm = {norm}");
    }

    #[test]
    fn contrastive_intervention_payload_scales_with_strength() {
        let device = cpu();
        let direction = ContrastiveDirection {
            layer: 5,
            vector: Tensor::new(&[1.0_f32, 0.0, 0.0], &device).unwrap(),
            is_normalised: true,
            n_positive: 1,
            n_negative: 1,
            position_strategy: PositionStrategy::Last,
        };
        let intervention = contrastive_intervention(&direction, 2.5_f32).unwrap();
        match intervention {
            Intervention::Add(payload) => {
                let values: Vec<f32> = payload.to_vec1().unwrap();
                assert!(
                    (values[0] - 2.5_f32).abs() < 1e-6,
                    "values[0] = {}",
                    values[0]
                );
                assert!(values[1].abs() < 1e-6);
                assert!(values[2].abs() < 1e-6);
            }
            // EXPLICIT: the helper is hard-coded to return Add; any other
            // variant is a bug.
            other => panic!("expected Intervention::Add, got {other:?}"),
        }
    }

    #[test]
    fn position_delta_places_vector_at_correct_index() {
        let device = cpu();
        let direction = Tensor::new(&[7.0_f32, 8.0, 9.0], &device).unwrap();
        let delta = position_delta(&direction, 2, 4).unwrap();
        assert_eq!(delta.dims(), &[1, 4, 3]);
        let squeezed = delta.squeeze(0).unwrap();
        let row0: Vec<f32> = squeezed.get(0).unwrap().to_vec1().unwrap();
        let row2: Vec<f32> = squeezed.get(2).unwrap().to_vec1().unwrap();
        let row3: Vec<f32> = squeezed.get(3).unwrap().to_vec1().unwrap();
        assert!(row0.iter().all(|&x| x.abs() < 1e-6));
        assert!((row2[0] - 7.0).abs() < 1e-6);
        assert!((row2[1] - 8.0).abs() < 1e-6);
        assert!((row2[2] - 9.0).abs() < 1e-6);
        assert!(row3.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn position_delta_errors_on_out_of_range() {
        let device = cpu();
        let direction = Tensor::new(&[1.0_f32, 2.0], &device).unwrap();
        let err = position_delta(&direction, 5, 4);
        assert!(err.is_err());
    }

    #[test]
    fn position_delta_errors_on_non_1d_direction() {
        let device = cpu();
        let direction = Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &device).unwrap();
        let err = position_delta(&direction, 0, 4);
        assert!(err.is_err());
    }
}
