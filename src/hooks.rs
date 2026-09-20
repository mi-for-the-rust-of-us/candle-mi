// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hook system for activation capture and intervention.
//!
//! Provides [`HookPoint`] (named locations in a forward pass),
//! [`HookSpec`] (what to capture and where to intervene), and
//! [`HookCache`] (captured tensors from a forward pass).
//!
//! See `design/hook-system.md` for the design rationale.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

use candle_core::Tensor;

use crate::error::{MIError, Result};
use crate::interp::intervention::{StateKnockoutSpec, StateSteeringSpec};

// ---------------------------------------------------------------------------
// HookPoint
// ---------------------------------------------------------------------------

/// Named location in a forward pass where activations can be captured
/// or interventions applied.
///
/// Mirrors the `TransformerLens` hook point naming convention via
/// [`Display`](std::fmt::Display) and [`FromStr`].
///
/// # String conversion
///
/// ```
/// use candle_mi::HookPoint;
///
/// let hook = HookPoint::AttnPattern(5);
/// assert_eq!(hook.to_string(), "blocks.5.attn.hook_pattern");
///
/// let parsed: HookPoint = "blocks.5.attn.hook_pattern".parse().unwrap();
/// assert_eq!(parsed, hook);
/// ```
///
/// Unknown strings parse as [`HookPoint::Custom`], providing an escape
/// hatch for backend-specific hook points.
///
/// # Ordering
///
/// [`Ord`] is derived so a hook point can key a [`BTreeMap`] directly. That is
/// what a caller under a determinism contract needs: iteration over a
/// [`HashMap`] is unordered, and without [`Ord`] the only total operation left
/// on a `#[non_exhaustive]` enum is [`to_string`](std::string::ToString::to_string).
///
/// The order is **total, but its relation to hook semantics is unspecified**.
/// Derived ordering follows variant declaration order, and `HookPoint` is
/// `#[non_exhaustive]`, so inserting a variant can reorder existing ones in a
/// patch release. Rely on it for within-run determinism and for map keys; never
/// persist it, compare it across versions, or read layer order into it. To order
/// by hook semantics, sort by the semantic fields explicitly.
///
/// [`BTreeMap`]: std::collections::BTreeMap
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HookPoint {
    // -- Embedding --
    /// After token embedding (`hook_embed`).
    Embed,

    // -- Per-layer: transformer --
    /// Residual stream before layer `i` (`blocks.{i}.hook_resid_pre`).
    ResidPre(usize),
    /// Query vectors in layer `i` (`blocks.{i}.attn.hook_q`).
    AttnQ(usize),
    /// Key vectors in layer `i` (`blocks.{i}.attn.hook_k`).
    AttnK(usize),
    /// Value vectors in layer `i` (`blocks.{i}.attn.hook_v`).
    AttnV(usize),
    /// Pre-softmax attention scores in layer `i` (`blocks.{i}.attn.hook_scores`).
    AttnScores(usize),
    /// Post-softmax attention pattern in layer `i` (`blocks.{i}.attn.hook_pattern`).
    AttnPattern(usize),
    /// Attention output in layer `i` (`blocks.{i}.hook_attn_out`).
    AttnOut(usize),
    /// Residual stream between attention and MLP in layer `i`
    /// (`blocks.{i}.hook_resid_mid`).
    ResidMid(usize),
    /// MLP pre-activation in layer `i` (`blocks.{i}.mlp.hook_pre`).
    MlpPre(usize),
    /// MLP post-activation in layer `i` (`blocks.{i}.mlp.hook_post`).
    MlpPost(usize),
    /// MLP output in layer `i` (`blocks.{i}.hook_mlp_out`).
    MlpOut(usize),
    /// Residual stream after full layer `i` (`blocks.{i}.hook_resid_post`).
    ResidPost(usize),

    // -- Final --
    /// After final layer norm (`hook_final_norm`).
    ///
    /// The last capturable point before the unembedding projection. **The
    /// logits are not a hook point**: they are the forward pass's output, read
    /// with [`HookCache::output`]. Project a captured `FinalNorm` (or any
    /// residual stream) to vocabulary space with
    /// [`MIBackend::project_to_vocab`](crate::MIBackend::project_to_vocab).
    FinalNorm,

    // -- RWKV-specific --
    /// RWKV recurrent state at layer `i` (`blocks.{i}.rwkv.hook_state`).
    RwkvState(usize),
    /// RWKV decay vector at layer `i` (`blocks.{i}.rwkv.hook_decay`).
    RwkvDecay(usize),
    /// RWKV effective attention at layer `i` (`blocks.{i}.rwkv.hook_effective_attn`).
    ///
    /// Shape: `[batch, heads, seq_query, seq_source]`.
    /// Derived from the WKV recurrence by computing how much each
    /// source position contributes to each query position's output.
    /// Normalised via `ReLU` + L1.
    RwkvEffectiveAttn(usize),

    // -- Escape hatch --
    /// Backend-specific hook point not covered by the standard enum.
    Custom(String),
}

impl HookPoint {
    /// Whether [`Intervention::PatchAt`] can be applied at this hook point.
    ///
    /// True for exactly the hook points whose activation is
    /// `[batch, seq_len, hidden]`, so that the sequence is dim 1 and a single
    /// position names one row unambiguously: [`Embed`](Self::Embed),
    /// [`ResidPre`](Self::ResidPre), [`AttnOut`](Self::AttnOut),
    /// [`ResidMid`](Self::ResidMid), [`MlpPre`](Self::MlpPre),
    /// [`MlpPost`](Self::MlpPost), [`MlpOut`](Self::MlpOut),
    /// [`ResidPost`](Self::ResidPost) and [`FinalNorm`](Self::FinalNorm).
    ///
    /// # Why the others are excluded
    ///
    /// - [`AttnQ`](Self::AttnQ), [`AttnK`](Self::AttnK) and
    ///   [`AttnV`](Self::AttnV) are `[batch, n_heads, seq_len, head_dim]` (and
    ///   `n_kv_heads` rather than `n_heads` for K and V, which are captured
    ///   before the grouped-query broadcast). Dim 1 is a head, so a positional
    ///   patch written there would silently overwrite a head.
    /// - [`AttnScores`](Self::AttnScores) and
    ///   [`AttnPattern`](Self::AttnPattern) are
    ///   `[batch, n_heads, seq_len, seq_len]`: two sequence axes, so "one
    ///   position" has no unambiguous meaning.
    /// - [`RwkvState`](Self::RwkvState), [`RwkvDecay`](Self::RwkvDecay) and
    ///   [`RwkvEffectiveAttn`](Self::RwkvEffectiveAttn) are state-shaped, not
    ///   sequence-major.
    /// - [`Custom`](Self::Custom) is backend-defined, so this crate cannot know
    ///   its layout.
    ///
    /// Patching a key or value is a coherent thing to want, but it is a
    /// different operation rather than a wider version of this one: those
    /// tensors are pre-broadcast, so a write lands on a KV head and fans out to
    /// `n_heads / n_kv_heads` query heads downstream.
    ///
    /// ```
    /// use candle_mi::HookPoint;
    ///
    /// assert!(HookPoint::ResidPost(5).accepts_positional_patch());
    /// assert!(!HookPoint::AttnPattern(5).accepts_positional_patch());
    /// ```
    #[must_use]
    pub const fn accepts_positional_patch(&self) -> bool {
        matches!(
            self,
            Self::Embed
                | Self::ResidPre(_)
                | Self::AttnOut(_)
                | Self::ResidMid(_)
                | Self::MlpPre(_)
                | Self::MlpPost(_)
                | Self::MlpOut(_)
                | Self::ResidPost(_)
                | Self::FinalNorm
        )
    }
}

impl fmt::Display for HookPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed => write!(f, "hook_embed"),
            Self::ResidPre(i) => write!(f, "blocks.{i}.hook_resid_pre"),
            Self::AttnQ(i) => write!(f, "blocks.{i}.attn.hook_q"),
            Self::AttnK(i) => write!(f, "blocks.{i}.attn.hook_k"),
            Self::AttnV(i) => write!(f, "blocks.{i}.attn.hook_v"),
            Self::AttnScores(i) => write!(f, "blocks.{i}.attn.hook_scores"),
            Self::AttnPattern(i) => write!(f, "blocks.{i}.attn.hook_pattern"),
            Self::AttnOut(i) => write!(f, "blocks.{i}.hook_attn_out"),
            Self::ResidMid(i) => write!(f, "blocks.{i}.hook_resid_mid"),
            Self::MlpPre(i) => write!(f, "blocks.{i}.mlp.hook_pre"),
            Self::MlpPost(i) => write!(f, "blocks.{i}.mlp.hook_post"),
            Self::MlpOut(i) => write!(f, "blocks.{i}.hook_mlp_out"),
            Self::ResidPost(i) => write!(f, "blocks.{i}.hook_resid_post"),
            Self::FinalNorm => write!(f, "hook_final_norm"),
            Self::RwkvState(i) => write!(f, "blocks.{i}.rwkv.hook_state"),
            Self::RwkvDecay(i) => write!(f, "blocks.{i}.rwkv.hook_decay"),
            Self::RwkvEffectiveAttn(i) => write!(f, "blocks.{i}.rwkv.hook_effective_attn"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Parse a `TransformerLens`-style string into a [`HookPoint`].
///
/// Unknown strings produce [`HookPoint::Custom`] rather than an error.
impl FromStr for HookPoint {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(parse_hook_string(s))
    }
}

/// Allow `hooks.capture("blocks.5.attn.hook_pattern")` via `Into<HookPoint>`.
impl From<&str> for HookPoint {
    fn from(s: &str) -> Self {
        parse_hook_string(s)
    }
}

/// Parse a hook string, falling back to [`HookPoint::Custom`] for unknown patterns.
fn parse_hook_string(s: &str) -> HookPoint {
    match s {
        "hook_embed" => return HookPoint::Embed,
        "hook_final_norm" => return HookPoint::FinalNorm,
        _ => {}
    }

    // Try "blocks.{layer}.{suffix}" pattern.
    if let Some(rest) = s.strip_prefix("blocks.")
        && let Some((layer_str, suffix)) = rest.split_once('.')
        && let Ok(layer) = layer_str.parse::<usize>()
    {
        return match suffix {
            "hook_resid_pre" => HookPoint::ResidPre(layer),
            "attn.hook_q" => HookPoint::AttnQ(layer),
            "attn.hook_k" => HookPoint::AttnK(layer),
            "attn.hook_v" => HookPoint::AttnV(layer),
            "attn.hook_scores" => HookPoint::AttnScores(layer),
            "attn.hook_pattern" => HookPoint::AttnPattern(layer),
            "hook_attn_out" => HookPoint::AttnOut(layer),
            "hook_resid_mid" => HookPoint::ResidMid(layer),
            "mlp.hook_pre" => HookPoint::MlpPre(layer),
            "mlp.hook_post" => HookPoint::MlpPost(layer),
            "hook_mlp_out" => HookPoint::MlpOut(layer),
            "hook_resid_post" => HookPoint::ResidPost(layer),
            "rwkv.hook_state" => HookPoint::RwkvState(layer),
            "rwkv.hook_decay" => HookPoint::RwkvDecay(layer),
            "rwkv.hook_effective_attn" => HookPoint::RwkvEffectiveAttn(layer),
            _ => HookPoint::Custom(s.to_string()),
        };
    }

    HookPoint::Custom(s.to_string())
}

// ---------------------------------------------------------------------------
// Intervention
// ---------------------------------------------------------------------------

/// An intervention to apply at a hook point during the forward pass.
///
/// Interventions modify activations in-place as they flow through the model.
/// They are specified as part of a [`HookSpec`] and applied by the backend
/// at the corresponding [`HookPoint`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Intervention {
    /// Replace the tensor entirely with a provided value.
    ///
    /// Whole-tensor. To overwrite one sequence position and leave the rest of
    /// the activation alone, use [`PatchAt`](Self::PatchAt) rather than
    /// capturing the activation, splicing a row in and handing the whole
    /// tensor back here.
    Replace(Tensor),

    /// Replace a single sequence position, leaving every other position
    /// untouched.
    ///
    /// The positional counterpart to [`Replace`](Self::Replace), and the
    /// standard causal instrument of activation patching: run the recipient's
    /// forward pass, but at one hook point and one position substitute a row
    /// taken from a donor pass.
    ///
    /// # Shapes
    ///
    /// - activation at the hook point: `[batch, seq_len, hidden]`
    /// - `value`: `[hidden]`, `[1, 1, hidden]` or `[batch, 1, hidden]`
    /// - result: `[batch, seq_len, hidden]`, contiguous
    ///
    /// A `[hidden]` or `[1, 1, hidden]` value applies to every batch row, the
    /// same way [`Add`](Self::Add) broadcasts. `[batch, 1, hidden]` gives each
    /// batch row its own replacement. A row extracted from a donor capture with
    /// `donor.narrow(1, position, 1)` is already `[1, 1, hidden]`, so it can be
    /// passed straight in.
    ///
    /// # Accepted hook points
    ///
    /// Only those for which
    /// [`HookPoint::accepts_positional_patch`] is true, which is every hook
    /// point whose activation is `[batch, seq_len, hidden]`. Anywhere else the
    /// forward pass fails with
    /// [`MIError::Intervention`] rather than
    /// writing at dim 1 regardless, which at an attention hook point would
    /// overwrite a head and produce a plausible figure instead of an error.
    ///
    /// # Dtype
    ///
    /// A `value` whose dtype differs from the activation's is converted, so an
    /// `F32` donor row patches into a `BF16` forward pass. This mirrors
    /// [`Add`](Self::Add).
    ///
    /// # Gradients
    ///
    /// The write is a masked select, which records a backward op, so a patch
    /// inside a tracked forward pass does not break the gradient chain.
    PatchAt {
        /// Sequence position to overwrite. Must be in `0..seq_len`.
        position: usize,
        /// Replacement row: `[hidden]`, `[1, 1, hidden]` or `[batch, 1, hidden]`.
        value: Tensor,
    },

    /// Add a vector to the activation (e.g., residual stream steering).
    ///
    /// To fire at a single sequence position (zero elsewhere), build the
    /// broadcast payload with `steering::position_delta` (requires a backend
    /// feature). To *overwrite* a position rather than add to it, use
    /// [`PatchAt`](Self::PatchAt).
    Add(Tensor),

    /// Apply a pre-softmax knockout mask.
    ///
    /// The mask tensor contains `0.0` for positions to keep and
    /// `-inf` for positions to knock out. Added to attention scores.
    Knockout(Tensor),

    /// Scale attention weights by a constant factor.
    Scale(f64),

    /// Zero the tensor at this hook point.
    Zero,
}

// ---------------------------------------------------------------------------
// Intervention application
// ---------------------------------------------------------------------------

/// Apply a single [`Intervention`] to a tensor.
///
/// Used by backend implementations at each hook point that supports
/// interventions (e.g., Embed, `AttnScores`, `AttnPattern`).
///
/// `point` is the hook point the tensor was taken at. It is needed because dim
/// 1 does not mean the same thing everywhere: it is the sequence in a
/// `[batch, seq_len, hidden]` activation but a head in a
/// `[batch, n_heads, seq_len, head_dim]` one, so a positional intervention has
/// to know where it is before it writes. See
/// [`HookPoint::accepts_positional_patch`].
///
/// # Shapes
/// - `tensor`: any shape — the activation at the hook point.
/// - returns: same shape as `tensor`.
///
/// # Errors
///
/// Returns [`MIError::Model`] if the underlying tensor operation fails.
/// Returns [`MIError::Intervention`] if [`Intervention::PatchAt`] is applied at
/// a hook point whose activation is not `[batch, seq_len, hidden]`.
/// Returns [`MIError::Intervention`] if [`Intervention::PatchAt`] is applied to
/// a tensor that is not rank 3.
/// Returns [`MIError::Intervention`] if an [`Intervention::PatchAt`] position is
/// outside `0..seq_len`.
/// Returns [`MIError::Intervention`] if an [`Intervention::PatchAt`] value does
/// not have one of the accepted shapes.
#[cfg(any(
    feature = "transformer",
    feature = "rwkv",
    feature = "diffusion",
    feature = "stoicheia"
))]
pub(crate) fn apply_intervention(
    tensor: &Tensor,
    point: &HookPoint,
    intervention: &Intervention,
) -> Result<Tensor> {
    match intervention {
        Intervention::Replace(replacement) => Ok(replacement.clone()),
        Intervention::PatchAt { position, value } => patch_at(tensor, point, *position, value),
        Intervention::Add(delta) => {
            // Convert delta to tensor's dtype if mismatched (e.g., F32 injection
            // into BF16 forward pass). This supports CLT injection where steering
            // vectors are accumulated in F32 for numerical stability.
            let delta = if delta.dtype() == tensor.dtype() {
                delta
            } else {
                &delta.to_dtype(tensor.dtype())?
            };
            Ok(tensor.broadcast_add(delta)?)
        }
        Intervention::Knockout(mask) => Ok(tensor.broadcast_add(mask)?),
        Intervention::Scale(factor) => Ok((tensor * *factor)?),
        Intervention::Zero => Ok(tensor.zeros_like()?),
    }
}

/// Apply the standard capture-then-intervene protocol at one hook point.
///
/// Every backend fires this at each of its hook points: the activation is
/// cloned into `cache` when the point is captured, then each registered
/// [`Intervention`] is applied in turn, mutating `tensor` in place so the rest
/// of the forward pass sees the intervened value.
///
/// **Both halves are load-bearing.** A backend that captures but never applies
/// interventions compiles, runs, and silently returns the unmodified baseline
/// from every causal experiment, so a measured effect of zero is
/// indistinguishable from a real null. `GenericRwkv` and both `stoicheia`
/// backends did exactly that until v0.2.0; see
/// `docs/audit/V0_2_0_AUDIT_2026-09-20.md` finding 1. Sharing one helper is what
/// keeps a new backend from reintroducing the same silent gap, and
/// `BACKENDS.md` makes the conformance test that catches it mandatory.
///
/// # Shapes
/// - `tensor`: any shape -- the activation at `point`, mutated in place.
///
/// # Errors
///
/// Returns [`MIError::Model`] if an intervention's tensor operation fails, for
/// example an [`Intervention::Add`] whose delta cannot broadcast to the
/// activation.
/// Returns [`MIError::Intervention`] if an intervention is invalid at `point`,
/// such as [`Intervention::PatchAt`] where the activation is not
/// `[batch, seq_len, hidden]`.
// The by-value `HookPoint` lets call sites pass a freshly-built variant without
// `&`; capturing still needs one clone either way.
#[cfg(any(
    feature = "transformer",
    feature = "rwkv",
    feature = "diffusion",
    feature = "stoicheia"
))]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn hook_point(
    tensor: &mut Tensor,
    point: HookPoint,
    hooks: &HookSpec,
    cache: &mut HookCache,
) -> Result<()> {
    if hooks.is_captured(&point) {
        cache.store(point.clone(), tensor.clone());
    }
    for intervention in hooks.interventions_at(&point) {
        *tensor = apply_intervention(tensor, &point, intervention)?;
    }
    Ok(())
}

/// Capture a diagnostic read-out, and reject any intervention aimed at it.
///
/// Some hook points expose a tensor that nothing downstream consumes: RWKV's
/// [`HookPoint::RwkvState`], [`HookPoint::RwkvDecay`] and
/// [`HookPoint::RwkvEffectiveAttn`] are computed for observation, and the
/// forward pass has already used whatever they were derived from. Mutating them
/// would change nothing.
///
/// Such a point must therefore **refuse** an intervention rather than accept one
/// and discard it. Silently accepting is precisely the failure this release
/// removes, and relocating it from "the backend forgot" to "the tensor was
/// dead" would be no better: a causal experiment would still report a null it
/// never tested. The error names the supported alternative where one exists.
///
/// # Shapes
/// - `tensor`: any shape -- the read-out at `point`; never modified.
///
/// # Errors
///
/// Returns [`MIError::Intervention`] if any intervention targets `point`.
// The by-value `HookPoint` matches `hook_point`, so call sites read alike.
// Gated to `rwkv` because that is the only backend with diagnostic read-out
// points today; widen the gate when a second one needs it.
#[cfg(feature = "rwkv")]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn hook_point_readonly(
    tensor: &Tensor,
    point: HookPoint,
    hooks: &HookSpec,
    cache: &mut HookCache,
    alternative: &str,
) -> Result<()> {
    if hooks.has_intervention_at(&point) {
        return Err(MIError::Intervention(format!(
            "hook point `{point}` is a diagnostic read-out and cannot be \
             intervened on (nothing downstream reads it, so the edit would be \
             silently discarded); {alternative}"
        )));
    }
    if hooks.is_captured(&point) {
        cache.store(point, tensor.clone());
    }
    Ok(())
}

/// Overwrite one sequence position of a `[batch, seq_len, hidden]` activation.
///
/// The implementation of [`Intervention::PatchAt`]. Split out of
/// [`apply_intervention`] so that its match arm stays a single line.
///
/// # Shapes
/// - `tensor`: `[batch, seq_len, hidden]`
/// - `value`: `[hidden]`, `[1, 1, hidden]` or `[batch, 1, hidden]`
/// - returns: `[batch, seq_len, hidden]`, contiguous
///
/// # Errors
///
/// Returns [`MIError::Intervention`] if `point` does not accept a positional
/// patch.
/// Returns [`MIError::Intervention`] if `tensor` is not rank 3.
/// Returns [`MIError::Intervention`] if `position` is outside `0..seq_len`.
/// Returns [`MIError::Intervention`] if `value` has none of the accepted shapes.
/// Returns [`MIError::Intervention`] if `seq_len` or `position` exceeds `u32`,
/// which the selector's index range is built over.
/// Returns [`MIError::Model`] if the underlying tensor operation fails.
#[cfg(any(
    feature = "transformer",
    feature = "rwkv",
    feature = "diffusion",
    feature = "stoicheia"
))]
fn patch_at(tensor: &Tensor, point: &HookPoint, position: usize, value: &Tensor) -> Result<Tensor> {
    if !point.accepts_positional_patch() {
        return Err(MIError::Intervention(format!(
            "intervention PatchAt not supported at hook point `{point}` \
             (activation is not [batch, seq_len, hidden]; see \
             HookPoint::accepts_positional_patch)"
        )));
    }

    // The policy above is by hook point; this is the same question asked of the
    // tensor itself, so a backend storing an unexpected rank at an accepting
    // point errors here rather than writing at the wrong axis.
    let tensor_dims = tensor.dims();
    let &[batch, seq_len, hidden] = tensor_dims else {
        return Err(MIError::Intervention(format!(
            "intervention PatchAt needs a rank-3 [batch, seq_len, hidden] \
             activation at hook point `{point}` (got shape {tensor_dims:?})"
        )));
    };

    if position >= seq_len {
        return Err(MIError::Intervention(format!(
            "patch position {position} out of bounds (seq_len={seq_len})"
        )));
    }

    // Normalise every accepted value shape to `[batch, 1, hidden]`. The two
    // batch-free shapes broadcast across the batch, matching how `Add` treats a
    // bare direction.
    let value_dims = value.dims();
    let row = match *value_dims {
        [h] if h == hidden => value
            .reshape((1, 1, hidden))?
            .broadcast_as((batch, 1, hidden))?,
        [1, 1, h] if h == hidden => value.broadcast_as((batch, 1, hidden))?,
        [b, 1, h] if b == batch && h == hidden => value.clone(),
        _ => {
            return Err(MIError::Intervention(format!(
                "patch value shape {value_dims:?} unusable (expected [hidden], \
                 [1, 1, hidden] or [{batch}, 1, hidden] with hidden={hidden})"
            )));
        }
    };

    // Convert the row to the activation's dtype if mismatched (e.g., an F32
    // donor row patched into a BF16 forward pass), mirroring the `Add` arm.
    let row = if row.dtype() == tensor.dtype() {
        row
    } else {
        row.to_dtype(tensor.dtype())?
    };

    // Written as a masked select rather than the obvious `Tensor::slice_scatter`,
    // which routes through candle's `copy_strided_src`, whose CUDA path sizes the
    // copy from the *storage* rather than from the source view:
    //
    //     to_copy = min(dst.len() - dst_offset, src.len() - src_offset)
    //         -- candle-core `cuda_backend::slice_src_and_dst`
    //
    // A donor row is normally a view into a captured activation (this is exactly
    // what `FullActivationCache::get_position` returns), so `src.len()` is the
    // whole donor and the copy overruns into the *following* positions. The
    // symptom is a patch that also overwrites every position after it, silently
    // and only on GPU. Reported as candle#3940; `tests/validate_patch_at.rs` is
    // the end-to-end guard. `where_cond` touches each element once through its
    // own layout and has no such dependence.
    //
    // CONTIGUOUS: every operand is materialised to the same contiguous shape, so
    // no operand reaches the kernel as a stride-0 broadcast view or as whatever
    // layout an earlier intervention happened to leave behind. `contiguous()` is
    // a clone when the tensor already is, so the common path costs nothing.
    let full = (batch, seq_len, hidden);
    let replacement = row.broadcast_as(full)?.contiguous()?;
    let base = tensor.contiguous()?;

    // `Tensor::arange` builds the index range at `u32`, so both bounds have to
    // fit it. A sequence long enough to fail this cannot be held in memory, but
    // the conversion is fallible and is not worth an `as` cast to hide.
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|e| MIError::Intervention(format!("seq_len {seq_len} exceeds u32: {e}")))?;
    let position_u32 = u32::try_from(position)
        .map_err(|e| MIError::Intervention(format!("position {position} exceeds u32: {e}")))?;
    let selector = Tensor::arange(0_u32, seq_len_u32, tensor.device())?
        .eq(position_u32)?
        .reshape((1, seq_len, 1))?
        .broadcast_as(full)?
        .contiguous()?;

    Ok(selector.where_cond(&replacement, &base)?)
}

// ---------------------------------------------------------------------------
// HookSpec
// ---------------------------------------------------------------------------

/// Declares which activations to capture and which interventions to apply.
///
/// Passed to [`MIBackend::forward`](crate::MIBackend::forward). When empty,
/// the forward pass has negligible overhead — a few microseconds of
/// per-hook-point `is_captured` checks (which return immediately on the empty
/// `HashSet`), and a small placeholder allocation for the returned
/// [`HookCache`]. No activations are cloned and no captures are stored. See
/// `docs/hook-architecture-diagnostic.md` for measured numbers on Llama-3.2-1B.
///
/// # Cloning
///
/// [`Clone`] is a **guarantee**, not an accident of the derive: a spec holds
/// hook points and intervention descriptions, never activations, so cloning one
/// is cheap. Callers may keep a table of per-step specs and hand out clones.
///
/// # Example
///
/// ```
/// use candle_mi::{HookPoint, HookSpec};
///
/// let mut hooks = HookSpec::new();
/// hooks.capture(HookPoint::AttnPattern(5))
///      .capture("blocks.5.hook_resid_post");
/// ```
#[derive(Debug, Clone, Default)]
pub struct HookSpec {
    /// Hook points to capture during the forward pass.
    captures: HashSet<HookPoint>,
    /// Interventions to apply, stored as (`hook_point`, intervention) pairs.
    interventions: Vec<(HookPoint, Intervention)>,
    /// RWKV state knockout specification (skip kv write at specified positions).
    state_knockout: Option<StateKnockoutSpec>,
    /// RWKV state steering specification (scale kv write at specified positions).
    state_steering: Option<StateSteeringSpec>,
}

impl HookSpec {
    /// Create an empty hook specification (no captures, no interventions).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request capture of the activation at the given hook point.
    pub fn capture<H: Into<HookPoint>>(&mut self, hook: H) -> &mut Self {
        self.captures.insert(hook.into());
        self
    }

    /// Request capture of every hook point in an iterator.
    ///
    /// The bulk form of [`capture`](Self::capture). Building a spec from a
    /// `&[HookPoint]` otherwise needs a `for` loop with a clone per element,
    /// repeated for every tapped forward pass.
    ///
    /// ```
    /// use candle_mi::{HookPoint, HookSpec};
    ///
    /// let mut hooks = HookSpec::new();
    /// hooks.capture_all((0..4).map(HookPoint::ResidPost))
    ///      .capture_all(["hook_embed", "hook_final_norm"]);
    /// assert_eq!(hooks.num_captures(), 6);
    /// ```
    ///
    /// See also the [`FromIterator`] impl, for building a spec rather than
    /// extending one.
    pub fn capture_all<I>(&mut self, hooks: I) -> &mut Self
    where
        I: IntoIterator,
        I::Item: Into<HookPoint>,
    {
        self.captures.extend(hooks.into_iter().map(Into::into));
        self
    }

    /// Register an intervention at the given hook point.
    pub fn intervene<H: Into<HookPoint>>(
        &mut self,
        hook: H,
        intervention: Intervention,
    ) -> &mut Self {
        self.interventions.push((hook.into(), intervention));
        self
    }

    /// Check whether a specific hook point should be captured.
    #[must_use]
    pub fn is_captured(&self, hook: &HookPoint) -> bool {
        self.captures.contains(hook)
    }

    /// Check whether this spec has no captures, no interventions, and no
    /// state specs (knockout/steering).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.captures.is_empty()
            && self.interventions.is_empty()
            && self.state_knockout.is_none()
            && self.state_steering.is_none()
    }

    /// Number of requested captures.
    #[must_use]
    pub fn num_captures(&self) -> usize {
        self.captures.len()
    }

    /// Iterate over the requested capture points.
    ///
    /// The counterpart to [`num_captures`](Self::num_captures), which reports a
    /// count of a collection the caller could not otherwise walk.
    ///
    /// **Order is arbitrary** (the backing store is a [`HashSet`]). For a
    /// deterministic walk, collect into a [`BTreeSet`]: [`HookPoint`] implements
    /// [`Ord`], and a [`BTreeSet`]'s iteration order cannot depend on the order
    /// things were inserted into it.
    ///
    /// ```
    /// use std::collections::BTreeSet;
    /// use candle_mi::{HookPoint, HookSpec};
    ///
    /// let mut hooks = HookSpec::new();
    /// hooks.capture(HookPoint::ResidPost(1)).capture(HookPoint::ResidPost(0));
    ///
    /// let requested: BTreeSet<&HookPoint> = hooks.captures().collect();
    /// assert_eq!(requested.len(), 2);
    /// ```
    ///
    /// [`BTreeSet`]: std::collections::BTreeSet
    pub fn captures(&self) -> impl Iterator<Item = &HookPoint> {
        self.captures.iter()
    }

    /// Number of registered interventions.
    #[must_use]
    pub const fn num_interventions(&self) -> usize {
        self.interventions.len()
    }

    /// Iterate over interventions registered at a specific hook point.
    pub fn interventions_at(&self, hook: &HookPoint) -> impl Iterator<Item = &Intervention> {
        self.interventions
            .iter()
            .filter(move |(h, _)| h == hook)
            .map(|(_, intervention)| intervention)
    }

    /// Check whether any intervention targets the given hook point.
    #[must_use]
    pub fn has_intervention_at(&self, hook: &HookPoint) -> bool {
        self.interventions.iter().any(|(h, _)| h == hook)
    }

    /// Set an RWKV state knockout specification.
    ///
    /// At specified token positions, the WKV recurrence skips the kv write,
    /// effectively making those tokens invisible to all future positions.
    pub fn set_state_knockout(&mut self, spec: StateKnockoutSpec) -> &mut Self {
        self.state_knockout = Some(spec);
        self
    }

    /// Set an RWKV state steering specification.
    ///
    /// At specified token positions, the WKV recurrence scales the kv write
    /// by the given factor, amplifying or dampening the token's contribution.
    pub fn set_state_steering(&mut self, spec: StateSteeringSpec) -> &mut Self {
        self.state_steering = Some(spec);
        self
    }

    /// Get the state knockout specification, if any.
    #[must_use]
    pub const fn state_knockout(&self) -> Option<&StateKnockoutSpec> {
        self.state_knockout.as_ref()
    }

    /// Get the state steering specification, if any.
    #[must_use]
    pub const fn state_steering(&self) -> Option<&StateSteeringSpec> {
        self.state_steering.as_ref()
    }

    /// Merge all captures and interventions from another [`HookSpec`] into this one.
    ///
    /// Useful for combining multiple intervention sources (e.g., suppress +
    /// inject in CLT steering).
    pub fn extend(&mut self, other: &Self) -> &mut Self {
        self.captures.extend(other.captures.iter().cloned());
        self.interventions
            .extend(other.interventions.iter().cloned());
        self
    }
}

/// Build a capture-only [`HookSpec`] from an iterator of hook points.
///
/// Deliberately **not** paired with an [`Extend`] impl: [`HookSpec`] already has
/// an inherent [`extend`](HookSpec::extend) that merges another spec, and
/// inherent methods win method resolution, so an [`Extend`] impl would make
/// `spec.extend(some_iterator)` fail to compile against the inherent signature.
/// Use [`capture_all`](HookSpec::capture_all) to extend from an iterator.
///
/// ```
/// use candle_mi::{HookPoint, HookSpec};
///
/// let hooks: HookSpec = (0..4).map(HookPoint::ResidPost).collect();
/// assert_eq!(hooks.num_captures(), 4);
/// assert_eq!(hooks.num_interventions(), 0);
/// ```
impl FromIterator<HookPoint> for HookSpec {
    fn from_iter<I: IntoIterator<Item = HookPoint>>(iter: I) -> Self {
        Self {
            captures: iter.into_iter().collect(),
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// HookCache
// ---------------------------------------------------------------------------

/// Tensors captured during a forward pass, plus the output logits.
///
/// Returned by [`MIBackend::forward`](crate::MIBackend::forward). Use
/// [`get`](Self::get) to retrieve activations at specific hook points.
///
/// # Example
///
/// ```
/// use candle_mi::{HookCache, HookPoint};
/// use candle_core::{Device, Tensor};
///
/// let logits = Tensor::zeros((1, 10, 32000), candle_core::DType::F32, &Device::Cpu).unwrap();
/// let mut cache = HookCache::new(logits);
///
/// // Store a captured activation
/// let pattern = Tensor::zeros((1, 8, 10, 10), candle_core::DType::F32, &Device::Cpu).unwrap();
/// cache.store(HookPoint::AttnPattern(5), pattern);
///
/// // Retrieve captured activations
/// let output = cache.output();
/// let attn = cache.get(&HookPoint::AttnPattern(5)).unwrap();
/// ```
#[derive(Debug)]
pub struct HookCache {
    /// Output tensor from the forward pass (typically logits).
    output: Tensor,
    /// Captured activations keyed by hook point.
    captures: HashMap<HookPoint, Tensor>,
}

impl HookCache {
    /// Create a new cache with the given output tensor and no captures.
    #[must_use]
    pub fn new(output: Tensor) -> Self {
        Self {
            output,
            captures: HashMap::new(),
        }
    }

    /// The output tensor from the forward pass: **the logit tap**.
    ///
    /// The logits are reached here rather than through a [`HookPoint`], because
    /// they are the forward pass's output rather than an intermediate
    /// activation. [`HookPoint::FinalNorm`] is the last capturable point before
    /// the unembedding projection.
    ///
    /// # Shapes
    /// - returns: `[batch, seq, vocab_size]`
    #[must_use]
    pub const fn output(&self) -> &Tensor {
        &self.output
    }

    /// Consume the cache and return the output tensor (the logits, see
    /// [`output`](Self::output)).
    ///
    /// Mutually exclusive with [`into_captures`](Self::into_captures), since
    /// both consume the cache.
    ///
    /// # Shapes
    /// - returns: `[batch, seq, vocab_size]`
    #[must_use]
    pub fn into_output(self) -> Tensor {
        self.output
    }

    /// Retrieve a captured tensor by hook point.
    #[must_use]
    pub fn get(&self, hook: &HookPoint) -> Option<&Tensor> {
        self.captures.get(hook)
    }

    /// Retrieve a captured tensor, returning an error if not found.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the hook point was not captured.
    pub fn require(&self, hook: &HookPoint) -> Result<&Tensor> {
        self.captures
            .get(hook)
            .ok_or_else(|| MIError::Hook(format!("hook point `{hook}` was not captured")))
    }

    /// Store a captured activation. Called by backend implementations.
    pub fn store(&mut self, hook: HookPoint, tensor: Tensor) {
        self.captures.insert(hook, tensor);
    }

    /// Replace the output tensor (e.g., after computing final logits).
    ///
    /// This allows the forward pass to collect captures into a cache
    /// initialized with a placeholder, then set the real output at the end.
    pub fn set_output(&mut self, output: Tensor) {
        self.output = output;
    }

    /// Number of captured tensors (excludes the output).
    #[must_use]
    pub fn num_captures(&self) -> usize {
        self.captures.len()
    }

    /// Iterate over every captured activation.
    ///
    /// The counterpart to [`num_captures`](Self::num_captures), which reports a
    /// count of a collection the caller could not otherwise walk. Without this,
    /// a harness wanting *everything that was captured* has to keep its own copy
    /// of the request and re-derive the keys, discovering absence one
    /// [`get`](Self::get) at a time.
    ///
    /// # Shapes
    /// - returns: `(hook point, tensor)` pairs, each tensor's shape being the
    ///   one documented for its [`HookPoint`]
    ///
    /// **Order is arbitrary** (the backing store is a [`HashMap`]). For a
    /// deterministic walk, collect into a [`BTreeMap`]: [`HookPoint`] implements
    /// [`Ord`], and a [`BTreeMap`]'s iteration order cannot depend on the order
    /// things were inserted into it.
    ///
    /// ```
    /// use std::collections::BTreeMap;
    /// use candle_mi::{HookCache, HookPoint};
    /// use candle_core::{DType, Device, Tensor};
    ///
    /// # fn main() -> candle_mi::Result<()> {
    /// let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, &Device::Cpu)?);
    /// cache.store(HookPoint::ResidPost(1), Tensor::zeros(2, DType::F32, &Device::Cpu)?);
    /// cache.store(HookPoint::ResidPost(0), Tensor::zeros(2, DType::F32, &Device::Cpu)?);
    ///
    /// let by_hook: BTreeMap<&HookPoint, &Tensor> = cache.captures().collect();
    /// assert_eq!(by_hook.keys().next(), Some(&&HookPoint::ResidPost(0)));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`BTreeMap`]: std::collections::BTreeMap
    pub fn captures(&self) -> impl Iterator<Item = (&HookPoint, &Tensor)> {
        self.captures.iter()
    }

    /// Consume the cache and iterate over every captured activation by value.
    ///
    /// Mutually exclusive with [`into_output`](Self::into_output), since both
    /// consume the cache. To keep both, clone the output first: candle's
    /// `Tensor` is reference-counted, so `cache.output().clone()` costs a
    /// refcount, not a copy of the logits.
    ///
    /// **Order is arbitrary**, as for [`captures`](Self::captures).
    ///
    /// # Shapes
    /// - returns: `(hook point, tensor)` pairs, each tensor's shape being the
    ///   one documented for its [`HookPoint`]
    pub fn into_captures(self) -> impl Iterator<Item = (HookPoint, Tensor)> {
        self.captures.into_iter()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use candle_core::{DType, Device};

    use super::*;

    #[test]
    fn hook_point_display_roundtrip() {
        let cases: Vec<(HookPoint, &str)> = vec![
            (HookPoint::Embed, "hook_embed"),
            (HookPoint::FinalNorm, "hook_final_norm"),
            (HookPoint::ResidPre(0), "blocks.0.hook_resid_pre"),
            (HookPoint::AttnQ(3), "blocks.3.attn.hook_q"),
            (HookPoint::AttnK(3), "blocks.3.attn.hook_k"),
            (HookPoint::AttnV(3), "blocks.3.attn.hook_v"),
            (HookPoint::AttnScores(7), "blocks.7.attn.hook_scores"),
            (HookPoint::AttnPattern(5), "blocks.5.attn.hook_pattern"),
            (HookPoint::AttnOut(2), "blocks.2.hook_attn_out"),
            (HookPoint::ResidMid(11), "blocks.11.hook_resid_mid"),
            (HookPoint::MlpPre(1), "blocks.1.mlp.hook_pre"),
            (HookPoint::MlpPost(1), "blocks.1.mlp.hook_post"),
            (HookPoint::MlpOut(4), "blocks.4.hook_mlp_out"),
            (HookPoint::ResidPost(9), "blocks.9.hook_resid_post"),
            (HookPoint::RwkvState(6), "blocks.6.rwkv.hook_state"),
            (HookPoint::RwkvDecay(6), "blocks.6.rwkv.hook_decay"),
            (
                HookPoint::RwkvEffectiveAttn(6),
                "blocks.6.rwkv.hook_effective_attn",
            ),
        ];

        for (hook, expected_str) in cases {
            // Display
            assert_eq!(
                hook.to_string(),
                expected_str,
                "Display failed for {hook:?}"
            );
            // FromStr roundtrip
            let parsed: HookPoint = expected_str.parse().unwrap();
            assert_eq!(parsed, hook, "FromStr failed for {expected_str:?}");
            // From<&str>
            let from_str: HookPoint = HookPoint::from(expected_str);
            assert_eq!(from_str, hook, "From<&str> failed for {expected_str:?}");
        }
    }

    #[test]
    fn unknown_string_becomes_custom() {
        let hook: HookPoint = "some.unknown.hook".parse().unwrap();
        assert_eq!(hook, HookPoint::Custom("some.unknown.hook".to_string()));
    }

    /// One instance of every `HookPoint` variant, in declaration order.
    ///
    /// `Custom` is last because it is the last variant; the ordering tests
    /// assert that rather than assume it.
    fn one_of_each_variant() -> Vec<HookPoint> {
        vec![
            HookPoint::Embed,
            HookPoint::ResidPre(0),
            HookPoint::AttnQ(1),
            HookPoint::AttnK(1),
            HookPoint::AttnV(1),
            HookPoint::AttnScores(2),
            HookPoint::AttnPattern(2),
            HookPoint::AttnOut(3),
            HookPoint::ResidMid(3),
            HookPoint::MlpPre(4),
            HookPoint::MlpPost(4),
            HookPoint::MlpOut(5),
            HookPoint::ResidPost(5),
            HookPoint::FinalNorm,
            HookPoint::RwkvState(6),
            HookPoint::RwkvDecay(6),
            HookPoint::RwkvEffectiveAttn(6),
            HookPoint::Custom("some.custom.hook".to_string()),
        ]
    }

    /// Position of a variant in the enum's declaration order.
    ///
    /// `#[non_exhaustive]` does not apply within the defining crate, so this
    /// match is exhaustive: adding a variant stops it compiling, which is the
    /// reminder to extend `one_of_each_variant`. Without that, the helper's
    /// claims would silently stop holding and the ordering tests below would
    /// stop covering the new variant.
    fn declaration_rank(hook: &HookPoint) -> usize {
        match hook {
            HookPoint::Embed => 0,
            HookPoint::ResidPre(_) => 1,
            HookPoint::AttnQ(_) => 2,
            HookPoint::AttnK(_) => 3,
            HookPoint::AttnV(_) => 4,
            HookPoint::AttnScores(_) => 5,
            HookPoint::AttnPattern(_) => 6,
            HookPoint::AttnOut(_) => 7,
            HookPoint::ResidMid(_) => 8,
            HookPoint::MlpPre(_) => 9,
            HookPoint::MlpPost(_) => 10,
            HookPoint::MlpOut(_) => 11,
            HookPoint::ResidPost(_) => 12,
            HookPoint::FinalNorm => 13,
            HookPoint::RwkvState(_) => 14,
            HookPoint::RwkvDecay(_) => 15,
            HookPoint::RwkvEffectiveAttn(_) => 16,
            HookPoint::Custom(_) => 17,
        }
    }

    #[test]
    fn one_of_each_variant_covers_the_enum_in_declaration_order() {
        let ranks: Vec<usize> = one_of_each_variant().iter().map(declaration_rank).collect();
        assert_eq!(
            ranks,
            (0..ranks.len()).collect::<Vec<usize>>(),
            "one_of_each_variant must list every variant exactly once, in declaration order"
        );
    }

    #[test]
    fn hook_point_ord_is_total() {
        let hooks = one_of_each_variant();

        for a in &hooks {
            for b in &hooks {
                assert_eq!(
                    usize::from(a < b) + usize::from(a == b) + usize::from(a > b),
                    1,
                    "trichotomy failed for {a:?} vs {b:?}"
                );
                // `Ord` agrees with `PartialOrd`. The derive guarantees this;
                // the assertion guards a future hand-written impl.
                assert_eq!(Some(a.cmp(b)), a.partial_cmp(b), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn hook_point_keys_a_btree_map_deterministically() {
        let forward: BTreeMap<HookPoint, usize> = one_of_each_variant()
            .into_iter()
            .enumerate()
            .map(|(i, hook)| (hook, i))
            .collect();
        assert_eq!(forward.len(), one_of_each_variant().len());

        // Insertion order cannot reach a `BTreeMap`'s iteration order: the same
        // set inserted in reverse yields the same key sequence. This is the
        // property a determinism contract needs from `captures()`.
        let mut reversed = one_of_each_variant();
        reversed.reverse();
        let backward: BTreeMap<HookPoint, usize> = reversed
            .into_iter()
            .enumerate()
            .map(|(i, hook)| (hook, i))
            .collect();

        let forward_keys: Vec<&HookPoint> = forward.keys().collect();
        let backward_keys: Vec<&HookPoint> = backward.keys().collect();
        assert_eq!(forward_keys, backward_keys);
    }

    #[test]
    fn hook_point_custom_sorts_last() {
        let mut hooks = one_of_each_variant();
        hooks.sort();
        assert!(
            matches!(hooks.last(), Some(HookPoint::Custom(_))),
            "expected `Custom` last, got {:?}",
            hooks.last()
        );
    }

    #[test]
    fn hook_spec_capture_and_query() {
        let mut spec = HookSpec::new();
        assert!(spec.is_empty());

        spec.capture(HookPoint::AttnPattern(5));
        spec.capture("blocks.3.hook_resid_post");

        assert!(!spec.is_empty());
        assert_eq!(spec.num_captures(), 2);
        assert!(spec.is_captured(&HookPoint::AttnPattern(5)));
        assert!(spec.is_captured(&HookPoint::ResidPost(3)));
        assert!(!spec.is_captured(&HookPoint::Embed));
    }

    #[test]
    fn hook_spec_captures_lists_every_request() {
        let mut spec = HookSpec::new();
        spec.capture(HookPoint::AttnPattern(5));
        spec.capture("blocks.3.hook_resid_post");
        spec.capture(HookPoint::Embed);

        let listed: BTreeSet<&HookPoint> = spec.captures().collect();
        assert_eq!(listed.len(), spec.num_captures());
        assert!(listed.contains(&HookPoint::AttnPattern(5)));
        assert!(listed.contains(&HookPoint::ResidPost(3)));
        assert!(listed.contains(&HookPoint::Embed));
    }

    #[test]
    fn hook_spec_capture_all_and_from_iterator() {
        let wanted: Vec<HookPoint> = (0..4).map(HookPoint::ResidPost).collect();

        // Bulk form matches the one-at-a-time form.
        let mut one_by_one = HookSpec::new();
        for hook in &wanted {
            one_by_one.capture(hook.clone());
        }
        let mut bulk = HookSpec::new();
        bulk.capture_all(wanted.clone());
        assert_eq!(bulk.num_captures(), one_by_one.num_captures());
        for hook in &wanted {
            assert!(bulk.is_captured(hook), "{hook} missing after capture_all");
        }

        // `capture_all` takes anything `Into<HookPoint>`, like `capture`.
        let mut strings = HookSpec::new();
        strings.capture_all(["hook_embed", "hook_final_norm"]);
        assert!(strings.is_captured(&HookPoint::Embed));
        assert!(strings.is_captured(&HookPoint::FinalNorm));

        // `FromIterator` builds a capture-only spec with nothing else set.
        let collected: HookSpec = wanted.iter().cloned().collect();
        assert_eq!(collected.num_captures(), wanted.len());
        assert_eq!(collected.num_interventions(), 0);
        assert!(collected.state_knockout().is_none());
        assert!(collected.state_steering().is_none());
    }

    /// A `HookCache` holding one zero tensor per given hook point.
    fn cache_with(hooks: &[HookPoint]) -> HookCache {
        let placeholder =
            Tensor::zeros(1, DType::F32, &Device::Cpu).expect("failed to create placeholder");
        let mut cache = HookCache::new(placeholder);
        for (i, hook) in hooks.iter().enumerate() {
            let tensor =
                Tensor::zeros(i + 1, DType::F32, &Device::Cpu).expect("failed to create capture");
            cache.store(hook.clone(), tensor);
        }
        cache
    }

    #[test]
    fn hook_cache_captures_enumerates_everything_stored() {
        let stored = [
            HookPoint::ResidPost(2),
            HookPoint::ResidPost(0),
            HookPoint::AttnPattern(1),
        ];
        let cache = cache_with(&stored);

        let by_hook: BTreeMap<&HookPoint, &Tensor> = cache.captures().collect();
        assert_eq!(by_hook.len(), cache.num_captures());
        for hook in &stored {
            assert!(by_hook.contains_key(hook), "{hook} missing from captures()");
        }

        // Walking `captures()` replaces per-key absence probing: what comes out
        // is exactly what went in, so a caller needs no "hook not captured"
        // error path to enumerate a cache.
        for (hook, tensor) in cache.captures() {
            assert_eq!(
                tensor.dims(),
                cache
                    .require(hook)
                    .expect("enumerated hook must resolve")
                    .dims()
            );
        }
    }

    #[test]
    fn hook_cache_captures_collect_deterministically() {
        let forward = cache_with(&[
            HookPoint::ResidPost(2),
            HookPoint::ResidPost(0),
            HookPoint::AttnPattern(1),
        ]);
        let backward = cache_with(&[
            HookPoint::AttnPattern(1),
            HookPoint::ResidPost(0),
            HookPoint::ResidPost(2),
        ]);

        let forward_keys: Vec<&HookPoint> = forward
            .captures()
            .collect::<BTreeMap<_, _>>()
            .into_keys()
            .collect();
        let backward_keys: Vec<&HookPoint> = backward
            .captures()
            .collect::<BTreeMap<_, _>>()
            .into_keys()
            .collect();
        assert_eq!(forward_keys, backward_keys);
    }

    #[test]
    fn hook_cache_into_captures_yields_owned_tensors() {
        let stored = [HookPoint::Embed, HookPoint::FinalNorm];
        let cache = cache_with(&stored);
        let n = cache.num_captures();

        let owned: BTreeMap<HookPoint, Tensor> = cache.into_captures().collect();
        assert_eq!(owned.len(), n);
        for hook in &stored {
            assert!(
                owned.contains_key(hook),
                "{hook} missing from into_captures()"
            );
        }
    }

    #[test]
    fn hook_spec_intervention_query() {
        let mut spec = HookSpec::new();
        spec.intervene(HookPoint::AttnScores(5), Intervention::Zero);
        spec.intervene(HookPoint::AttnScores(5), Intervention::Scale(2.0));
        spec.intervene(HookPoint::ResidPost(10), Intervention::Zero);

        assert_eq!(spec.num_interventions(), 3);
        assert!(spec.has_intervention_at(&HookPoint::AttnScores(5)));
        assert!(!spec.has_intervention_at(&HookPoint::Embed));

        let at_5: Vec<_> = spec.interventions_at(&HookPoint::AttnScores(5)).collect();
        assert_eq!(at_5.len(), 2);
    }

    #[test]
    fn accepts_positional_patch_decides_every_variant() {
        // Every variant with the answer this crate commits to. Asserted below
        // to be exactly `one_of_each_variant()`, which `declaration_rank` keeps
        // complete, so a new `HookPoint` has to be given an answer here rather
        // than inheriting one.
        let table: Vec<(HookPoint, bool)> = vec![
            (HookPoint::Embed, true),
            (HookPoint::ResidPre(0), true),
            (HookPoint::AttnQ(1), false),
            (HookPoint::AttnK(1), false),
            (HookPoint::AttnV(1), false),
            (HookPoint::AttnScores(2), false),
            (HookPoint::AttnPattern(2), false),
            (HookPoint::AttnOut(3), true),
            (HookPoint::ResidMid(3), true),
            (HookPoint::MlpPre(4), true),
            (HookPoint::MlpPost(4), true),
            (HookPoint::MlpOut(5), true),
            (HookPoint::ResidPost(5), true),
            (HookPoint::FinalNorm, true),
            (HookPoint::RwkvState(6), false),
            (HookPoint::RwkvDecay(6), false),
            (HookPoint::RwkvEffectiveAttn(6), false),
            (HookPoint::Custom("some.custom.hook".to_string()), false),
        ];

        let listed: Vec<HookPoint> = table.iter().map(|(hook, _)| hook.clone()).collect();
        assert_eq!(
            listed,
            one_of_each_variant(),
            "the PatchAt policy table must list every HookPoint variant, in declaration order"
        );

        for (hook, accepted) in table {
            assert_eq!(
                hook.accepts_positional_patch(),
                accepted,
                "wrong PatchAt policy for {hook}"
            );
        }
    }

    /// `Intervention::PatchAt` behaviour, nested so one `cfg` gate covers the
    /// lot: `apply_intervention` is compiled only when a backend is enabled.
    #[cfg(any(
        feature = "transformer",
        feature = "rwkv",
        feature = "diffusion",
        feature = "stoicheia"
    ))]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod patch_at {
        use candle_core::{DType, Device, Tensor};

        use crate::error::MIError;
        use crate::hooks::{HookPoint, Intervention, apply_intervention};

        /// A `[1, 4, 3]` activation whose rows run `[0, 1, 2]` to `[9, 10, 11]`.
        fn resid() -> Tensor {
            Tensor::new(
                &[[
                    [0.0_f32, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                ]],
                &Device::Cpu,
            )
            .unwrap()
        }

        /// A `[2, 2, 2]` activation, for the batch cases.
        fn batched_resid() -> Tensor {
            Tensor::new(
                &[[[0.0_f32, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]],
                &Device::Cpu,
            )
            .unwrap()
        }

        /// The donor row `[100, 200, 300]`, shaped `[hidden]`.
        fn donor_row() -> Tensor {
            Tensor::new(&[100.0_f32, 200.0, 300.0], &Device::Cpu).unwrap()
        }

        /// Flatten to a `Vec<f32>` so whole activations compare in one assert.
        fn flat(tensor: &Tensor) -> Vec<f32> {
            tensor.flatten_all().unwrap().to_vec1().unwrap()
        }

        /// Apply one `PatchAt` and unwrap, for the cases expected to succeed.
        fn patch(base: &Tensor, point: &HookPoint, position: usize, value: Tensor) -> Tensor {
            apply_intervention(base, point, &Intervention::PatchAt { position, value }).unwrap()
        }

        /// Apply one `PatchAt` and unwrap the error, for the rejection cases.
        fn patch_err(base: &Tensor, point: &HookPoint, position: usize, value: Tensor) -> MIError {
            apply_intervention(base, point, &Intervention::PatchAt { position, value })
                .expect_err("expected PatchAt to be rejected")
        }

        #[test]
        fn patches_only_the_target_position() {
            let base = resid();
            let patched = patch(&base, &HookPoint::ResidPost(3), 2, donor_row());

            assert_eq!(patched.dims(), base.dims());
            assert_eq!(
                flat(&patched),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0, 9.0, 10.0, 11.0
                ]
            );
        }

        #[test]
        fn the_result_is_contiguous() {
            // The patched activation flows straight on into the rest of the
            // forward pass, so the documented `# Shapes` promise that it comes
            // back contiguous is a contract, not an implementation detail.
            let patched = patch(&resid(), &HookPoint::ResidPost(0), 0, donor_row());
            assert!(patched.is_contiguous());
        }

        #[test]
        fn a_bare_and_a_unit_value_agree() {
            let base = resid();
            let unit = donor_row().reshape((1, 1, 3)).unwrap();

            let from_bare = flat(&patch(&base, &HookPoint::ResidMid(1), 1, donor_row()));
            let from_unit = flat(&patch(&base, &HookPoint::ResidMid(1), 1, unit));

            assert_eq!(from_bare, from_unit);
        }

        #[test]
        fn a_bare_value_broadcasts_across_the_batch() {
            let value = Tensor::new(&[9.0_f32, 9.0], &Device::Cpu).unwrap();
            let patched = patch(&batched_resid(), &HookPoint::ResidPost(0), 0, value);

            assert_eq!(flat(&patched), vec![9.0, 9.0, 2.0, 3.0, 9.0, 9.0, 6.0, 7.0]);
        }

        #[test]
        fn a_batched_value_gives_each_row_its_own_replacement() {
            let value = Tensor::new(&[[[9.0_f32, 9.0]], [[8.0, 8.0]]], &Device::Cpu).unwrap();
            let patched = patch(&batched_resid(), &HookPoint::ResidPost(0), 0, value);

            assert_eq!(flat(&patched), vec![9.0, 9.0, 2.0, 3.0, 8.0, 8.0, 6.0, 7.0]);
        }

        #[test]
        fn converts_a_mismatched_value_dtype() {
            let value = donor_row().to_dtype(DType::F64).unwrap();
            let patched = patch(&resid(), &HookPoint::ResidPost(0), 0, value);

            assert_eq!(patched.dtype(), DType::F32);
            assert_eq!(
                flat(&patched),
                vec![
                    100.0, 200.0, 300.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0
                ]
            );
        }

        #[test]
        fn patches_from_a_donor_row_that_is_an_offset_view() {
            // `FullActivationCache::get_position` hands back
            // `donor.narrow(0, p, 1)?.squeeze(0)?`, a view whose storage offset
            // is non-zero and whose storage holds the *whole* donor activation.
            // A freshly built value tensor does not exercise that.
            let base = resid();
            let donor = Tensor::new(
                &[
                    [900.0_f32, 901.0, 902.0],
                    [910.0, 911.0, 912.0],
                    [920.0, 921.0, 922.0],
                    [930.0, 931.0, 932.0],
                ],
                &Device::Cpu,
            )
            .unwrap();
            let donor_row_2 = donor.narrow(0, 2, 1).unwrap().squeeze(0).unwrap();

            let patched = patch(&base, &HookPoint::ResidPost(0), 2, donor_row_2);

            assert_eq!(
                flat(&patched),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 920.0, 921.0, 922.0, 9.0, 10.0, 11.0
                ]
            );
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_patches_from_a_donor_row_that_is_an_offset_view() {
            let device = Device::new_cuda(0).expect("no CUDA device");
            let (seq_len, hidden) = (6_usize, 2048_usize);
            let base = Tensor::rand(0.0_f32, 1.0_f32, (1, seq_len, hidden), &device).unwrap();
            let donor = Tensor::rand(0.0_f32, 1.0_f32, (seq_len, hidden), &device).unwrap();
            let donor_row_4 = donor.narrow(0, 4, 1).unwrap().squeeze(0).unwrap();

            let patched = apply_intervention(
                &base,
                &HookPoint::ResidPost(15),
                &Intervention::PatchAt {
                    position: 4,
                    value: donor_row_4,
                },
            )
            .unwrap();

            for pos in 0..seq_len {
                let got: Vec<f32> = patched.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap();
                let want: Vec<f32> = if pos == 4 {
                    donor.get(4).unwrap().to_vec1().unwrap()
                } else {
                    base.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap()
                };
                assert_eq!(got, want, "row {pos} differs");
            }
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_patches_only_the_target_position_at_model_shape() {
            // Llama-3.2-1B's residual stream shape for a 6-token prompt.
            let device = Device::new_cuda(0).expect("no CUDA device");
            let (seq_len, hidden) = (6_usize, 2048_usize);
            let base = Tensor::rand(0.0_f32, 1.0_f32, (1, seq_len, hidden), &device).unwrap();
            let value = Tensor::rand(0.0_f32, 1.0_f32, hidden, &device).unwrap();

            let patched = apply_intervention(
                &base,
                &HookPoint::ResidPost(15),
                &Intervention::PatchAt {
                    position: 4,
                    value: value.clone(),
                },
            )
            .unwrap();

            // Every row but 4 must be untouched, and row 4 must be `value`.
            for pos in 0..seq_len {
                let got: Vec<f32> = patched.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap();
                let want: Vec<f32> = if pos == 4 {
                    value.to_vec1().unwrap()
                } else {
                    base.get(0).unwrap().get(pos).unwrap().to_vec1().unwrap()
                };
                assert_eq!(got, want, "row {pos} differs");
            }
        }

        #[test]
        #[ignore = "requires a CUDA device"]
        fn cuda_patches_only_the_target_position() {
            let device = Device::new_cuda(0).expect("no CUDA device");
            let base = Tensor::new(
                &[[
                    [0.0_f32, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                ]],
                &device,
            )
            .unwrap();
            let value = Tensor::new(&[100.0_f32, 200.0, 300.0], &device).unwrap();

            let patched = apply_intervention(
                &base,
                &HookPoint::ResidPost(3),
                &Intervention::PatchAt { position: 2, value },
            )
            .unwrap();

            assert_eq!(
                flat(&patched),
                vec![
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0, 300.0, 9.0, 10.0, 11.0
                ]
            );
        }

        #[test]
        fn rejects_a_position_past_the_end() {
            let err = patch_err(&resid(), &HookPoint::ResidPost(0), 4, donor_row());

            assert!(matches!(err, MIError::Intervention(_)), "got {err:?}");
            assert!(err.to_string().contains("seq_len=4"), "{err}");
        }

        #[test]
        fn rejects_every_attention_hook_point() {
            // The regression this policy exists for. Dim 1 is a head at all
            // five, so a positional write would overwrite a head and return a
            // plausible figure. The activation passed here is rank 3, so the
            // rejection is by hook point and not by luck of shape.
            for point in [
                HookPoint::AttnQ(0),
                HookPoint::AttnK(0),
                HookPoint::AttnV(0),
                HookPoint::AttnScores(0),
                HookPoint::AttnPattern(0),
            ] {
                let err = patch_err(&resid(), &point, 1, donor_row());

                assert!(
                    matches!(err, MIError::Intervention(_)),
                    "{point}: got {err:?}"
                );
                assert!(
                    err.to_string().contains("accepts_positional_patch"),
                    "{point}: {err}"
                );
            }
        }

        #[test]
        fn rejects_a_custom_hook_point() {
            let point = HookPoint::Custom("some.backend.hook".to_string());
            let err = patch_err(&resid(), &point, 1, donor_row());

            assert!(matches!(err, MIError::Intervention(_)), "got {err:?}");
        }

        #[test]
        fn rejects_a_rank_four_activation_at_an_accepting_point() {
            // Defence in depth behind the hook-point policy: a backend storing
            // an unexpected rank must error rather than write at the wrong axis.
            let base = Tensor::zeros((1, 2, 2, 2), DType::F32, &Device::Cpu).unwrap();
            let value = Tensor::new(&[9.0_f32, 9.0], &Device::Cpu).unwrap();
            let err = patch_err(&base, &HookPoint::ResidPost(0), 0, value);

            assert!(err.to_string().contains("rank-3"), "{err}");
        }

        #[test]
        fn rejects_a_value_of_the_wrong_shape() {
            let value = Tensor::new(&[1.0_f32, 2.0], &Device::Cpu).unwrap();
            let err = patch_err(&resid(), &HookPoint::ResidPost(0), 0, value);

            assert!(matches!(err, MIError::Intervention(_)), "got {err:?}");
            assert!(err.to_string().contains("expected [hidden]"), "{err}");
        }
    }
}
