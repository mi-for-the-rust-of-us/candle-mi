// SPDX-License-Identifier: MIT OR Apache-2.0

//! The closed-loop **replacement model** of a cross-layer transcoder, and the faithfulness
//! readings taken from it.
//!
//! A transcoder explains a model only as far as the model, run with its MLP outputs REPLACED by
//! the transcoder's reconstruction, still does what the model does. Reconstruction error (`FVU`)
//! does not measure that: a transcoder can reconstruct each layer well and still change the
//! model's output, because errors compound through the layers above. This module runs the
//! replacement and reads the output.
//!
//! # Closed loop
//!
//! [`Splice::Replaced`] runs one pass per layer. Pass `t` runs the model with layers `< t`
//! already replaced, captures the residual stream entering layer `t`'s MLP (`ResidMid(t)`), and
//! decodes layer `t`'s replacement from features encoded at those SPLICED residuals; a last pass
//! replaces every layer. So the transcoder at layer `t` reads what the replacement model itself
//! feeds it, as the replacement model of "On the Biology of a Large Language Model" does — not
//! what the clean model would have. A reconstruction of layer `t` reads only source layers
//! `<= t`, so the inputs of the layers above, still zero at pass `t`, do not touch it.
//! [`Splice::Ablated`] zeroes every MLP output: the floor that [`loss_recovered`] measures
//! against.
//!
//! # Confined to a domain
//!
//! A transcoder trained on some positions only -- a grid's non-padding cells, say -- is not a model
//! of the others. [`replacement_logits_where`] runs the same closed loop with the splice confined to
//! marked positions; elsewhere every layer keeps the model's own MLP output, computed on the spliced
//! residual. Splicing the unmarked positions too measures the reconstruction outside its domain,
//! which attention carries to every position.
//!
//! # The forward is the caller's
//!
//! [`replacement_logits`] takes the model's forward as a closure, because what counts as the
//! model's plain forward is the caller's decision: a self-conditioned `OthelloGpt`, for example,
//! may be run without a carry ([`MIBackend::forward`](crate::MIBackend::forward)) or with a
//! "no carry yet" grid, and the two are different models of the same weights.
//! [`replacement_logits_for`] is the [`MIBackend::forward`](crate::MIBackend::forward) case.
//!
//! # Units
//!
//! [`Reconstruct`] works in the model's RAW activation units. [`ScaledClt`] adapts a
//! [`TrainableClt`] trained on per-layer-scaled activations, and runs it detached and in chunks
//! of [`CHUNK`] positions, so an evaluation over a whole batch neither records a backward graph
//! nor holds one giant activation tensor.

use candle_core::{D, DType, Tensor};

use super::train::TrainableClt;
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec, Intervention};

/// Positions per chunk when a transcoder runs over a whole batch: bounds the peak to a few
/// hundred MB (`2,048 x 6 x 4,096` `F32` activations is ~200 MB) whatever the batch size.
pub const CHUNK: usize = 2048;

/// A transcoder seen from the replacement model: every layer's MLP output, reconstructed from
/// every layer's MLP input, in the model's raw units and without a backward graph.
pub trait Reconstruct {
    /// Number of layers read and written.
    fn n_layers(&self) -> usize;

    /// The reconstruction of every layer's MLP output.
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]` -- each layer's MLP input (`ResidMid`), raw units
    /// - returns: `[n_positions, n_layers, d_model]` -- each layer's reconstructed MLP output
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` has the wrong shape, [`MIError::Model`] on tensor
    /// failure.
    fn reconstruct(&self, x: &Tensor) -> Result<Tensor>;
}

/// A [`TrainableClt`] seen in raw units.
///
/// The transcoder was trained on per-layer-scaled activations (`x / scale_in[l]`, `target /
/// scale_out[l]`): inputs are divided by `scale_in`, reconstructions multiplied by `scale_out`,
/// and it runs detached and in chunks of [`CHUNK`] positions.
pub struct ScaledClt<'a> {
    /// The transcoder.
    clt: &'a TrainableClt,
    /// Per-layer input scales.
    scale_in: Vec<f32>,
    /// Per-layer output scales.
    scale_out: Vec<f32>,
}

impl<'a> ScaledClt<'a> {
    /// A transcoder with its training scales.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a scale slice does not hold one positive, finite value per
    /// layer.
    pub fn new(clt: &'a TrainableClt, scale_in: &[f32], scale_out: &[f32]) -> Result<Self> {
        let layers = clt.config().n_layers;
        for (name, scales) in [("scale_in", scale_in), ("scale_out", scale_out)] {
            if scales.len() != layers || scales.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
                return Err(MIError::Config(format!(
                    "{name} must hold {layers} positive finite values (got {scales:?})"
                )));
            }
        }
        Ok(Self {
            clt,
            scale_in: scale_in.to_vec(),
            scale_out: scale_out.to_vec(),
        })
    }

    /// A transcoder trained on raw activations (every scale 1).
    #[must_use]
    pub fn unscaled(clt: &'a TrainableClt) -> Self {
        let layers = clt.config().n_layers;
        Self {
            clt,
            scale_in: vec![1.0; layers],
            scale_out: vec![1.0; layers],
        }
    }

    /// The reconstruction in raw units, with which (layer, feature) pairs fired at least once.
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]`, raw units
    /// - returns: (`[n_positions, n_layers, d_model]` raw reconstruction, `[n_layers, n_features]`
    ///   0/1 fired mask)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` has the wrong shape, [`MIError::Model`] on tensor
    /// failure.
    pub fn reconstruct_and_fired(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let c = self.clt.config();
        let device = x.device();
        let s_in = Tensor::from_slice(&self.scale_in, (1, c.n_layers, 1), device)?;
        let s_out = Tensor::from_slice(&self.scale_out, (1, c.n_layers, 1), device)?;
        let total = x.dim(0)?;
        let mut pieces = Vec::new();
        let mut fired = Tensor::zeros((c.n_layers, c.n_features), DType::F32, device)?;
        // EXPLICIT: `fired` is a running maximum over chunks; a fold would hide the state
        for start in (0..total).step_by(CHUNK) {
            let chunk = x
                .narrow(0, start, CHUNK.min(total - start))?
                .broadcast_div(&s_in)?;
            let (recon, acts) = self.clt.forward_detached(&chunk)?;
            fired = fired.maximum(&acts.gt(0.0)?.to_dtype(DType::F32)?.max(0)?)?;
            pieces.push(recon.broadcast_mul(&s_out)?);
        }
        Ok((Tensor::cat(&pieces, 0)?, fired))
    }
}

impl Reconstruct for ScaledClt<'_> {
    fn n_layers(&self) -> usize {
        self.clt.config().n_layers
    }

    fn reconstruct(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.reconstruct_and_fired(x)?.0)
    }
}

/// How [`replacement_logits`] treats the model's MLP outputs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Splice {
    /// The model as it is.
    Clean,
    /// Every MLP output replaced by the transcoder's closed-loop reconstruction.
    Replaced,
    /// Every MLP output zeroed: the floor of [`loss_recovered`].
    Ablated,
}

/// The model's logits with its MLP outputs treated as `splice` says.
///
/// Closed loop for [`Splice::Replaced`] (see the [module documentation](self)). `forward` runs
/// the model on the caller's input under a hook spec — `|hooks| model.forward(&ids, hooks)` for a
/// [`MIBackend`](crate::MIBackend), or whatever the caller's plain forward is.
///
/// # Shapes
/// - returns: the forward's output, `[batch, seq, vocab]` for a language model
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the forward does not provide `ResidMid(t)` for a layer.
/// Returns [`MIError::Model`] on a tensor operation failure.
/// Returns the forward's and the transcoder's own errors when either fails.
pub fn replacement_logits<F>(forward: F, clt: &impl Reconstruct, splice: Splice) -> Result<Tensor>
where
    F: Fn(&HookSpec) -> Result<HookCache>,
{
    let layers = clt.n_layers();
    let mut spec = HookSpec::new();
    match splice {
        Splice::Clean => {}
        Splice::Ablated => {
            for l in 0..layers {
                spec.intervene(HookPoint::MlpOut(l), Intervention::Zero);
            }
        }
        Splice::Replaced => {
            let mut replaced: Vec<Tensor> = Vec::with_capacity(layers);
            let mut inputs: Option<Tensor> = None;
            // EXPLICIT: pass t needs the replacements of passes < t, a sequential dependency
            for t in 0..layers {
                let mut pass = HookSpec::new();
                pass.capture(HookPoint::ResidMid(t));
                for (l, r) in replaced.iter().enumerate() {
                    pass.intervene(HookPoint::MlpOut(l), Intervention::Replace(r.clone()));
                }
                let cache = forward(&pass)?;
                let resid = cache.get(&HookPoint::ResidMid(t)).ok_or_else(|| {
                    MIError::Hook(format!("hook point ResidMid({t}) not captured"))
                })?;
                let (batch, seq, d) = resid.dims3()?;
                let rows = batch * seq;
                // PROMOTE: the transcoder reads F32 whatever dtype the model runs in
                let column = resid.to_dtype(DType::F32)?.reshape((rows, 1, d))?;
                let base = match inputs.take() {
                    Some(x) => x,
                    None => Tensor::zeros((rows, layers, d), DType::F32, resid.device())?,
                };
                let x = base.slice_assign(&[0..rows, t..t + 1, 0..d], &column)?;
                let layer = clt.reconstruct(&x)?.narrow(1, t, 1)?;
                replaced.push(layer.reshape((batch, seq, d))?.to_dtype(resid.dtype())?);
                inputs = Some(x);
            }
            for (l, r) in replaced.into_iter().enumerate() {
                spec.intervene(HookPoint::MlpOut(l), Intervention::Replace(r));
            }
        }
    }
    Ok(forward(&spec)?.output().clone())
}

/// [`replacement_logits`] with the backend's own forward
/// ([`MIBackend::forward`](crate::MIBackend::forward)).
///
/// # Shapes
/// - `input_ids`: `[batch, seq]`
/// - returns: `[batch, seq, vocab]`
///
/// # Errors
///
/// As [`replacement_logits`].
pub fn replacement_logits_for(
    backend: &impl crate::MIBackend,
    input_ids: &Tensor,
    clt: &impl Reconstruct,
    splice: Splice,
) -> Result<Tensor> {
    replacement_logits(|hooks| backend.forward(input_ids, hooks), clt, splice)
}

/// [`replacement_logits`] with the splice confined to the positions `at` marks: elsewhere every
/// layer runs the model's own MLP, on the residual the splice has produced.
///
/// A transcoder trained on some positions only -- a grid's non-padding cells, say -- is not a
/// model of the others, and splicing them anyway measures its reconstruction outside its domain
/// (which, through attention, reaches every position). `at` marks the domain. Both
/// [`Splice::Replaced`] and [`Splice::Ablated`] run closed loop here: pass `t` captures
/// `ResidMid(t)` and the model's own `MlpOut(t)` with the layers below already treated, and layer
/// `t` becomes the treated output where `at` is nonzero and the model's own where it is zero.
/// [`Splice::Clean`] ignores `at`. With every position marked, this is [`replacement_logits`];
/// with none, the clean model.
///
/// # Shapes
/// - `at`: `[batch, seq]`, any dtype -- nonzero where the transcoder splices
/// - returns: the forward's output, `[batch, seq, vocab]` for a language model
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the forward does not provide `ResidMid(t)` or `MlpOut(t)` for a
/// layer.
/// Returns [`MIError::Config`] if `at` is not `[batch, seq]` of the residual stream.
/// Returns [`MIError::Model`] on a tensor operation failure.
/// Returns the forward's and the transcoder's own errors when either fails.
pub fn replacement_logits_where<F>(
    forward: F,
    clt: &impl Reconstruct,
    splice: Splice,
    at: &Tensor,
) -> Result<Tensor>
where
    F: Fn(&HookSpec) -> Result<HookCache>,
{
    if splice == Splice::Clean {
        return Ok(forward(&HookSpec::new())?.output().clone());
    }
    let layers = clt.n_layers();
    let mut treated: Vec<Tensor> = Vec::with_capacity(layers);
    let mut inputs: Option<Tensor> = None;
    // EXPLICIT: pass t needs the treated layers of passes < t, a sequential dependency
    for t in 0..layers {
        let mut pass = HookSpec::new();
        pass.capture(HookPoint::ResidMid(t))
            .capture(HookPoint::MlpOut(t));
        for (l, r) in treated.iter().enumerate() {
            pass.intervene(HookPoint::MlpOut(l), Intervention::Replace(r.clone()));
        }
        let cache = forward(&pass)?;
        let captured = |hook: HookPoint| {
            cache
                .get(&hook)
                .cloned()
                .ok_or_else(|| MIError::Hook(format!("hook point {hook:?} not captured")))
        };
        let resid = captured(HookPoint::ResidMid(t))?;
        let own = captured(HookPoint::MlpOut(t))?;
        let spliced = splice_mask(at, &resid)?;
        let layer = if splice == Splice::Replaced {
            let (layer, x) = reconstruct_layer(clt, &resid, t, inputs.take())?;
            inputs = Some(x);
            layer
        } else {
            resid.zeros_like()?
        };
        // A select, not a 0/1 product: an out-of-domain reconstruction that is not finite must
        // not reach the kept positions as `0 * inf = NaN`.
        treated.push(spliced.where_cond(&layer, &own)?);
    }
    let mut spec = HookSpec::new();
    for (l, r) in treated.into_iter().enumerate() {
        spec.intervene(HookPoint::MlpOut(l), Intervention::Replace(r));
    }
    Ok(forward(&spec)?.output().clone())
}

/// The splice mask of [`replacement_logits_where`]: `1` where the transcoder splices, `0` where the
/// model keeps its own MLP, as `U8` broadcast to the residual's `[batch, seq, d_model]`.
fn splice_mask(at: &Tensor, resid: &Tensor) -> Result<Tensor> {
    let (batch, seq, _) = resid.dims3()?;
    if at.dims() != [batch, seq] {
        return Err(MIError::Config(format!(
            "splice mask shape {:?} mismatched (expected [{batch}, {seq}], the residual stream's)",
            at.dims()
        )));
    }
    // PROMOTE: the mask may come in any dtype; compared in F32, it becomes the U8 a select takes
    Ok(at
        .to_dtype(DType::F32)?
        .ne(0.0)?
        .unsqueeze(2)?
        .broadcast_as(resid.shape())?)
}

/// Layer `t`'s closed-loop reconstruction from the spliced residual `resid` (`ResidMid(t)`), with
/// the inputs gathered by the passes below (`None` at the first); returns it in the residual's
/// shape and dtype, and the inputs with column `t` filled.
fn reconstruct_layer(
    clt: &impl Reconstruct,
    resid: &Tensor,
    t: usize,
    inputs: Option<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, seq, d) = resid.dims3()?;
    let rows = batch * seq;
    // PROMOTE: the transcoder reads F32 whatever dtype the model runs in
    let column = resid.to_dtype(DType::F32)?.reshape((rows, 1, d))?;
    let base = match inputs {
        Some(x) => x,
        None => Tensor::zeros((rows, clt.n_layers(), d), DType::F32, resid.device())?,
    };
    let x = base.slice_assign(&[0..rows, t..t + 1, 0..d], &column)?;
    let layer = clt
        .reconstruct(&x)?
        .narrow(1, t, 1)?
        .reshape((batch, seq, d))?
        .to_dtype(resid.dtype())?;
    Ok((layer, x))
}

/// Loss recovered by a replacement: `1 - (spliced - clean) / (ablated - clean)`, where each is the
/// cross-entropy at the same positions. 1 is perfect; 0 is no better than zeroing the MLPs.
///
/// ```
/// use candle_mi::clt::replacement::loss_recovered;
/// assert_eq!(loss_recovered(0.5, 0.5, 2.5), 1.0);
/// assert_eq!(loss_recovered(0.5, 1.0, 2.5), 0.75);
/// assert_eq!(loss_recovered(0.5, 2.5, 2.5), 0.0);
/// ```
#[must_use]
pub fn loss_recovered(clean: f64, spliced: f64, ablated: f64) -> f64 {
    1.0 - (spliced - clean) / (ablated - clean)
}

/// Fraction of variance unexplained, per layer: the mean squared error over (positions, width)
/// divided by the target's variance around its per-layer mean.
///
/// # Shapes
/// - `recon`: `[n_positions, n_layers, d_model]`
/// - `target`: `[n_positions, n_layers, d_model]`
/// - returns: one value per layer
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure (including a shape mismatch).
///
/// ```
/// use candle_core::{Device, Tensor};
/// use candle_mi::clt::replacement::fvu_per_layer;
///
/// let target = Tensor::new(&[[[1.0f32, 3.0]], [[3.0, 1.0]]], &Device::Cpu)?; // [2, 1, 2]
/// let mean = Tensor::new(&[[[2.0f32, 2.0]], [[2.0, 2.0]]], &Device::Cpu)?;
/// assert_eq!(fvu_per_layer(&target, &target)?, [0.0]);
/// assert_eq!(fvu_per_layer(&mean, &target)?, [1.0]);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
pub fn fvu_per_layer(recon: &Tensor, target: &Tensor) -> Result<Vec<f32>> {
    let err = (recon - target)?.sqr()?.mean(0)?.mean(D::Minus1)?;
    let centred = target.broadcast_sub(&target.mean_keepdim(0)?)?;
    let var = centred.sqr()?.mean(0)?.mean(D::Minus1)?;
    Ok(err.div(&var)?.to_vec1::<f32>()?)
}

/// The logits at `positions` (flat indices into `batch x seq`), as `[n_positions, vocab]`.
fn logits_at(logits: &Tensor, positions: &[u32]) -> Result<Tensor> {
    let vocab = logits.dim(D::Minus1)?;
    let index = Tensor::from_slice(positions, positions.len(), logits.device())?;
    // PROMOTE: the read-out is taken in F32 whatever dtype the model runs in
    Ok(logits
        .to_dtype(DType::F32)?
        .reshape(((), vocab))?
        .index_select(&index, 0)?)
}

/// Mean cross-entropy of `targets` at `positions` (flat indices into `batch x seq`).
///
/// # Shapes
/// - `logits`: `[batch, seq, vocab]`
///
/// # Errors
///
/// Returns [`MIError::Config`] if `positions` and `targets` differ in length, [`MIError::Model`]
/// on tensor failure (including an index out of range).
///
/// ```
/// use candle_core::{Device, Tensor};
/// use candle_mi::clt::replacement::cross_entropy_at;
///
/// // Two positions, vocabulary of 2; uniform logits give ln 2 whatever the target.
/// let logits = Tensor::zeros((1, 2, 2), candle_core::DType::F32, &Device::Cpu)?;
/// let ce = cross_entropy_at(&logits, &[0, 1], &[1, 0])?;
/// assert!((ce - std::f64::consts::LN_2).abs() < 1e-6);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
pub fn cross_entropy_at(logits: &Tensor, positions: &[u32], targets: &[u32]) -> Result<f64> {
    if positions.len() != targets.len() {
        return Err(MIError::Config(format!(
            "positions and targets differ in length ({} vs {})",
            positions.len(),
            targets.len()
        )));
    }
    let at = logits_at(logits, positions)?;
    // A terminal read-out: no gradient is ever wanted here, so the fused kernel is fine.
    let logp = candle_nn::ops::log_softmax(&at, D::Minus1)?;
    let target = Tensor::from_slice(targets, (targets.len(), 1), logits.device())?;
    let ce = logp.gather(&target, 1)?.neg()?.mean_all()?;
    Ok(ce.to_dtype(DType::F64)?.to_scalar::<f64>()?)
}

/// The top token at each of `positions` (flat indices into `batch x seq`).
///
/// # Shapes
/// - `logits`: `[batch, seq, vocab]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure (including an index out of range).
///
/// ```
/// use candle_core::{Device, Tensor};
/// use candle_mi::clt::replacement::top_tokens_at;
///
/// let logits = Tensor::new(&[[[0.0f32, 2.0, 1.0], [3.0, 0.0, 0.0]]], &Device::Cpu)?;
/// assert_eq!(top_tokens_at(&logits, &[1, 0])?, [0, 1]);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
pub fn top_tokens_at(logits: &Tensor, positions: &[u32]) -> Result<Vec<u32>> {
    Ok(logits_at(logits, positions)?
        .argmax(D::Minus1)?
        .to_vec1::<u32>()?)
}

/// Share of (layer, feature) pairs that never fired.
///
/// From a 0/1 fired mask such as [`ScaledClt::reconstruct_and_fired`] returns. A feature that
/// fires on none of the evaluated positions counts as dead, so a small evaluation overcounts rare
/// features.
///
/// # Shapes
/// - `fired`: `[n_layers, n_features]`, 0 or 1
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
///
/// ```
/// use candle_core::{Device, Tensor};
/// use candle_mi::clt::replacement::dead_fraction;
///
/// let fired = Tensor::new(&[[1.0f32, 0.0, 0.0, 1.0]], &Device::Cpu)?;
/// assert_eq!(dead_fraction(&fired)?, 0.5);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
pub fn dead_fraction(fired: &Tensor) -> Result<f64> {
    let live = fired.to_dtype(DType::F64)?.mean_all()?.to_scalar::<f64>()?;
    Ok(1.0 - live)
}
