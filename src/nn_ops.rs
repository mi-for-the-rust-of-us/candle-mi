// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backward-safe wrappers around `candle_nn`'s fused kernels.
//!
//! `candle_nn::ops::softmax_last_dim` and the fused `layer_norm` / `rms_norm`
//! are built with `apply_op*_no_bwd`: their output records **no** backprop op,
//! so it is a graph *leaf*.  `backward()` reaches it, finds nothing above it,
//! and silently stops — every parameter upstream trains as if frozen, with no
//! error, no warning, and a loss that still decreases (see
//! `docs/dogfooding-feedbacks/trainable-backbones.md`).
//!
//! Each helper here dispatches on [`Tensor::track_op`]: an inference forward
//! (no `Var` anywhere upstream) takes candle's fused kernel, byte-identical to
//! calling it directly, so every existing parity baseline keeps its meaning; a
//! forward under a `VarMap` takes the composed form, which carries a backward.
//! The predicate is sound by construction — candle records a `BackpropOp` only
//! when an input already tracks, so a pure-inference forward never starts
//! tracking.  The only cost is one boolean test per call site.

use std::sync::OnceLock;

use candle_core::{D, DType, Module, Tensor, Var};
use candle_nn::{LayerNorm, RmsNorm};

use crate::error::Result;

/// Whether the installed `candle-nn`'s fused `softmax_last_dim` records a backward op.
static FUSED_SOFTMAX_DIFFERENTIABLE: OnceLock<bool> = OnceLock::new();
/// Whether the installed `candle-nn`'s fused `layer_norm` records a backward op.
static FUSED_LAYER_NORM_DIFFERENTIABLE: OnceLock<bool> = OnceLock::new();
/// Whether the installed `candle-nn`'s fused `rotary_emb::rope` records a backward op.
static FUSED_ROPE_DIFFERENTIABLE: OnceLock<bool> = OnceLock::new();

/// Probes, ONCE per process, whether a fused `candle-nn` op carries gradients.
///
/// Stock `candle-nn` builds its fused `softmax_last_dim` / `layer_norm` with
/// `apply_op*_no_bwd`: `backward()` returns `Ok` and the input simply never appears in the
/// gradient store — the C1 failure, a silent wrong answer with a green light. A patched or
/// future `candle-nn` whose fused ops implement `CustomOp::bwd` carries them. **Which world
/// this process is in cannot be known at compile time** (the crate is version-ranged), so it
/// is measured: a four-element CPU graph, one forward, one backward, one lookup. The result
/// decides the dispatch below for the life of the process; the cost is microseconds, once.
///
/// This is the difference between "fast when the runtime supports it" and "wrong when it
/// doesn't": with the probe, a training run on stock `candle-nn` silently takes the composed
/// path and stays CORRECT, and the same binary against a bwd-carrying `candle-nn` takes the
/// fused kernels and their analytic backward.
fn fused_carries_gradients(cell: &OnceLock<bool>, run: fn(&Tensor) -> Result<Tensor>) -> bool {
    *cell.get_or_init(|| {
        let probe = || -> Result<bool> {
            let x = Tensor::new(&[[0.1_f32, 0.2, 0.3, 0.4]], &candle_core::Device::Cpu)?;
            let v = Var::from_tensor(&x)?;
            let y = run(v.as_tensor())?;
            let grads = y.sum_all()?.backward()?;
            Ok(grads.get(&v).is_some())
        };
        probe().unwrap_or(false)
    })
}

/// The softmax probe body: the fused op, applied to the tracked probe tensor.
fn probe_softmax(xs: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::softmax_last_dim(xs)?)
}

/// The `RoPE` probe body: the fused op on a minimal `[1, 1, seq, head_dim]` input.
///
/// The shared prober hands a `[1, 4]` tensor, which `rope` cannot take, so this
/// reshapes it to the rank-4 layout and builds `cos`/`sin` of the matching
/// `[seq, head_dim / 2]` shape. Only op registration is under test, so the
/// angle values are arbitrary.
fn probe_rope(xs: &Tensor) -> Result<Tensor> {
    let device = xs.device();
    // [1, 4] -> [1, 1, 2, 2]: batch 1, one head, seq 2, head_dim 2.
    let x = xs.reshape((1, 1, 2, 2))?;
    let cos = Tensor::ones((2, 1), DType::F32, device)?;
    let sin = Tensor::zeros((2, 1), DType::F32, device)?;
    Ok(candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)?)
}

/// The layer-norm probe body: the fused op with weight and bias, the gradient-barrier
/// configuration (weight-only `LayerNorm` never took the fused kernel in the first place).
fn probe_layer_norm(xs: &Tensor) -> Result<Tensor> {
    let hidden = xs.dim(D::Minus1)?;
    let device = xs.device();
    let weight = Tensor::ones(hidden, DType::F32, device)?;
    let bias = Tensor::zeros(hidden, DType::F32, device)?;
    // EXPLICIT: eps value is irrelevant to the probe; only op registration is under test.
    Ok(candle_nn::ops::layer_norm(xs, &weight, &bias, 1e-5)?)
}

/// Softmax over the last dimension, differentiable when the graph is tracked.
///
/// Dispatches on [`Tensor::track_op`]: an inference forward takes candle's
/// fused kernel, unchanged; a forward under a `VarMap` takes the composed
/// form, which carries a backward.  Both subtract the row maximum before
/// exponentiating, so the two paths agree to `F32` rounding.
///
/// # Shapes
/// - `xs`: `[.., n]` -- softmax is taken over the final axis
/// - returns: `[.., n]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    if xs.track_op() && !fused_carries_gradients(&FUSED_SOFTMAX_DIFFERENTIABLE, probe_softmax) {
        Ok(candle_nn::ops::softmax(xs, D::Minus1)?)
    } else {
        Ok(candle_nn::ops::softmax_last_dim(xs)?)
    }
}

/// Softmax over the last dimension, computed in `F32` and returned at the
/// input's dtype.
///
/// The attention pattern of every backbone in the crate is produced this way:
/// a softmax over `BF16` or `F16` scores can overflow to `NaN`, so the reduction
/// runs at `F32` and the result is cast back. Three backbones carried their own
/// copy of this promote/softmax/demote sequence
/// (`diffusion::mdlm`, `diffusion::othello`, `transformer::attention`); the
/// sequence is identical in all three, so this is a pure extraction and the
/// arithmetic is unchanged.
///
/// Dispatches through [`softmax_last_dim`], so it stays backward-safe under a
/// `VarMap` and byte-identical to the fused kernel for inference.
///
/// # Shapes
/// - `scores`: any shape; the softmax is over the last dimension.
/// - returns: same shape and dtype as `scores`.
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn softmax_last_dim_f32(scores: &Tensor) -> Result<Tensor> {
    let original_dtype = scores.dtype();
    if original_dtype == DType::F32 {
        return softmax_last_dim(scores);
    }
    // PROMOTE: softmax over a lower-precision dtype can produce NaN; compute in F32
    let pattern = softmax_last_dim(&scores.to_dtype(DType::F32)?)?;
    Ok(pattern.to_dtype(original_dtype)?)
}

/// `LayerNorm` forward, differentiable when the graph is tracked.
///
/// `candle_nn::LayerNorm::forward` takes the fused (no-backward) kernel
/// exactly when the input is contiguous, the norm removes the mean, **and** a
/// bias is present — so weight-only `LayerNorm` is accidentally differentiable
/// while with-bias `LayerNorm` is not, invisibly at the call site.  This
/// helper makes the choice explicit: an untracked input takes
/// `LayerNorm::forward` unchanged; a tracked input takes the composed form
/// (the same formula as `candle_nn`'s own non-fused fall-through), which
/// carries a backward.
///
/// # Shapes
/// - `xs`: `[.., hidden]` -- normalized over the final axis
/// - returns: `[.., hidden]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn layer_norm(norm: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    if xs.track_op() && !fused_carries_gradients(&FUSED_LAYER_NORM_DIFFERENTIABLE, probe_layer_norm)
    {
        layer_norm_composed(norm, xs)
    } else {
        Ok(norm.forward(xs)?)
    }
}

/// `RmsNorm` forward, differentiable when the graph is tracked.
///
/// `candle_nn::RmsNorm::forward` takes the fused (no-backward) kernel whenever
/// the input is contiguous.  A tracked input is routed to
/// `RmsNorm::forward_diff` — `candle_nn`'s own composed path, which carries a
/// backward; an untracked input takes `RmsNorm::forward` unchanged.
///
/// # Shapes
/// - `xs`: `[.., hidden]` -- normalized over the final axis
/// - returns: `[.., hidden]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn rms_norm(norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
    if xs.track_op() {
        Ok(norm.forward_diff(xs)?)
    } else {
        Ok(norm.forward(xs)?)
    }
}

/// Rotary position embedding, differentiable when the graph is tracked.
///
/// `candle_nn::rotary_emb::rope` is `apply_op3_no_bwd`, so its output records
/// **no** backprop op and does not even set `track_op`. Every `Q` and `K` in
/// `GenericTransformer` (requires `transformer`) and `GenericMdlm` (requires
/// `diffusion`) passes through it, so before
/// v0.2.0 a `VarMap`-backed forward lost its gradient at `RoPE`: the
/// projections, the embeddings and every earlier layer trained as if frozen,
/// while the loss still fell through the `V` path and the residual stream.
///
/// This is the same C1 failure the module doc describes, and it survived the
/// v0.1.20 sweep because the all-parameters-receive-a-gradient test runs on
/// `OthelloGpt`, which uses learned absolute positions and never touches
/// `RoPE`.
///
/// `candle_nn::rotary_emb::rope_slow` is the composed equivalent and produces
/// **identical values**, verified elementwise, so the dispatch costs nothing
/// but a boolean test and inference stays byte-identical.
///
/// # Shapes
/// - `xs`: `[batch, n_heads, seq_len, head_dim]`
/// - `cos` / `sin`: `[seq_len, head_dim / 2]`
/// - returns: `[batch, n_heads, seq_len, head_dim]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
pub fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if xs.track_op() && !fused_carries_gradients(&FUSED_ROPE_DIFFERENTIABLE, probe_rope) {
        Ok(candle_nn::rotary_emb::rope_slow(xs, cos, sin)?)
    } else {
        Ok(candle_nn::rotary_emb::rope(xs, cos, sin)?)
    }
}

/// Composed (differentiable) `LayerNorm`: the formula from `candle_nn`'s own
/// non-fused fall-through, built from primitive ops so every step records a
/// backward.
///
/// # Shapes
/// - `xs`: `[.., hidden]` -- normalized over the final axis
/// - returns: `[.., hidden]`
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) on tensor failures.
fn layer_norm_composed(norm: &LayerNorm, xs: &Tensor) -> Result<Tensor> {
    let x_dtype = xs.dtype();
    let internal_dtype = if matches!(x_dtype, DType::F16 | DType::BF16) {
        DType::F32
    } else {
        x_dtype
    };
    let hidden_size = xs.dim(D::Minus1)?;
    // PROMOTE: mean/variance accumulation over F16/BF16 loses precision;
    // compute in F32 exactly as candle-nn's composed path does
    let x = xs.to_dtype(internal_dtype)?;
    let x = if norm.remove_mean() {
        // CAST: usize → f64, hidden dimension fits exactly in the f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let mean = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean)?
    } else {
        x
    };
    // CAST: usize → f64, hidden dimension fits exactly in the f64 mantissa
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + norm.eps())?.sqrt()?)?;
    let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(norm.weight())?;
    match norm.bias() {
        None => Ok(x),
        Some(bias) => Ok(x.broadcast_add(bias)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    /// Fresh leaf `Var` of shape `[2, 8]` on CPU.
    fn leaf() -> Var {
        Var::randn(0f32, 1f32, (2, 8), &Device::Cpu).unwrap()
    }

    /// Max absolute element-wise difference between two same-shape tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// A weight + bias `LayerNorm` (the fused, gradient-barrier configuration).
    fn layer_norm_with_bias() -> LayerNorm {
        let weight = Tensor::rand(0.5f32, 1.5f32, 8, &Device::Cpu).unwrap();
        let bias = Tensor::rand(-0.5f32, 0.5f32, 8, &Device::Cpu).unwrap();
        LayerNorm::new(weight, bias, 1e-5)
    }

    fn rms_norm_layer() -> RmsNorm {
        let weight = Tensor::rand(0.5f32, 1.5f32, 8, &Device::Cpu).unwrap();
        RmsNorm::new(weight, 1e-5)
    }

    #[test]
    fn softmax_gradient_reaches_leaf() {
        let v = leaf();
        let out = softmax_last_dim(v.as_tensor()).unwrap();
        assert!(out.track_op(), "tracked input must produce tracked output");
        let grads = out.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(&v).is_some(), "softmax barrier: no gradient");
    }

    #[test]
    fn layer_norm_gradient_reaches_leaf() {
        let ln = layer_norm_with_bias();
        let v = leaf();
        let out = layer_norm(&ln, v.as_tensor()).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(&v).is_some(), "layer_norm barrier: no gradient");
    }

    #[test]
    fn rms_norm_gradient_reaches_leaf() {
        let rms = rms_norm_layer();
        let v = leaf();
        let out = rms_norm(&rms, v.as_tensor()).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(&v).is_some(), "rms_norm barrier: no gradient");
    }

    #[test]
    fn untracked_input_stays_untracked() {
        let xs = Tensor::randn(0f32, 1f32, (2, 8), &Device::Cpu).unwrap();
        assert!(!xs.track_op());
        assert!(!softmax_last_dim(&xs).unwrap().track_op());
        assert!(!layer_norm(&layer_norm_with_bias(), &xs).unwrap().track_op());
        assert!(!rms_norm(&rms_norm_layer(), &xs).unwrap().track_op());
    }

    #[test]
    fn softmax_paths_agree() {
        let v = leaf();
        let tracked = softmax_last_dim(v.as_tensor()).unwrap();
        let untracked = softmax_last_dim(&v.as_tensor().detach()).unwrap();
        assert!(max_abs_diff(&tracked, &untracked) < 1e-6);
    }

    #[test]
    fn layer_norm_paths_agree() {
        let ln = layer_norm_with_bias();
        let v = leaf();
        let tracked = layer_norm(&ln, v.as_tensor()).unwrap();
        let untracked = layer_norm(&ln, &v.as_tensor().detach()).unwrap();
        assert!(max_abs_diff(&tracked, &untracked) < 1e-6);
    }

    #[test]
    fn rms_norm_paths_agree() {
        let rms = rms_norm_layer();
        let v = leaf();
        let tracked = rms_norm(&rms, v.as_tensor()).unwrap();
        let untracked = rms_norm(&rms, &v.as_tensor().detach()).unwrap();
        assert!(max_abs_diff(&tracked, &untracked) < 1e-6);
    }

    #[test]
    fn weight_only_layer_norm_paths_agree() {
        let weight = Tensor::rand(0.5f32, 1.5f32, 8, &Device::Cpu).unwrap();
        let ln = LayerNorm::new_no_bias(weight, 1e-5);
        let v = leaf();
        let tracked = layer_norm(&ln, v.as_tensor()).unwrap();
        let grads = tracked.sum_all().unwrap().backward().unwrap();
        assert!(grads.get(&v).is_some());
        let untracked = layer_norm(&ln, &v.as_tensor().detach()).unwrap();
        assert!(max_abs_diff(&tracked, &untracked) < 1e-6);
    }

    /// Build a `[1, 1, 2, 2]` input plus matching `cos`/`sin` for `rope`.
    fn rope_inputs() -> (Var, Tensor, Tensor) {
        let dev = candle_core::Device::Cpu;
        let x = Tensor::new(&[[[[0.1f32, 0.2], [0.3, 0.4]]]], &dev).unwrap();
        let v = Var::from_tensor(&x).unwrap();
        let cos = Tensor::new(&[[0.8f32], [0.6]], &dev).unwrap();
        let sin = Tensor::new(&[[0.2f32], [0.4]], &dev).unwrap();
        (v, cos, sin)
    }

    /// `RoPE` must carry a gradient under a `VarMap`.
    ///
    /// The regression: `candle_nn::rotary_emb::rope` is `apply_op3_no_bwd`, so
    /// a tracked input came out as a graph leaf and `backward()` stopped at
    /// `RoPE` -- silently, with the loss still falling through the `V` path.
    /// Every `Q` and `K` in `GenericTransformer` and `GenericMdlm` goes through
    /// here. The v0.1.20 sweep missed it because the 29/29-gradient test runs on
    /// `OthelloGpt`, which has learned absolute positions and no `RoPE`.
    #[test]
    fn rope_carries_a_gradient_when_tracked() {
        let (v, cos, sin) = rope_inputs();
        let tracked = rope(&v.as_tensor().contiguous().unwrap(), &cos, &sin).unwrap();

        assert!(
            tracked.track_op(),
            "a tracked input must produce a tracked output"
        );
        let grads = tracked.sum_all().unwrap().backward().unwrap();
        assert!(
            grads.get(&v).is_some(),
            "backward() must reach the input through rope"
        );
    }

    /// And the tracked path must agree with the fused kernel, so routing every
    /// backbone through the wrapper cannot move an inference parity baseline.
    #[test]
    fn rope_tracked_and_fused_paths_agree() {
        let (v, cos, sin) = rope_inputs();
        let tracked = rope(&v.as_tensor().contiguous().unwrap(), &cos, &sin).unwrap();
        let untracked = rope(&v.as_tensor().detach().contiguous().unwrap(), &cos, &sin).unwrap();

        assert!(
            !untracked.track_op(),
            "the detached input must take the fused path"
        );
        assert!(
            max_abs_diff(&tracked, &untracked) < 1e-6,
            "composed and fused rope must agree"
        );
    }
}
