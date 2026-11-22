// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PlainTransformer`: a backbone for hand-weighted (constructed) models.
//!
//! A pre-norm residual transformer in which **every** component is an explicit
//! choice: the norm (none by default), the activation (`ReLU` by default), the
//! positional embedding, the biases, the head dimension, the MLP width and the
//! attention scale. Nothing is derived behind the caller's back, so a model
//! whose weights were set by construction (`Tracr`-style: named subspaces of
//! the residual stream, attention that reads and writes chosen subspaces, MLPs
//! that compute lookups) runs exactly as written, and is read through the same
//! hook surface as every trained backbone in the crate.
//!
//! ```text
//! x      = tok_emb[ids] (+ pos_emb[0..seq])
//! for each layer i:
//!   x    = x + attention_i(norm1_i(x))
//!   x    = x + W_proj_i · act(W_fc_i · norm2_i(x) + b_fc_i) + b_proj_i
//! logits = unembed · norm_f(x)                     (every position)
//! ```
//!
//! Build one from tensors in memory with [`PlainTransformer::from_tensors`]
//! (strict: a missing, extra, misnamed, mis-shaped or non-`F32` tensor is an
//! error, never silently ignored), or from a `VarBuilder` with
//! [`PlainTransformer::load`]. The expected tensor names and shapes are
//! [`PlainTransformerConfig::weight_shapes`].
//!
//! **No padding mask.** The only mask is the optional causal one. With
//! `causal = false`, real tokens attend to padding, so batch only sequences of
//! the same length, or run them one at a time.
//!
//! Design and validation: `design/plain-transformer.md`. Enabled by the `plain`
//! feature.

use std::collections::{BTreeSet, HashMap};

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, RmsNorm, VarBuilder};

use crate::backend::MIBackend;
use crate::config::{Activation, NormType};
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `PlainTransformer` configuration: every dimension and component explicit.
///
/// Build with [`new`](Self::new), which takes the dimensions that have no
/// sensible default, then the chainable `with_*` setters. The defaults suit a
/// constructed model: no norm, `ReLU`, no positional embedding, no biases,
/// bidirectional attention, scale `1 / sqrt(head_dim)`.
///
/// ```
/// use candle_mi::{Activation, NormType, PlainTransformerConfig};
///
/// // A Tracr-shaped model: 2 layers, 3 heads of 4 dims in a 10-dim residual,
/// // ReLU MLP of width 6, 8 positions, zero biases present.
/// let cfg = PlainTransformerConfig::new(5, 2, 10, 3, 4, 6)?
///     .with_output_size(3)
///     .with_positions(8)
///     .with_bias(true);
/// assert_eq!(cfg.activation, Activation::Relu);
/// assert_eq!(cfg.norm, None);
/// assert!(!cfg.causal);
///
/// // OthelloGpt's shape: LayerNorm, exact GELU, mlp = 4 * d_model.
/// let gpt = PlainTransformerConfig::new(62, 8, 512, 8, 64, 2048)?
///     .with_positions(60)
///     .with_norm(NormType::LayerNorm)
///     .with_activation(Activation::Gelu)
///     .with_bias(true);
/// assert_eq!(gpt.attn_scale, 0.125);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PlainTransformerConfig {
    /// Number of rows of the input token embedding `tok_emb` (the input
    /// vocabulary). Every input id must be below it.
    pub vocab_size: usize,
    /// Number of rows of `unembed`: the width of the logits, and what
    /// [`MIBackend::vocab_size`] returns. Equal to `vocab_size` unless set; a
    /// `Tracr` model unembeds onto its output space, whose size differs.
    pub output_size: usize,
    /// Number of rows of the learned positional embedding `pos_emb`, the
    /// longest sequence the model accepts. `None` means no positional embedding
    /// at all (and no length limit).
    pub max_positions: Option<usize>,
    /// Number of transformer blocks (each: attention, then MLP).
    pub n_layer: usize,
    /// Width of the residual stream.
    pub d_model: usize,
    /// Number of attention heads per layer.
    pub n_head: usize,
    /// Per-head dimension of queries, keys and values. `n_head * head_dim` need
    /// not equal `d_model`.
    pub head_dim: usize,
    /// Width of each MLP's hidden layer (the neurons, [`HookPoint::MlpAct`]).
    pub mlp_hidden: usize,
    /// MLP activation. Default [`Activation::Relu`].
    pub activation: Activation,
    /// Norm at the three norm sites (before attention, before the MLP, before
    /// the unembed). `None` (the default) makes all three the identity.
    /// [`NormType::LayerNorm`] (weight and bias) and [`NormType::RmsNorm`]
    /// (weight) are supported.
    pub norm: Option<NormType>,
    /// Norm epsilon, used only when `norm` is set. Default `1e-5`.
    pub norm_eps: f64,
    /// Whether attention is causally masked. Default `false` (bidirectional),
    /// `Tracr`'s default.
    pub causal: bool,
    /// Multiplier applied to the queries before `q·kᵀ`. Default
    /// `1.0 / (head_dim as f64).sqrt()`, computed as `OthelloGpt` computes it.
    /// Kept as `f64` and applied as `q * attn_scale`, exactly as `OthelloGpt`
    /// does, so the two backbones scale identically to the last bit.
    pub attn_scale: f64,
    /// Whether the q, k, v and o projections and both MLP linears have biases.
    /// Default `false`. The unembed never has one.
    pub bias: bool,
}

impl PlainTransformerConfig {
    /// A config with the given dimensions and the constructed-model defaults.
    ///
    /// `output_size` defaults to `vocab_size`, and `attn_scale` to
    /// `1.0 / (head_dim as f64).sqrt()`. Assigning `head_dim` directly later
    /// does not update `attn_scale`; set it too.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any dimension is zero.
    pub fn new(
        vocab_size: usize,
        n_layer: usize,
        d_model: usize,
        n_head: usize,
        head_dim: usize,
        mlp_hidden: usize,
    ) -> Result<Self> {
        let config = Self {
            vocab_size,
            output_size: vocab_size,
            max_positions: None,
            n_layer,
            d_model,
            n_head,
            head_dim,
            mlp_hidden,
            activation: Activation::Relu,
            norm: None,
            norm_eps: 1e-5,
            causal: false,
            attn_scale: default_attn_scale(head_dim),
            bias: false,
        };
        config.validate()?;
        Ok(config)
    }

    /// Set the logit width (rows of `unembed`).
    #[must_use]
    pub const fn with_output_size(mut self, output_size: usize) -> Self {
        self.output_size = output_size;
        self
    }

    /// Add a learned positional embedding with `max_positions` rows.
    #[must_use]
    pub const fn with_positions(mut self, max_positions: usize) -> Self {
        self.max_positions = Some(max_positions);
        self
    }

    /// Set the MLP activation.
    #[must_use]
    pub const fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Put `norm` at all three norm sites.
    #[must_use]
    pub const fn with_norm(mut self, norm: NormType) -> Self {
        self.norm = Some(norm);
        self
    }

    /// Set the norm epsilon (used only when a norm is set).
    #[must_use]
    pub const fn with_norm_eps(mut self, norm_eps: f64) -> Self {
        self.norm_eps = norm_eps;
        self
    }

    /// Causally mask the attention.
    #[must_use]
    pub const fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set the query multiplier applied before `q·kᵀ`.
    #[must_use]
    pub const fn with_attn_scale(mut self, attn_scale: f64) -> Self {
        self.attn_scale = attn_scale;
        self
    }

    /// Give the q, k, v, o projections and both MLP linears biases.
    #[must_use]
    pub const fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Width of the concatenated heads, `n_head * head_dim`.
    #[must_use]
    pub const fn attn_width(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Check the config is one `PlainTransformer` can build.
    ///
    /// Run by [`new`](Self::new), [`PlainTransformer::from_tensors`] and
    /// [`PlainTransformer::load`], so a config edited field by field after
    /// construction is still checked before use.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a dimension (or `max_positions`, when
    /// set) is zero, if `attn_scale` is not finite, if `norm` is a variant this
    /// backbone does not implement, or if `norm_eps` is not a positive finite
    /// number while a norm is set.
    pub fn validate(&self) -> Result<()> {
        let dims = [
            ("vocab_size", self.vocab_size),
            ("output_size", self.output_size),
            ("n_layer", self.n_layer),
            ("d_model", self.d_model),
            ("n_head", self.n_head),
            ("head_dim", self.head_dim),
            ("mlp_hidden", self.mlp_hidden),
        ];
        for (name, value) in dims {
            if value == 0 {
                return Err(MIError::Config(format!(
                    "{name} is zero (every PlainTransformer dimension must be at least 1)"
                )));
            }
        }
        if self.max_positions == Some(0) {
            return Err(MIError::Config(
                "max_positions is Some(0) (use None for no positional embedding)".to_string(),
            ));
        }
        if !self.attn_scale.is_finite() {
            return Err(MIError::Config(format!(
                "attn_scale {} is not finite",
                self.attn_scale
            )));
        }
        if let Some(norm) = self.norm {
            match norm {
                // EXPLICIT: the two supported norms need no further check here
                NormType::LayerNorm | NormType::RmsNorm => {}
                NormType::GemmaRmsNorm => {
                    return Err(MIError::Config(
                        "norm GemmaRmsNorm is not supported by PlainTransformer \
                         (supported: LayerNorm, RmsNorm, or None)"
                            .to_string(),
                    ));
                }
            }
            if !(self.norm_eps.is_finite() && self.norm_eps > 0.0) {
                return Err(MIError::Config(format!(
                    "norm_eps {} is not a positive finite number",
                    self.norm_eps
                )));
            }
        }
        Ok(())
    }

    /// Every tensor a `PlainTransformer` with this config expects: name and
    /// shape, in a fixed order (embeddings, blocks in index order, final norm,
    /// unembed).
    ///
    /// Shapes follow the `PyTorch` `nn.Linear` convention, `weight` is
    /// `[out, in]`. Head `h` owns rows `h * head_dim .. (h + 1) * head_dim` of
    /// `blocks.{i}.attn.{q,k,v}.weight`, and the same columns of
    /// `blocks.{i}.attn.o.weight`.
    ///
    /// | Name | Shape | Present when |
    /// |---|---|---|
    /// | `tok_emb.weight` | `[vocab_size, d_model]` | always |
    /// | `pos_emb.weight` | `[max_positions, d_model]` | `max_positions` set |
    /// | `blocks.{i}.ln1.weight` / `.bias` | `[d_model]` | `norm` set / `LayerNorm` |
    /// | `blocks.{i}.attn.{q,k,v}.weight` | `[n_head * head_dim, d_model]` | always |
    /// | `blocks.{i}.attn.{q,k,v}.bias` | `[n_head * head_dim]` | `bias` |
    /// | `blocks.{i}.attn.o.weight` | `[d_model, n_head * head_dim]` | always |
    /// | `blocks.{i}.attn.o.bias` | `[d_model]` | `bias` |
    /// | `blocks.{i}.ln2.weight` / `.bias` | `[d_model]` | `norm` set / `LayerNorm` |
    /// | `blocks.{i}.mlp.fc.weight` | `[mlp_hidden, d_model]` | always |
    /// | `blocks.{i}.mlp.fc.bias` | `[mlp_hidden]` | `bias` |
    /// | `blocks.{i}.mlp.proj.weight` | `[d_model, mlp_hidden]` | always |
    /// | `blocks.{i}.mlp.proj.bias` | `[d_model]` | `bias` |
    /// | `ln_f.weight` / `.bias` | `[d_model]` | `norm` set / `LayerNorm` |
    /// | `unembed.weight` | `[output_size, d_model]` | always |
    #[must_use]
    pub fn weight_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let d = self.d_model;
        let a = self.attn_width();
        let m = self.mlp_hidden;
        let norm_weight = self.norm.is_some();
        let norm_bias = matches!(self.norm, Some(NormType::LayerNorm));

        let mut shapes = Vec::new();
        let norm = |shapes: &mut Vec<(String, Vec<usize>)>, prefix: &str| {
            if norm_weight {
                shapes.push((format!("{prefix}.weight"), vec![d]));
            }
            if norm_bias {
                shapes.push((format!("{prefix}.bias"), vec![d]));
            }
        };

        shapes.push(("tok_emb.weight".to_string(), vec![self.vocab_size, d]));
        if let Some(positions) = self.max_positions {
            shapes.push(("pos_emb.weight".to_string(), vec![positions, d]));
        }
        for i in 0..self.n_layer {
            norm(&mut shapes, &format!("blocks.{i}.ln1"));
            for proj in ["q", "k", "v"] {
                shapes.push((format!("blocks.{i}.attn.{proj}.weight"), vec![a, d]));
                if self.bias {
                    shapes.push((format!("blocks.{i}.attn.{proj}.bias"), vec![a]));
                }
            }
            shapes.push((format!("blocks.{i}.attn.o.weight"), vec![d, a]));
            if self.bias {
                shapes.push((format!("blocks.{i}.attn.o.bias"), vec![d]));
            }
            norm(&mut shapes, &format!("blocks.{i}.ln2"));
            shapes.push((format!("blocks.{i}.mlp.fc.weight"), vec![m, d]));
            if self.bias {
                shapes.push((format!("blocks.{i}.mlp.fc.bias"), vec![m]));
            }
            shapes.push((format!("blocks.{i}.mlp.proj.weight"), vec![d, m]));
            if self.bias {
                shapes.push((format!("blocks.{i}.mlp.proj.bias"), vec![d]));
            }
        }
        norm(&mut shapes, "ln_f");
        shapes.push(("unembed.weight".to_string(), vec![self.output_size, d]));
        shapes
    }
}

/// `1.0 / (head_dim as f64).sqrt()`: the default attention scale, computed the
/// way `OthelloGpt` computes its own (in `f64`), so the two agree to the bit.
fn default_attn_scale(head_dim: usize) -> f64 {
    // CAST: usize → f64, head_dim fits in f64 mantissa
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let head_dim = head_dim as f64;
    1.0 / head_dim.sqrt()
}

// ---------------------------------------------------------------------------
// Norm
// ---------------------------------------------------------------------------

/// One norm site: `LayerNorm` or `RmsNorm`, or nothing (the identity).
enum PlainNorm {
    /// `LayerNorm`, weight and bias.
    Layer(LayerNorm),
    /// `RmsNorm`, weight only.
    Rms(RmsNorm),
}

impl PlainNorm {
    /// Load the norm at `vb` for `config`, or `None` when the config has none.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if a weight fails to load.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &PlainTransformerConfig, vb: VarBuilder<'_>) -> Result<Option<Self>> {
        let d = config.d_model;
        match config.norm {
            None => Ok(None),
            Some(NormType::LayerNorm) => {
                let cfg = candle_nn::LayerNormConfig {
                    eps: config.norm_eps,
                    ..Default::default()
                };
                Ok(Some(Self::Layer(candle_nn::layer_norm(d, cfg, vb)?)))
            }
            Some(NormType::RmsNorm) => Ok(Some(Self::Rms(candle_nn::rms_norm(
                d,
                config.norm_eps,
                vb,
            )?))),
            // `validate` rejects every other variant before any load.
            Some(other) => Err(MIError::Config(format!(
                "norm {other:?} is not supported by PlainTransformer"
            ))),
        }
    }

    /// Apply the norm through the backward-safe `nn_ops` wrappers.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, d_model]`
    /// - returns: `[batch, seq, d_model]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Layer(norm) => crate::nn_ops::layer_norm(norm, xs),
            Self::Rms(norm) => crate::nn_ops::rms_norm(norm, xs),
        }
    }
}

/// Apply an optional norm: the identity when there is none.
///
/// The identity is a shallow `clone` (a reference-count bump), not a copy, and
/// adds no operation to the forward pass.
fn apply_norm(norm: Option<&PlainNorm>, xs: &Tensor) -> Result<Tensor> {
    norm.map_or_else(|| Ok(xs.clone()), |norm| norm.forward(xs))
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// One block: optional norm, multi-head attention, residual; optional norm,
/// MLP, residual.
struct PlainBlock {
    /// Norm before attention, or `None` for the identity.
    ln1: Option<PlainNorm>,
    /// Norm before the MLP, or `None` for the identity.
    ln2: Option<PlainNorm>,
    /// Q, K and V concatenated along the output axis at load:
    /// `d_model -> 3 * n_head * head_dim`. One matmul, then split, which is
    /// exactly what `OthelloGpt` runs (see the module docs on parity).
    qkv: Linear,
    /// Attention output projection, `n_head * head_dim -> d_model`.
    o: Linear,
    /// MLP up projection, `d_model -> mlp_hidden`.
    fc: Linear,
    /// MLP down projection, `mlp_hidden -> d_model`.
    proj: Linear,
    /// Number of heads.
    n_head: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Query multiplier, `f64` as in `OthelloGpt`.
    attn_scale: f64,
    /// Whether attention is causally masked.
    causal: bool,
    /// MLP activation.
    activation: Activation,
}

impl PlainBlock {
    /// Load block `i` from the `blocks.{i}` namespace.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if a weight fails to load or has the wrong
    /// shape.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &PlainTransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let d = config.d_model;
        let a = config.attn_width();
        let m = config.mlp_hidden;
        let attn = vb.pp("attn");

        // Q, K, V are separate tensors in the checkpoint (what a construction
        // writes) and one projection in the forward. `Tensor::cat` along the
        // output axis of three `[a, d]` weights is the `[3a, d]` weight
        // `OthelloGpt` stores as `attn.qkv.weight`, element for element.
        let q_w = attn.pp("q").get((a, d), "weight")?;
        let k_w = attn.pp("k").get((a, d), "weight")?;
        let v_w = attn.pp("v").get((a, d), "weight")?;
        let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let qkv_b = if config.bias {
            let q_b = attn.pp("q").get(a, "bias")?;
            let k_b = attn.pp("k").get(a, "bias")?;
            let v_b = attn.pp("v").get(a, "bias")?;
            Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0)?)
        } else {
            None
        };

        Ok(Self {
            ln1: PlainNorm::load(config, vb.pp("ln1"))?,
            ln2: PlainNorm::load(config, vb.pp("ln2"))?,
            qkv: Linear::new(qkv_w, qkv_b),
            o: load_linear(a, d, config.bias, attn.pp("o"))?,
            fc: load_linear(d, m, config.bias, vb.pp("mlp").pp("fc"))?,
            proj: load_linear(m, d, config.bias, vb.pp("mlp").pp("proj"))?,
            n_head: config.n_head,
            head_dim: config.head_dim,
            attn_scale: config.attn_scale,
            causal: config.causal,
            activation: config.activation,
        })
    }

    /// Multi-head self-attention, with the attention hooks.
    ///
    /// The operations, and their order, are `OthelloGpt`'s composed path: one
    /// `qkv` matmul, split, heads to dim 1, `AttnQ`/`AttnK`/`AttnV`, `q` scaled
    /// by `attn_scale` (as `f64`), `q·kᵀ`, the causal mask from
    /// `crate::util::masks::create_causal_mask`, `AttnScores`, the shared
    /// `F32` softmax, `AttnPattern`, `pattern·v`, heads merged, `o`.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, d_model]`
    /// - returns: `[batch, seq, d_model]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures and
    /// [`MIError::Intervention`] if an intervention is invalid at its hook
    /// point.
    fn attention(
        &self,
        xs: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let width = self.n_head * self.head_dim;

        let qkv = self.qkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, width)?;
        let k = qkv.narrow(D::Minus1, width, width)?;
        let v = qkv.narrow(D::Minus1, 2 * width, width)?;

        // [batch, seq, n_head, head_dim] -> [batch, n_head, seq, head_dim]
        let mut q = q
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((batch, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;

        hook_point(&mut q, HookPoint::AttnQ(layer_idx), hooks, cache)?;
        hook_point(&mut k, HookPoint::AttnK(layer_idx), hooks, cache)?;
        hook_point(&mut v, HookPoint::AttnV(layer_idx), hooks, cache)?;

        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let k_t = k.contiguous()?.transpose(2, 3)?;
        // The scale multiplies the queries (`q * f64`), as in `OthelloGpt`: the same
        // operation on the same operand, so the scores agree to the bit. `AttnQ` above
        // is captured before the scaling; `AttnScores` below holds the scaled scores.
        let q = (q * self.attn_scale)?;
        let mut scores = q.matmul(&k_t)?;
        if self.causal {
            let mask =
                crate::util::masks::create_causal_mask(seq_len, scores.device(), scores.dtype())?;
            scores = scores.broadcast_add(&mask)?;
        }
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        let mut pattern = crate::nn_ops::softmax_last_dim_f32(&scores)?;
        hook_point(
            &mut pattern,
            HookPoint::AttnPattern(layer_idx),
            hooks,
            cache,
        )?;

        // CONTIGUOUS: ensure contiguous layout for the pattern·value matmul
        let v = v.contiguous()?;
        let attn = pattern.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, width))?;

        Ok(self.o.forward(&attn)?)
    }

    /// Run the block with hook support.
    ///
    /// Hooks fire where `OthelloGpt` fires them, plus `MlpAct` on the neurons.
    /// `MlpPost` and `MlpOut` are both the MLP output (there is no post-MLP
    /// norm), as in `OthelloGpt`.
    ///
    /// # Shapes
    /// - `hidden_in`: `[batch, seq, d_model]`
    /// - returns: `[batch, seq, d_model]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures and
    /// [`MIError::Intervention`] if an intervention is invalid at its hook
    /// point.
    fn forward(
        &self,
        hidden_in: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut hidden = hidden_in.clone();
        hook_point(&mut hidden, HookPoint::ResidPre(layer_idx), hooks, cache)?;

        // --- attention sublayer ---
        let residual = hidden.clone();
        let normed1 = apply_norm(self.ln1.as_ref(), &residual)?;
        let mut attn = self.attention(&normed1, layer_idx, hooks, cache)?;
        hook_point(&mut attn, HookPoint::AttnOut(layer_idx), hooks, cache)?;
        hidden = (residual + attn)?;
        hook_point(&mut hidden, HookPoint::ResidMid(layer_idx), hooks, cache)?;

        // --- MLP sublayer ---
        let residual2 = hidden.clone();
        let mut normed2 = apply_norm(self.ln2.as_ref(), &hidden)?;
        hook_point(&mut normed2, HookPoint::MlpPre(layer_idx), hooks, cache)?;
        let up = self.fc.forward(&normed2)?;
        let mut act = crate::nn_ops::activation(&up, self.activation)?;
        hook_point(&mut act, HookPoint::MlpAct(layer_idx), hooks, cache)?;
        let mut mlp_out = self.proj.forward(&act)?;
        hook_point(&mut mlp_out, HookPoint::MlpPost(layer_idx), hooks, cache)?;
        // No post-MLP norm: the MLP output IS the residual contribution.
        hook_point(&mut mlp_out, HookPoint::MlpOut(layer_idx), hooks, cache)?;
        hidden = (residual2 + mlp_out)?;
        hook_point(&mut hidden, HookPoint::ResidPost(layer_idx), hooks, cache)?;

        Ok(hidden)
    }
}

/// Load an `nn.Linear`-convention layer (`weight` `[out, in]`), with or
/// without a bias.
///
/// # Errors
///
/// Returns [`MIError::Model`] if a weight fails to load or has the wrong shape.
#[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
fn load_linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder<'_>) -> Result<Linear> {
    if bias {
        Ok(candle_nn::linear(in_dim, out_dim, vb)?)
    } else {
        Ok(candle_nn::linear_no_bias(in_dim, out_dim, vb)?)
    }
}

// ---------------------------------------------------------------------------
// PlainTransformer
// ---------------------------------------------------------------------------

/// A transformer backbone for hand-weighted (constructed) models, with full
/// hook support.
///
/// See the [module docs](self) for the architecture and
/// [`PlainTransformerConfig`] for the options.
///
/// **[`MIBackend::vocab_size`] is the logit width, `output_size`**, which is
/// what the trait's `forward` contract names (`[batch, seq, vocab_size]`
/// logits). The input vocabulary is `config().vocab_size`. Code that uses
/// `vocab_size()` as an *input* id bound (as the MDLM and `OthelloGpt` examples
/// do, for their `[MASK]` id) is wrong for a model whose two sizes differ.
pub struct PlainTransformer {
    /// Input token embedding, `[vocab_size, d_model]`.
    tok_emb: Embedding,
    /// Learned positional embedding, `[max_positions, d_model]`, when present.
    pos_emb: Option<Tensor>,
    /// The blocks, in order.
    blocks: Vec<PlainBlock>,
    /// Norm before the unembed, or `None` for the identity.
    ln_f: Option<PlainNorm>,
    /// Unembedding, `[output_size, d_model]`, no bias.
    unembed: Linear,
    /// The configuration the model was built with.
    config: PlainTransformerConfig,
}

impl PlainTransformer {
    /// Build a model from tensors in memory: the path for constructed models.
    ///
    /// **Strict.** The tensor names must be exactly
    /// [`PlainTransformerConfig::weight_shapes`]: every expected tensor present,
    /// nothing else present, every shape as listed, every tensor `F32`. A
    /// violation is an error that lists every offending name with what was
    /// expected and what was given, so a typo in a hand-written key can never
    /// silently leave a head at zero. Tensors may live on any device; they are
    /// moved to `device`.
    ///
    /// # Shapes
    /// - `tensors`: the shapes in [`PlainTransformerConfig::weight_shapes`]
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` `U32` ids to `[batch, seq, output_size]` logits.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the config is invalid, or if a tensor is
    /// missing, unexpected, of the wrong shape or not `F32`.
    /// Returns [`MIError::Model`] if moving a tensor to `device` fails.
    pub fn from_tensors(
        config: PlainTransformerConfig,
        tensors: HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        check_tensors(&config, &tensors)?;
        Self::load(
            config,
            VarBuilder::from_tensors(tensors, DType::F32, device),
        )
    }

    /// Build a model from a [`VarBuilder`] (for instance over a `safetensors`
    /// file).
    ///
    /// Reads every tensor in [`PlainTransformerConfig::weight_shapes`] and fails
    /// on a missing or mis-shaped one. A `VarBuilder` cannot list its keys, so,
    /// unlike [`from_tensors`](Self::from_tensors), this cannot reject a tensor
    /// it was not asked for: an extra or misnamed key in the source goes
    /// unnoticed here (and the correctly named one it was meant to be is then
    /// reported missing). Prefer `from_tensors` for constructed models.
    ///
    /// Q, K and V are read as three tensors and concatenated into one
    /// projection, so the forward runs one matmul for the three.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` `U32` ids to `[batch, seq, output_size]` logits.
    ///
    /// # Memory
    /// Holds every weight once, at the `VarBuilder`'s dtype and device, plus a
    /// transient copy of each layer's Q, K and V while they are concatenated.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the config is invalid.
    /// Returns [`MIError::Model`] if a tensor is missing or has the wrong shape.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(config: PlainTransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let d = config.d_model;

        let tok_emb = Embedding::new(vb.pp("tok_emb").get((config.vocab_size, d), "weight")?, d);
        let pos_emb = match config.max_positions {
            Some(positions) => Some(vb.pp("pos_emb").get((positions, d), "weight")?),
            None => None,
        };

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(PlainBlock::load(&config, vb.pp(format!("blocks.{i}")))?);
        }

        let ln_f = PlainNorm::load(&config, vb.pp("ln_f"))?;
        let unembed = candle_nn::linear_no_bias(d, config.output_size, vb.pp("unembed"))?;

        Ok(Self {
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            unembed,
            config,
        })
    }

    /// The configuration the model was built with.
    #[must_use]
    pub const fn config(&self) -> &PlainTransformerConfig {
        &self.config
    }

    /// Reject input ids the model cannot take, before they reach the lookup.
    ///
    /// The id-range check is not redundant with candle's: `index_select` raises
    /// an error on CPU, but the CUDA kernel does not bounds-check, so an
    /// out-of-range id there would read past the table, a silent wrong answer on
    /// GPU only. The cost is one `max_all` and a scalar read.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if `input_ids` is not rank 2 or not `U32`, is
    /// empty, is longer than `max_positions`, or holds an id `>= vocab_size`.
    fn validate_input(&self, input_ids: &Tensor) -> Result<usize> {
        let (batch, seq_len) = input_ids.dims2()?;
        if input_ids.dtype() != DType::U32 {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "input_ids dtype {:?} is not U32",
                input_ids.dtype()
            ))));
        }
        if batch == 0 || seq_len == 0 {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "input_ids is empty (dims [{batch}, {seq_len}])"
            ))));
        }
        if let Some(positions) = self.config.max_positions
            && seq_len > positions
        {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "seq_len {seq_len} exceeds max_positions {positions}"
            ))));
        }
        let max_id: u32 = input_ids.flatten_all()?.max_all()?.to_vec0()?;
        // CAST: u32 → usize, lossless widening on every supported target
        #[allow(clippy::as_conversions)]
        let max_id_usize = max_id as usize;
        if max_id_usize >= self.config.vocab_size {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "input id {max_id} out of range (vocab_size {})",
                self.config.vocab_size
            ))));
        }
        Ok(seq_len)
    }

    /// Final norm (if any) then unembed, with the `FinalNorm` hook.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, d_model]`
    /// - returns: `[batch, seq, output_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures and
    /// [`MIError::Intervention`] if an intervention is invalid at `FinalNorm`.
    fn head_forward(
        &self,
        hidden: &Tensor,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut xs = apply_norm(self.ln_f.as_ref(), hidden)?;
        hook_point(&mut xs, HookPoint::FinalNorm, hooks, cache)?;
        Ok(self.unembed.forward(&xs)?)
    }
}

impl MIBackend for PlainTransformer {
    fn num_layers(&self) -> usize {
        self.config.n_layer
    }

    fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    /// The logit width, `output_size` (see the type's docs).
    // Deliberate, and the lint names the trap exactly: the trait's `vocab_size`
    // is the logit width (`forward` returns `[batch, seq, vocab_size]`), which
    // here is `output_size`, while `config.vocab_size` is the input table.
    #[allow(clippy::misnamed_getters)]
    fn vocab_size(&self) -> usize {
        self.config.output_size
    }

    fn num_heads(&self) -> usize {
        self.config.n_head
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let seq_len = self.validate_input(input_ids)?;
        let device = input_ids.device();

        let mut hidden = self.tok_emb.forward(input_ids)?;
        if let Some(pos_emb) = &self.pos_emb {
            let pos = pos_emb.narrow(0, 0, seq_len)?;
            hidden = hidden.broadcast_add(&pos)?;
        }

        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);
        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, layer_idx, hooks, &mut cache)?;
        }

        let logits = self.head_forward(&hidden, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = apply_norm(self.ln_f.as_ref(), hidden)?;
        Ok(self.unembed.forward(&xs)?)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        // CAST: u32 → usize, lossless widening on every supported target
        #[allow(clippy::as_conversions)]
        let id = token_id as usize;
        if id >= self.config.vocab_size {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "token id {token_id} out of range (vocab_size {})",
                self.config.vocab_size
            ))));
        }
        let device = self.tok_emb.embeddings().device();
        let ids = Tensor::new(&[token_id], device)?;
        let emb = self.tok_emb.forward(&ids)?; // [1, d_model]
        Ok(emb.squeeze(0)?) // [d_model]
    }
}

// ---------------------------------------------------------------------------
// Strict tensor check (from_tensors)
// ---------------------------------------------------------------------------

/// Check `tensors` against `config.weight_shapes()` exactly, reporting every
/// problem at once.
///
/// # Errors
///
/// Returns [`MIError::Config`] listing every missing, unexpected, mis-shaped
/// and non-`F32` tensor, each with what was expected and what was given.
fn check_tensors(config: &PlainTransformerConfig, tensors: &HashMap<String, Tensor>) -> Result<()> {
    let expected = config.weight_shapes();
    let expected_names: BTreeSet<&str> = expected.iter().map(|(name, _)| name.as_str()).collect();

    let mut problems: Vec<String> = Vec::new();
    for (name, shape) in &expected {
        match tensors.get(name) {
            None => problems.push(format!(
                "missing tensor `{name}` (expected shape {shape:?})"
            )),
            Some(tensor) => {
                if tensor.dims() != shape.as_slice() {
                    problems.push(format!(
                        "tensor `{name}` has shape {:?} (expected {shape:?})",
                        tensor.dims()
                    ));
                }
                if tensor.dtype() != DType::F32 {
                    problems.push(format!(
                        "tensor `{name}` has dtype {:?} (expected F32)",
                        tensor.dtype()
                    ));
                }
            }
        }
    }
    // Sorted, so the report is deterministic (HashMap iteration is not).
    let unexpected: BTreeSet<&str> = tensors
        .keys()
        // BORROW: explicit .as_str() to compare against the expected-name set
        .map(String::as_str)
        .filter(|name| !expected_names.contains(name))
        .collect();
    for name in unexpected {
        problems.push(format!(
            "unexpected tensor `{name}` (not in PlainTransformerConfig::weight_shapes for this config)"
        ));
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(MIError::Config(format!(
            "PlainTransformer tensors do not match the config ({} problem(s)): {}",
            problems.len(),
            problems.join("; ")
        )))
    }
}

#[cfg(test)]
mod tests;
