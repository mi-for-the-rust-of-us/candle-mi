// SPDX-License-Identifier: MIT OR Apache-2.0

//! `OthelloGpt`: a plain GPT-2-style bidirectional backbone.
//!
//! A faithful Rust port of the `OthelloMDLM` world model (a nanoGPT/minGPT
//! lineage transformer) used in masked-diffusion Othello probing studies.
//! Unlike the [`GenericMdlm`](super::mdlm::GenericMdlm) `DiT`, this backbone
//! has **no** `adaLN` conditioning, **no** rotary embeddings, and **no**
//! weight-only `LayerNorm`.  Instead it is the classic GPT-2 recipe:
//!
//! - **learned absolute** positional embeddings (`nn.Embedding`), added to the
//!   token embedding — there is no `RoPE`;
//! - **full** `LayerNorm` (weight *and* bias) at every norm site;
//! - **with-bias** fused QKV, attention output, and both MLP linears;
//! - an exact (erf) `GELU` MLP — *not* the tanh approximation;
//! - an **untied** vocabulary head with no bias.
//!
//! Attention is bidirectional by default (`causal = false`, the masked-diffusion
//! setting); the [`causal`](OthelloGptConfig::causal) flag enables a causal mask
//! so the same module can also load an autoregressive Othello-GPT control.
//!
//! The backbone is feature-gated behind `diffusion` because its reason to exist
//! is the masked-diffusion `SUBS` sampler and the diffusion logit-lens in
//! [`sample`](super::sample), both of which operate on any
//! [`MIBackend`].

#[cfg(test)]
use std::collections::HashMap;

use candle_core::{D, DType, Device, Module, Tensor, Var};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, VarMap};
use serde_json::Value;

use crate::backend::MIBackend;
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// `OthelloGpt` model configuration (a GPT-2-style backbone).
///
/// Mirrors the upstream `OthelloMDLMConfig`.  The released world model is
/// `vocab_size = 62`, `block_size = 60`, `n_layer = 8`, `n_head = 8`,
/// `n_embd = 512`, `causal = false`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OthelloGptConfig {
    /// Vocabulary size (60 move cells + pad + `[MASK]` for the world model).
    pub vocab_size: usize,
    /// Maximum sequence length covered by the learned positional embedding.
    pub block_size: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Hidden dimension (`d_model`); must be divisible by `n_head`.
    pub n_embd: usize,
    /// Per-head dimension (`n_embd / n_head`).
    pub head_dim: usize,
    /// Feed-forward expansion ratio (GPT-2 uses `4`; not stored upstream).
    pub mlp_ratio: usize,
    /// `LayerNorm` epsilon (GPT-2 uses `1e-5`).
    pub norm_eps: f64,
    /// Whether to apply a causal attention mask.  `false` (bidirectional) for
    /// the masked-diffusion world model; `true` for an autoregressive control.
    pub causal: bool,
    /// Whether the model carries a self-conditioning input: a second token grid
    /// holding the model's own previous clean prediction, embedded through
    /// `self_cond_emb` and added to the residual stream before the first hook.
    ///
    /// `false` by default, and with it off the forward pass is bit-identical to
    /// a model built without the feature.  See
    /// [`forward_with_self_cond`](OthelloGpt::forward_with_self_cond).
    pub self_conditioning: bool,
}

impl OthelloGptConfig {
    /// Construct a config from explicit dimensions, deriving `head_dim`.
    ///
    /// [`self_conditioning`](Self::self_conditioning) defaults to `false`; set
    /// the field directly, or use
    /// [`with_self_conditioning`](Self::with_self_conditioning).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `n_head`
    /// is zero or `n_embd` is not divisible by `n_head`.
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        causal: bool,
    ) -> Result<Self> {
        if n_head == 0 || !n_embd.is_multiple_of(n_head) {
            return Err(MIError::Config(format!(
                "n_embd {n_embd} not divisible by n_head {n_head}"
            )));
        }
        Ok(Self {
            vocab_size,
            block_size,
            n_layer,
            n_head,
            n_embd,
            head_dim: n_embd / n_head,
            mlp_ratio: 4,
            norm_eps: 1e-5,
            causal,
            self_conditioning: false,
        })
    }

    /// Enable or disable the self-conditioning input, consuming `self`.
    ///
    /// Chainable alternative to setting the field, kept so that turning the
    /// feature on does not require naming every other field:
    ///
    /// ```
    /// use candle_mi::OthelloGptConfig;
    /// let cfg = OthelloGptConfig::new(62, 60, 8, 8, 512, false)?
    ///     .with_self_conditioning(true);
    /// assert!(cfg.self_conditioning);
    /// # Ok::<(), candle_mi::MIError>(())
    /// ```
    #[must_use]
    pub const fn with_self_conditioning(mut self, enabled: bool) -> Self {
        self.self_conditioning = enabled;
        self
    }

    /// Override the feed-forward expansion ratio (GPT-2 uses `4`).
    ///
    /// [`new`](Self::new) does not take it, and the struct is
    /// `#[non_exhaustive]`, so this is the construction path for a backbone
    /// whose MLP is not 4x. Assigning the field on an existing value works too.
    #[must_use]
    pub const fn with_mlp_ratio(mut self, mlp_ratio: usize) -> Self {
        self.mlp_ratio = mlp_ratio;
        self
    }

    /// Override the `LayerNorm` epsilon (GPT-2 uses `1e-5`).
    #[must_use]
    pub const fn with_norm_eps(mut self, norm_eps: f64) -> Self {
        self.norm_eps = norm_eps;
        self
    }

    /// Parse an [`OthelloGptConfig`] from a companion `config.json` value.
    ///
    /// The converter (`scripts/convert_othello_mdlm.py`) writes this file from
    /// the checkpoint's `config` dict, so the keys mirror `OthelloMDLMConfig`:
    /// `vocab_size`, `block_size`, `n_layer`, `n_head`, `n_embd`, the optional
    /// `causal` (default `false`), and the optional `self_conditioning`
    /// (default `false`).
    ///
    /// Both optional keys read as `false` when absent, so a `config.json`
    /// written before either existed parses unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a required
    /// key is missing or not a non-negative integer, or if `n_embd` is not
    /// divisible by `n_head`.
    pub fn from_hf_config(config: &Value) -> Result<Self> {
        let vocab_size = crate::config::get_usize_in(config, "vocab_size", "OthelloGpt config")?;
        let block_size = crate::config::get_usize_in(config, "block_size", "OthelloGpt config")?;
        let n_layer = crate::config::get_usize_in(config, "n_layer", "OthelloGpt config")?;
        let n_head = crate::config::get_usize_in(config, "n_head", "OthelloGpt config")?;
        let n_embd = crate::config::get_usize_in(config, "n_embd", "OthelloGpt config")?;
        let causal = crate::config::get_bool_or(config, "causal", false);
        let self_conditioning = crate::config::get_bool_or(config, "self_conditioning", false);
        Self::new(vocab_size, block_size, n_layer, n_head, n_embd, causal)
            .map(|c| c.with_self_conditioning(self_conditioning))
    }
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// A single GPT-2-style block: pre-`LayerNorm`, bidirectional (or causal)
/// self-attention, pre-`LayerNorm`, and an exact-`GELU` MLP, each with a
/// residual connection.
struct OthelloBlock {
    /// Pre-attention `LayerNorm` (weight + bias).
    ln1: LayerNorm,
    /// Pre-MLP `LayerNorm` (weight + bias).
    ln2: LayerNorm,
    /// Fused QKV projection: `n_embd → 3 * n_embd` (with bias).
    qkv: Linear,
    /// Attention output projection: `n_embd → n_embd` (with bias).
    proj: Linear,
    /// MLP up-projection: `n_embd → mlp_ratio * n_embd` (with bias).
    mlp_fc: Linear,
    /// MLP down-projection: `mlp_ratio * n_embd → n_embd` (with bias).
    mlp_proj: Linear,
    /// Number of attention heads.
    n_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Hidden dimension (`n_heads * head_dim`).
    hidden_dim: usize,
    /// Attention scale `1 / sqrt(head_dim)`.
    scale: f64,
    /// Whether to apply a causal attention mask.
    causal: bool,
}

impl OthelloBlock {
    /// Load a block from the `blocks.{i}` namespace.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight
    /// fails to load.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &OthelloGptConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.n_embd;
        let inter = config.mlp_ratio * h;
        // CAST: usize → f64, head_dim fits in f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        let ln_cfg = candle_nn::LayerNormConfig {
            eps: config.norm_eps,
            ..Default::default()
        };

        Ok(Self {
            ln1: candle_nn::layer_norm(h, ln_cfg, vb.pp("ln1"))?,
            ln2: candle_nn::layer_norm(h, ln_cfg, vb.pp("ln2"))?,
            qkv: candle_nn::linear(h, 3 * h, vb.pp("attn").pp("qkv"))?,
            proj: candle_nn::linear(h, h, vb.pp("attn").pp("proj"))?,
            mlp_fc: candle_nn::linear(h, inter, vb.pp("mlp").pp("0"))?,
            mlp_proj: candle_nn::linear(inter, h, vb.pp("mlp").pp("2"))?,
            n_heads: config.n_head,
            head_dim: config.head_dim,
            hidden_dim: h,
            scale,
            causal: config.causal,
        })
    }

    /// Self-attention sublayer (bidirectional unless `causal`).
    ///
    /// Captures `AttnQ`/`AttnK`/`AttnV` (post-reshape), `AttnScores`
    /// (post-scale, post-mask), and `AttnPattern` (post-softmax).  No `RoPE`
    /// is applied — positional information is carried by the learned absolute
    /// embedding added at the model input.
    ///
    /// # Shapes
    /// - `xs`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn attention(
        &self,
        xs: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let hidden = self.hidden_dim;

        let qkv = self.qkv.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, hidden)?;
        let k = qkv.narrow(D::Minus1, hidden, hidden)?;
        let v = qkv.narrow(D::Minus1, 2 * hidden, hidden)?;

        // [batch, seq, n_heads, head_dim] → [batch, n_heads, seq, head_dim]
        let mut q = q
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        hook_point(&mut q, HookPoint::AttnQ(layer_idx), hooks, cache)?;
        hook_point(&mut k, HookPoint::AttnK(layer_idx), hooks, cache)?;
        hook_point(&mut v, HookPoint::AttnV(layer_idx), hooks, cache)?;

        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let k_t = k.contiguous()?.transpose(2, 3)?;
        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let q = q.contiguous()?;
        let mut scores = (q.matmul(&k_t)? * self.scale)?;
        if self.causal {
            let mask =
                crate::util::masks::create_causal_mask(seq_len, scores.device(), scores.dtype())?;
            scores = scores.broadcast_add(&mask)?;
        }
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        // Softmax in F32 (no-op promote on the F32 default path; defensive for
        // lower-precision loads).
        // Promote/softmax/demote, shared so the three backbones cannot drift.
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
            .reshape((batch, seq_len, hidden))?;

        Ok(self.proj.forward(&attn)?)
    }

    /// Exact-`GELU` MLP: `proj(gelu_erf(fc(x)))`.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.mlp_fc.forward(x)?;
        // Exact (erf) GELU — PyTorch `nn.GELU()` default. `Tensor::gelu()` is the
        // tanh approximation and must NOT be used here.
        let act = up.gelu_erf()?;
        Ok(self.mlp_proj.forward(&act)?)
    }

    /// Run the full block forward with hook support.
    ///
    /// Hook semantics follow the `TransformerLens` "added to the residual
    /// stream" convention: `AttnOut` / `MlpOut` capture the (ungated) sublayer
    /// contributions actually summed into the residual.
    ///
    /// # Shapes
    /// - `hidden_in`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
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
        // Backward-safe dispatch: with-bias `LayerNorm` takes candle's fused
        // (no-backward) kernel; route through `nn_ops` so training works.
        let normed1 = crate::nn_ops::layer_norm(&self.ln1, &residual)?;
        let mut attn = self.attention(&normed1, layer_idx, hooks, cache)?;
        hook_point(&mut attn, HookPoint::AttnOut(layer_idx), hooks, cache)?;
        hidden = (residual + attn)?;
        hook_point(&mut hidden, HookPoint::ResidMid(layer_idx), hooks, cache)?;

        // --- MLP sublayer ---
        let residual2 = hidden.clone();
        let mut normed2 = crate::nn_ops::layer_norm(&self.ln2, &hidden)?;
        hook_point(&mut normed2, HookPoint::MlpPre(layer_idx), hooks, cache)?;
        let mut mlp_out = self.mlp(&normed2)?;
        hook_point(&mut mlp_out, HookPoint::MlpPost(layer_idx), hooks, cache)?;
        // No gating: the MLP output IS the residual contribution.
        hook_point(&mut mlp_out, HookPoint::MlpOut(layer_idx), hooks, cache)?;
        hidden = (residual2 + mlp_out)?;
        hook_point(&mut hidden, HookPoint::ResidPost(layer_idx), hooks, cache)?;

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// OthelloGpt
// ---------------------------------------------------------------------------

/// `OthelloGpt` backend: a plain GPT-2-style bidirectional transformer with
/// full hook support.
///
/// Load via [`OthelloGpt::load`] from a [`VarBuilder`] over a converted
/// `safetensors` checkpoint (see `scripts/convert_othello_mdlm.py`).  The
/// weight keys match the upstream `OthelloMDLM` state dict verbatim, so no
/// transposes or renames happen at load time.
pub struct OthelloGpt {
    /// Token embedding (`tok_emb.weight`).
    tok_emb: Embedding,
    /// Self-conditioning embedding (`self_cond_emb.weight`):
    /// `[vocab_size + 1, n_embd]`, present only when
    /// [`OthelloGptConfig::self_conditioning`] is set.  Row `vocab_size` is the
    /// reserved `NONE` id.
    self_cond_emb: Option<Embedding>,
    /// Learned absolute positional embedding (`pos_emb.weight`):
    /// `[block_size, n_embd]`.
    pos_emb: Tensor,
    /// Transformer blocks.
    blocks: Vec<OthelloBlock>,
    /// Final `LayerNorm` (`ln_f`, weight + bias).
    ln_f: LayerNorm,
    /// Untied vocabulary head (`head.weight`, no bias).
    head: Linear,
    /// Model configuration.
    config: OthelloGptConfig,
}

impl OthelloGpt {
    /// Load an `OthelloGpt` model from a [`VarBuilder`].
    ///
    /// The caller constructs the `VarBuilder` (buffered or mmap) and provides
    /// the parsed [`OthelloGptConfig`].  Weight keys are read verbatim from the
    /// upstream state dict: `tok_emb.weight`, `pos_emb.weight`,
    /// `blocks.{i}.{ln1,ln2,attn.qkv,attn.proj,mlp.0,mlp.2}`, `ln_f`, and
    /// `head.weight`.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` token ids to `[batch, seq, vocab_size]` logits.
    ///
    /// # Memory
    /// Loads the single converted `safetensors` (~100 MB for the released
    /// 25.3 M-param world model) through the caller's `VarBuilder`: near-zero
    /// extra copy with the `mmap` feature, or ~100 MB CPU when buffered.
    ///
    /// **Loading is not initializing.** Over an *empty* `VarMap`-backed
    /// `VarBuilder`, `load` creates every missing tensor with `VarBuilder::get`'s
    /// default init — `Const(0.)` — so `tok_emb.weight` and `pos_emb.weight`
    /// come out as exact zeros (no token identity, no position information).
    /// For a from-scratch trainable model use [`init`](Self::init), which
    /// applies the GPT-2 recipe from an explicit seed.
    ///
    /// **A checkpoint predating self-conditioning still loads.** When
    /// [`self_conditioning`](OthelloGptConfig::self_conditioning) is set but the
    /// checkpoint has no `self_cond_emb.weight`, the table is created as exact
    /// zeros at the `VarBuilder`'s dtype and device rather than failing. That is
    /// what lets an anchor checkpoint trained without the table keep its numbers
    /// to the last digit: a zero table contributes a zero vector at every
    /// position, so the forward is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight
    /// fails to load or a dimension is inconsistent with the checkpoint.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(config: OthelloGptConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.n_embd;

        let tok_emb = Embedding::new(vb.pp("tok_emb").get((config.vocab_size, h), "weight")?, h);
        let pos_emb = vb.pp("pos_emb").get((config.block_size, h), "weight")?;

        let self_cond_emb = if config.self_conditioning {
            // `+ 1` for the reserved `NONE` row at index `vocab_size`.
            let rows = config.vocab_size + 1;
            // A safetensors-backed `VarBuilder` errors on a missing tensor, so
            // the absent case is handled explicitly rather than relying on a
            // default init (which only a `VarMap` backend provides). Built at
            // `vb.dtype()`, not `F32`, so a `BF16` model does not silently get
            // an `F32` table.
            let weight = if vb.contains_tensor(SELF_COND_WEIGHT) {
                vb.pp("self_cond_emb").get((rows, h), "weight")?
            } else {
                Tensor::zeros((rows, h), vb.dtype(), vb.device())?
            };
            Some(Embedding::new(weight, h))
        } else {
            // The mirror of the case above, and the more dangerous one: a
            // checkpoint that *has* the table loaded under a config that says
            // the feature is off would run as a non-self-conditioned model and
            // return different logits with no error. Refuse instead.
            if vb.contains_tensor(SELF_COND_WEIGHT) {
                return Err(MIError::Config(format!(
                    "checkpoint contains `{SELF_COND_WEIGHT}` but the config has \
                     self_conditioning = false; set it to true, or the model would \
                     silently run without the channel it was trained with"
                )));
            }
            None
        };

        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(OthelloBlock::load(&config, vb.pp(format!("blocks.{i}")))?);
        }

        let ln_cfg = candle_nn::LayerNormConfig {
            eps: config.norm_eps,
            ..Default::default()
        };
        let ln_f = candle_nn::layer_norm(h, ln_cfg, vb.pp("ln_f"))?;
        let head = candle_nn::linear_no_bias(h, config.vocab_size, vb.pp("head"))?;

        Ok(Self {
            tok_emb,
            self_cond_emb,
            pos_emb,
            blocks,
            ln_f,
            head,
            config,
        })
    }

    /// Initialize a from-scratch `OthelloGpt` over `varmap` at `F32`, with the
    /// GPT-2 recipe, reproducible from `(config, seed)` alone.
    ///
    /// The `F32` shim over [`init_with_dtype`](Self::init_with_dtype), which
    /// documents the recipe in full and is the entry point to use when training
    /// at `BF16`.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` token ids to `[batch, seq, vocab_size]` logits.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor creation failure.
    /// Returns [`MIError::Model`] if `varmap`'s lock is poisoned.
    pub fn init(
        config: OthelloGptConfig,
        varmap: &VarMap,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        Self::init_with_dtype(config, varmap, device, seed, DType::F32)
    }

    /// Initialize a from-scratch `OthelloGpt` over `varmap` at `dtype`, with the
    /// GPT-2 recipe, reproducible from `(config, seed)` alone.
    ///
    /// Where [`load`](Self::load) *reads* weights that already exist, this
    /// *creates* them: `N(0, 0.02)` for embeddings and linear weights, zero
    /// biases, `LayerNorm` weight `1` / bias `0` — drawn from an explicitly
    /// seeded, algorithm-frozen generator (`crate::util::rng`), independent of
    /// the device RNG, so two runs with the same `(config, seed)` produce
    /// byte-identical weights.  Every parameter is registered in `varmap`, so
    /// `varmap.all_vars()` hands the full trainable set to an optimizer, and the
    /// forward pass is tracked end-to-end (see the `nn_ops` module).
    /// Same-named entries already present in `varmap` are replaced; pass a fresh
    /// `VarMap` unless re-initialization is intended.
    ///
    /// Every parameter is **created** at `dtype`, not merely requested at it.
    /// That distinction matters: `VarMap::get` validates shape only, so a
    /// pre-inserted `F32` tensor is handed back unchanged even under a `BF16`
    /// `VarBuilder`, which would yield a silently `F32` model. Passing
    /// `DType::BF16` halves activation bytes, the knob that moves the training
    /// batch-size ceiling.
    ///
    /// The Gaussian draws are generated at `f32` and then cast, so the RNG
    /// stream — and therefore the model a given seed produces — does not depend
    /// on `dtype`; only the storage precision does.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` token ids to `[batch, seq, vocab_size]` logits.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor creation failure.
    /// Returns [`MIError::Model`] if `varmap`'s lock is poisoned.
    pub fn init_with_dtype(
        config: OthelloGptConfig,
        varmap: &VarMap,
        device: &Device,
        seed: u64,
        dtype: DType,
    ) -> Result<Self> {
        let mut rng = crate::util::rng::seeded(seed);
        {
            let mut data = varmap.data().lock().map_err(|_| {
                MIError::Model(candle_core::Error::Msg(
                    "varmap lock poisoned (a thread panicked while holding it)".to_string(),
                ))
            })?;
            for (name, dims) in weight_shapes(&config) {
                let tensor = if is_zero_init(&name) {
                    Tensor::zeros(dims, dtype, device)?
                } else if is_norm_weight(&name) {
                    Tensor::ones(dims, dtype, device)?
                } else {
                    let n = dims.iter().product();
                    let samples = crate::util::randn::randn_f32(&mut rng, n, 0.02);
                    // Not a PROMOTE: the draws are `f32` so the stream stays
                    // dtype-independent, and this stores them at the model's
                    // precision — `F32` (a no-op) or narrower, never widening.
                    Tensor::from_vec(samples, dims, device)?.to_dtype(dtype)?
                };
                data.insert(name, Var::from_tensor(&tensor)?);
            }
        } // release the lock — `load` re-enters it through the VarBuilder

        Self::load(config, VarBuilder::from_varmap(varmap, dtype, device))
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &OthelloGptConfig {
        &self.config
    }

    /// The reserved `NONE` id for the self-conditioning grid: `vocab_size`.
    ///
    /// Use it at positions where the model has no previous prediction, such as
    /// decode round 0. Its embedding row is zero at creation, so a grid of all
    /// `NONE` reproduces the no-self-conditioning forward exactly.
    #[must_use]
    pub const fn self_cond_none_id(&self) -> u32 {
        // CAST: usize → u32, vocab sizes are far below u32::MAX (62 upstream)
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        {
            self.config.vocab_size as u32
        }
    }

    /// Forward pass with an optional self-conditioning input.
    ///
    /// `self_cond_ids` holds the model's own previous clean prediction, one id
    /// per position, or [`self_cond_none_id`](Self::self_cond_none_id) where
    /// there is none. It is embedded through `self_cond_emb` and **added to the
    /// residual stream before [`HookPoint::Embed`] fires**, so the logit lens
    /// and every capture read the stream the logits actually came from.
    ///
    /// This is the mechanism of Chen, Zhang and Hinton (2022): a masked
    /// diffusion model that can see its own draft can revise it, instead of
    /// committing irrevocably one position at a time.
    ///
    /// [`MIBackend::forward`] is exactly this method with `None`, and with
    /// `None` -- or with a grid of all `NONE` against a freshly created table --
    /// the result is bit-identical to a model built without the feature.
    ///
    /// The carry is a token grid, so it is detached by construction: no
    /// gradient flows back through it into the forward that produced it.
    ///
    /// # Shapes
    /// - `input_ids`: `[batch, seq]` `U32`
    /// - `self_cond_ids`: `[batch, seq]` `U32`, the same dims as `input_ids`,
    ///   every id in `0..=vocab_size`
    /// - returns: [`HookCache`] whose output is `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if `seq` exceeds
    /// `block_size`, if `self_cond_ids` is given while
    /// [`self_conditioning`](OthelloGptConfig::self_conditioning) is off, if it
    /// is not `U32`, if its dims differ from `input_ids`, or if any id exceeds
    /// `vocab_size`.
    /// Returns [`MIError::Intervention`] if a
    /// registered intervention is invalid at its hook point.
    pub fn forward_with_self_cond(
        &self,
        input_ids: &Tensor,
        self_cond_ids: Option<&Tensor>,
        hooks: &HookSpec,
    ) -> Result<HookCache> {
        let device = input_ids.device();
        let (_batch, seq_len) = input_ids.dims2()?;
        if seq_len > self.config.block_size {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "seq_len {seq_len} exceeds block_size {} (no positional embedding)",
                self.config.block_size
            ))));
        }

        // Token embedding + learned absolute positions (broadcast over batch).
        let mut hidden = self.tok_emb.forward(input_ids)?;

        // Self-conditioning, added BEFORE the positional add and before the
        // `Embed` hook. Skipped entirely when absent, so the no-carry path adds
        // no op at all rather than adding a zero — `x + 0.0` is bitwise `x` for
        // every finite `x` except `-0.0`, and "bit-identical" should not rest on
        // that.
        if let Some(ids) = self_cond_ids {
            let table = self.self_cond_emb.as_ref().ok_or_else(|| {
                MIError::Model(candle_core::Error::Msg(
                    "self_cond_ids given but self_conditioning is off in the config \
                     (enable it with OthelloGptConfig::with_self_conditioning)"
                        .to_string(),
                ))
            })?;
            self.validate_self_cond(ids, input_ids)?;
            hidden = hidden.add(&table.forward(ids)?)?;
        }

        let pos = self.pos_emb.narrow(0, 0, seq_len)?;
        hidden = hidden.broadcast_add(&pos)?;

        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);
        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, layer_idx, hooks, &mut cache)?;
        }

        let logits = self.head_forward(&hidden, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    /// Reject a malformed self-conditioning grid before it reaches the lookup.
    ///
    /// The id-range check is not redundant with candle's own. `index_select`
    /// raises `InvalidIndex` on CPU, but the CUDA kernel does not bounds-check,
    /// so an out-of-range id there reads whatever follows the table: a silent
    /// wrong answer on GPU only, which is the failure mode this crate can least
    /// afford. The cost is one `max_all` plus a scalar read, not a full copy.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if `ids` is not
    /// `U32`, has different dims from `input_ids`, or contains an id greater
    /// than `vocab_size`.
    fn validate_self_cond(&self, ids: &Tensor, input_ids: &Tensor) -> Result<()> {
        if ids.dtype() != DType::U32 {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "self_cond_ids dtype {:?} is not U32",
                ids.dtype()
            ))));
        }
        if ids.dims() != input_ids.dims() {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "self_cond_ids dims {:?} differ from input_ids dims {:?}",
                ids.dims(),
                input_ids.dims()
            ))));
        }
        let max_id: u32 = ids.flatten_all()?.max_all()?.to_vec0()?;
        let none_id = self.self_cond_none_id();
        if max_id > none_id {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "self_cond_ids contains id {max_id} (max allowed {none_id}, the NONE row)"
            ))));
        }
        Ok(())
    }

    /// Final `LayerNorm` then untied head, with the `FinalNorm` hook.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor
    /// failures.
    fn head_forward(
        &self,
        hidden: &Tensor,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut xs = crate::nn_ops::layer_norm(&self.ln_f, hidden)?;
        hook_point(&mut xs, HookPoint::FinalNorm, hooks, cache)?;
        Ok(self.head.forward(&xs)?)
    }
}

impl MIBackend for OthelloGpt {
    fn num_layers(&self) -> usize {
        self.config.n_layer
    }

    fn hidden_size(&self) -> usize {
        self.config.n_embd
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_heads(&self) -> usize {
        self.config.n_head
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        self.forward_with_self_cond(input_ids, None, hooks)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = crate::nn_ops::layer_norm(&self.ln_f, hidden)?;
        Ok(self.head.forward(&xs)?)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        let device = self.tok_emb.embeddings().device();
        let ids = Tensor::new(&[token_id], device)?;
        let emb = self.tok_emb.forward(&ids)?; // [1, hidden]
        Ok(emb.squeeze(0)?) // [hidden]
    }
}

// ---------------------------------------------------------------------------
// Weight-shape table (shared by `init` and the synthetic test loader)
// ---------------------------------------------------------------------------

/// The full `(name, shape)` table of an `OthelloGpt` checkpoint for `config`,
/// in a fixed order (embeddings, blocks in index order, final norm, head).
///
/// Single source of truth shared by [`OthelloGpt::init`] and the synthetic
/// test loader, so the initializer and the loader cannot drift apart.  The
/// fixed order also makes `init`'s seeded draws reproducible.
fn weight_shapes(config: &OthelloGptConfig) -> Vec<(String, Vec<usize>)> {
    let h = config.n_embd;
    let inter = config.mlp_ratio * h;
    // 2 embeddings + 12 per block + ln_f weight/bias + head.
    let mut shapes = Vec::with_capacity(5 + 12 * config.n_layer);

    shapes.push(("tok_emb.weight".to_string(), vec![config.vocab_size, h]));
    shapes.push(("pos_emb.weight".to_string(), vec![config.block_size, h]));
    for i in 0..config.n_layer {
        shapes.push((format!("blocks.{i}.ln1.weight"), vec![h]));
        shapes.push((format!("blocks.{i}.ln1.bias"), vec![h]));
        shapes.push((format!("blocks.{i}.ln2.weight"), vec![h]));
        shapes.push((format!("blocks.{i}.ln2.bias"), vec![h]));
        shapes.push((format!("blocks.{i}.attn.qkv.weight"), vec![3 * h, h]));
        shapes.push((format!("blocks.{i}.attn.qkv.bias"), vec![3 * h]));
        shapes.push((format!("blocks.{i}.attn.proj.weight"), vec![h, h]));
        shapes.push((format!("blocks.{i}.attn.proj.bias"), vec![h]));
        shapes.push((format!("blocks.{i}.mlp.0.weight"), vec![inter, h]));
        shapes.push((format!("blocks.{i}.mlp.0.bias"), vec![inter]));
        shapes.push((format!("blocks.{i}.mlp.2.weight"), vec![h, inter]));
        shapes.push((format!("blocks.{i}.mlp.2.bias"), vec![h]));
    }
    shapes.push(("ln_f.weight".to_string(), vec![h]));
    shapes.push(("ln_f.bias".to_string(), vec![h]));
    shapes.push(("head.weight".to_string(), vec![config.vocab_size, h]));

    // Appended LAST, deliberately. The table's order is what makes `init`'s
    // seeded draws reproducible, so a new entry inserted anywhere else would
    // shift the RNG stream for every tensor after it and silently change the
    // weights a given seed produces. `self_cond_emb` is zero-initialized and so
    // draws nothing today, but placing it at the end means that stays true even
    // if its init rule is ever changed. See `seeded-init-rng-stability.md`.
    if config.self_conditioning {
        shapes.push((SELF_COND_WEIGHT.to_string(), vec![config.vocab_size + 1, h]));
    }

    shapes
}

/// Checkpoint key of the self-conditioning table, used by the shape table, the
/// loader and the initializer so the three cannot disagree.
const SELF_COND_WEIGHT: &str = "self_cond_emb.weight";

/// Whether `name` is a `LayerNorm` weight — initialized to ones by
/// [`OthelloGpt::init`] (GPT-2 recipe), unlike embedding/linear weights.
fn is_norm_weight(name: &str) -> bool {
    name == "ln_f.weight" || name.ends_with(".ln1.weight") || name.ends_with(".ln2.weight")
}

/// Whether `name` is initialized to exact zeros by [`OthelloGpt::init`].
///
/// Biases, per the GPT-2 recipe, plus the self-conditioning table. The latter
/// is not a style choice: a zero table makes
/// [`forward_with_self_cond`](OthelloGpt::forward_with_self_cond) with no carry
/// bit-identical to [`MIBackend::forward`], which is what lets a checkpoint
/// trained without the feature keep its published numbers exactly.
// `.bias` is a checkpoint tensor-name suffix, not a file extension — the
// case-sensitivity lint does not apply to it.
#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn is_zero_init(name: &str) -> bool {
    name.ends_with(".bias") || name == SELF_COND_WEIGHT
}

// ---------------------------------------------------------------------------
// Synthetic VarBuilder for tests
// ---------------------------------------------------------------------------

/// Build an in-memory [`VarBuilder`] of zero-initialised weights matching the
/// `OthelloGpt` layout for `config`.  Test-only: exercises the full load and
/// forward path without any download.
#[cfg(test)]
fn synthetic_var_builder(
    config: &OthelloGptConfig,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, dims) in weight_shapes(config) {
        tensors.insert(name, Tensor::zeros(dims, DType::F32, device)?);
    }
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn tiny_config() -> OthelloGptConfig {
        // 2 layers, 2 heads, hidden 8 — small enough to run instantly.
        OthelloGptConfig::new(12, 6, 2, 2, 8, false).unwrap()
    }

    /// All `(name, values)` pairs of a fresh seeded `init`, sorted by name.
    fn init_weights(seed: u64) -> Vec<(String, Vec<f32>)> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let _model = OthelloGpt::init(tiny_config(), &varmap, &dev, seed).unwrap();
        // Single-statement lock so the guard is dropped before the sort.
        let mut out: Vec<(String, Vec<f32>)> = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| {
                let values = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
                (name.clone(), values)
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    #[test]
    fn init_is_reproducible_from_config_and_seed() {
        let a = init_weights(42);
        let b = init_weights(42);
        let c = init_weights(43);
        assert_eq!(a.len(), 29, "2-layer config must create 29 parameters");
        assert_eq!(a, b, "same (config, seed) must reproduce identical weights");
        assert_ne!(a, c, "different seeds must produce different weights");
    }

    // Exact-value checks on deliberately constant inits (zeros / ones).
    // Branch membership is decided here by name PATTERN, independently of
    // `is_norm_weight` (which `init` itself uses) — and the per-branch counts
    // are pinned, so a classifier bug cannot silently satisfy both sides.
    // `.bias` is a tensor-name suffix, not a file extension.
    #[allow(clippy::float_cmp, clippy::case_sensitive_file_extension_comparisons)]
    #[test]
    fn init_follows_gpt2_recipe() {
        let (mut biases, mut norm_weights, mut drawn) = (0usize, 0usize, 0usize);
        for (name, values) in &init_weights(7) {
            if name.ends_with(".bias") {
                biases += 1;
                assert!(values.iter().all(|v| *v == 0.0), "{name}: bias not zero");
            } else if name.contains(".ln") || name.starts_with("ln_f") {
                norm_weights += 1;
                assert!(values.iter().all(|v| *v == 1.0), "{name}: norm not ones");
            } else {
                drawn += 1;
                // Embedding / linear weights are N(0, 0.02) draws — in
                // particular NOT the `Const(0.)` that `load` over an empty
                // `VarMap` produces (the §5.1 gap this API closes).
                assert!(values.iter().any(|v| *v != 0.0), "{name}: all zeros");
            }
        }
        // tiny_config (2 layers): 6 biases/block + ln_f.bias = 13; ln1/ln2
        // weights + ln_f.weight = 5; embeddings + qkv/proj/mlp linears + head
        // = 11.
        assert_eq!(
            (biases, norm_weights, drawn),
            (13, 5, 11),
            "recipe branch counts drifted"
        );
    }

    #[test]
    fn init_model_forward_produces_logits() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_config(), &varmap, &dev, 7).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3]], &dev).unwrap();
        let cache = MIBackend::forward(&model, &ids, &HookSpec::new()).unwrap();
        assert_eq!(cache.output().dims(), &[1, 3, 12]);
    }

    // `project_to_vocab` must preserve rank. The trait documented rank 2 only,
    // while this backend's `layer_norm` + `Linear` accepted rank 3 all along --
    // silent tolerance a probe could not rely on, since the two `stoicheia`
    // backends rejected the same input. The contract is now explicit, and this
    // asserts it here so a future rank-2 assertion cannot land unnoticed. See
    // docs/dogfooding-feedbacks/interp-api-forces-stringly-typed-hook-handling.md.
    #[test]
    fn project_to_vocab_preserves_rank() {
        use candle_core::IndexOp;

        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_config(), &varmap, &dev, 11).unwrap();
        // `tiny_config` is vocab 12, hidden 8.
        let (batch, seq, hidden_size) = (2, 3, 8);
        let hidden = Tensor::randn(0.0_f32, 1.0, (batch, seq, hidden_size), &dev).unwrap();

        let seq_logits = MIBackend::project_to_vocab(&model, &hidden).unwrap();
        assert_eq!(seq_logits.dims(), &[batch, seq, 12]);

        // Every position must equal the rank-2 projection of that same position,
        // so rank preservation cannot be satisfied by computing something else.
        for b in 0..batch {
            for s in 0..seq {
                let position = hidden.i((b, s, ..)).unwrap().unsqueeze(0).unwrap();
                let expected = MIBackend::project_to_vocab(&model, &position).unwrap();
                assert_eq!(expected.dims(), &[1, 12]);
                let actual = seq_logits.i((b, s, ..)).unwrap().unsqueeze(0).unwrap();
                let diff = (actual - expected)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                assert!(diff < 1e-5, "position [{b}][{s}] differs by {diff}");
            }
        }
    }

    // The §2 regression of docs/dogfooding-feedbacks/trainable-backbones.md:
    // before the `nn_ops` dispatch, backward() through this exact forward
    // silently stopped at the first fused op and only `head.weight` (1 of 29
    // parameters) received a gradient — while the loss still decreased. This
    // test is the measurement itself: build over a `VarMap`, run the real
    // `MIBackend::forward`, backward, and assert EVERY parameter receives a
    // gradient. Being a count over all vars, it fails loudly on any future
    // fused-op barrier anywhere in the forward (attention softmax, ln1/ln2,
    // ln_f), not just the sites known today.
    #[test]
    fn backward_reaches_every_parameter() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_config(), &varmap, &dev, 0).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let cache = MIBackend::forward(&model, &ids, &HookSpec::new()).unwrap();
        let output = cache.output();
        assert!(
            output.track_op(),
            "forward over a VarMap must produce a tracked output"
        );
        let grads = output.sum_all().unwrap().backward().unwrap();

        // Block-scoped lock so the guard is dropped before the asserts.
        let (n_params, mut missing) = {
            let data = varmap.data().lock().unwrap();
            let missing: Vec<String> = data
                .iter()
                .filter(|(_, var)| grads.get(var.as_tensor()).is_none())
                .map(|(name, _)| name.clone())
                .collect();
            (data.len(), missing)
        };
        missing.sort();
        assert_eq!(n_params, 29, "tiny 2-layer config must have 29 parameters");
        assert!(
            missing.is_empty(),
            "parameters receiving no gradient (a fused-op barrier regressed): {missing:?}"
        );
    }

    // The dtype counterpart of the gradient count above, and the reason
    // `init_with_dtype` *creates* tensors rather than only passing `dtype` down
    // to `load`: `VarMap::get` validates shape ONLY (candle-nn 0.11,
    // var_map.rs:103-111), returning any pre-inserted tensor unchanged. A dtype
    // parameter alone would therefore hand back the F32 vars `init` had already
    // inserted, producing a silently F32 model under a BF16 `VarBuilder` — and a
    // bf16 batch sweep would then measure nothing while appearing to run. Being
    // a count over all vars, this fails on any future path that reintroduces a
    // hardcoded dtype.
    #[test]
    fn init_with_dtype_creates_every_parameter_at_the_requested_dtype() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model =
            OthelloGpt::init_with_dtype(tiny_config(), &varmap, &dev, 0, DType::BF16).unwrap();

        // Block-scoped lock so the guard is dropped before the asserts.
        let (n_params, mut wrong) = {
            let data = varmap.data().lock().unwrap();
            let wrong: Vec<String> = data
                .iter()
                .filter(|(_, var)| var.dtype() != DType::BF16)
                .map(|(name, var)| format!("{name} ({:?})", var.dtype()))
                .collect();
            (data.len(), wrong)
        };
        wrong.sort();
        assert_eq!(n_params, 29, "tiny 2-layer config must have 29 parameters");
        assert!(
            wrong.is_empty(),
            "parameters not created at the requested dtype: {wrong:?}"
        );

        // The model itself is the other half of the assertion: `load` read every
        // var back through the `BF16` `VarBuilder` without a shape or dtype
        // error, so the weights the forward will use are the ones checked above.
        assert_eq!(model.config().n_layer, 2);

        // Not asserted here: that the forward *returns* BF16 logits. candle
        // 0.11's CPU backend has no BF16 matmul ("unsupported dtype BF16 for op
        // matmul"), so a bf16 forward is a GPU-only path and does not belong in
        // a CPU unit test. It is exercised by the bf16 batch sweep on the GPU.
    }

    /// `init` must remain the exact `F32` shim, so existing seeds and every
    /// parity baseline derived from them keep their meaning.
    #[test]
    fn init_matches_init_with_dtype_at_f32() {
        let dev = Device::Cpu;
        let shim = {
            let varmap = VarMap::new();
            OthelloGpt::init(tiny_config(), &varmap, &dev, 3).unwrap();
            varmap
        };
        let explicit = {
            let varmap = VarMap::new();
            OthelloGpt::init_with_dtype(tiny_config(), &varmap, &dev, 3, DType::F32).unwrap();
            varmap
        };

        let dump = |vm: &VarMap| -> Vec<(String, Vec<f32>)> {
            let mut out: Vec<(String, Vec<f32>)> = vm
                .data()
                .lock()
                .unwrap()
                .iter()
                .map(|(name, var)| {
                    let values = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
                    (name.clone(), values)
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        assert_eq!(dump(&shim), dump(&explicit));
    }

    #[test]
    fn config_derives_head_dim() {
        let cfg = OthelloGptConfig::new(62, 60, 8, 8, 512, false).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.mlp_ratio, 4);
        assert!(!cfg.causal);
    }

    #[test]
    fn config_rejects_indivisible_head_count() {
        assert!(OthelloGptConfig::new(62, 60, 8, 7, 512, false).is_err());
    }

    #[test]
    fn config_parses_companion_json() {
        let json = serde_json::json!({
            "vocab_size": 62,
            "block_size": 60,
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "dropout": 0.0,
            "causal": false
        });
        let cfg = OthelloGptConfig::from_hf_config(&json).unwrap();
        assert_eq!(cfg.vocab_size, 62);
        assert_eq!(cfg.block_size, 60);
        assert_eq!(cfg.n_layer, 8);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn config_missing_key_errors() {
        let json = serde_json::json!({ "vocab_size": 62 });
        assert!(OthelloGptConfig::from_hf_config(&json).is_err());
    }

    #[test]
    fn forward_runs_and_shapes_match() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.hidden_size(), 8);
        assert_eq!(model.vocab_size(), 12);
        assert_eq!(model.num_heads(), 2);

        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let hooks = HookSpec::new();
        let cache = model.forward(&input, &hooks).unwrap();
        let (batch, seq, vocab) = cache.output().dims3().unwrap();
        assert_eq!((batch, seq, vocab), (1, 4, 12));
    }

    #[test]
    fn causal_mask_is_upper_triangular() {
        let device = Device::Cpu;
        let mask = crate::util::masks::create_causal_mask(3, &device, DType::F32).unwrap();
        assert_eq!(mask.dims4().unwrap(), (1, 1, 3, 3));
        let values: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        let neg = f32::NEG_INFINITY;
        // Row i forbids columns j > i (strictly-upper triangle is -inf).
        assert_eq!(values, vec![0.0, neg, neg, 0.0, 0.0, neg, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn causal_model_runs_and_shapes_match() {
        // The autoregressive Othello-GPT control loads via the same module with
        // `causal = true`, exercising the causal-mask path.
        let device = Device::Cpu;
        let cfg = OthelloGptConfig::new(12, 6, 2, 2, 8, true).unwrap();
        assert!(cfg.causal);
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
        let cache = model.forward(&input, &HookSpec::new()).unwrap();
        assert_eq!(cache.output().dims3().unwrap(), (1, 4, 12));
    }

    #[test]
    fn hooks_capture_standard_points() {
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();
        let mut hooks = HookSpec::new();
        hooks
            .capture(HookPoint::Embed)
            .capture(HookPoint::ResidPost(0))
            .capture(HookPoint::ResidPost(1))
            .capture(HookPoint::AttnOut(0))
            .capture(HookPoint::MlpOut(1))
            .capture(HookPoint::FinalNorm);
        let cache = model.forward(&input, &hooks).unwrap();

        assert!(cache.get(&HookPoint::Embed).is_some());
        assert!(cache.get(&HookPoint::ResidPost(0)).is_some());
        assert!(cache.get(&HookPoint::ResidPost(1)).is_some());
        assert!(cache.get(&HookPoint::AttnOut(0)).is_some());
        assert!(cache.get(&HookPoint::MlpOut(1)).is_some());
        assert!(cache.get(&HookPoint::FinalNorm).is_some());

        // ResidPost has the residual-stream shape [batch, seq, hidden].
        let rp = cache.get(&HookPoint::ResidPost(0)).unwrap();
        assert_eq!(rp.dims3().unwrap(), (1, 3, 8));
    }

    /// `BACKENDS.md`'s conformance case for `OthelloGpt`: `Intervention::Zero`
    /// at `ResidPost(0)` must change the output, which proves the hook is wired
    /// to the tensor the forward pass actually carries forward.
    ///
    /// Seeded weights via `init`, not `synthetic_var_builder`, and that choice
    /// is the test. The synthetic builder zero-initialises every tensor, so the
    /// residual at `ResidPost(0)` is already zero and zeroing it again is a
    /// no-op: the assertion would pass whether or not the intervention ran, and
    /// the test would certify nothing. `intervention_add_propagates_downstream`
    /// below can use that rig precisely because `Add` is visible against zeros.
    ///
    /// Three backends failed this conformance case silently for releases
    /// because nothing asserted it (fixed in v0.2.0); `OthelloGpt` was never
    /// among them, and this pins that.
    #[test]
    fn honours_intervention_at_resid_post() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_config(), &varmap, &device, 7).unwrap();
        let input = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();

        let logits = |hooks: &HookSpec| -> Vec<f32> {
            model
                .forward(&input, hooks)
                .unwrap()
                .output()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };

        let baseline = logits(&HookSpec::new());

        let mut hooks = HookSpec::new();
        hooks.intervene(HookPoint::ResidPost(0), crate::hooks::Intervention::Zero);
        let treated = logits(&hooks);

        assert_ne!(
            baseline, treated,
            "Intervention::Zero at ResidPost(0) must change the output"
        );
    }

    #[test]
    fn intervention_add_propagates_downstream() {
        // The P4 pattern: add a steering vector at ResidPost(layer) and verify
        // it flows into the next block. We capture ResidPre(1) — the residual
        // entering block 1, i.e. block 0's output *after* the ResidPost(0)
        // intervention — and compare it to the un-steered baseline.
        let device = Device::Cpu;
        let cfg = tiny_config();
        let vb = synthetic_var_builder(&cfg, &device).unwrap();
        let model = OthelloGpt::load(cfg, vb).unwrap();

        let input = Tensor::new(&[[1u32, 2, 3]], &device).unwrap();

        let mut base_hooks = HookSpec::new();
        base_hooks.capture(HookPoint::ResidPre(1));
        let base = model.forward(&input, &base_hooks).unwrap();
        let base_pre: Vec<f32> = base
            .get(&HookPoint::ResidPre(1))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let steer = Tensor::ones(8, DType::F32, &device).unwrap();
        let mut hooks = HookSpec::new();
        hooks.capture(HookPoint::ResidPre(1)).intervene(
            HookPoint::ResidPost(0),
            crate::hooks::Intervention::Add(steer),
        );
        let steered = model.forward(&input, &hooks).unwrap();
        let steered_pre: Vec<f32> = steered
            .get(&HookPoint::ResidPre(1))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // With zero weights the residual is 0 until the +1 steer at ResidPost(0),
        // so block 1 sees 1.0 everywhere while the baseline sees 0.0.
        for (b, s) in base_pre.iter().zip(steered_pre.iter()) {
            assert!((b - 0.0).abs() < 1e-6, "baseline ResidPre(1) should be 0");
            assert!((s - 1.0).abs() < 1e-6, "steered ResidPre(1) should be 1");
        }
    }

    // -- Self-conditioning -------------------------------------------------

    fn tiny_self_cond_config() -> OthelloGptConfig {
        tiny_config().with_self_conditioning(true)
    }

    fn logits_of(cache: &HookCache) -> Vec<f32> {
        cache.output().flatten_all().unwrap().to_vec1().unwrap()
    }

    /// The switch-off guarantee, and the reason the table is zero-initialised:
    /// `forward`, `forward_with_self_cond(None)` and an all-`NONE` grid must
    /// agree **bit for bit**, so a checkpoint trained without the feature keeps
    /// its published numbers when the flag is turned on.
    #[test]
    fn self_conditioning_off_is_bit_identical() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_self_cond_config(), &varmap, &dev, 7).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let hooks = HookSpec::new();

        let none_grid = Tensor::full(model.self_cond_none_id(), (1, 4), &dev).unwrap();

        let a = logits_of(&model.forward(&ids, &hooks).unwrap());
        let b = logits_of(&model.forward_with_self_cond(&ids, None, &hooks).unwrap());
        let c = logits_of(
            &model
                .forward_with_self_cond(&ids, Some(&none_grid), &hooks)
                .unwrap(),
        );

        assert_eq!(a, b, "MIBackend::forward must equal the None path exactly");
        assert_eq!(
            a, c,
            "an all-NONE grid must equal the no-carry forward exactly"
        );
    }

    /// The table is wired in, not dead: a non-`NONE` carry moves the logits.
    /// Paired with the test above, this is what distinguishes "off by default"
    /// from "silently ignored".
    #[test]
    fn non_none_self_cond_changes_logits() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_self_cond_config(), &varmap, &dev, 7).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let hooks = HookSpec::new();

        // A freshly created table is zeros, so give one row a signal first.
        let rows = model.config().vocab_size + 1;
        let mut filled = vec![0.0f32; rows * model.config().n_embd];
        // Row 0 only: a constant is enough to make the lookup non-zero, and it
        // avoids an index-to-float cast that buys the test nothing.
        for v in filled.iter_mut().take(model.config().n_embd) {
            *v = 0.25;
        }
        {
            let mut data = varmap.data().lock().unwrap();
            let t = Tensor::from_vec(filled, (rows, model.config().n_embd), &dev).unwrap();
            data.insert(SELF_COND_WEIGHT.to_string(), Var::from_tensor(&t).unwrap());
        }
        let model = OthelloGpt::load(
            tiny_self_cond_config(),
            VarBuilder::from_varmap(&varmap, DType::F32, &dev),
        )
        .unwrap();

        let baseline = logits_of(&model.forward(&ids, &hooks).unwrap());
        // Row 0 is the one we filled; NONE is row `vocab_size`.
        let carry = Tensor::zeros((1, 4), DType::U32, &dev).unwrap();
        let treated = logits_of(
            &model
                .forward_with_self_cond(&ids, Some(&carry), &hooks)
                .unwrap(),
        );

        assert_ne!(baseline, treated, "a non-NONE carry must change the logits");
    }

    /// Appending `self_cond_emb.weight` **last** in `weight_shapes` must leave
    /// the seeded RNG stream for every other tensor untouched, so a seed that
    /// produced a given model before the feature existed still does.
    #[test]
    fn self_conditioning_does_not_move_the_seeded_stream() {
        let dev = Device::Cpu;
        let dump = |cfg: OthelloGptConfig| -> Vec<(String, Vec<f32>)> {
            let varmap = VarMap::new();
            OthelloGpt::init(cfg, &varmap, &dev, 11).unwrap();
            let mut out: Vec<(String, Vec<f32>)> = varmap
                .data()
                .lock()
                .unwrap()
                .iter()
                .map(|(n, v)| {
                    (
                        n.clone(),
                        v.as_tensor().flatten_all().unwrap().to_vec1().unwrap(),
                    )
                })
                .collect();
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };

        let without = dump(tiny_config());
        let with: Vec<(String, Vec<f32>)> = dump(tiny_self_cond_config())
            .into_iter()
            .filter(|(n, _)| n != SELF_COND_WEIGHT)
            .collect();

        assert_eq!(
            without, with,
            "enabling self-conditioning must not change any other tensor for a given seed"
        );
    }

    /// A checkpoint saved before the feature existed loads into a
    /// `self_conditioning: true` config with a zero table, and forwards the same.
    #[test]
    fn checkpoint_without_the_table_loads_as_zeros() {
        let dev = Device::Cpu;
        // Shapes of the *old* layout: no `self_cond_emb.weight`.
        let vb = synthetic_var_builder(&tiny_config(), &dev).unwrap();
        let model = OthelloGpt::load(tiny_self_cond_config(), vb).unwrap();

        let table = model.self_cond_emb.as_ref().expect("table must exist");
        let values: Vec<f32> = table.embeddings().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(table.embeddings().dims(), &[13, 8], "vocab_size + 1 rows");
        assert!(
            values.iter().all(|v| v.abs() < f32::EPSILON),
            "absent table must be zeros"
        );
    }

    /// `backward()` must reach the rows the carry actually used. candle's
    /// embedding backward produces a gradient for the WHOLE table with zeros in
    /// unused rows, so this asserts on row *values*, not on `grads.get`.
    #[test]
    fn backward_reaches_used_self_cond_rows_only() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_self_cond_config(), &varmap, &dev, 3).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        // Rows 0 and 5 are used; NONE (row 12) is deliberately absent.
        let carry = Tensor::new(&[[0u32, 5, 0, 5]], &dev).unwrap();

        let cache = model
            .forward_with_self_cond(&ids, Some(&carry), &HookSpec::new())
            .unwrap();
        let grads = cache.output().sum_all().unwrap().backward().unwrap();

        let table = model.self_cond_emb.as_ref().unwrap().embeddings();
        let grad = grads
            .get(table)
            .expect("self_cond_emb must receive a gradient");
        let rows: Vec<Vec<f32>> = grad.to_vec2().unwrap();

        let row_sum = |r: &Vec<f32>| r.iter().map(|v| v.abs()).sum::<f32>();
        assert!(
            row_sum(&rows[0]) > 0.0,
            "row 0 was used and must have a gradient"
        );
        assert!(
            row_sum(&rows[5]) > 0.0,
            "row 5 was used and must have a gradient"
        );
        assert!(
            row_sum(&rows[12]) < f32::EPSILON,
            "the NONE row was not used and must have a zero gradient"
        );
    }

    /// Every rejection path, so none of them can become a silent accept.
    #[test]
    fn malformed_self_cond_is_rejected() {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let model = OthelloGpt::init(tiny_self_cond_config(), &varmap, &dev, 1).unwrap();
        let ids = Tensor::new(&[[1u32, 2, 3, 4]], &dev).unwrap();
        let hooks = HookSpec::new();

        // Wrong dtype.
        let f32_grid = Tensor::zeros((1, 4), DType::F32, &dev).unwrap();
        assert!(
            model
                .forward_with_self_cond(&ids, Some(&f32_grid), &hooks)
                .is_err(),
            "a non-U32 grid must be rejected"
        );

        // Wrong dims.
        let short = Tensor::zeros((1, 3), DType::U32, &dev).unwrap();
        assert!(
            model
                .forward_with_self_cond(&ids, Some(&short), &hooks)
                .is_err(),
            "a grid whose dims differ from input_ids must be rejected"
        );

        // Out-of-range id (NONE is 12, so 13 is past the table).
        let oob = Tensor::new(&[[13u32, 0, 0, 0]], &dev).unwrap();
        assert!(
            model
                .forward_with_self_cond(&ids, Some(&oob), &hooks)
                .is_err(),
            "an id past the NONE row must be rejected"
        );

        // Carry supplied while the feature is off.
        let plain_varmap = VarMap::new();
        let plain = OthelloGpt::init(tiny_config(), &plain_varmap, &dev, 1).unwrap();
        let ok_grid = Tensor::zeros((1, 4), DType::U32, &dev).unwrap();
        assert!(
            plain
                .forward_with_self_cond(&ids, Some(&ok_grid), &hooks)
                .is_err(),
            "a carry must be rejected when self_conditioning is off, never ignored"
        );
    }

    /// The companion `config.json` key round-trips, and is absent-means-off.
    #[test]
    fn self_conditioning_parses_from_companion_json() {
        let base = serde_json::json!({
            "vocab_size": 62, "block_size": 60, "n_layer": 8,
            "n_head": 8, "n_embd": 512,
        });
        assert!(
            !OthelloGptConfig::from_hf_config(&base)
                .unwrap()
                .self_conditioning,
            "absent `self_conditioning` must read as false"
        );

        let mut on = base.clone();
        on["self_conditioning"] = serde_json::json!(true);
        assert!(
            OthelloGptConfig::from_hf_config(&on)
                .unwrap()
                .self_conditioning
        );
    }
}
