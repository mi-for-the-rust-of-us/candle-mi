// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MDLM` masked-diffusion `DiT` backend.
//!
//! A faithful Rust port of `kuleshov-group/mdlm-owt` (Sahoo et al., `NeurIPS`
//! 2024), validated against the flash-attn-free fp32 reference
//! `TheQweaker/mdlm-owt-noflash`.  Unlike the causal
//! `GenericTransformer`, `MDLM` is a
//! **bidirectional** `DiT`: each block applies `adaLN` modulation (constant
//! here, since the checkpoint is time-independent), full self-attention with
//! no causal mask, weight-only `LayerNorm`, and a plain `GELU`-tanh MLP.
//!
//! Output logits are raw — the `SUBS` masking (forbidding the `[MASK]` token)
//! belongs to the diffusion sampler, not the forward pass.

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};

use crate::backend::MIBackend;
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};

use super::config::MdlmConfig;
use super::rope::MdlmRope;

/// Frequency-embedding size of the upstream `TimestepEmbedder` (`sigma_map`).
const FREQ_EMBED: usize = 256;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `adaLN` modulation: `x * (1 + scale) + shift`.
///
/// `shift` and `scale` are `[1, hidden]` and broadcast over the leading
/// dimensions of `x` (numpy right-alignment), so this works for both
/// `[batch, seq, hidden]` block activations and `[batch, hidden]` logit-lens
/// inputs.
///
/// # Shapes
/// - `x`: `[.., hidden]`
/// - `shift` / `scale`: `[1, hidden]`
/// - returns: same shape as `x`
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scaled = x.broadcast_mul(&(scale + 1.0)?)?;
    Ok(scaled.broadcast_add(shift)?)
}

/// Load a weight-only `LayerNorm` (no bias, mean-subtracting, `eps`).
///
/// `MDLM` stores only the scale (`norm.weight`) and applies
/// `F.layer_norm(x) * weight` — exactly `candle_nn::LayerNorm::new_no_bias`.
///
/// # Errors
///
/// Returns [`MIError::Model`] if the weight tensor
/// cannot be loaded.
#[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
fn load_layer_norm(vb: VarBuilder<'_>, hidden: usize, eps: f64) -> Result<LayerNorm> {
    let weight = vb.get(hidden, "weight")?;
    Ok(LayerNorm::new_no_bias(weight, eps))
}

/// Compute the constant `adaLN` conditioning vector `c = silu(sigma_map(0))`.
///
/// The released checkpoint is time-independent (`time_conditioning = false`),
/// so the network zeroes the diffusion time internally and the conditioning
/// is a single fixed vector.  Evaluating the `TimestepEmbedder` at `t = 0`
/// (`timestep_embedding(0) = [cos(0)…, sin(0)…] = [1…, 0…]`) and applying
/// `silu` reproduces it once, at load time.
///
/// # Shapes
/// - returns: `[1, cond_dim]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failures.
#[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
fn compute_conditioning(
    vb_sigma: VarBuilder<'_>,
    cond_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let fc1 = candle_nn::linear(FREQ_EMBED, cond_dim, vb_sigma.pp("mlp").pp("0"))?;
    let fc2 = candle_nn::linear(cond_dim, cond_dim, vb_sigma.pp("mlp").pp("2"))?;

    // timestep_embedding(0, 256) = cat([cos(0)×128, sin(0)×128]) = [1×128, 0×128].
    let emb: Vec<f32> = (0..FREQ_EMBED)
        .map(|i| if i < FREQ_EMBED / 2 { 1.0 } else { 0.0 })
        .collect();
    let emb = Tensor::from_vec(emb, (1, FREQ_EMBED), device)?.to_dtype(dtype)?;

    let h1 = candle_nn::ops::silu(&fc1.forward(&emb)?)?;
    let sig = fc2.forward(&h1)?;
    Ok(candle_nn::ops::silu(&sig)?)
}

// ---------------------------------------------------------------------------
// DiTBlock
// ---------------------------------------------------------------------------

/// A single `MDLM` `DiT` block: `adaLN`-modulated bidirectional attention and
/// `GELU`-tanh MLP, each gated by a constant conditioning vector.
struct DiTBlock {
    /// Pre-attention weight-only `LayerNorm`.
    norm1: LayerNorm,
    /// Pre-MLP weight-only `LayerNorm`.
    norm2: LayerNorm,
    /// `adaLN` modulation projection: `cond_dim → 6 * hidden` (with bias).
    adaln: Linear,
    /// Fused QKV projection: `hidden → 3 * hidden` (no bias).
    attn_qkv: Linear,
    /// Attention output projection: `hidden → hidden` (no bias).
    attn_out: Linear,
    /// MLP up-projection: `hidden → mlp_ratio * hidden` (with bias).
    mlp_fc: Linear,
    /// MLP down-projection: `mlp_ratio * hidden → hidden` (with bias).
    mlp_proj: Linear,
    /// Number of attention heads.
    n_heads: usize,
    /// Per-head dimension.
    head_dim: usize,
    /// Hidden dimension (`n_heads * head_dim`).
    hidden_dim: usize,
    /// Attention scale `1 / sqrt(head_dim)`.
    scale: f64,
}

impl DiTBlock {
    /// Load a `DiT` block from the `backbone.blocks.{i}` namespace.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight fails to load.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &MdlmConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.hidden_dim;
        let inter = config.mlp_ratio * h;
        // CAST: usize → f64, head_dim fits in f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = 1.0 / (config.head_dim as f64).sqrt();

        Ok(Self {
            norm1: load_layer_norm(vb.pp("norm1"), h, config.norm_eps)?,
            norm2: load_layer_norm(vb.pp("norm2"), h, config.norm_eps)?,
            adaln: candle_nn::linear(config.cond_dim, 6 * h, vb.pp("adaLN_modulation"))?,
            attn_qkv: candle_nn::linear_no_bias(h, 3 * h, vb.pp("attn_qkv"))?,
            attn_out: candle_nn::linear_no_bias(h, h, vb.pp("attn_out"))?,
            mlp_fc: candle_nn::linear(h, inter, vb.pp("mlp").pp("0"))?,
            mlp_proj: candle_nn::linear(inter, h, vb.pp("mlp").pp("2"))?,
            n_heads: config.n_heads,
            head_dim: config.head_dim,
            hidden_dim: h,
            scale,
        })
    }

    /// Bidirectional self-attention sublayer (pre-gate).
    ///
    /// Captures `AttnQ`/`AttnK`/`AttnV` (post-reshape, pre-`RoPE`),
    /// `AttnScores` (post-scale), and `AttnPattern` (post-softmax).  No causal
    /// mask is applied — attention is full bidirectional.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn attention(
        &self,
        xs: &Tensor,
        rope: &MdlmRope,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let hidden = self.hidden_dim;

        let qkv = self.attn_qkv.forward(xs)?;
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

        let q = rope.apply(&q)?;
        let k = rope.apply(&k)?;

        // Attention scores (bidirectional — no causal mask).
        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let k_t = k.contiguous()?.transpose(2, 3)?;
        let q = q.contiguous()?;
        let mut scores = (q.matmul(&k_t)? * self.scale)?;
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        // Softmax in F32 (no-op promote on the F32 default path; defensive for
        // lower-precision loads).
        let original_dtype = scores.dtype();
        // PROMOTE: softmax over a lower-precision dtype can produce NaN; compute in F32
        let scores_f32 = if original_dtype == DType::F32 {
            scores
        } else {
            scores.to_dtype(DType::F32)?
        };
        // Backward-safe dispatch: fused kernel for inference, composed form
        // when the graph is tracked (training over a `VarMap`).
        let mut pattern = crate::nn_ops::softmax_last_dim(&scores_f32)?;
        if original_dtype != DType::F32 {
            pattern = pattern.to_dtype(original_dtype)?;
        }
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

        Ok(self.attn_out.forward(&attn)?)
    }

    /// Plain `GELU`-tanh MLP: `proj(gelu(fc(x)))`.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn mlp(&self, x: &Tensor) -> Result<Tensor> {
        let up = self.mlp_fc.forward(x)?;
        let act = up.gelu()?; // tanh approximation, matching nn.GELU(approximate='tanh')
        Ok(self.mlp_proj.forward(&act)?)
    }

    /// Run the full block forward with hook support.
    ///
    /// Hook semantics: `AttnOut` / `MlpOut` capture the **gated** residual
    /// contributions (`gate_msa * attn`, `gate_mlp * mlp`), matching the
    /// `TransformerLens` "added to the residual stream" convention.
    ///
    /// # Shapes
    /// - `hidden_in`: `[batch, seq, hidden]`
    /// - `cond`: `[1, cond_dim]`
    /// - returns: `[batch, seq, hidden]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn forward(
        &self,
        hidden_in: &Tensor,
        cond: &Tensor,
        rope: &MdlmRope,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut hidden = hidden_in.clone();
        hook_point(&mut hidden, HookPoint::ResidPre(layer_idx), hooks, cache)?;
        let residual = hidden.clone();

        // adaLN modulation parameters from the constant conditioning vector:
        // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        let mods = self.adaln.forward(cond)?;
        let hidden_dim = self.hidden_dim;
        let shift_msa = mods.narrow(D::Minus1, 0, hidden_dim)?;
        let scale_msa = mods.narrow(D::Minus1, hidden_dim, hidden_dim)?;
        let gate_msa = mods.narrow(D::Minus1, 2 * hidden_dim, hidden_dim)?;
        let shift_mlp = mods.narrow(D::Minus1, 3 * hidden_dim, hidden_dim)?;
        let scale_mlp = mods.narrow(D::Minus1, 4 * hidden_dim, hidden_dim)?;
        let gate_mlp = mods.narrow(D::Minus1, 5 * hidden_dim, hidden_dim)?;

        // --- attention sublayer ---
        let normed1 = modulate(&self.norm1.forward(&residual)?, &shift_msa, &scale_msa)?;
        let attn = self.attention(&normed1, rope, layer_idx, hooks, cache)?;
        let mut attn_contrib = attn.broadcast_mul(&gate_msa)?;
        hook_point(
            &mut attn_contrib,
            HookPoint::AttnOut(layer_idx),
            hooks,
            cache,
        )?;
        hidden = (residual + attn_contrib)?;
        hook_point(&mut hidden, HookPoint::ResidMid(layer_idx), hooks, cache)?;

        // --- MLP sublayer ---
        let residual2 = hidden.clone();
        let mut normed2 = modulate(&self.norm2.forward(&hidden)?, &shift_mlp, &scale_mlp)?;
        hook_point(&mut normed2, HookPoint::MlpPre(layer_idx), hooks, cache)?;
        let mut mlp_out = self.mlp(&normed2)?;
        hook_point(&mut mlp_out, HookPoint::MlpPost(layer_idx), hooks, cache)?;
        let mut mlp_contrib = mlp_out.broadcast_mul(&gate_mlp)?;
        hook_point(&mut mlp_contrib, HookPoint::MlpOut(layer_idx), hooks, cache)?;
        hidden = (residual2 + mlp_contrib)?;
        hook_point(&mut hidden, HookPoint::ResidPost(layer_idx), hooks, cache)?;

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// DDitFinalLayer
// ---------------------------------------------------------------------------

/// `MDLM` output head: weight-only `LayerNorm`, `adaLN` shift/scale, then the
/// untied vocabulary projection.
struct DDitFinalLayer {
    /// Final weight-only `LayerNorm`.
    norm_final: LayerNorm,
    /// `adaLN` modulation projection: `cond_dim → 2 * hidden` (shift + scale).
    adaln: Linear,
    /// Untied vocabulary projection: `hidden → vocab_size` (with bias).
    linear: Linear,
    /// Hidden dimension (for slicing the modulation parameters).
    hidden_dim: usize,
}

impl DDitFinalLayer {
    /// Load the output head from the `backbone.output_layer` namespace.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if any weight fails to load.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &MdlmConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let h = config.hidden_dim;
        Ok(Self {
            norm_final: load_layer_norm(vb.pp("norm_final"), h, config.norm_eps)?,
            adaln: candle_nn::linear(config.cond_dim, 2 * h, vb.pp("adaLN_modulation"))?,
            linear: candle_nn::linear(h, config.vocab_size, vb.pp("linear"))?,
            hidden_dim: h,
        })
    }

    /// Modulated, normalized hidden state (pre-projection).
    ///
    /// # Shapes
    /// - `hidden`: `[.., hidden]`
    /// - `cond`: `[1, cond_dim]`
    /// - returns: `[.., hidden]`
    fn modulated(&self, hidden: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let mods = self.adaln.forward(cond)?;
        let hidden_dim = self.hidden_dim;
        let shift = mods.narrow(D::Minus1, 0, hidden_dim)?;
        let scale = mods.narrow(D::Minus1, hidden_dim, hidden_dim)?;
        modulate(&self.norm_final.forward(hidden)?, &shift, &scale)
    }

    /// Full output head with the `FinalNorm` hook (post-modulation, pre-projection).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden]`
    /// - returns: `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn forward(
        &self,
        hidden: &Tensor,
        cond: &Tensor,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let mut xs = self.modulated(hidden, cond)?;
        hook_point(&mut xs, HookPoint::FinalNorm, hooks, cache)?;
        Ok(self.linear.forward(&xs)?)
    }

    /// Logit projection without hooks (for [`MIBackend::project_to_vocab`]).
    ///
    /// Rank-preserving: `modulated` is rank-agnostic and its `shift` / `scale`
    /// broadcast from `[1, hidden]`, so a rank-3 hidden state projects too.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, hidden]` or `[batch, seq, hidden]`
    /// - returns: `[batch, vocab_size]` or `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor failures.
    fn project(&self, hidden: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let xs = self.modulated(hidden, cond)?;
        Ok(self.linear.forward(&xs)?)
    }
}

// ---------------------------------------------------------------------------
// GenericMdlm
// ---------------------------------------------------------------------------

/// `MDLM` masked-diffusion backend.
///
/// Loads `kuleshov-group/mdlm-owt`-format weights (prefix `backbone.`) and runs
/// the bidirectional `DiT` forward pass with full hook support, exactly
/// matching the fp32 reference.
pub struct GenericMdlm {
    /// Token embedding (raw `backbone.vocab_embed.embedding` parameter).
    vocab_embed: Embedding,
    /// `DiT` blocks.
    blocks: Vec<DiTBlock>,
    /// Output head (`backbone.output_layer`).
    output_layer: DDitFinalLayer,
    /// Rotary position embedding cache.
    rope: MdlmRope,
    /// Constant `adaLN` conditioning vector `c`: `[1, cond_dim]`.
    cond: Tensor,
    /// Model configuration.
    config: MdlmConfig,
}

impl GenericMdlm {
    /// Load an `MDLM` model from a [`VarBuilder`].
    ///
    /// The caller constructs the `VarBuilder` (buffered or mmap) and provides
    /// the parsed [`MdlmConfig`].  The constant conditioning vector and the
    /// `RoPE` cache are pre-computed here.
    ///
    /// # Shapes
    /// - returns: a model whose [`forward`](MIBackend::forward) maps
    ///   `[batch, seq]` token ids to `[batch, seq, vocab_size]` logits.
    ///
    /// # Memory
    /// Loads the single `mdlm-owt` `safetensors` (~648 MB) through the caller's
    /// `VarBuilder`: near-zero extra copy with the `mmap` feature, or ~648 MB
    /// CPU when buffered.  The conditioning vector and `RoPE` cache are negligible.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the checkpoint sets
    /// `time_conditioning = true` (unsupported), or
    /// [`MIError::Model`] if weight loading fails or a
    /// dimension is inconsistent with the checkpoint.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(
        config: MdlmConfig,
        device: &Device,
        dtype: DType,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // The conditioning vector is precomputed once at `t = 0`; that is only
        // correct for a time-independent checkpoint (`time_conditioning = false`,
        // as in the released `mdlm-owt`).  Reject the time-conditioned case rather
        // than silently producing wrong logits.
        if config.time_conditioning {
            return Err(MIError::Config(
                "MDLM time_conditioning=true is unsupported (the constant-conditioning \
                 fast path assumes a time-independent checkpoint)"
                    .into(),
            ));
        }

        let vb_b = vb.pp("backbone");

        let vocab_embed = Embedding::new(
            vb_b.pp("vocab_embed")
                .get((config.vocab_size, config.hidden_dim), "embedding")?,
            config.hidden_dim,
        );

        let cond = compute_conditioning(vb_b.pp("sigma_map"), config.cond_dim, device, dtype)?;

        let mut blocks = Vec::with_capacity(config.n_blocks);
        for i in 0..config.n_blocks {
            blocks.push(DiTBlock::load(&config, vb_b.pp(format!("blocks.{i}")))?);
        }

        let output_layer = DDitFinalLayer::load(&config, vb_b.pp("output_layer"))?;

        let rope = MdlmRope::new(
            config.head_dim,
            config.model_length,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            vocab_embed,
            blocks,
            output_layer,
            rope,
            cond,
            config,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &MdlmConfig {
        &self.config
    }
}

impl MIBackend for GenericMdlm {
    fn num_layers(&self) -> usize {
        self.config.n_blocks
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_dim
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_heads(&self) -> usize {
        self.config.n_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let mut hidden = self.vocab_embed.forward(input_ids)?;
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(
                &hidden, &self.cond, &self.rope, layer_idx, hooks, &mut cache,
            )?;
        }

        let logits = self
            .output_layer
            .forward(&hidden, &self.cond, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output_layer.project(hidden, &self.cond)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        let device = self.vocab_embed.embeddings().device();
        let ids = Tensor::new(&[token_id], device)?;
        let emb = self.vocab_embed.forward(&ids)?; // [1, hidden]
        Ok(emb.squeeze(0)?) // [hidden]
    }
}
