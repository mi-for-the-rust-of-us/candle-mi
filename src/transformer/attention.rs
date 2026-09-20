// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-head attention with GQA, separate/fused QKV, and hooks.
//!
//! Supports grouped-query attention (GQA), multi-head attention (MHA),
//! and multi-query attention (MQA) via the `num_kv_heads` configuration.

use candle_core::{D, Module, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};

use crate::config::{QkvLayout, TransformerConfig};
use crate::error::Result;
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};

use super::rope::RopeCache;

// ---------------------------------------------------------------------------
// QKV projection — separate or fused
// ---------------------------------------------------------------------------

/// QKV projection, either separate or fused.
enum QkvProj {
    /// Three separate linear layers: `q_proj`, `k_proj`, `v_proj`.
    Separate {
        /// Query projection.
        q_proj: Linear,
        /// Key projection.
        k_proj: Linear,
        /// Value projection.
        v_proj: Linear,
    },
    /// Single fused linear layer `qkv_proj`, split via `narrow()`.
    Fused {
        /// Fused QKV projection.
        qkv_proj: Linear,
        /// Query dimension (= `num_attention_heads * head_dim`).
        q_dim: usize,
        /// Key/value dimension (= `num_kv_heads * head_dim`).
        kv_dim: usize,
    },
}

impl QkvProj {
    /// Project input to Q, K, V tensors.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden_size]`
    /// - returns: `(Q, K, V)` each `[batch, seq, proj_dim]`
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let q = q_proj.forward(x)?;
                let k = k_proj.forward(x)?;
                let v = v_proj.forward(x)?;
                Ok((q, k, v))
            }
            Self::Fused {
                qkv_proj,
                q_dim,
                kv_dim,
            } => {
                let qkv = qkv_proj.forward(x)?;
                let q = qkv.narrow(D::Minus1, 0, *q_dim)?;
                let k = qkv.narrow(D::Minus1, *q_dim, *kv_dim)?;
                let v = qkv.narrow(D::Minus1, q_dim + kv_dim, *kv_dim)?;
                Ok((q, k, v))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Multi-head attention layer with hook points.
pub struct Attention {
    /// QKV projection (separate or fused).
    qkv: QkvProj,
    /// Output projection.
    o_proj: Linear,
    /// Number of query heads.
    num_attention_heads: usize,
    /// Number of key/value heads.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Attention scale factor: `1/sqrt(head_dim)` or `1/sqrt(query_pre_attn_scalar)`.
    scale: f64,
    /// Optional attention logit soft-capping value (Gemma 2).
    attn_logit_softcapping: Option<f64>,
    /// Optional per-head-dim `RMSNorm` applied to `Q` before `RoPE` (`Qwen3`).
    /// `Some(_)` when [`TransformerConfig::use_qk_norm`] is `true`; `None`
    /// otherwise.  Weight shape: `[head_dim]`.
    q_norm: Option<RmsNorm>,
    /// Optional per-head-dim `RMSNorm` applied to `K` before `RoPE` (`Qwen3`).
    /// `Some(_)` when [`TransformerConfig::use_qk_norm`] is `true`; `None`
    /// otherwise.  Weight shape: `[head_dim]`.
    k_norm: Option<RmsNorm>,
}

impl Attention {
    /// Load attention weights from a [`VarBuilder`].
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weight loading fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(config: &TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let qkv = match config.qkv_layout {
            QkvLayout::Separate => {
                let q_proj = if config.qkv_bias {
                    candle_nn::linear(config.hidden_size, q_dim, vb.pp("q_proj"))?
                } else {
                    candle_nn::linear_no_bias(config.hidden_size, q_dim, vb.pp("q_proj"))?
                };
                let k_proj = if config.qkv_bias {
                    candle_nn::linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?
                } else {
                    candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?
                };
                let v_proj = if config.qkv_bias {
                    candle_nn::linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?
                } else {
                    candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?
                };
                QkvProj::Separate {
                    q_proj,
                    k_proj,
                    v_proj,
                }
            }
            QkvLayout::Fused => {
                let total_dim = q_dim + 2 * kv_dim;
                let qkv_proj = if config.qkv_bias {
                    candle_nn::linear(config.hidden_size, total_dim, vb.pp("qkv_proj"))?
                } else {
                    candle_nn::linear_no_bias(config.hidden_size, total_dim, vb.pp("qkv_proj"))?
                };
                QkvProj::Fused {
                    qkv_proj,
                    q_dim,
                    kv_dim,
                }
            }
        };

        let o_proj = if config.o_proj_bias {
            candle_nn::linear(q_dim, config.hidden_size, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?
        };

        // CAST: usize → f64, head_dim fits in f64 mantissa
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = config.query_pre_attn_scalar.map_or_else(
            || 1.0 / (config.head_dim as f64).sqrt(),
            |scalar| 1.0 / scalar.sqrt(),
        );

        // Qwen3-style QK norm: per-head-dim `RMSNorm` of Q and K before RoPE.
        // The weight tensors live at `self_attn.q_norm.weight` /
        // `self_attn.k_norm.weight` with shape `[head_dim]`.
        let (q_norm, k_norm) = if config.use_qk_norm {
            let q_norm = candle_nn::rms_norm(config.head_dim, config.qk_norm_eps, vb.pp("q_norm"))?;
            let k_norm = candle_nn::rms_norm(config.head_dim, config.qk_norm_eps, vb.pp("k_norm"))?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };

        Ok(Self {
            qkv,
            o_proj,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale,
            attn_logit_softcapping: config.attn_logit_softcapping,
            q_norm,
            k_norm,
        })
    }

    /// Run the attention forward pass with hook capture and intervention.
    ///
    /// For `Qwen3`-family models (`TransformerConfig::use_qk_norm == true`),
    /// `RMSNorm` is applied per head dimension to `Q` and `K` after the
    /// projection-and-reshape step and **before** the
    /// [`HookPoint::AttnQ`] / [`HookPoint::AttnK`] capture/intervention
    /// block, so those hooks see the same post-`QK`-norm tensors that
    /// actually feed into `RoPE`.  For non-`Qwen3` models (`q_norm` /
    /// `k_norm` are `None`), behaviour is unchanged.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden_size]`
    /// - `mask`: `[1, 1, seq, seq]` — causal (or sliding window) mask
    /// - returns: `[batch, seq, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures.
    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: &RopeCache,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // --- QKV projection ---
        let (q, k, v) = self.qkv.forward(x)?;

        // Reshape to [batch, seq, n_heads, head_dim] then transpose to [batch, n_heads, seq, head_dim]
        let mut q = q
            .reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QK norm (Qwen3-style): per-head-dim `RMSNorm` of Q / K before
        // RoPE.  No-op when `q_norm` / `k_norm` are `None`.  Applied here so
        // the AttnQ / AttnK hooks below capture post-QK-norm tensors,
        // matching the values that actually feed into `RoPE` and attention.
        if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            q = qn.forward(&q)?;
            k = kn.forward(&k)?;
        }

        // Hook: capture and/or intervene on Q, K, V after reshape (before RoPE)
        hook_point(&mut q, HookPoint::AttnQ(layer_idx), hooks, cache)?;
        hook_point(&mut k, HookPoint::AttnK(layer_idx), hooks, cache)?;
        hook_point(&mut v, HookPoint::AttnV(layer_idx), hooks, cache)?;

        // --- Apply RoPE ---
        let q = rope.apply(&q, 0)?;
        let k = rope.apply(&k, 0)?;

        // --- GQA: expand K, V from n_kv_heads to n_heads ---
        let k = repeat_kv(k, self.num_attention_heads, self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_attention_heads, self.num_kv_heads)?;

        // --- Attention scores ---
        // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
        let k_t = k.contiguous()?.transpose(2, 3)?;
        let q = q.contiguous()?;

        let mut scores = q.matmul(&k_t)?;
        scores = (scores * self.scale)?;

        // Hook: AttnScores — capture and/or intervene (knockout)
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        // Optional soft-capping (Gemma 2)
        if let Some(cap) = self.attn_logit_softcapping {
            scores = ((scores / cap)?.tanh()? * cap)?;
        }

        // Apply causal mask
        scores = scores.broadcast_add(mask)?;

        // Softmax
        // PROMOTE: softmax over F16/BF16 can produce NaN; compute in F32
        // Promote/softmax/demote, shared so the three backbones cannot drift.
        let mut pattern = crate::nn_ops::softmax_last_dim_f32(&scores)?;

        // Hook: AttnPattern — capture and/or intervene (steering)
        hook_point(
            &mut pattern,
            HookPoint::AttnPattern(layer_idx),
            hooks,
            cache,
        )?;

        // --- Attention output ---
        // CONTIGUOUS: ensure contiguous layout for matmul
        let v = v.contiguous()?;
        let attn_output = pattern.matmul(&v)?;

        // Reshape back to [batch, seq, n_heads * head_dim]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;

        // Output projection
        Ok(self.o_proj.forward(&attn_output)?)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Expand key/value tensors from `n_kv_heads` to `n_heads` for GQA.
///
/// # Shapes
/// - `x`: `[batch, n_kv_heads, seq, head_dim]`
/// - returns: `[batch, n_heads, seq, head_dim]`
///
/// When `n_heads == n_kv_heads` (MHA), returns the input unchanged.
fn repeat_kv(x: Tensor, n_heads: usize, n_kv_heads: usize) -> Result<Tensor> {
    if n_heads == n_kv_heads {
        return Ok(x);
    }
    let repeats = n_heads / n_kv_heads;
    let (batch, _kv_heads, seq_len, head_dim) = x.dims4()?;

    // [batch, n_kv_heads, 1, seq, head_dim] → repeat → [batch, n_heads, seq, head_dim]
    let x = x
        .unsqueeze(2)?
        .expand((batch, n_kv_heads, repeats, seq_len, head_dim))?
        .reshape((batch, n_heads, seq_len, head_dim))?;
    Ok(x)
}
