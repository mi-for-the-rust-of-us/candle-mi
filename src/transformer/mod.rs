// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic transformer implementation.
//!
//! A single forward pass covers `LLaMA`, `Qwen2`, `Qwen3`, Gemma, Gemma 2,
//! `Phi-3`, `StarCoder2`, Mistral, and more — parameterized by
//! [`TransformerConfig`].
//!
//! `Qwen3` adds per-head-dim `RMSNorm` on `Q` and `K` before `RoPE`
//! ([`TransformerConfig::use_qk_norm`]); the rest of the forward pass is
//! shared with the other GQA-style decoder-only families.

pub(crate) mod attention;
pub(crate) mod mlp;
pub(crate) mod norm;
pub(crate) mod recurrent;
pub(crate) mod rope;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::backend::{self, MIBackend};
use crate::config::TransformerConfig;
use crate::error::Result;
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};
use crate::util::masks;

use self::attention::Attention;
use self::mlp::Mlp;
use self::norm::{Norm, create_norm};
use self::recurrent::RecurrentPassSpec;
use self::rope::RopeCache;

// ---------------------------------------------------------------------------
// TransformerLayer
// ---------------------------------------------------------------------------

/// A single transformer decoder layer.
struct TransformerLayer {
    /// Pre-attention norm (always present).
    input_norm: Norm,
    /// Self-attention block.
    attention: Attention,
    /// Post-attention norm (Gemma 2 only; `None` for standard 2-norm models).
    post_attention_norm: Option<Norm>,
    /// Pre-MLP norm (standard models: `post_attention_layernorm`;
    /// Gemma 2: `pre_feedforward_layernorm`).
    mid_norm: Norm,
    /// Post-MLP norm (Gemma 2 only; `None` for standard 2-norm models).
    post_feedforward_norm: Option<Norm>,
    /// MLP block.
    mlp: Mlp,
}

impl TransformerLayer {
    /// Load a single decoder layer from weights.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if weight loading fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    fn load(config: &TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let input_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let attention = Attention::load(config, vb.pp("self_attn"))?;

        let post_attention_norm = if config.use_post_norms {
            Some(create_norm(
                config.norm_type,
                config.hidden_size,
                config.norm_eps,
                vb.pp("post_attention_layernorm"),
            )?)
        } else {
            None
        };

        // For Gemma 2 (4-norm): this is "pre_feedforward_layernorm".
        // For standard models (2-norm): this is "post_attention_layernorm".
        let mid_norm_name = if config.use_post_norms {
            "pre_feedforward_layernorm"
        } else {
            "post_attention_layernorm"
        };
        let mid_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb.pp(mid_norm_name),
        )?;

        let post_feedforward_norm = if config.use_post_norms {
            Some(create_norm(
                config.norm_type,
                config.hidden_size,
                config.norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?)
        } else {
            None
        };

        let mlp = Mlp::load(config, vb.pp("mlp"))?;

        Ok(Self {
            input_norm,
            attention,
            post_attention_norm,
            mid_norm,
            post_feedforward_norm,
            mlp,
        })
    }
}

// ---------------------------------------------------------------------------
// GenericTransformer
// ---------------------------------------------------------------------------

/// Config-driven generic transformer backend.
///
/// One implementation covers `LLaMA`, `Qwen2`, `Qwen3`, Gemma 2, `Phi-3`,
/// `StarCoder2`, Mistral, and more.  Architecture differences are
/// captured in [`TransformerConfig`] fields; the forward pass branches
/// on these at runtime with zero overhead when the branch is not taken.
/// `Qwen3`'s per-head-dim `QK` norm runs only when
/// [`TransformerConfig::use_qk_norm`] is `true`.
pub struct GenericTransformer {
    /// Token embedding matrix.
    embed_tokens: Embedding,
    /// Decoder layers.
    layers: Vec<TransformerLayer>,
    /// Final normalization before LM head.
    final_norm: Norm,
    /// LM head (vocabulary projection).  `None` when tied to `embed_tokens`.
    lm_head: Option<Linear>,
    /// Pre-computed `RoPE` cos/sin cache.
    rope_cache: RopeCache,
    /// Model configuration.
    config: TransformerConfig,
}

impl GenericTransformer {
    /// Load a generic transformer from a [`VarBuilder`].
    ///
    /// The caller constructs the `VarBuilder` (safe or mmap) and provides
    /// the parsed `TransformerConfig`.  This function loads all weights
    /// and pre-computes the `RoPE` cache.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) if weight loading fails or dimensions
    /// are inconsistent.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(
        config: TransformerConfig,
        device: &Device,
        dtype: DType,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let vb_model = vb.pp("model");

        // --- Embedding ---
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        // --- Layers ---
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let vb_layer = vb_model.pp(format!("layers.{i}"));
            let layer = TransformerLayer::load(&config, vb_layer)?;
            layers.push(layer);
        }

        // --- Final norm ---
        let final_norm = create_norm(
            config.norm_type,
            config.hidden_size,
            config.norm_eps,
            vb_model.pp("norm"),
        )?;

        // --- LM head ---
        // Prefer a materialized `lm_head.weight` whenever it is present: it is
        // authoritative even if `tie_word_embeddings` is set (A2D-converted
        // diffusion checkpoints ship a separate head despite the tie flag).
        // Only tie to the embedding when the flag is set *and* no head ships;
        // otherwise load the head (an untied model missing it surfaces here).
        let lm_head = if config.tie_word_embeddings && !vb.contains_tensor("lm_head.weight") {
            None
        } else {
            Some(candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        // --- RoPE cache ---
        let rope_cache = RopeCache::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            // BORROW: RopeScaling carries Vec factors (longrope) so it is not
            // Copy; clone it out of the retained config.
            config.rope_scaling.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            rope_cache,
            config,
        })
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Forward pass through a range of layers (internal helper).
    ///
    /// Processes layers `start..end` with full hook support (capture
    /// and intervention at every hook point).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - returns: `[batch, seq, hidden_size]`
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_range(
        &self,
        mut hidden: Tensor,
        start: usize,
        end: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let layer_slice = self.layers.get(start..end).ok_or_else(|| {
            crate::error::MIError::Intervention(format!(
                "layer range {start}..{end} out of bounds (n_layers={})",
                self.layers.len()
            ))
        })?;
        for (offset, layer) in layer_slice.iter().enumerate() {
            let layer_idx = start + offset;

            // Hook: ResidPre
            hook_point(&mut hidden, HookPoint::ResidPre(layer_idx), hooks, cache)?;

            let residual = hidden.clone();

            // Pre-attention norm
            hidden = layer.input_norm.forward(&hidden)?;

            // Attention mask for this layer
            let mask = self.mask_for_layer(layer_idx, seq_len, device, dtype)?;

            // Self-attention (hooks for Q, K, V, Scores, Pattern handled inside)
            hidden = layer.attention.forward(
                &hidden,
                &mask,
                &self.rope_cache,
                layer_idx,
                hooks,
                cache,
            )?;

            // Optional post-attention norm (Gemma 2)
            if let Some(ref norm) = layer.post_attention_norm {
                hidden = norm.forward(&hidden)?;
            }

            // Hook: AttnOut
            hook_point(&mut hidden, HookPoint::AttnOut(layer_idx), hooks, cache)?;

            // Residual connection after attention
            hidden = (residual + &hidden)?;

            // Hook: ResidMid
            hook_point(&mut hidden, HookPoint::ResidMid(layer_idx), hooks, cache)?;

            let residual = hidden.clone();

            // Pre-MLP norm
            hidden = layer.mid_norm.forward(&hidden)?;

            // Hook: MlpPre
            hook_point(&mut hidden, HookPoint::MlpPre(layer_idx), hooks, cache)?;

            // MLP
            hidden = layer.mlp.forward(&hidden)?;

            // Hook: MlpPost
            hook_point(&mut hidden, HookPoint::MlpPost(layer_idx), hooks, cache)?;

            // Optional post-feedforward norm (Gemma 2)
            if let Some(ref norm) = layer.post_feedforward_norm {
                hidden = norm.forward(&hidden)?;
            }

            // Hook: MlpOut
            hook_point(&mut hidden, HookPoint::MlpOut(layer_idx), hooks, cache)?;

            // Residual connection after MLP
            hidden = (residual + &hidden)?;

            // Hook: ResidPost
            hook_point(&mut hidden, HookPoint::ResidPost(layer_idx), hooks, cache)?;
        }
        Ok(hidden)
    }

    /// Embed input tokens and apply embedding scale + Embed hook.
    ///
    /// Returns `(hidden, dtype, cache)`.
    ///
    /// # Shapes
    /// - `input_ids`: `[batch, seq]`
    /// - returns hidden: `[batch, seq, hidden_size]`
    fn embed_with_hooks(
        &self,
        input_ids: &Tensor,
        hooks: &HookSpec,
    ) -> Result<(Tensor, DType, HookCache)> {
        let device = input_ids.device();
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let dtype = hidden.dtype();

        if let Some(scale) = self.config.embedding_scale {
            hidden = (hidden * scale)?;
        }

        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        Ok((hidden, dtype, cache))
    }

    /// Apply final norm, logit projection, and softcapping.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]`
    /// - returns logits: `[batch, seq, vocab_size]`
    fn finalize_logits(
        &self,
        mut hidden: Tensor,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        hidden = self.final_norm.forward(&hidden)?;

        hook_point(&mut hidden, HookPoint::FinalNorm, hooks, cache)?;

        let mut logits = self.project_logits(&hidden)?;

        if let Some(cap) = self.config.final_logit_softcapping {
            logits = ((logits / cap)?.tanh()? * cap)?;
        }

        Ok(logits)
    }

    // --- Recurrent forward pass -------------------------------------------

    /// Forward pass with recurrent re-execution of a layer block.
    ///
    /// Runs layers `spec.loop_start..=spec.loop_end` a total of `spec.depth`
    /// times, with optional feedback injected between passes.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// embed → layers[0..loop_start)
    ///   → save_input (if feedback)
    ///   → layers[loop_start..=loop_end]  (pass 1)
    ///   → for pass 2..depth:
    ///       if feedback: hidden = saved_input + feedback entries
    ///       else:        hidden = previous pass output (true recurrence)
    ///       layers[loop_start..=loop_end]
    ///   → layers[(loop_end+1)..n_layers)
    ///   → norm → logits
    /// ```
    ///
    /// Hooks fire at every layer in every pass. For loop layers, later pass
    /// captures overwrite earlier ones in the [`HookCache`].
    ///
    /// # Shapes
    /// - `input_ids`: `[batch, seq]`
    /// - returns: [`HookCache`] containing logits at `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`](crate::MIError::Intervention) if the spec is invalid for this model,
    /// or [`MIError::Model`](crate::MIError::Model) on tensor operation failures.
    pub fn forward_recurrent(
        &self,
        input_ids: &Tensor,
        hooks: &HookSpec,
        spec: &RecurrentPassSpec,
    ) -> Result<HookCache> {
        let (mut hidden, dtype, mut cache) = self.embed_with_hooks(input_ids, hooks)?;
        let device = input_ids.device();
        let (_, seq_len, _) = hidden.dims3()?;

        spec.validate(self.config.num_layers, seq_len, self.config.hidden_size)?;

        // Pre-loop layers
        hidden = self.forward_layer_range(
            hidden,
            0,
            spec.loop_start,
            seq_len,
            device,
            dtype,
            hooks,
            &mut cache,
        )?;

        // Save input to the loop block (needed for feedback injection)
        let saved_input = if spec.feedback.is_empty() {
            None
        } else {
            Some(hidden.clone())
        };

        // Pass 1 through loop layers
        hidden = self.forward_layer_range(
            hidden,
            spec.loop_start,
            spec.loop_end + 1,
            seq_len,
            device,
            dtype,
            hooks,
            &mut cache,
        )?;

        // Recurrent passes 2..depth
        for _pass in 1..spec.depth {
            if let Some(ref saved) = saved_input {
                // With feedback: reset to saved pre-loop state + inject
                hidden = saved.clone();
                for entry in &spec.feedback {
                    let scaled = (&entry.vector * f64::from(entry.strength))?;
                    hidden = inject_feedback_at_position(&hidden, &scaled, entry.position)?;
                }
            }
            // Without feedback: hidden stays as previous pass output (true recurrence)

            hidden = self.forward_layer_range(
                hidden,
                spec.loop_start,
                spec.loop_end + 1,
                seq_len,
                device,
                dtype,
                hooks,
                &mut cache,
            )?;
        }

        // Post-loop layers
        hidden = self.forward_layer_range(
            hidden,
            spec.loop_end + 1,
            self.config.num_layers,
            seq_len,
            device,
            dtype,
            hooks,
            &mut cache,
        )?;

        let logits = self.finalize_logits(hidden, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    // --- Recurrent generation ---------------------------------------------

    /// Generate tokens with recurrent re-execution.
    ///
    /// Since candle-mi recomputes the full sequence at every step (no KV
    /// cache), the recurrent multi-pass applies at every generation step.
    ///
    /// - **Prefill-only** (`sustained: false`): feedback at original
    ///   prompt positions only.
    /// - **Sustained** (`sustained: true`): feedback is also injected
    ///   at the current last-token position at each step.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`](crate::MIError::Intervention) if the spec is invalid, or
    /// [`MIError::Model`](crate::MIError::Model) on tensor/generation failures.
    pub fn generate_recurrent(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        stop_tokens: &[u32],
        spec: &RecurrentPassSpec,
    ) -> Result<Vec<u32>> {
        let device = self.embed_tokens.embeddings().device();
        let mut tokens = prompt_tokens.to_vec();

        for _ in 0..max_tokens {
            let input_tensor = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

            // Build effective spec for this step
            let effective_spec = if spec.sustained && !spec.feedback.is_empty() {
                let mut step_spec = spec.clone();
                // Add feedback at the current last position (sustained pressure)
                let last_pos = tokens.len() - 1;
                for entry in &spec.feedback {
                    step_spec.feedback.push(recurrent::RecurrentFeedbackEntry {
                        position: last_pos,
                        vector: entry.vector.clone(),
                        strength: entry.strength,
                    });
                }
                step_spec
            } else {
                spec.clone()
            };

            let hook_cache =
                self.forward_recurrent(&input_tensor, &HookSpec::new(), &effective_spec)?;

            // Extract last-token logits
            let logits = hook_cache.output();
            let seq_len = logits.dim(1)?;
            let last_logits = logits.i((.., seq_len - 1, ..))?.squeeze(1)?;
            let last_logits_flat = last_logits.flatten_all()?;

            let next_token = backend::sample_token(&last_logits_flat, temperature)?;
            if stop_tokens.contains(&next_token) {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Project hidden states to vocabulary logits (internal helper).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, hidden_size]` or `[batch, hidden_size]`
    /// - returns: `[batch, seq, vocab_size]` or `[batch, vocab_size]`
    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        if let Some(head) = &self.lm_head {
            Ok(head.forward(hidden)?)
        } else {
            // Tied embeddings: logits = hidden @ embed_tokens^T
            // broadcast_matmul handles [batch, seq, d] @ [d, vocab] → [batch, seq, vocab]
            let embed_weight = self.embed_tokens.embeddings();
            let logits = hidden.broadcast_matmul(&embed_weight.t()?)?;
            Ok(logits)
        }
    }

    /// Create the appropriate attention mask for a given layer.
    ///
    /// Returns a causal mask with optional sliding window.
    fn mask_for_layer(
        &self,
        layer_idx: usize,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        // Masked-diffusion LMs (e.g. Dream) run the decoder non-causally: every
        // position attends to every other, so no causal/sliding mask applies.
        if self.config.bidirectional {
            return masks::create_bidirectional_mask(seq_len, device, dtype);
        }

        let use_sliding = match (
            self.config.sliding_window,
            self.config.alternating_sliding_window,
        ) {
            (Some(_), true) => layer_idx.is_multiple_of(2), // Gemma 2: even layers
            (Some(_), false) => true,                       // Mistral: all layers
            (None, _) => false,
        };

        if use_sliding && let Some(window) = self.config.sliding_window {
            return create_sliding_window_mask(seq_len, window, device, dtype);
        }

        masks::create_causal_mask(seq_len, device, dtype)
    }
}

// ---------------------------------------------------------------------------
// MIBackend implementation
// ---------------------------------------------------------------------------

impl MIBackend for GenericTransformer {
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let (hidden, dtype, mut cache) = self.embed_with_hooks(input_ids, hooks)?;
        let (_, seq_len, _) = hidden.dims3()?;

        // --- All layers ---
        let hidden = self.forward_layer_range(
            hidden,
            0,
            self.config.num_layers,
            seq_len,
            device,
            dtype,
            hooks,
            &mut cache,
        )?;

        let logits = self.finalize_logits(hidden, hooks, &mut cache)?;
        cache.set_output(logits);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.final_norm.forward(hidden)?;
        self.project_logits(&normed)
    }

    fn embedding_vector(&self, token_id: u32) -> Result<Tensor> {
        let device = self.embed_tokens.embeddings().device();
        let ids = Tensor::new(&[token_id], device)?;
        let emb = self.embed_tokens.forward(&ids)?; // [1, d_model]
        Ok(emb.squeeze(0)?) // [d_model]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Inject a vector at a specific sequence position in a hidden state.
///
/// # Shapes
/// - `hidden`: `[batch, seq_len, d_model]`
/// - `vector`: `[d_model]` (pre-scaled)
/// - returns: `[batch, seq_len, d_model]` with `hidden[:, position, :] += vector`
fn inject_feedback_at_position(
    hidden: &Tensor,
    vector: &Tensor,
    position: usize,
) -> Result<Tensor> {
    let seq_len = hidden.dim(1)?;
    let d_model = hidden.dim(2)?;

    // Build a [1, seq_len, d_model] delta tensor that is zero everywhere
    // except at the target position.
    let mut delta_data = vec![0.0_f32; seq_len * d_model];
    // PROMOTE: feedback vector may be BF16 from embedding; F32 for host-side copy
    let vec_f32: Vec<f32> = vector
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let start = position * d_model;
    let dest = delta_data.get_mut(start..start + d_model).ok_or_else(|| {
        crate::error::MIError::Intervention(format!(
            "feedback position {position} out of bounds (seq_len={seq_len})"
        ))
    })?;
    dest.copy_from_slice(&vec_f32);

    let delta = Tensor::from_vec(delta_data, (1, seq_len, d_model), hidden.device())?
        .to_dtype(hidden.dtype())?;

    Ok(hidden.broadcast_add(&delta)?)
}

/// Create a sliding window causal mask.
///
/// # Shapes
/// - returns: `[1, 1, seq_len, seq_len]`
///
/// Positions beyond the window or in the future are set to `-inf`.
fn create_sliding_window_mask(
    seq_len: usize,
    window: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let idx = i * seq_len + j;
            if j > i || i.saturating_sub(j) > window {
                // idx is always < seq_len * seq_len by construction
                if let Some(cell) = mask_data.get_mut(idx) {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)?)
}
