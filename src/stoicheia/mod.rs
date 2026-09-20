// SPDX-License-Identifier: MIT OR Apache-2.0

//! `AlgZoo` model backends and MI tooling — stoicheia (στοιχεῖα, "elements").
//!
//! ## Phase A — Backends
//!
//! Two [`MIBackend`] implementations for ARC's
//! [`AlgZoo`](https://github.com/alignment-research-center/alg-zoo) tiny models:
//!
//! - [`StoicheiaRnn`] — single-layer `ReLU` RNN (continuous input)
//! - [`StoicheiaTransformer`] — attention-only transformer (discrete input)
//!
//! These models have 8–1,408 parameters and solve algorithmic tasks.
//! They are designed as "model organisms" for exhaustive mechanistic
//! interpretability.
//!
//! ## Phase B — MI Analysis (ekthesis)
//!
//! Six analysis modules for `ReLU` RNN mechanistic understanding:
//!
//! - [`fast`] — raw `f32` forward pass bypassing `Tensor` overhead
//! - [`standardize`] — weight rescaling so `|W_ih[j]| = 1`
//! - [`piecewise`] — `ReLU` activation region enumeration
//! - [`ablation`] — single-neuron and pairwise zero-ablation
//! - [`probing`] — neuron functional classification
//! - [`surprise`] — ARC's information-theoretic surprise accounting
//!
//! # Weight loading
//!
//! Weights are loaded from `safetensors` or `PyTorch` `.pth` files.
//! Format detection is automatic based on file extension:
//!
//! ```rust,ignore
//! // Both work — format is detected from the extension
//! let model = StoicheiaRnn::load(config, "model.safetensors", &device)?;
//! let model = StoicheiaRnn::load(config, "model.pth", &device)?;
//! ```
//!
//! `.pth` files are converted in memory via
//! [anamnesis](https://crates.io/crates/anamnesis)' pickle VM — no manual
//! preprocessing step required.
//!
//! # Validation
//!
//! The full `AlgZoo` corpus (6,960 `.pth` files, 4 tasks, hidden sizes
//! 2–32, sequence lengths 2–10) was loaded and converted with zero
//! failures. Cross-validation against `PyTorch` reference outputs
//! confirms numerical agreement:
//!
//! | Backend | Fixture | Tolerance | Accuracy |
//! |---------|---------|-----------|----------|
//! | `StoicheiaRnn` | M₂,₂ (10 params) | < 1e-4 | 100% on 10K samples |
//! | `StoicheiaTransformer` | h4n4 (176 params) | < 1e-2 | exact match |
//!
//! The wider transformer tolerance (1e-2) reflects larger logit
//! magnitudes (~1,000×); relative precision is equivalent.

pub mod ablation;
pub mod config;
pub mod fast;
pub mod piecewise;
pub mod probing;
pub mod standardize;
pub mod surprise;
pub mod tasks;

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::backend::MIBackend;
use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookPoint, HookSpec, hook_point};

pub use config::{StoicheiaArch, StoicheiaConfig, StoicheiaOutput, StoicheiaTask};

// ---------------------------------------------------------------------------
// Agnostic weight loading
// ---------------------------------------------------------------------------

/// Load weight bytes in safetensors format, auto-detecting file format.
///
/// - `.safetensors` — read directly
/// - `.pth` / `.pkl` — convert in memory via `anamnesis`' pickle VM
///
/// # Errors
///
/// Returns [`MIError::Io`](crate::MIError::Io) if the file cannot be read.
/// Returns [`MIError::Config`] if the file
/// extension is not recognized.
/// Returns [`MIError::Model`] if `.pth` parsing
/// or safetensors conversion fails.
fn load_weight_bytes(path: &Path) -> Result<Vec<u8>> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Ok(std::fs::read(path)?),
        Some("pth" | "pkl") => {
            let parsed = anamnesis::parse_pth(path).map_err(|e| {
                MIError::Model(candle_core::Error::Msg(format!(
                    "failed to parse .pth file: {e}"
                )))
            })?;
            parsed.to_safetensors_bytes().map_err(|e| {
                MIError::Model(candle_core::Error::Msg(format!(
                    "failed to convert .pth to safetensors: {e}"
                )))
            })
        }
        Some(ext) => Err(MIError::Config(format!(
            "unsupported weight file extension: .{ext} \
             (expected .safetensors, .pth, or .pkl)"
        ))),
        None => Err(MIError::Config(
            "weight file has no extension \
             (expected .safetensors, .pth, or .pkl)"
                .into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// StoicheiaRnn
// ---------------------------------------------------------------------------

/// Single-layer `ReLU` RNN backend for `AlgZoo` continuous tasks.
///
/// Architecture (from `AlgZoo`'s `architectures.py`):
///
/// ```text
/// For each timestep t:
///     pre_act_t = W_ih @ x_t + W_hh @ h_{t-1}    // [batch, H]
///     h_t = relu(pre_act_t)                        // [batch, H]
/// output = W_oh @ h_final                          // [batch, output_size]
/// ```
///
/// Where `x_t` is scalar (`input_size = 1`), so `W_ih` has shape `[H, 1]`.
///
/// # Hook points
///
/// | Hook | Shape | Description |
/// |------|-------|-------------|
/// | `Custom("rnn.hook_pre_activation.{t}")` | `[batch, H]` | Before ReLU at timestep `t` |
/// | `Custom("rnn.hook_hidden.{t}")` | `[batch, H]` | Hidden state after timestep `t` |
/// | `Custom("rnn.hook_final_state")` | `[batch, H]` | Final hidden state |
/// | `Custom("rnn.hook_output")` | `[batch, output_size]` | After output projection |
pub struct StoicheiaRnn {
    /// Input-to-hidden weights: `[H, 1]`.
    weight_ih: Tensor,
    /// Hidden-to-hidden weights: `[H, H]`.
    weight_hh: Tensor,
    /// Output projection weights: `[output_size, H]`.
    weight_oh: Tensor,
    /// Model configuration.
    config: StoicheiaConfig,
}

impl StoicheiaRnn {
    /// Load an `AlgZoo` RNN from a weight file.
    ///
    /// Accepts `.safetensors`, `.pth`, or `.pkl` files. Format is detected
    /// from the file extension; `.pth`/`.pkl` files are converted in memory
    /// via `anamnesis`.
    ///
    /// The weight file must contain:
    /// - `rnn.weight_ih_l0`: `[H, 1]`
    /// - `rnn.weight_hh_l0`: `[H, H]`
    /// - `linear.weight`: `[output_size, H]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weights are
    /// missing or have wrong shapes.
    /// Returns [`MIError::Config`] if the file
    /// extension is not recognized.
    #[allow(clippy::similar_names)]
    pub fn load(config: StoicheiaConfig, path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let buffer = load_weight_bytes(path.as_ref())?;
        let vb = VarBuilder::from_buffered_safetensors(buffer, DType::F32, device)?;

        let weight_ih = vb.get((config.hidden_size, 1), "rnn.weight_ih_l0")?;
        let weight_hh = vb.get((config.hidden_size, config.hidden_size), "rnn.weight_hh_l0")?;
        let weight_oh = vb.get((config.output_size(), config.hidden_size), "linear.weight")?;

        Ok(Self {
            weight_ih,
            weight_hh,
            weight_oh,
            config,
        })
    }
}

// -- Weight accessors (Phase B) ------------------------------------------

impl StoicheiaRnn {
    /// Access the input-to-hidden weight tensor.
    ///
    /// # Shapes
    /// - returns: `[hidden_size, 1]`
    #[must_use]
    pub const fn weight_ih(&self) -> &Tensor {
        &self.weight_ih
    }

    /// Access the hidden-to-hidden weight tensor.
    ///
    /// # Shapes
    /// - returns: `[hidden_size, hidden_size]`
    #[must_use]
    pub const fn weight_hh(&self) -> &Tensor {
        &self.weight_hh
    }

    /// Access the output projection weight tensor.
    ///
    /// # Shapes
    /// - returns: `[output_size, hidden_size]`
    #[must_use]
    pub const fn weight_oh(&self) -> &Tensor {
        &self.weight_oh
    }

    /// Access the model configuration.
    #[must_use]
    pub const fn config(&self) -> &StoicheiaConfig {
        &self.config
    }
}

impl MIBackend for StoicheiaRnn {
    fn num_layers(&self) -> usize {
        1
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.output_size()
    }

    fn num_heads(&self) -> usize {
        0
    }

    fn forward(&self, input: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input.device();
        let (batch_size, seq_len) = input.dims2()?;
        let h = self.config.hidden_size;

        // Initialize hidden state to zeros: [batch, H]
        let mut hidden = Tensor::zeros((batch_size, h), DType::F32, device)?;

        // Placeholder output — replaced at the end
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Pre-scan: determine which timesteps need the hook protocol run.
        // This avoids per-timestep String allocation when hooks are empty
        // (zero-overhead guarantee) or when only specific timesteps are
        // hooked (one allocation per timestep at scan time, not per
        // forward pass iteration).
        //
        // The scan covers interventions as well as captures: a timestep that is
        // only intervened still has to build its `HookPoint` in the loop, and
        // scanning for captures alone is exactly how this backend came to ignore
        // interventions silently (audit finding 1, v0.2.0).
        let has_hooks = !hooks.is_empty();
        let (hooked_pre_act, hooked_hidden) = if has_hooks {
            let pre_act: std::collections::HashSet<usize> = (0..seq_len)
                .filter(|t| {
                    let point = HookPoint::Custom(format!("rnn.hook_pre_activation.{t}"));
                    hooks.is_captured(&point) || hooks.has_intervention_at(&point)
                })
                .collect();
            let hid: std::collections::HashSet<usize> = (0..seq_len)
                .filter(|t| {
                    let point = HookPoint::Custom(format!("rnn.hook_hidden.{t}"));
                    hooks.is_captured(&point) || hooks.has_intervention_at(&point)
                })
                .collect();
            (pre_act, hid)
        } else {
            (
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            )
        };

        // RNN loop: one timestep at a time
        for t in 0..seq_len {
            // x_t: [batch, 1] — scalar input per timestep
            // INDEX: t is bounded by seq_len from dims2()
            let x_t = input.i((.., t..=t))?;

            // pre_act = x_t @ W_ih^T + h_{t-1} @ W_hh^T
            // x_t @ W_ih^T: [batch, 1] @ [1, H] → [batch, H]
            let ih = x_t.matmul(&self.weight_ih.t()?)?;
            // h_{t-1} @ W_hh^T: [batch, H] @ [H, H] → [batch, H]
            let hh = hidden.matmul(&self.weight_hh.t()?)?;
            let mut pre_act = (ih + hh)?;

            // Hook: pre-activation at timestep t (no allocation when unhooked)
            if hooked_pre_act.contains(&t) {
                hook_point(
                    &mut pre_act,
                    HookPoint::Custom(format!("rnn.hook_pre_activation.{t}")),
                    hooks,
                    &mut cache,
                )?;
            }

            // h_t = relu(pre_act)
            hidden = pre_act.relu()?;

            // Hook: hidden state at timestep t (no allocation when unhooked).
            // An intervention here propagates into the next timestep through
            // the recurrence, which is the point of steering an RNN.
            if hooked_hidden.contains(&t) {
                hook_point(
                    &mut hidden,
                    HookPoint::Custom(format!("rnn.hook_hidden.{t}")),
                    hooks,
                    &mut cache,
                )?;
            }
        }

        // Hook: final hidden state (allocated only when hooks are present)
        if has_hooks {
            hook_point(
                &mut hidden,
                HookPoint::Custom("rnn.hook_final_state".into()),
                hooks,
                &mut cache,
            )?;
        }

        // Output projection: [batch, H] @ [H, output_size] → [batch, output_size]
        let mut output = hidden.matmul(&self.weight_oh.t()?)?;

        // Hook: output (allocated only when hooks are present)
        if has_hooks {
            hook_point(
                &mut output,
                HookPoint::Custom("rnn.hook_output".into()),
                hooks,
                &mut cache,
            )?;
        }

        // Unsqueeze to [batch, 1, output_size] to match MIBackend convention
        let output_3d = output.unsqueeze(1)?;
        cache.set_output(output_3d);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        // Output linear projection, rank-preserving: [.., H] → [.., output_size].
        // `broadcast_matmul` (not `matmul`) so a rank-3 [batch, seq, H] residual
        // stream projects too; it falls through to `matmul` when neither side
        // broadcasts, so the rank-2 path is unchanged.
        Ok(hidden.broadcast_matmul(&self.weight_oh.t()?)?)
    }
}

// ---------------------------------------------------------------------------
// StoicheiaTransformer
// ---------------------------------------------------------------------------

/// Attention layer for `AlgZoo`'s attention-only transformer.
///
/// `PyTorch` packs Q, K, V into a single `in_proj_weight` of shape `[3*H, H]`.
struct AttentionLayer {
    /// Packed Q, K, V projection: `[3*H, H]`.
    in_proj_weight: Tensor,
    /// Output projection: `[H, H]`.
    out_proj_weight: Tensor,
    /// Hidden size.
    hidden_size: usize,
}

impl AttentionLayer {
    /// Run one attention layer (full bidirectional, single head, no causal mask).
    ///
    /// The [`HookPoint::AttnScores`] and [`HookPoint::AttnPattern`] hooks fire
    /// *inside* this function rather than on its return values, because the
    /// caller receives them only after `attn_output` has already been computed.
    /// Firing them outside would let an intervention mutate a tensor nothing
    /// reads, which is the silent no-op this release exists to remove.
    ///
    /// # Shapes
    /// - `hidden`: `[batch, seq, H]`
    /// - returns: `attn_output` at `[batch, seq, H]`; `scores` and `pattern`
    ///   (both `[batch, 1, seq, seq]`, pre- and post-softmax) reach the caller
    ///   through `cache` when captured.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`](crate::MIError::Model) on a tensor failure.
    /// Returns [`MIError::Intervention`](crate::MIError::Intervention) if an
    /// intervention is invalid at one of the hook points fired here.
    fn forward(
        &self,
        hidden: &Tensor,
        layer_idx: usize,
        hooks: &HookSpec,
        cache: &mut HookCache,
    ) -> Result<Tensor> {
        let dim = self.hidden_size;

        // Project Q, K, V from packed in_proj_weight
        // hidden @ in_proj_weight^T: [batch, seq, H] @ [H, 3H] → [batch, seq, 3H]
        let qkv = hidden.broadcast_matmul(&self.in_proj_weight.t()?)?;

        // Split into Q, K, V each [batch, seq, H]
        let query = qkv.narrow(2, 0, dim)?;
        let key = qkv.narrow(2, dim, dim)?;
        let value = qkv.narrow(2, 2 * dim, dim)?;

        // Attention scores: Q @ K^T / sqrt(H)
        // [batch, seq, H] @ [batch, H, seq] → [batch, seq, seq]
        // CAST: usize → f64, hidden_size for attention scale sqrt
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let scale = (dim as f64).sqrt();
        let scores_2d = (query.matmul(&key.t()?)? / scale)?;

        // Unsqueeze to [batch, 1, seq, seq] for head dimension
        let mut scores = scores_2d.unsqueeze(1)?;

        // Hook: AttnScores — fired pre-softmax, so a knockout mask added here
        // is still seen by the softmax below.
        hook_point(&mut scores, HookPoint::AttnScores(layer_idx), hooks, cache)?;

        // Softmax over last dimension (no causal mask — full bidirectional).
        // Backward-safe dispatch: fused kernel for inference, composed form
        // when the graph is tracked (training over a `VarMap`).
        let mut pattern = crate::nn_ops::softmax_last_dim(&scores)?;

        // Hook: AttnPattern — fired before the weighted sum, so an intervention
        // here changes what the layer actually attends to.
        hook_point(
            &mut pattern,
            HookPoint::AttnPattern(layer_idx),
            hooks,
            cache,
        )?;

        // Weighted sum: pattern @ V
        // [batch, 1, seq, seq] → squeeze → [batch, seq, seq]
        let pattern_2d = pattern.squeeze(1)?;
        // [batch, seq, seq] @ [batch, seq, H] → [batch, seq, H]
        let attn_out = pattern_2d.matmul(&value)?;

        // Output projection: [batch, seq, H] @ [H, H] → [batch, seq, H]
        let projected = attn_out.broadcast_matmul(&self.out_proj_weight.t()?)?;

        Ok(projected)
    }
}

/// Attention-only transformer backend for `AlgZoo` discrete tasks.
///
/// Architecture (from `AlgZoo`'s `architectures.py`):
///
/// ```text
/// x = embed(input) + pos_embed(positions)
/// for each attention layer:
///     x = x + attention(x, x, x)       // residual, full bidirectional
/// output = unembed(x[:, -1])            // last position only
/// ```
///
/// No MLP blocks, no layer normalization, no causal mask.
///
/// # Hook points
///
/// | Hook | Shape | Description |
/// |------|-------|-------------|
/// | `Embed` | `[batch, seq, H]` | After token + positional embedding |
/// | `ResidPre(i)` | `[batch, seq, H]` | Before attention layer `i` |
/// | `AttnScores(i)` | `[batch, 1, seq, seq]` | Pre-softmax attention |
/// | `AttnPattern(i)` | `[batch, 1, seq, seq]` | Post-softmax attention |
/// | `AttnOut(i)` | `[batch, seq, H]` | Attention output (before residual add) |
/// | `ResidPost(i)` | `[batch, seq, H]` | After residual add |
pub struct StoicheiaTransformer {
    /// Token embedding.
    embed: Embedding,
    /// Positional embedding.
    pos_embed: Embedding,
    /// Attention layers.
    attns: Vec<AttentionLayer>,
    /// Unembedding weights: `[output_size, H]`.
    unembed_weight: Tensor,
    /// Model configuration.
    config: StoicheiaConfig,
}

impl StoicheiaTransformer {
    /// Load an `AlgZoo` attention-only transformer from a weight file.
    ///
    /// Accepts `.safetensors`, `.pth`, or `.pkl` files. Format is detected
    /// from the file extension; `.pth`/`.pkl` files are converted in memory
    /// via `anamnesis`.
    ///
    /// The weight file must contain:
    /// - `embed.weight`: `[input_range, H]`
    /// - `pos_embed.weight`: `[seq_len, H]`
    /// - `attns.{i}.in_proj_weight`: `[3*H, H]` for each layer
    /// - `attns.{i}.out_proj.weight`: `[H, H]` for each layer
    /// - `unembed.weight`: `[output_size, H]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weights are
    /// missing or have wrong shapes.
    /// Returns [`MIError::Config`] if the file
    /// extension is not recognized.
    pub fn load(config: StoicheiaConfig, path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let buffer = load_weight_bytes(path.as_ref())?;
        let vb = VarBuilder::from_buffered_safetensors(buffer, DType::F32, device)?;

        let h = config.hidden_size;

        let embed = Embedding::new(vb.get((config.input_range, h), "embed.weight")?, h);
        let pos_embed = Embedding::new(vb.get((config.seq_len, h), "pos_embed.weight")?, h);

        let mut attns = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let in_proj_weight = vb.get((3 * h, h), &format!("attns.{i}.in_proj_weight"))?;
            let out_proj_weight = vb.get((h, h), &format!("attns.{i}.out_proj.weight"))?;
            attns.push(AttentionLayer {
                in_proj_weight,
                out_proj_weight,
                hidden_size: h,
            });
        }

        let unembed_weight = vb.get((config.output_size(), h), "unembed.weight")?;

        Ok(Self {
            embed,
            pos_embed,
            attns,
            unembed_weight,
            config,
        })
    }
}

impl MIBackend for StoicheiaTransformer {
    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.output_size()
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        let device = input_ids.device();
        let (batch, seq_len) = input_ids.dims2()?;

        // Placeholder output — replaced at the end
        let mut cache = HookCache::new(Tensor::zeros(1, DType::F32, device)?);

        // Token embedding + positional embedding
        let token_emb = self.embed.forward(input_ids)?;
        // CAST: usize → u32, seq_len fits in u32 (`AlgZoo` max seq_len = 10)
        #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let pos_ids = Tensor::new(&positions[..], device)?
            .unsqueeze(0)?
            .expand((batch, seq_len))?;
        let pos_emb = self.pos_embed.forward(&pos_ids)?;
        let mut hidden = (token_emb + pos_emb)?;

        // Hook: Embed
        hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

        // Attention layers with residual connections
        for (i, attn) in self.attns.iter().enumerate() {
            // Hook: ResidPre
            hook_point(&mut hidden, HookPoint::ResidPre(i), hooks, &mut cache)?;

            // `AttnScores` and `AttnPattern` fire inside this call; see
            // `AttentionLayer::forward`.
            let mut attn_out = attn.forward(&hidden, i, hooks, &mut cache)?;

            // Hook: AttnOut — the sublayer contribution, before the residual add
            hook_point(&mut attn_out, HookPoint::AttnOut(i), hooks, &mut cache)?;

            // Residual connection
            hidden = (hidden + attn_out)?;

            // Hook: ResidPost
            hook_point(&mut hidden, HookPoint::ResidPost(i), hooks, &mut cache)?;
        }

        // Unembed last position only: [batch, H] → [batch, output_size]
        // INDEX: seq_len-1 is valid because seq_len >= 1 from dims2()
        let last_hidden = hidden.i((.., seq_len - 1, ..))?;
        let output = last_hidden.matmul(&self.unembed_weight.t()?)?;

        // Unsqueeze to [batch, 1, output_size] to match MIBackend convention
        let output_3d = output.unsqueeze(1)?;
        cache.set_output(output_3d);
        Ok(cache)
    }

    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        // Unembedding projection, rank-preserving: [.., H] → [.., output_size].
        // `broadcast_matmul` (not `matmul`) so a rank-3 [batch, seq, H] residual
        // stream projects too; it falls through to `matmul` when neither side
        // broadcasts, so the rank-2 path is unchanged.
        Ok(hidden.broadcast_matmul(&self.unembed_weight.t()?)?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::hooks::Intervention;

    /// A 2-hidden, 3-step RNN with non-zero weights, built without a file.
    fn tiny_rnn() -> StoicheiaRnn {
        let dev = Device::Cpu;
        let config = StoicheiaConfig::from_task(StoicheiaTask::Median, 2, 3);
        StoicheiaRnn {
            weight_ih: Tensor::new(&[[0.5f32], [-0.25]], &dev).unwrap(),
            weight_hh: Tensor::new(&[[0.3f32, 0.1], [0.2, 0.4]], &dev).unwrap(),
            weight_oh: Tensor::new(&[[0.7f32, -0.6]], &dev).unwrap(),
            config,
        }
    }

    /// A 1-layer, 2-hidden attention-only transformer, built without a file.
    fn tiny_transformer() -> StoicheiaTransformer {
        let dev = Device::Cpu;
        let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 2, 3);
        let embed_w = Tensor::new(&[[0.4f32, -0.2], [0.1, 0.6], [-0.3, 0.5]], &dev).unwrap();
        let pos_w = Tensor::new(&[[0.2f32, 0.1], [-0.1, 0.3], [0.5, -0.4]], &dev).unwrap();
        let attn = AttentionLayer {
            in_proj_weight: Tensor::new(
                &[
                    [0.3f32, 0.1],
                    [-0.2, 0.4],
                    [0.5, -0.3],
                    [0.2, 0.6],
                    [-0.4, 0.2],
                    [0.1, -0.5],
                ],
                &dev,
            )
            .unwrap(),
            out_proj_weight: Tensor::new(&[[0.6f32, -0.1], [0.3, 0.45]], &dev).unwrap(),
            hidden_size: 2,
        };
        StoicheiaTransformer {
            embed: Embedding::new(embed_w, 2),
            pos_embed: Embedding::new(pos_w, 2),
            attns: vec![attn],
            unembed_weight: Tensor::new(&[[0.5f32, 0.2], [-0.3, 0.7], [0.1, -0.4]], &dev).unwrap(),
            config,
        }
    }

    fn output_vec(cache: &HookCache) -> Vec<f32> {
        cache.output().flatten_all().unwrap().to_vec1().unwrap()
    }

    /// `BACKENDS.md`'s conformance case for `StoicheiaTransformer`.
    ///
    /// Until v0.2.0 this backend ran `is_captured` without ever consulting
    /// `interventions_at`, so this assertion failed silently: the "treated"
    /// output was the baseline, and any causal experiment measured zero.
    #[test]
    fn transformer_honours_intervention_at_resid_post() {
        let model = tiny_transformer();
        let dev = Device::Cpu;
        let ids = Tensor::new(&[[0u32, 1, 2]], &dev).unwrap();

        let baseline = output_vec(&model.forward(&ids, &HookSpec::new()).unwrap());

        let mut hooks = HookSpec::new();
        hooks.intervene(HookPoint::ResidPost(0), Intervention::Zero);
        let treated = output_vec(&model.forward(&ids, &hooks).unwrap());

        assert_ne!(
            baseline, treated,
            "Intervention::Zero at ResidPost(0) must change the output"
        );
    }

    /// The `AttnPattern` hook fires inside `AttentionLayer::forward`, so an
    /// intervention there must reach the weighted sum. Firing it on the
    /// returned tensor instead would leave the output untouched.
    #[test]
    fn transformer_honours_intervention_inside_attention() {
        let model = tiny_transformer();
        let dev = Device::Cpu;
        let ids = Tensor::new(&[[0u32, 1, 2]], &dev).unwrap();

        let baseline = output_vec(&model.forward(&ids, &HookSpec::new()).unwrap());

        let mut hooks = HookSpec::new();
        hooks.intervene(HookPoint::AttnPattern(0), Intervention::Zero);
        let treated = output_vec(&model.forward(&ids, &hooks).unwrap());

        assert_ne!(
            baseline, treated,
            "zeroing the attention pattern must change the output"
        );
    }

    /// Captures must keep working unchanged alongside the new intervention path.
    #[test]
    fn transformer_capture_still_works() {
        let model = tiny_transformer();
        let dev = Device::Cpu;
        let ids = Tensor::new(&[[0u32, 1, 2]], &dev).unwrap();

        let mut hooks = HookSpec::new();
        hooks.capture(HookPoint::ResidPost(0));
        hooks.capture(HookPoint::AttnPattern(0));
        let cache = model.forward(&ids, &hooks).unwrap();

        assert!(cache.get(&HookPoint::ResidPost(0)).is_some());
        assert!(cache.get(&HookPoint::AttnPattern(0)).is_some());
    }

    /// The RNN's per-timestep `Custom` hooks are pre-scanned for allocation
    /// reasons; the scan must cover interventions, not captures alone.
    #[test]
    fn rnn_honours_intervention_at_hidden_state() {
        let model = tiny_rnn();
        let dev = Device::Cpu;
        let input = Tensor::new(&[[1.0f32, 2.0, 3.0]], &dev).unwrap();

        let baseline = output_vec(&model.forward(&input, &HookSpec::new()).unwrap());

        let mut hooks = HookSpec::new();
        hooks.intervene(
            HookPoint::Custom("rnn.hook_hidden.0".into()),
            Intervention::Zero,
        );
        let treated = output_vec(&model.forward(&input, &hooks).unwrap());

        assert_ne!(
            baseline, treated,
            "zeroing the hidden state at t=0 must propagate through the recurrence"
        );
    }

    /// With no hooks registered the forward is untouched, so parity baselines
    /// recorded before v0.2.0 keep their meaning.
    #[test]
    fn empty_hookspec_leaves_output_unchanged() {
        let model = tiny_transformer();
        let dev = Device::Cpu;
        let ids = Tensor::new(&[[0u32, 1, 2]], &dev).unwrap();

        let a = output_vec(&model.forward(&ids, &HookSpec::new()).unwrap());
        let mut unrelated = HookSpec::new();
        unrelated.capture(HookPoint::Embed);
        let b = output_vec(&model.forward(&ids, &unrelated).unwrap());

        assert_eq!(a, b, "capturing must not perturb the forward pass");
    }
}
