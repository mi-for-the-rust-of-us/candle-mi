// SPDX-License-Identifier: MIT OR Apache-2.0

//! MLP variants for the generic transformer.
//!
//! Supports three layouts:
//! - [`GatedSeparate`](MlpLayout::GatedSeparate): `down(act(gate(x)) * up(x))`
//! - [`GatedFused`](MlpLayout::GatedFused): fused gate+up, then split
//! - [`Plain`](MlpLayout::Plain): `proj(act(fc(x)))`

use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::{Activation, MlpLayout, TransformerConfig};
use crate::error::Result;

// ---------------------------------------------------------------------------
// Mlp
// ---------------------------------------------------------------------------

/// MLP (feed-forward network) layer, parameterized by layout and activation.
pub struct Mlp {
    /// MLP variant (determines forward pass structure).
    variant: MlpVariant,
    /// Activation function.
    activation: Activation,
}

/// Internal representation of the MLP variant.
enum MlpVariant {
    /// Gated MLP with separate gate and up projections.
    GatedSeparate {
        /// Gate projection: `[hidden_size, intermediate_size]`.
        gate_proj: Linear,
        /// Up projection: `[hidden_size, intermediate_size]`.
        up_proj: Linear,
        /// Down projection: `[intermediate_size, hidden_size]`.
        down_proj: Linear,
    },
    /// Gated MLP with fused gate+up projection (Phi-3).
    GatedFused {
        /// Fused gate+up: `[hidden_size, 2 * intermediate_size]`.
        gate_up_proj: Linear,
        /// Down projection: `[intermediate_size, hidden_size]`.
        down_proj: Linear,
        /// Intermediate size (for splitting the fused output).
        intermediate_size: usize,
    },
    /// Plain (non-gated) MLP (`StarCoder2`).
    Plain {
        /// First projection: `[hidden_size, intermediate_size]`.
        fc: Linear,
        /// Second projection: `[intermediate_size, hidden_size]`.
        proj: Linear,
    },
}

impl Mlp {
    /// Load MLP weights from a [`VarBuilder`].
    ///
    /// The weight names depend on the layout:
    /// - `GatedSeparate`: `gate_proj`, `up_proj`, `down_proj`
    /// - `GatedFused`: `gate_up_proj`, `down_proj`
    /// - `Plain`: `c_fc`, `c_proj`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if weight loading fails.
    #[allow(clippy::needless_pass_by_value)] // VarBuilder is candle's pass-by-value convention
    pub fn load(config: &TransformerConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let bias = config.mlp_bias;

        let variant = match config.mlp_layout {
            MlpLayout::GatedSeparate => {
                let gate_proj = load_linear(hidden, inter, bias, vb.pp("gate_proj"))?;
                let up_proj = load_linear(hidden, inter, bias, vb.pp("up_proj"))?;
                let down_proj = load_linear(inter, hidden, bias, vb.pp("down_proj"))?;
                MlpVariant::GatedSeparate {
                    gate_proj,
                    up_proj,
                    down_proj,
                }
            }
            MlpLayout::GatedFused => {
                let gate_up_proj = load_linear(hidden, 2 * inter, bias, vb.pp("gate_up_proj"))?;
                let down_proj = load_linear(inter, hidden, bias, vb.pp("down_proj"))?;
                MlpVariant::GatedFused {
                    gate_up_proj,
                    down_proj,
                    intermediate_size: inter,
                }
            }
            MlpLayout::Plain => {
                let fc = load_linear(hidden, inter, bias, vb.pp("c_fc"))?;
                let proj = load_linear(inter, hidden, bias, vb.pp("c_proj"))?;
                MlpVariant::Plain { fc, proj }
            }
        };

        Ok(Self {
            variant,
            activation: config.activation,
        })
    }

    /// Run the MLP forward pass.
    ///
    /// # Shapes
    /// - `x`: `[batch, seq, hidden_size]`
    /// - returns: `[batch, seq, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.variant {
            MlpVariant::GatedSeparate {
                gate_proj,
                up_proj,
                down_proj,
            } => {
                let gate = apply_activation(&gate_proj.forward(x)?, self.activation)?;
                let up = up_proj.forward(x)?;
                Ok(down_proj.forward(&(gate * up)?)?)
            }
            MlpVariant::GatedFused {
                gate_up_proj,
                down_proj,
                intermediate_size,
            } => {
                let gate_up = gate_up_proj.forward(x)?;
                let gate = gate_up.narrow(candle_core::D::Minus1, 0, *intermediate_size)?;
                let up = gate_up.narrow(
                    candle_core::D::Minus1,
                    *intermediate_size,
                    *intermediate_size,
                )?;
                let gate = apply_activation(&gate, self.activation)?;
                Ok(down_proj.forward(&(gate * up)?)?)
            }
            MlpVariant::Plain { fc, proj } => {
                let hidden = apply_activation(&fc.forward(x)?, self.activation)?;
                Ok(proj.forward(&hidden)?)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply the selected activation function (the dispatch shared with every
/// backbone, in `crate::nn_ops`).
fn apply_activation(x: &Tensor, activation: Activation) -> Result<Tensor> {
    crate::nn_ops::activation(x, activation)
}

/// Load a linear layer with or without bias.
#[allow(clippy::needless_pass_by_value)] // VarBuilder convention
fn load_linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder<'_>) -> Result<Linear> {
    if bias {
        Ok(candle_nn::linear(in_dim, out_dim, vb)?)
    } else {
        Ok(candle_nn::linear_no_bias(in_dim, out_dim, vb)?)
    }
}
