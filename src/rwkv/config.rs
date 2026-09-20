// SPDX-License-Identifier: MIT OR Apache-2.0

//! RWKV configuration and `HuggingFace` `config.json` parsing.
//!
//! [`RwkvConfig`] captures the configuration axes that distinguish RWKV
//! versions (v6 Finch, v7 Goose).  Parsed from `HuggingFace` `config.json`
//! via [`from_hf_config`](RwkvConfig::from_hf_config).
//!
//! # Usage
//!
//! ```
//! use candle_mi::rwkv::RwkvConfig;
//!
//! let config_str = r#"{"model_type": "rwkv6", "hidden_size": 2048,
//!     "num_hidden_layers": 24, "num_attention_heads": 64,
//!     "vocab_size": 65536}"#;
//! let json: serde_json::Value = serde_json::from_str(config_str).unwrap();
//! let config = RwkvConfig::from_hf_config(&json).unwrap();
//! assert_eq!(config.num_layers, 24);
//! assert_eq!(config.num_heads, 32);
//! ```

use std::fmt;

use serde_json::Value;

use crate::config::{get_bool_or, get_f64_or, get_usize, get_usize_or};
use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Supported RWKV model types
// ---------------------------------------------------------------------------

/// `model_type` strings accepted by [`RwkvConfig::from_hf_config`].
pub const SUPPORTED_RWKV_MODEL_TYPES: &[&str] = &["rwkv6", "rwkv7"];

// ---------------------------------------------------------------------------
// RwkvVersion
// ---------------------------------------------------------------------------

/// RWKV architecture version.
///
/// Determines the recurrence formula (WKV kernel), token shift mechanism,
/// and channel mix variant.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RwkvVersion {
    /// RWKV-6 "Finch": data-dependent decay via `LoRA`, receptance-gated FFN.
    V6,
    /// RWKV-7 "Goose": generalized delta rule (`diag` + rank-1 state transition).
    V7,
}

impl fmt::Display for RwkvVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V6 => write!(f, "RWKV-6 (Finch)"),
            Self::V7 => write!(f, "RWKV-7 (Goose)"),
        }
    }
}

// ---------------------------------------------------------------------------
// RwkvLoraDims
// ---------------------------------------------------------------------------

/// Low-rank projection dimensions used in RWKV data-dependent mixing.
///
/// For RWKV-6 these are hardcoded (not in `config.json`):
/// `time_mix_extra_dim` = 32, `time_decay_extra_dim` = 64.
///
/// For RWKV-7 they appear explicitly in `config.json`:
/// `decay_low_rank_dim`, `a_low_rank_dim`, `v_low_rank_dim`, `gate_low_rank_dim`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RwkvLoraDims {
    // --- V6 fields ---
    /// Dimension for the 5-component time-mix `LoRA` projection (V6 only; 0 for V7).
    pub time_mix_extra_dim: usize,
    /// Dimension for the time-decay `LoRA` projection (V6 only; 0 for V7).
    pub time_decay_extra_dim: usize,

    // --- V7 fields ---
    /// Low-rank dimension for the decay `LoRA` (V7 only; 0 for V6).
    pub decay_low_rank_dim: usize,
    /// Low-rank dimension for the rank-1 state transition `LoRA` (V7 only; 0 for V6).
    pub a_low_rank_dim: usize,
    /// Low-rank dimension for the value-residual `LoRA` (V7 only; 0 for V6).
    pub v_low_rank_dim: usize,
    /// Low-rank dimension for the output gate `LoRA` (V7 only; 0 for V6).
    pub gate_low_rank_dim: usize,
}

// ---------------------------------------------------------------------------
// RwkvConfig
// ---------------------------------------------------------------------------

/// Configuration for a generic RWKV gated-linear RNN model.
///
/// Parsed from `HuggingFace` `config.json` via
/// [`from_hf_config`](Self::from_hf_config).
///
/// # Supported versions
///
/// | Version | Key config traits |
/// |---------|------------------|
/// | RWKV-6 (Finch) | Data-dependent decay (`LoRA`), receptance-gated FFN |
/// | RWKV-7 (Goose) | Generalized delta rule, plain FFN |
///
/// # `config.json` field reference (RWKV-6)
///
/// | Field | `config.json` key | Notes |
/// |-------|-------------------|-------|
/// | `hidden_size` | `hidden_size` | |
/// | `num_layers` | `num_hidden_layers` | |
/// | `head_dim` | `num_attention_heads` | Confusingly named: this is `head_size`, not head count |
/// | `vocab_size` | `vocab_size` | |
/// | `norm_eps` | `layer_norm_epsilon` | Default: 1e-5 |
/// | `head_size_divisor` | `head_size_divisor` | Default: 8; scales GroupNorm eps |
/// | `rescale_every` | `rescale_every` | Default: 6 |
/// | `intermediate_size` | `intermediate_size` | Default: `(hidden * 7/2) / 32 * 32` |
/// | `tie_word_embeddings` | `tie_word_embeddings` | Default: false |
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RwkvConfig {
    /// Architecture version.
    pub version: RwkvVersion,

    // --- Dimensions ----------------------------------------------------------
    /// Hidden dimension (`d_model`).
    pub hidden_size: usize,
    /// Number of RWKV blocks (layers).
    pub num_layers: usize,
    /// Per-head dimension (typically 64 for all current models).
    pub head_dim: usize,
    /// Number of heads (`hidden_size / head_dim`).
    pub num_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,

    // --- Normalization -------------------------------------------------------
    /// Epsilon for `LayerNorm` layers.
    pub norm_eps: f64,

    // --- FFN -----------------------------------------------------------------
    /// MLP intermediate dimension.
    ///
    /// RWKV-6: computed as `(hidden_size * 7/2) / 32 * 32` if not in config.
    /// RWKV-7: explicit in config (`hidden_size * hidden_ratio`).
    pub intermediate_size: usize,

    // --- Version-specific ----------------------------------------------------
    /// Rescale hidden states every N layers (v6: 6, v7: `None`).
    pub rescale_every: Option<usize>,
    /// Head-size divisor for `GroupNorm` epsilon scaling (v6: 8, v7: `None`).
    ///
    /// `GroupNorm` eps = `norm_eps * head_size_divisor^2`.
    pub head_size_divisor: Option<usize>,
    /// Low-rank projection dimensions for data-dependent mixing.
    pub lora_dims: RwkvLoraDims,
    /// Hidden ratio for FFN (v7 only; v6 uses the implicit 3.5x formula).
    pub hidden_ratio: Option<f64>,

    // --- Embeddings ----------------------------------------------------------
    /// Whether the LM head shares weights with the token embedding.
    pub tie_word_embeddings: bool,
}

impl RwkvConfig {
    /// Parse an [`RwkvConfig`] from a `HuggingFace` `config.json` value.
    ///
    /// Dispatches on the `model_type` field to a version-specific parser.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `model_type` is missing, unsupported,
    /// or if required fields are absent.
    pub fn from_hf_config(config: &Value) -> Result<Self> {
        let model_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| MIError::Config("missing 'model_type' field".into()))?;

        match model_type {
            "rwkv6" => Self::parse_rwkv6(config),
            "rwkv7" => Self::parse_rwkv7(config),
            other => Err(MIError::Config(format!(
                "unsupported RWKV model_type: '{other}'"
            ))),
        }
    }

    /// Compute the `GroupNorm` epsilon: `norm_eps * head_size_divisor^2`.
    ///
    /// Falls back to `norm_eps` if `head_size_divisor` is `None`.
    #[must_use]
    // CAST: usize → f64, head_size_divisor is small (typically 8 or 16)
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    pub fn group_norm_eps(&self) -> f64 {
        self.head_size_divisor
            .map_or(self.norm_eps, |d| self.norm_eps * (d as f64).powi(2))
    }
}

// ---------------------------------------------------------------------------
// Per-version config parsers
// ---------------------------------------------------------------------------

impl RwkvConfig {
    /// Parse an RWKV-6 "Finch" config.
    ///
    /// # Notes
    ///
    /// The `HuggingFace` RWKV-6 config uses `num_attention_heads` to store
    /// what is actually `head_size` (the per-head dimension, typically 64),
    /// **not** the number of heads.  The actual head count is computed as
    /// `hidden_size / head_size`.
    ///
    /// The `LoRA` dimensions (`time_mix_extra_dim` = 32, `time_decay_extra_dim` = 64)
    /// are hardcoded and do not appear in `config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    fn parse_rwkv6(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;

        // HF config confusingly stores head_size in "num_attention_heads".
        let head_dim = get_usize(config, "num_attention_heads")?;

        if head_dim == 0 {
            return Err(MIError::Config(
                "head_dim (num_attention_heads) is 0".into(),
            ));
        }
        let num_heads = hidden_size / head_dim;

        // intermediate_size: explicit in config, or computed as (hidden * 3.5) rounded to 32
        let intermediate_size =
            get_usize_or(config, "intermediate_size", (hidden_size * 7 / 2) / 32 * 32);

        Ok(Self {
            version: RwkvVersion::V6,
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            head_dim,
            num_heads,
            vocab_size: get_usize(config, "vocab_size")?,
            norm_eps: get_f64_or(config, "layer_norm_epsilon", 1e-5),
            intermediate_size,
            rescale_every: Some(get_usize_or(config, "rescale_every", 6)),
            head_size_divisor: Some(get_usize_or(config, "head_size_divisor", 8)),
            lora_dims: RwkvLoraDims {
                time_mix_extra_dim: 32,
                time_decay_extra_dim: 64,
                decay_low_rank_dim: 0,
                a_low_rank_dim: 0,
                v_low_rank_dim: 0,
                gate_low_rank_dim: 0,
            },
            hidden_ratio: None,
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),
        })
    }

    /// Parse an RWKV-7 "Goose" config (fla / `HuggingFace` format).
    ///
    /// # Notes
    ///
    /// RWKV-7 uses `head_dim` directly (not the confusing `num_attention_heads`
    /// alias from v6).  `LoRA` dimensions are explicit in `config.json`; when
    /// absent they are computed from `hidden_size` per the fla library defaults.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required dimension fields are missing.
    // CAST: usize → f64 for LoRA dim computation; f64 → usize for rounding back
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::as_conversions
    )]
    fn parse_rwkv7(config: &Value) -> Result<Self> {
        let hidden_size = get_usize(config, "hidden_size")?;
        let head_dim = get_usize_or(config, "head_dim", 64);

        if head_dim == 0 {
            return Err(MIError::Config("head_dim is 0".into()));
        }
        let num_heads = hidden_size / head_dim;

        // intermediate_size: explicit in config, or computed from hidden_ratio
        let hidden_ratio = config
            .get("hidden_ratio")
            .and_then(Value::as_f64)
            .unwrap_or(4.0);

        let intermediate_size = get_usize_or(
            config,
            "intermediate_size",
            Self::round_to_32((hidden_size as f64 * hidden_ratio) as usize),
        );

        // LoRA dims: explicit in config or fla defaults.
        // fla default: max(32, round_to_32(2.5 * sqrt(hidden) * factor))
        // where factor = head_dim / 64.
        let factor = head_dim as f64 / 64.0;
        let sqrt_h = (hidden_size as f64).sqrt();

        let decay_low_rank_dim = get_usize_or(
            config,
            "decay_low_rank_dim",
            Self::fla_lora_default(2.5, sqrt_h, factor),
        );
        let a_low_rank_dim = get_usize_or(
            config,
            "a_low_rank_dim",
            Self::fla_lora_default(2.5, sqrt_h, factor),
        );
        let v_low_rank_dim = get_usize_or(
            config,
            "v_low_rank_dim",
            Self::fla_lora_default(1.7, sqrt_h, factor),
        );
        let gate_low_rank_dim = get_usize_or(
            config,
            "gate_low_rank_dim",
            Self::fla_lora_default(5.0, sqrt_h, 1.0),
        );

        Ok(Self {
            version: RwkvVersion::V7,
            hidden_size,
            num_layers: get_usize(config, "num_hidden_layers")?,
            head_dim,
            num_heads,
            vocab_size: get_usize(config, "vocab_size")?,
            norm_eps: get_f64_or(config, "norm_eps", 1e-5),
            intermediate_size,
            rescale_every: None,
            head_size_divisor: None,
            lora_dims: RwkvLoraDims {
                time_mix_extra_dim: 0,
                time_decay_extra_dim: 0,
                decay_low_rank_dim,
                a_low_rank_dim,
                v_low_rank_dim,
                gate_low_rank_dim,
            },
            hidden_ratio: Some(hidden_ratio),
            tie_word_embeddings: get_bool_or(config, "tie_word_embeddings", false),
        })
    }

    /// Round `n` up to the nearest multiple of 32.
    const fn round_to_32(n: usize) -> usize {
        n.div_ceil(32) * 32
    }

    /// Compute fla-style default `LoRA` dimension:
    /// `max(32, round_to_32(scale * sqrt_h * factor))`.
    // CAST: f64 → usize, result is a small positive integer after rounding
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::as_conversions
    )]
    fn fla_lora_default(scale: f64, sqrt_h: f64, factor: f64) -> usize {
        let raw = (scale * sqrt_h * factor / 32.0).round() as usize * 32;
        raw.max(32)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Sample RWKV-6 config matching RWKV/v6-Finch-1B6-HF.
    fn rwkv6_config_json() -> Value {
        serde_json::json!({
            "model_type": "rwkv6",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "vocab_size": 65536,
            "layer_norm_epsilon": 1e-5,
            "head_size_divisor": 8,
            "rescale_every": 6,
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn parse_rwkv6_basic() {
        let config = RwkvConfig::from_hf_config(&rwkv6_config_json()).unwrap();
        assert_eq!(config.version, RwkvVersion::V6);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.vocab_size, 65536);
        assert!((config.norm_eps - 1e-5).abs() < f64::EPSILON);
        // intermediate_size = (2048 * 7/2) / 32 * 32 = 7168 / 32 * 32 = 7168
        assert_eq!(config.intermediate_size, 7168);
        assert_eq!(config.rescale_every, Some(6));
        assert_eq!(config.head_size_divisor, Some(8));
        assert_eq!(config.lora_dims.time_mix_extra_dim, 32);
        assert_eq!(config.lora_dims.time_decay_extra_dim, 64);
        assert!(config.hidden_ratio.is_none());
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn rwkv6_group_norm_eps() {
        let config = RwkvConfig::from_hf_config(&rwkv6_config_json()).unwrap();
        // GroupNorm eps = 1e-5 * 8^2 = 1e-5 * 64 = 6.4e-4
        let expected = 1e-5 * 64.0;
        assert!((config.group_norm_eps() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn rwkv6_explicit_intermediate_size() {
        let json = serde_json::json!({
            "model_type": "rwkv6",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "vocab_size": 65536,
            "intermediate_size": 8192
        });
        let config = RwkvConfig::from_hf_config(&json).unwrap();
        assert_eq!(config.intermediate_size, 8192);
    }

    /// Sample RWKV-7 config matching RWKV/RWKV7-Goose-World3-1.5B-HF.
    fn rwkv7_config_json() -> Value {
        serde_json::json!({
            "model_type": "rwkv7",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "head_dim": 64,
            "vocab_size": 65536,
            "norm_eps": 1e-5,
            "intermediate_size": 8192,
            "hidden_ratio": 4.0,
            "decay_low_rank_dim": 96,
            "a_low_rank_dim": 96,
            "v_low_rank_dim": 64,
            "gate_low_rank_dim": 256,
            "tie_word_embeddings": false
        })
    }

    #[test]
    fn parse_rwkv7_basic() {
        let config = RwkvConfig::from_hf_config(&rwkv7_config_json()).unwrap();
        assert_eq!(config.version, RwkvVersion::V7);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.vocab_size, 65536);
        assert!((config.norm_eps - 1e-5).abs() < f64::EPSILON);
        assert_eq!(config.intermediate_size, 8192);
        assert!(config.rescale_every.is_none());
        assert!(config.head_size_divisor.is_none());
        assert_eq!(config.lora_dims.decay_low_rank_dim, 96);
        assert_eq!(config.lora_dims.a_low_rank_dim, 96);
        assert_eq!(config.lora_dims.v_low_rank_dim, 64);
        assert_eq!(config.lora_dims.gate_low_rank_dim, 256);
        // V6 fields should be zero
        assert_eq!(config.lora_dims.time_mix_extra_dim, 0);
        assert_eq!(config.lora_dims.time_decay_extra_dim, 0);
        assert_eq!(config.hidden_ratio, Some(4.0));
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn rwkv7_group_norm_eps() {
        let config = RwkvConfig::from_hf_config(&rwkv7_config_json()).unwrap();
        // V7: head_size_divisor is None, so group_norm_eps = norm_eps * head_dim
        // (from the fla code: eps = head_dim * norm_eps)
        // Actually with head_size_divisor=None, group_norm_eps() falls back to norm_eps.
        // The V7 code will use head_dim * norm_eps directly.
        assert!((config.group_norm_eps() - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn rwkv7_default_lora_dims() {
        // When LoRA dims are absent, fla defaults should be computed
        let json = serde_json::json!({
            "model_type": "rwkv7",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "vocab_size": 65536
        });
        let config = RwkvConfig::from_hf_config(&json).unwrap();
        // sqrt(2048) ≈ 45.25, factor = 64/64 = 1.0
        // decay: round(2.5 * 45.25 * 1.0 / 32) * 32 = round(3.53) * 32 = 4 * 32 = 128
        assert!(config.lora_dims.decay_low_rank_dim >= 32);
        assert!(config.lora_dims.gate_low_rank_dim >= 32);
    }

    #[test]
    fn unsupported_model_type_errors() {
        let json = serde_json::json!({ "model_type": "gpt2" });
        let result = RwkvConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn missing_model_type_errors() {
        let json = serde_json::json!({ "hidden_size": 2048 });
        let result = RwkvConfig::from_hf_config(&json);
        assert!(result.is_err());
    }

    #[test]
    fn version_display() {
        assert_eq!(RwkvVersion::V6.to_string(), "RWKV-6 (Finch)");
        assert_eq!(RwkvVersion::V7.to_string(), "RWKV-7 (Goose)");
    }
}
