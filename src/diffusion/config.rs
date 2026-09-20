// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration for the `MDLM` masked-diffusion backend.
//!
//! [`MdlmConfig`] is parsed from a `kuleshov-group/mdlm-owt`-style
//! `config.json` (model type `"mdlm"`).  Unlike
//! [`TransformerConfig`](crate::TransformerConfig), `MDLM` is a bidirectional
//! `DiT` with `adaLN` conditioning, so it carries its own small config rather
//! than reusing the causal-decoder one.

use serde_json::Value;

use crate::error::{MIError, Result};

/// `MDLM` masked-diffusion model configuration.
///
/// Parsed from the upstream `config.json`.  The released `mdlm-owt`
/// checkpoint is time-independent (`time_conditioning = false`), so the
/// `DiT` conditioning vector is a constant computed once at load time.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MdlmConfig {
    /// Hidden dimension (`d_model`; upstream key `hidden_dim`).
    pub hidden_dim: usize,
    /// Number of `DiT` blocks (upstream key `n_blocks`).
    pub n_blocks: usize,
    /// Number of attention heads (upstream key `n_heads`).
    pub n_heads: usize,
    /// Per-head dimension (`hidden_dim / n_heads`).
    pub head_dim: usize,
    /// Conditioning dimension fed to the `adaLN` modulation (upstream key `cond_dim`).
    pub cond_dim: usize,
    /// Vocabulary size, including the `[MASK]` token (upstream key `vocab_size`).
    pub vocab_size: usize,
    /// Maximum sequence length (upstream key `model_length`).
    pub model_length: usize,
    /// Feed-forward expansion ratio (`MDLM` uses `4`; not stored in `config.json`).
    pub mlp_ratio: usize,
    /// Rotary base frequency (`MDLM` hard-codes `10000`).
    pub rope_theta: f64,
    /// `LayerNorm` epsilon (`MDLM` uses `1e-5`).
    pub norm_eps: f64,
    /// Token id of the absorbing `[MASK]` state (`vocab_size - 1`).
    pub mask_token_id: u32,
    /// Whether the network conditions on diffusion time.  The released
    /// checkpoint sets this `false`, so the conditioning vector is constant.
    pub time_conditioning: bool,
}

impl MdlmConfig {
    /// Parse an [`MdlmConfig`] from a `HuggingFace` `config.json` value.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a required key
    /// (`hidden_dim`, `n_blocks`, `n_heads`, `cond_dim`, `vocab_size`) is
    /// missing or not a non-negative integer, or if `hidden_dim` is not
    /// divisible by `n_heads`.
    pub fn from_hf_config(config: &Value) -> Result<Self> {
        let hidden_dim = get_usize(config, "hidden_dim")?;
        let n_heads = get_usize(config, "n_heads")?;
        if n_heads == 0 || !hidden_dim.is_multiple_of(n_heads) {
            return Err(MIError::Config(format!(
                "hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )));
        }
        let vocab_size = get_usize(config, "vocab_size")?;
        // The absorbing [MASK] state is the final vocab index (GPT-2's 50257
        // tokens 0..=50256 plus [MASK] = 50257, for vocab_size 50258).
        let mask_token_id = u32::try_from(vocab_size.saturating_sub(1)).map_err(|e| {
            MIError::Config(format!("vocab_size {vocab_size} does not fit in u32: {e}"))
        })?;

        Ok(Self {
            hidden_dim,
            n_blocks: get_usize(config, "n_blocks")?,
            n_heads,
            head_dim: hidden_dim / n_heads,
            cond_dim: get_usize(config, "cond_dim")?,
            vocab_size,
            model_length: get_usize_or(config, "model_length", 1024),
            mlp_ratio: 4,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            mask_token_id,
            time_conditioning: get_bool_or(config, "time_conditioning", false),
        })
    }
}

/// Read a required non-negative integer config field as `usize`.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the key is absent or
/// not a `u64`.
fn get_usize(config: &Value, key: &str) -> Result<usize> {
    let value = config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| MIError::Config(format!("missing or non-integer `{key}` in MDLM config")))?;
    // CAST: u64 → usize, model dimensions fit in usize on 64-bit targets
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    Ok(value as usize)
}

/// Read an optional non-negative integer config field as `usize`, falling
/// back to `default` when absent.
fn get_usize_or(config: &Value, key: &str, default: usize) -> usize {
    config
        .get(key)
        .and_then(Value::as_u64)
        .map_or(default, |v| {
            // CAST: u64 → usize, model dimensions fit in usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            {
                v as usize
            }
        })
}

/// Read an optional boolean config field, falling back to `default` when absent.
fn get_bool_or(config: &Value, key: &str, default: bool) -> bool {
    config.get(key).and_then(Value::as_bool).unwrap_or(default)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn owt_config() -> Value {
        serde_json::json!({
            "model_type": "mdlm",
            "hidden_dim": 768,
            "n_blocks": 12,
            "n_heads": 12,
            "cond_dim": 128,
            "vocab_size": 50258,
            "model_length": 1024,
            "time_conditioning": false
        })
    }

    #[test]
    fn parses_mdlm_owt_config() {
        let cfg = MdlmConfig::from_hf_config(&owt_config()).unwrap();
        assert_eq!(cfg.hidden_dim, 768);
        assert_eq!(cfg.n_blocks, 12);
        assert_eq!(cfg.n_heads, 12);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.cond_dim, 128);
        assert_eq!(cfg.vocab_size, 50258);
        assert_eq!(cfg.model_length, 1024);
        assert_eq!(cfg.mask_token_id, 50257);
        assert!(!cfg.time_conditioning);
    }

    #[test]
    fn rejects_indivisible_head_count() {
        let mut cfg = owt_config();
        cfg["n_heads"] = serde_json::json!(7);
        assert!(MdlmConfig::from_hf_config(&cfg).is_err());
    }

    #[test]
    fn missing_required_key_errors() {
        let mut cfg = owt_config();
        // BORROW: remove a required key to exercise the error path.
        cfg.as_object_mut().unwrap().remove("hidden_dim");
        assert!(MdlmConfig::from_hf_config(&cfg).is_err());
    }
}
