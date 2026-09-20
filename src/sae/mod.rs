// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse Autoencoder (SAE) support.
//!
//! Loads pre-trained SAE weights from SAELens-format safetensors + `cfg.json`
//! or from Gemma Scope NPZ archives, encodes model activations into sparse
//! feature vectors, decodes back to activation space, and produces steering
//! vectors for injection.
//!
//! Each SAE targets a single hook point in the model (e.g., `resid_post` at
//! layer 5). Multiple SAEs can be loaded independently for different hook
//! points.
//!
//! # SAE Architecture
//!
//! A Sparse Autoencoder implements:
//! ```text
//! Encode:  features = activation_fn(x @ W_enc + b_enc)
//! Decode:  x_hat = features @ W_dec + b_dec
//! ```
//!
//! Supported activation functions:
//! - **`ReLU`**: `features = ReLU(pre_acts)`
//! - **`JumpReLU`**: `features = pre_acts * (pre_acts > threshold)`
//! - **`TopK`**: keep only the top-k pre-activations, zero the rest
//!
//! # Weight File Formats
//!
//! ## `SAELens` safetensors format
//!
//! Each SAE directory contains:
//! - `cfg.json`: configuration (`d_in`, `d_sae`, architecture, `hook_name`, ...)
//! - `sae_weights.safetensors` (or `model.safetensors`): weight tensors
//!
//! ## Gemma Scope NPZ format
//!
//! A single `params.npz` file (`NumPy` ZIP archive) containing:
//! - `W_enc.npy`: shape `[d_in, d_sae]` — encoder weight matrix
//! - `W_dec.npy`: shape `[d_sae, d_in]` — decoder weight matrix
//! - `b_enc.npy`: shape `[d_sae]` — encoder bias
//! - `b_dec.npy`: shape `[d_in]` — decoder bias
//! - `threshold.npy`: shape `[d_sae]` — `JumpReLU` threshold (optional)
//!
//! Architecture is auto-detected: presence of `threshold` → `JumpReLU`,
//! otherwise `ReLU`.
//!
//! ## Tensor names (both formats)
//!
//! - `W_enc`: shape `[d_in, d_sae]` — encoder weight matrix
//! - `W_dec`: shape `[d_sae, d_in]` — decoder weight matrix
//! - `b_enc`: shape `[d_sae]` — encoder bias
//! - `b_dec`: shape `[d_in]` — decoder bias
//! - `threshold`: shape `[d_sae]` — `JumpReLU` threshold (optional)

// `npz` adapts anamnesis NPZ tensors to candle. Originally private to the
// `sae` module; now `pub(crate)` so the CLT GemmaScope loader can share
// the same NPZ → candle bridge without duplicating the F32/F64 conversion.
pub(crate) mod npz;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use safetensors::tensor::SafeTensors;
use tracing::info;

use crate::error::{MIError, Result};
use crate::hooks::{HookPoint, HookSpec, Intervention};
use crate::sparse::{FeatureId, SparseActivations};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Identifies a single SAE feature by its index within the dictionary.
// EXHAUSTIVE: a single-field value identity, not a growing record. Same
// reasoning as `CltFeatureId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SaeFeatureId {
    /// Feature index within the SAE dictionary (`0..d_sae`).
    pub index: usize,
}

impl std::fmt::Display for SaeFeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SAE:{}", self.index)
    }
}

impl FeatureId for SaeFeatureId {}

/// Activation function architecture for the SAE encoder.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaeArchitecture {
    /// Standard `ReLU`: `features = ReLU(W_enc @ x + b_enc)`.
    ReLU,
    /// `JumpReLU`: `features = pre_acts * (pre_acts > threshold)`.
    /// Requires a learned `threshold` tensor of shape `[d_sae]`.
    JumpReLU,
    /// `TopK`: keep only the top-k pre-activations, zero the rest.
    TopK {
        /// Number of features to keep active.
        k: usize,
    },
}

/// Input normalization strategy for SAE encoding.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizeActivations {
    /// No normalization.
    None,
    /// Normalize by expected average L2 norm (estimated during training).
    ExpectedAverageOnlyIn,
}

/// Strategy for `TopK` activation computation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopKStrategy {
    /// Automatically select based on device (CPU → direct, GPU → sort-based).
    Auto,
    /// Force CPU-side computation (transfer to CPU, compute mask, transfer back).
    Cpu,
    /// Force GPU-side sort-based computation.
    Gpu,
}

/// Configuration for a Sparse Autoencoder, parsed from `cfg.json`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SaeConfig {
    /// Input dimension (must match model hidden size at the hook point).
    pub d_in: usize,
    /// Dictionary size (number of SAE features).
    pub d_sae: usize,
    /// Encoder architecture (activation function).
    pub architecture: SaeArchitecture,
    /// `SAELens` hook name string (e.g., `"blocks.5.hook_resid_post"`).
    pub hook_name: String,
    /// Parsed hook point from the hook name.
    pub hook_point: HookPoint,
    /// Whether to subtract `b_dec` from input before encoding.
    pub apply_b_dec_to_input: bool,
    /// Input normalization strategy.
    pub normalize_activations: NormalizeActivations,
}

// ---------------------------------------------------------------------------
// Internal config parsing
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
#[allow(clippy::missing_docs_in_private_items)]
struct RawSaeConfig {
    d_in: usize,
    d_sae: usize,
    #[serde(default)]
    architecture: Option<String>,
    #[serde(default)]
    activation_fn_str: Option<String>,
    #[serde(default)]
    activation_fn_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    hook_name: Option<String>,
    #[serde(default)]
    hook_point: Option<String>,
    #[serde(default)]
    apply_b_dec_to_input: bool,
    #[serde(default)]
    normalize_activations: Option<String>,
}

/// Parse a `RawSaeConfig` into a validated `SaeConfig`.
fn parse_sae_config(raw: RawSaeConfig) -> Result<SaeConfig> {
    // Resolve architecture from `architecture` or `activation_fn_str`.
    let architecture = resolve_architecture(
        raw.architecture.as_deref(),
        raw.activation_fn_str.as_deref(),
        raw.activation_fn_kwargs.as_ref(),
    )?;

    // Resolve hook name from `hook_name` or `hook_point`.
    let hook_name = raw
        .hook_name
        .or(raw.hook_point)
        .unwrap_or_else(|| "unknown".to_owned());

    // Parse hook name to HookPoint via FromStr.
    let hook_point: HookPoint = hook_name
        .parse()
        .unwrap_or_else(|_: std::convert::Infallible| {
            // EXHAUSTIVE: Infallible can never happen, parse always succeeds
            unreachable!()
        });

    let normalize_activations = match raw.normalize_activations.as_deref() {
        Some("expected_average_only_in") => NormalizeActivations::ExpectedAverageOnlyIn,
        _ => NormalizeActivations::None,
    };

    Ok(SaeConfig {
        d_in: raw.d_in,
        d_sae: raw.d_sae,
        architecture,
        hook_name,
        hook_point,
        apply_b_dec_to_input: raw.apply_b_dec_to_input,
        normalize_activations,
    })
}

/// Resolve SAE architecture from config fields.
fn resolve_architecture(
    architecture: Option<&str>,
    activation_fn_str: Option<&str>,
    activation_fn_kwargs: Option<&serde_json::Value>,
) -> Result<SaeArchitecture> {
    // Check `architecture` field first.
    match architecture {
        Some("jumprelu") => return Ok(SaeArchitecture::JumpReLU),
        Some("topk") => {
            let k = extract_topk_k(activation_fn_kwargs)?;
            return Ok(SaeArchitecture::TopK { k });
        }
        Some("standard") | None => {} // fall through to activation_fn_str
        Some(other) => {
            return Err(MIError::Config(format!(
                "unsupported SAE architecture: {other:?}"
            )));
        }
    }

    // Fall back to `activation_fn_str`.
    match activation_fn_str {
        Some("relu") | None => Ok(SaeArchitecture::ReLU),
        Some("jumprelu") => Ok(SaeArchitecture::JumpReLU),
        Some("topk") => {
            let k = extract_topk_k(activation_fn_kwargs)?;
            Ok(SaeArchitecture::TopK { k })
        }
        Some(other) => Err(MIError::Config(format!(
            "unsupported SAE activation function: {other:?}"
        ))),
    }
}

/// Extract `k` from `activation_fn_kwargs.k`.
fn extract_topk_k(kwargs: Option<&serde_json::Value>) -> Result<usize> {
    let k = kwargs
        .and_then(|v| v.get("k"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            MIError::Config("TopK SAE requires activation_fn_kwargs.k in cfg.json".into())
        })?;
    let k_usize = usize::try_from(k)
        .map_err(|_| MIError::Config(format!("TopK k value {k} too large for usize")))?;
    Ok(k_usize)
}

// ---------------------------------------------------------------------------
// SparseAutoencoder
// ---------------------------------------------------------------------------

/// A Sparse Autoencoder for mechanistic interpretability.
///
/// Loads SAE weights from SAELens-format safetensors + `cfg.json`,
/// encodes model activations into sparse feature vectors, decodes
/// back to activation space, and produces steering vectors for injection.
///
/// Each SAE targets a single hook point in the model (e.g., `resid_post`
/// at layer 5). Multiple SAEs can be loaded independently for different
/// hook points.
///
/// # Example
///
/// ```no_run
/// # fn main() -> candle_mi::Result<()> {
/// use candle_mi::sae::SparseAutoencoder;
/// use candle_core::Device;
///
/// let sae = SparseAutoencoder::from_pretrained(
///     "jbloom/Gemma-2-2B-Residual-Stream-SAEs",
///     "gemma-2-2b-res-jb/blocks.20.hook_resid_post",
///     &Device::Cpu,
/// )?;
/// println!("SAE: d_in={}, d_sae={}", sae.d_in(), sae.d_sae());
/// # Ok(())
/// # }
/// ```
pub struct SparseAutoencoder {
    /// SAE configuration parsed from `cfg.json`.
    config: SaeConfig,
    /// Encoder weight matrix.
    ///
    /// # Shapes
    /// - `w_enc`: `[d_in, d_sae]`
    w_enc: Tensor,
    /// Decoder weight matrix.
    ///
    /// # Shapes
    /// - `w_dec`: `[d_sae, d_in]`
    w_dec: Tensor,
    /// Encoder bias vector.
    ///
    /// # Shapes
    /// - `b_enc`: `[d_sae]`
    b_enc: Tensor,
    /// Decoder bias vector.
    ///
    /// # Shapes
    /// - `b_dec`: `[d_in]`
    b_dec: Tensor,
    /// `JumpReLU` threshold (only present for `JumpReLU` architecture).
    ///
    /// # Shapes
    /// - `threshold`: `[d_sae]`
    threshold: Option<Tensor>,
}

impl SparseAutoencoder {
    // --- Loading ---

    /// Load an SAE from a local directory containing safetensors + `cfg.json`.
    ///
    /// Expects either `sae_weights.safetensors` or `model.safetensors`
    /// plus a `cfg.json` file.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `cfg.json` is missing or malformed.
    /// Returns [`MIError::Config`] if weight shapes don't match `cfg.json` dimensions.
    /// Returns [`MIError::Model`] on tensor loading failure.
    /// Returns [`MIError::Io`] if files cannot be read.
    pub fn from_local(dir: &Path, device: &Device) -> Result<Self> {
        // Parse cfg.json.
        let cfg_path = dir.join("cfg.json");
        if !cfg_path.exists() {
            return Err(MIError::Config(format!(
                "cfg.json not found in {}",
                dir.display()
            )));
        }
        let cfg_text = std::fs::read_to_string(&cfg_path)?;
        let raw: RawSaeConfig = serde_json::from_str(&cfg_text)
            .map_err(|e| MIError::Config(format!("failed to parse cfg.json: {e}")))?;
        let config = parse_sae_config(raw)?;

        info!(
            "SAE config: d_in={}, d_sae={}, arch={:?}, hook={}",
            config.d_in, config.d_sae, config.architecture, config.hook_name
        );

        // Find safetensors file.
        let weights_path = if dir.join("sae_weights.safetensors").exists() {
            dir.join("sae_weights.safetensors")
        } else if dir.join("model.safetensors").exists() {
            dir.join("model.safetensors")
        } else {
            return Err(MIError::Config(format!(
                "no safetensors file found in {}",
                dir.display()
            )));
        };

        // Load weights.
        let data = std::fs::read(&weights_path)?;
        let st = SafeTensors::deserialize(&data)
            .map_err(|e| MIError::Config(format!("failed to deserialize SAE weights: {e}")))?;

        let w_enc = load_tensor(&st, "W_enc", device)?;
        let w_dec = load_tensor(&st, "W_dec", device)?;
        let b_enc = load_tensor(&st, "b_enc", device)?;
        let b_dec = load_tensor(&st, "b_dec", device)?;
        let threshold = st
            .tensor("threshold")
            .ok()
            .map(|v| crate::util::safetensors_view::tensor_from_view(&v, device, "SAE"))
            .transpose()?;

        // PROMOTE: F32 for numerical stability in matmul and bias add
        let w_enc = w_enc.to_dtype(DType::F32)?;
        let w_dec = w_dec.to_dtype(DType::F32)?;
        let b_enc = b_enc.to_dtype(DType::F32)?;
        let b_dec = b_dec.to_dtype(DType::F32)?;
        let threshold = threshold.map(|t| t.to_dtype(DType::F32)).transpose()?;

        // Validate shapes.
        validate_shape(&w_enc, &[config.d_in, config.d_sae], "W_enc")?;
        validate_shape(&w_dec, &[config.d_sae, config.d_in], "W_dec")?;
        validate_shape(&b_enc, &[config.d_sae], "b_enc")?;
        validate_shape(&b_dec, &[config.d_in], "b_dec")?;
        if let Some(ref t) = threshold {
            validate_shape(t, &[config.d_sae], "threshold")?;
        }

        // Validate JumpReLU has threshold.
        if config.architecture == SaeArchitecture::JumpReLU && threshold.is_none() {
            return Err(MIError::Config(
                "JumpReLU SAE requires 'threshold' tensor in weights file".into(),
            ));
        }

        info!(
            "SAE loaded: {} weights on {:?}",
            weights_path.display(),
            device
        );

        Ok(Self {
            config,
            w_enc,
            w_dec,
            b_enc,
            b_dec,
            threshold,
        })
    }

    /// Load an SAE from a Gemma Scope NPZ file (`params.npz`).
    ///
    /// The NPZ file must contain `W_enc`, `W_dec`, `b_enc`, `b_dec` arrays,
    /// and optionally `threshold` (for `JumpReLU`). Config is inferred from
    /// tensor shapes since NPZ files have no `cfg.json`.
    ///
    /// # Arguments
    /// * `npz_path` — Path to the `params.npz` file
    /// * `hook_layer` — Which model layer this SAE hooks into
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if required tensors are missing or shapes
    /// are inconsistent.
    /// Returns [`MIError::Io`] if the file cannot be read.
    pub fn from_npz(npz_path: &Path, hook_layer: usize, device: &Device) -> Result<Self> {
        info!("Loading SAE from NPZ: {}", npz_path.display());
        let tensors = npz::load_npz(npz_path, device)?;

        let w_enc = tensors
            .get("W_enc")
            .ok_or_else(|| MIError::Config("NPZ missing W_enc".into()))?
            .to_dtype(DType::F32)?;
        let w_dec = tensors
            .get("W_dec")
            .ok_or_else(|| MIError::Config("NPZ missing W_dec".into()))?
            .to_dtype(DType::F32)?;
        let b_enc = tensors
            .get("b_enc")
            .ok_or_else(|| MIError::Config("NPZ missing b_enc".into()))?
            .to_dtype(DType::F32)?;
        let b_dec = tensors
            .get("b_dec")
            .ok_or_else(|| MIError::Config("NPZ missing b_dec".into()))?
            .to_dtype(DType::F32)?;
        let threshold = tensors
            .get("threshold")
            .map(|t| t.to_dtype(DType::F32))
            .transpose()?;

        // Infer dimensions from W_enc: [d_in, d_sae].
        let w_enc_dims = w_enc.dims();
        if w_enc_dims.len() != 2 {
            return Err(MIError::Config(format!(
                "W_enc expected 2 dims, got {}",
                w_enc_dims.len()
            )));
        }
        let d_in = *w_enc_dims
            .first()
            .ok_or_else(|| MIError::Config("W_enc has no dimensions".into()))?;
        let d_sae = *w_enc_dims
            .get(1)
            .ok_or_else(|| MIError::Config("W_enc has no second dimension".into()))?;

        // Validate shapes.
        validate_shape(&w_enc, &[d_in, d_sae], "W_enc")?;
        validate_shape(&w_dec, &[d_sae, d_in], "W_dec")?;
        validate_shape(&b_enc, &[d_sae], "b_enc")?;
        validate_shape(&b_dec, &[d_in], "b_dec")?;
        if let Some(ref t) = threshold {
            validate_shape(t, &[d_sae], "threshold")?;
        }

        // Auto-detect architecture: threshold present → JumpReLU, else ReLU.
        let architecture = if threshold.is_some() {
            SaeArchitecture::JumpReLU
        } else {
            SaeArchitecture::ReLU
        };

        let hook_name = format!("blocks.{hook_layer}.hook_resid_post");
        let hook_point = hook_name
            .parse::<HookPoint>()
            .map_err(|e| MIError::Config(format!("failed to parse hook name: {e}")))?;

        let config = SaeConfig {
            d_in,
            d_sae,
            architecture,
            hook_name,
            hook_point,
            apply_b_dec_to_input: false,
            normalize_activations: NormalizeActivations::None,
        };

        info!(
            "SAE from NPZ: d_in={d_in}, d_sae={d_sae}, arch={:?}, hook={}",
            config.architecture, config.hook_name
        );

        Ok(Self {
            config,
            w_enc,
            w_dec,
            b_enc,
            b_dec,
            threshold,
        })
    }

    /// Load an SAE from a `HuggingFace` repository containing an NPZ file.
    ///
    /// Downloads the NPZ file via `hf-fetch-model`, then delegates to
    /// [`from_npz`](Self::from_npz).
    ///
    /// # Arguments
    /// * `repo_id` — `HuggingFace` repository ID
    ///   (e.g., `"google/gemma-scope-2b-pt-res"`)
    /// * `npz_path` — Path within the repo to the NPZ file
    ///   (e.g., `"layer_0/width_16k/average_l0_105/params.npz"`)
    /// * `hook_layer` — Which model layer this SAE hooks into
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the file cannot be fetched.
    /// Returns [`MIError::Config`] if the NPZ format is invalid.
    pub fn from_pretrained_npz(
        repo_id: &str,
        npz_path: &str,
        hook_layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let fetch_config = crate::download::fetch_config_builder()
            .on_progress(|event| {
                tracing::info!(
                    filename = %event.filename,
                    percent = event.percent,
                    bytes_downloaded = event.bytes_downloaded,
                    bytes_total = event.bytes_total,
                    "SAE NPZ download progress",
                );
            })
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;

        info!("Downloading {npz_path} from {repo_id}");
        let local_path =
            hf_fetch_model::download_file_blocking(repo_id.to_owned(), npz_path, &fetch_config)
                .map_err(|e| MIError::Download(format!("failed to download NPZ: {e}")))?
                .into_inner();

        Self::from_npz(&local_path, hook_layer, device)
    }

    /// Load an SAE from a `HuggingFace` repository.
    ///
    /// Downloads safetensors + `cfg.json` via `hf-fetch-model`, then delegates
    /// to [`from_local`](Self::from_local).
    ///
    /// # Arguments
    /// * `repo_id` — `HuggingFace` repository ID
    ///   (e.g., `"jbloom/Gemma-2-2B-Residual-Stream-SAEs"`)
    /// * `sae_id` — Subdirectory within the repo
    ///   (e.g., `"gemma-2-2b-res-jb/blocks.20.hook_resid_post"`)
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if files cannot be fetched.
    /// Returns [`MIError::Config`] if the SAE format is invalid.
    pub fn from_pretrained(repo_id: &str, sae_id: &str, device: &Device) -> Result<Self> {
        let fetch_config = crate::download::fetch_config_builder()
            .on_progress(|event| {
                tracing::info!(
                    filename = %event.filename,
                    percent = event.percent,
                    bytes_downloaded = event.bytes_downloaded,
                    bytes_total = event.bytes_total,
                    "SAE download progress",
                );
            })
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;

        // Download cfg.json.
        let cfg_remote = format!("{sae_id}/cfg.json");
        info!("Downloading {cfg_remote} from {repo_id}");
        let cfg_path =
            hf_fetch_model::download_file_blocking(repo_id.to_owned(), &cfg_remote, &fetch_config)
                .map_err(|e| MIError::Download(format!("failed to download cfg.json: {e}")))?
                .into_inner();

        // Download weights: try sae_weights.safetensors, fall back to model.safetensors.
        let weights_remote = format!("{sae_id}/sae_weights.safetensors");
        info!("Downloading {weights_remote} from {repo_id}");
        let weights_path = hf_fetch_model::download_file_blocking(
            repo_id.to_owned(),
            &weights_remote,
            &fetch_config,
        )
        .or_else(|_| {
            let alt_remote = format!("{sae_id}/model.safetensors");
            info!("Trying {alt_remote} from {repo_id}");
            hf_fetch_model::download_file_blocking(repo_id.to_owned(), &alt_remote, &fetch_config)
        })
        .map_err(|e| MIError::Download(format!("failed to download SAE weights: {e}")))?
        .into_inner();

        // Both files are in cache; determine the common directory.
        let dir = cfg_path.parent().ok_or_else(|| {
            MIError::Config("cannot determine SAE directory from cfg.json path".into())
        })?;

        // Verify the weights file is in the same directory (or just load by path).
        // hf-fetch-model may place files in the same cache dir; if not, we need
        // to construct the dir from the weights path instead.
        if dir.join("sae_weights.safetensors").exists() || dir.join("model.safetensors").exists() {
            Self::from_local(dir, device)
        } else {
            // Files might be in different cache locations; load manually.
            let weights_dir = weights_path.parent().ok_or_else(|| {
                MIError::Config("cannot determine SAE directory from weights path".into())
            })?;
            // Copy cfg.json to weights dir if needed.
            let target_cfg = weights_dir.join("cfg.json");
            if !target_cfg.exists() {
                std::fs::copy(&cfg_path, &target_cfg)?;
            }
            Self::from_local(weights_dir, device)
        }
    }

    // --- Accessors ---

    /// Access the SAE configuration.
    #[must_use]
    pub const fn config(&self) -> &SaeConfig {
        &self.config
    }

    /// The hook point this SAE targets.
    #[must_use]
    pub const fn hook_point(&self) -> &HookPoint {
        &self.config.hook_point
    }

    /// Dictionary size (number of features).
    #[must_use]
    pub const fn d_sae(&self) -> usize {
        self.config.d_sae
    }

    /// Input dimension.
    #[must_use]
    pub const fn d_in(&self) -> usize {
        self.config.d_in
    }

    // --- Encoding ---

    /// Encode activations into SAE feature space (dense output).
    ///
    /// Applies the full encoder: `pre_acts = x @ W_enc + b_enc`, then the
    /// architecture-specific activation function (`ReLU`, `JumpReLU`, or `TopK`).
    ///
    /// Uses [`TopKStrategy::Auto`] for `TopK` SAEs.
    ///
    /// # Shapes
    /// - `x`: `[..., d_in]` — activations with any leading dimensions
    /// - returns: `[..., d_sae]` — encoded features (mostly sparse)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the last dimension of `x` != `d_in`.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        self.encode_with_strategy(x, &TopKStrategy::Auto)
    }

    /// Encode activations with an explicit [`TopKStrategy`].
    ///
    /// Same as [`encode()`](Self::encode) but allows overriding the `TopK`
    /// computation strategy.
    ///
    /// # Shapes
    /// - `x`: `[..., d_in]` — activations with any leading dimensions
    /// - returns: `[..., d_sae]` — encoded features (mostly sparse)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the last dimension of `x` != `d_in`.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode_with_strategy(&self, x: &Tensor, strategy: &TopKStrategy) -> Result<Tensor> {
        let dims = x.dims();
        let last_dim = *dims
            .last()
            .ok_or_else(|| MIError::Config("cannot encode empty tensor".into()))?;
        if last_dim != self.config.d_in {
            return Err(MIError::Config(format!(
                "input last dim {last_dim} != SAE d_in {}",
                self.config.d_in
            )));
        }

        // PROMOTE: F32 for matmul precision
        let x_f32 = x.to_dtype(DType::F32)?;

        // Optionally subtract b_dec from input (centering).
        let x_centered = if self.config.apply_b_dec_to_input {
            let b_dec = broadcast_bias(&self.b_dec, x_f32.dims())?;
            (&x_f32 - &b_dec)?
        } else {
            x_f32
        };

        // pre_acts = x @ W_enc + b_enc
        // [... , d_in] @ [d_in, d_sae] → [..., d_sae]
        let pre_acts = x_centered.broadcast_matmul(&self.w_enc)?;
        // Broadcast b_enc [d_sae] to match pre_acts leading dims.
        let b_enc = broadcast_bias(&self.b_enc, pre_acts.dims())?;
        let pre_acts = (&pre_acts + &b_enc)?;

        // Apply activation function.
        match &self.config.architecture {
            SaeArchitecture::ReLU => Ok(pre_acts.relu()?),
            SaeArchitecture::JumpReLU => {
                let threshold = self
                    .threshold
                    .as_ref()
                    .ok_or_else(|| MIError::Config("JumpReLU requires threshold tensor".into()))?;
                // Broadcast threshold to match pre_acts leading dims.
                let threshold = broadcast_bias(threshold, pre_acts.dims())?;
                // mask = (pre_acts > threshold), features = pre_acts * mask
                let mask = pre_acts.gt(&threshold)?;
                let mask_f32 = mask.to_dtype(DType::F32)?;
                Ok((&pre_acts * &mask_f32)?)
            }
            SaeArchitecture::TopK { k } => topk_activation(&pre_acts, *k, strategy),
        }
    }

    /// Encode a single activation vector into sparse SAE features.
    ///
    /// Returns only non-zero features, sorted by magnitude descending.
    ///
    /// # Shapes
    /// - `x`: `[d_in]` — single activation vector
    /// - returns: [`SparseActivations<SaeFeatureId>`] with `(SaeFeatureId, f32)` pairs
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` has wrong dimension.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode_sparse(&self, x: &Tensor) -> Result<SparseActivations<SaeFeatureId>> {
        let encoded = self.encode(&x.unsqueeze(0)?)?;
        let encoded_1d = encoded.squeeze(0)?;

        // Transfer to CPU for sparse extraction.
        let values: Vec<f32> = encoded_1d.to_vec1()?;

        let mut features: Vec<(SaeFeatureId, f32)> = values
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v > 0.0)
            .map(|(i, v)| (SaeFeatureId { index: i }, *v))
            .collect();

        // Sort by activation magnitude (descending).
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(SparseActivations { features })
    }

    // --- Decoding ---

    /// Decode SAE features back to activation space.
    ///
    /// # Shapes
    /// - `features`: `[..., d_sae]` — encoded feature activations
    /// - returns: `[..., d_in]` — reconstructed activations
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn decode(&self, features: &Tensor) -> Result<Tensor> {
        // x_hat = features @ W_dec + b_dec
        // [..., d_sae] @ [d_sae, d_in] → [..., d_in]
        let features_f32 = features.to_dtype(DType::F32)?;
        let decoded = features_f32.broadcast_matmul(&self.w_dec)?;
        let b_dec = broadcast_bias(&self.b_dec, decoded.dims())?;
        Ok((&decoded + &b_dec)?)
    }

    // --- Reconstruction ---

    /// Reconstruct activations through the SAE (encode then decode).
    ///
    /// # Shapes
    /// - `x`: `[..., d_in]` — original activations
    /// - returns: `[..., d_in]` — reconstructed activations
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the last dimension of `x` != `d_in`.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn reconstruct(&self, x: &Tensor) -> Result<Tensor> {
        let encoded = self.encode(x)?;
        self.decode(&encoded)
    }

    /// Compute reconstruction MSE loss.
    ///
    /// # Shapes
    /// - `x`: `[..., d_in]` — original activations
    /// - returns: scalar `f64` mean squared error
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the last dimension of `x` != `d_in`.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn reconstruction_error(&self, x: &Tensor) -> Result<f64> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let x_hat = self.reconstruct(&x_f32)?;
        let diff = (&x_f32 - &x_hat)?;
        let mse: f32 = diff.sqr()?.mean_all()?.to_scalar()?;
        Ok(f64::from(mse))
    }

    // --- Steering ---

    /// Extract a single feature's decoder vector (steering direction).
    ///
    /// # Shapes
    /// - returns: `[d_in]` — decoder vector on the SAE's device
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `feature_idx` >= `d_sae`.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn decoder_vector(&self, feature_idx: usize) -> Result<Tensor> {
        if feature_idx >= self.config.d_sae {
            return Err(MIError::Config(format!(
                "feature index {feature_idx} out of range (d_sae={})",
                self.config.d_sae
            )));
        }
        // W_dec: [d_sae, d_in] → row feature_idx → [d_in]
        Ok(self.w_dec.get(feature_idx)?)
    }

    /// Build a [`HookSpec`] that injects SAE decoder vectors into the model.
    ///
    /// Creates an [`Intervention::Add`] at this SAE's hook point with the
    /// accumulated (scaled) decoder vectors placed at the given position.
    ///
    /// # Shapes
    /// - Internally constructs `[1, seq_len, d_in]` with the vector at `position`.
    ///
    /// # Arguments
    /// * `features` — List of `(feature_index, strength)` pairs
    /// * `position` — Token position in the sequence to inject at
    /// * `seq_len` — Total sequence length
    /// * `device` — Device to construct injection tensors on
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any `feature_index` >= `d_sae`.
    /// Returns [`MIError::Model`] on tensor construction failure.
    pub fn prepare_hook_injection(
        &self,
        features: &[(usize, f32)],
        position: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<HookSpec> {
        let d_in = self.config.d_in;

        // Accumulate weighted decoder vectors.
        let mut accumulated = Tensor::zeros(d_in, DType::F32, device)?;
        for &(feature_idx, strength) in features {
            let dec_vec = self.decoder_vector(feature_idx)?;
            let dec_vec = dec_vec.to_device(device)?;
            let scaled = (&dec_vec * f64::from(strength))?;
            accumulated = (&accumulated + &scaled)?;
        }

        // [1, seq_len, d_in] carrying the accumulated vector at `position`.
        // The shared helper also bounds-checks `position`, which the former
        // inline narrow/cat did not.
        let injection = crate::util::inject::position_delta(&accumulated, position, seq_len)?;

        let mut hooks = HookSpec::new();
        hooks.intervene(self.config.hook_point.clone(), Intervention::Add(injection));
        Ok(hooks)
    }
}

// ---------------------------------------------------------------------------
// TopK activation
// ---------------------------------------------------------------------------

/// Apply top-k activation: keep only the k largest values, zero the rest.
///
/// # Shapes
/// - `pre_acts`: `[..., d_sae]` — pre-activation values
/// - returns: same shape, with all but top-k values zeroed per last-dim slice
///
/// # Strategy
/// - [`TopKStrategy::Auto`]: CPU → direct iteration, GPU → sort-based
/// - [`TopKStrategy::Cpu`]: force CPU path
/// - [`TopKStrategy::Gpu`]: force GPU sort-based path
fn topk_activation(pre_acts: &Tensor, k: usize, strategy: &TopKStrategy) -> Result<Tensor> {
    let use_cpu = match strategy {
        TopKStrategy::Cpu => true,
        TopKStrategy::Gpu => false,
        TopKStrategy::Auto => matches!(pre_acts.device(), Device::Cpu),
    };

    if use_cpu {
        topk_cpu(pre_acts, k)
    } else {
        topk_gpu(pre_acts, k)
    }
}

/// `TopK` via CPU-side partial sort.
fn topk_cpu(pre_acts: &Tensor, k: usize) -> Result<Tensor> {
    let device = pre_acts.device().clone();
    let shape = pre_acts.dims().to_vec();
    let d_sae = *shape
        .last()
        .ok_or_else(|| MIError::Config("cannot apply TopK to empty tensor".into()))?;

    // Flatten to 2D: [n, d_sae]
    let n: usize = shape.iter().take(shape.len() - 1).product();
    let flat = pre_acts.reshape((n, d_sae))?.to_dtype(DType::F32)?;
    let flat_cpu = flat.to_device(&Device::Cpu)?;

    let mut result_data: Vec<f32> = Vec::with_capacity(n * d_sae);

    for row_idx in 0..n {
        let row = flat_cpu.get(row_idx)?;
        let mut row_vec: Vec<f32> = row.to_vec1()?;

        // Find the k-th largest value via partial sort.
        let k_clamped = k.min(d_sae);
        if k_clamped > 0 && k_clamped < d_sae {
            // Partial sort: put the k largest elements at the front.
            let mut indices: Vec<usize> = (0..d_sae).collect();
            #[allow(clippy::indexing_slicing)]
            // CONTIGUOUS: indices and row_vec are both exactly d_sae elements
            indices.select_nth_unstable_by(k_clamped - 1, |&a, &b| {
                let va = row_vec.get(b).copied().unwrap_or(f32::NEG_INFINITY);
                let vb = row_vec.get(a).copied().unwrap_or(f32::NEG_INFINITY);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });
            let threshold_idx = indices.get(k_clamped - 1).copied().unwrap_or(0);
            let threshold = row_vec.get(threshold_idx).copied().unwrap_or(0.0);

            // Zero values below threshold.
            for v in &mut row_vec {
                if *v < threshold {
                    *v = 0.0;
                }
            }

            // If there are ties at threshold, we might keep more than k.
            // Count how many are >= threshold and zero extras from the end.
            let active: usize = row_vec.iter().filter(|&&v| v >= threshold).count();
            if active > k_clamped {
                let mut excess = active - k_clamped;
                for v in row_vec.iter_mut().rev() {
                    if excess == 0 {
                        break;
                    }
                    if (*v - threshold).abs() < f32::EPSILON {
                        *v = 0.0;
                        excess -= 1;
                    }
                }
            }
        } else if k_clamped == 0 {
            row_vec.fill(0.0);
        }
        // k_clamped >= d_sae: keep all values

        result_data.extend_from_slice(&row_vec);
    }

    let result = Tensor::from_vec(result_data, (n, d_sae), &device)?;
    result.reshape(shape.as_slice()).map_err(Into::into)
}

/// `TopK` via GPU sort-based masking.
fn topk_gpu(pre_acts: &Tensor, k: usize) -> Result<Tensor> {
    let shape = pre_acts.dims().to_vec();
    let d_sae = *shape
        .last()
        .ok_or_else(|| MIError::Config("cannot apply TopK to empty tensor".into()))?;

    let k_clamped = k.min(d_sae);
    if k_clamped == 0 {
        return Ok(pre_acts.zeros_like()?);
    }
    if k_clamped >= d_sae {
        return Ok(pre_acts.clone());
    }

    // Flatten to 2D for sort_last_dim.
    let n: usize = shape.iter().take(shape.len() - 1).product();
    let flat = pre_acts.reshape((n, d_sae))?.to_dtype(DType::F32)?;

    // Sort descending along last dim.
    let (sorted_vals, _sorted_indices) = flat.sort_last_dim(false)?;

    // Get the k-th largest value per row: [n, 1]
    let kth_vals = sorted_vals.narrow(1, k_clamped - 1, 1)?;

    // Mask: keep values >= kth value.
    let mask = flat.ge(&kth_vals)?;
    let mask_f32 = mask.to_dtype(DType::F32)?;

    let result = (&flat * &mask_f32)?;
    result.reshape(shape.as_slice()).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Broadcast a 1D bias `[dim]` to match an arbitrary target shape.
///
/// For example, bias `[d_sae]` with target shape `[batch, seq, d_sae]`
/// is reshaped to `[1, 1, d_sae]` then broadcast to `[batch, seq, d_sae]`.
///
/// # Shapes
/// - `bias`: `[dim]` — 1D bias vector
/// - `target_shape`: the shape to broadcast into (last dim must match)
/// - returns: bias with same shape as `target_shape`
fn broadcast_bias(bias: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let ndim = target_shape.len();
    if ndim <= 1 {
        return Ok(bias.clone());
    }
    // Reshape [dim] → [1, 1, ..., dim] with (ndim - 1) leading 1s.
    let mut shape = vec![1_usize; ndim];
    let last_dim = *target_shape
        .last()
        .ok_or_else(|| MIError::Config("cannot broadcast bias to empty shape".into()))?;
    if let Some(slot) = shape.last_mut() {
        *slot = last_dim;
    }
    let reshaped = bias.reshape(shape.as_slice())?;
    Ok(reshaped.broadcast_as(target_shape)?)
}

/// Load a named tensor from safetensors.
fn load_tensor(st: &SafeTensors<'_>, name: &str, device: &Device) -> Result<Tensor> {
    let view = st
        .tensor(name)
        .map_err(|e| MIError::Config(format!("tensor '{name}' not found: {e}")))?;
    crate::util::safetensors_view::tensor_from_view(&view, device, "SAE")
}

/// Validate that a tensor has the expected shape.
fn validate_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.dims() != expected {
        return Err(MIError::Config(format!(
            "SAE tensor '{name}' shape mismatch: expected {expected:?}, got {:?}",
            tensor.dims()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sae_feature_id_display() {
        let fid = SaeFeatureId { index: 42 };
        assert_eq!(fid.to_string(), "SAE:42");
    }

    #[test]
    fn resolve_architecture_relu_default() {
        let arch = resolve_architecture(None, None, None).unwrap();
        assert_eq!(arch, SaeArchitecture::ReLU);
    }

    #[test]
    fn resolve_architecture_relu_explicit() {
        let arch = resolve_architecture(Some("standard"), Some("relu"), None).unwrap();
        assert_eq!(arch, SaeArchitecture::ReLU);
    }

    #[test]
    fn resolve_architecture_jumprelu() {
        let arch = resolve_architecture(Some("jumprelu"), None, None).unwrap();
        assert_eq!(arch, SaeArchitecture::JumpReLU);
    }

    #[test]
    fn resolve_architecture_jumprelu_from_activation() {
        let arch = resolve_architecture(None, Some("jumprelu"), None).unwrap();
        assert_eq!(arch, SaeArchitecture::JumpReLU);
    }

    #[test]
    fn resolve_architecture_topk() {
        let kwargs = serde_json::json!({"k": 32});
        let arch = resolve_architecture(Some("topk"), None, Some(&kwargs)).unwrap();
        assert_eq!(arch, SaeArchitecture::TopK { k: 32 });
    }

    #[test]
    fn resolve_architecture_topk_from_activation() {
        let kwargs = serde_json::json!({"k": 64});
        let arch = resolve_architecture(None, Some("topk"), Some(&kwargs)).unwrap();
        assert_eq!(arch, SaeArchitecture::TopK { k: 64 });
    }

    #[test]
    fn resolve_architecture_topk_missing_k() {
        let result = resolve_architecture(Some("topk"), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_architecture_unknown() {
        let result = resolve_architecture(Some("gated"), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn parse_config_minimal() {
        let json = r#"{
            "d_in": 2304,
            "d_sae": 16384,
            "hook_name": "blocks.5.hook_resid_post"
        }"#;
        let raw: RawSaeConfig = serde_json::from_str(json).unwrap();
        let config = parse_sae_config(raw).unwrap();
        assert_eq!(config.d_in, 2304);
        assert_eq!(config.d_sae, 16384);
        assert_eq!(config.architecture, SaeArchitecture::ReLU);
        assert_eq!(config.hook_point, HookPoint::ResidPost(5));
        assert!(!config.apply_b_dec_to_input);
    }

    #[test]
    fn parse_config_jumprelu() {
        let json = r#"{
            "d_in": 2304,
            "d_sae": 16384,
            "architecture": "jumprelu",
            "hook_name": "blocks.20.hook_resid_post",
            "apply_b_dec_to_input": true,
            "normalize_activations": "expected_average_only_in"
        }"#;
        let raw: RawSaeConfig = serde_json::from_str(json).unwrap();
        let config = parse_sae_config(raw).unwrap();
        assert_eq!(config.architecture, SaeArchitecture::JumpReLU);
        assert_eq!(config.hook_point, HookPoint::ResidPost(20));
        assert!(config.apply_b_dec_to_input);
        assert_eq!(
            config.normalize_activations,
            NormalizeActivations::ExpectedAverageOnlyIn
        );
    }

    #[test]
    fn parse_config_topk() {
        let json = r#"{
            "d_in": 2304,
            "d_sae": 65536,
            "activation_fn_str": "topk",
            "activation_fn_kwargs": {"k": 32},
            "hook_name": "blocks.10.hook_resid_post"
        }"#;
        let raw: RawSaeConfig = serde_json::from_str(json).unwrap();
        let config = parse_sae_config(raw).unwrap();
        assert_eq!(config.architecture, SaeArchitecture::TopK { k: 32 });
    }

    #[test]
    fn topk_cpu_basic() {
        let data = Tensor::new(&[[5.0_f32, 3.0, 1.0, 4.0, 2.0]], &Device::Cpu).unwrap();
        let result = topk_cpu(&data, 2).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![5.0, 0.0, 0.0, 4.0, 0.0]);
    }

    #[test]
    fn topk_cpu_all_kept() {
        let data = Tensor::new(&[[1.0_f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let result = topk_cpu(&data, 5).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn topk_cpu_none_kept() {
        let data = Tensor::new(&[[1.0_f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let result = topk_cpu(&data, 0).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn topk_cpu_batched() {
        let data = Tensor::new(
            &[[5.0_f32, 3.0, 1.0, 4.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
            &Device::Cpu,
        )
        .unwrap();
        let result = topk_cpu(&data, 3).unwrap();
        let vals: Vec<Vec<f32>> = result.to_vec2().unwrap();
        assert_eq!(vals[0], vec![5.0, 3.0, 0.0, 4.0, 0.0]);
        assert_eq!(vals[1], vec![0.0, 0.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn sparse_activations_sae() {
        let features = vec![
            (SaeFeatureId { index: 5 }, 3.0),
            (SaeFeatureId { index: 2 }, 2.0),
            (SaeFeatureId { index: 8 }, 1.0),
        ];
        let sparse = SparseActivations { features };
        assert_eq!(sparse.len(), 3);
        assert!(!sparse.is_empty());
    }

    #[test]
    fn sparse_activations_truncate_sae() {
        let features = vec![
            (SaeFeatureId { index: 5 }, 3.0),
            (SaeFeatureId { index: 2 }, 2.0),
            (SaeFeatureId { index: 8 }, 1.0),
        ];
        let mut sparse = SparseActivations { features };
        sparse.truncate(2);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.features[0].0.index, 5);
        assert_eq!(sparse.features[1].0.index, 2);
    }

    #[test]
    fn encode_decode_roundtrip_shapes() {
        // Create a tiny SAE for shape testing.
        let d_in = 4;
        let d_sae = 8;
        let device = Device::Cpu;

        let w_enc = Tensor::randn(0.0_f32, 1.0, (d_in, d_sae), &device).unwrap();
        let w_dec = Tensor::randn(0.0_f32, 1.0, (d_sae, d_in), &device).unwrap();
        let b_enc = Tensor::zeros(d_sae, DType::F32, &device).unwrap();
        let b_dec = Tensor::zeros(d_in, DType::F32, &device).unwrap();

        let sae = SparseAutoencoder {
            config: SaeConfig {
                d_in,
                d_sae,
                architecture: SaeArchitecture::ReLU,
                hook_name: "blocks.0.hook_resid_post".into(),
                hook_point: HookPoint::ResidPost(0),
                apply_b_dec_to_input: false,
                normalize_activations: NormalizeActivations::None,
            },
            w_enc,
            w_dec,
            b_enc,
            b_dec,
            threshold: None,
        };

        // Test 1D input.
        let x1 = Tensor::randn(0.0_f32, 1.0, (d_in,), &device).unwrap();
        let encoded = sae.encode(&x1.unsqueeze(0).unwrap()).unwrap();
        assert_eq!(encoded.dims(), &[1, d_sae]);

        // Test 2D input.
        let x2 = Tensor::randn(0.0_f32, 1.0, (3, d_in), &device).unwrap();
        let encoded = sae.encode(&x2).unwrap();
        assert_eq!(encoded.dims(), &[3, d_sae]);
        let decoded = sae.decode(&encoded).unwrap();
        assert_eq!(decoded.dims(), &[3, d_in]);

        // Test 3D input.
        let x3 = Tensor::randn(0.0_f32, 1.0, (2, 5, d_in), &device).unwrap();
        let encoded = sae.encode(&x3).unwrap();
        assert_eq!(encoded.dims(), &[2, 5, d_sae]);
        let decoded = sae.decode(&encoded).unwrap();
        assert_eq!(decoded.dims(), &[2, 5, d_in]);

        // Test reconstruction.
        let x_hat = sae.reconstruct(&x2).unwrap();
        assert_eq!(x_hat.dims(), &[3, d_in]);

        // Test reconstruction error.
        let mse = sae.reconstruction_error(&x2).unwrap();
        assert!(mse >= 0.0);
    }

    #[test]
    fn encode_sparse_basic() {
        let d_in = 4;
        let d_sae = 8;
        let device = Device::Cpu;

        // Use identity-like encoder to get predictable output.
        let mut w_enc_data = vec![0.0_f32; d_in * d_sae];
        // Map input dim 0 → feature 0, dim 1 → feature 1, etc.
        for i in 0..d_in {
            w_enc_data[i * d_sae + i] = 1.0;
        }
        let w_enc = Tensor::from_vec(w_enc_data, (d_in, d_sae), &device).unwrap();
        let w_dec = Tensor::randn(0.0_f32, 1.0, (d_sae, d_in), &device).unwrap();
        let b_enc = Tensor::zeros(d_sae, DType::F32, &device).unwrap();
        let b_dec = Tensor::zeros(d_in, DType::F32, &device).unwrap();

        let sae = SparseAutoencoder {
            config: SaeConfig {
                d_in,
                d_sae,
                architecture: SaeArchitecture::ReLU,
                hook_name: "blocks.0.hook_resid_post".into(),
                hook_point: HookPoint::ResidPost(0),
                apply_b_dec_to_input: false,
                normalize_activations: NormalizeActivations::None,
            },
            w_enc,
            w_dec,
            b_enc,
            b_dec,
            threshold: None,
        };

        let x = Tensor::new(&[2.0_f32, -1.0, 3.0, 0.5], &device).unwrap();
        let sparse = sae.encode_sparse(&x).unwrap();

        // Only positive values should appear: 2.0, 3.0, 0.5
        assert_eq!(sparse.len(), 3);
        // Should be sorted descending.
        assert_eq!(sparse.features[0].0.index, 2); // 3.0
        assert_eq!(sparse.features[1].0.index, 0); // 2.0
        assert_eq!(sparse.features[2].0.index, 3); // 0.5
    }

    #[test]
    fn decoder_vector_basic() {
        let d_in = 4;
        let d_sae = 8;
        let device = Device::Cpu;

        let w_dec = Tensor::randn(0.0_f32, 1.0, (d_sae, d_in), &device).unwrap();
        let sae = SparseAutoencoder {
            config: SaeConfig {
                d_in,
                d_sae,
                architecture: SaeArchitecture::ReLU,
                hook_name: "blocks.0.hook_resid_post".into(),
                hook_point: HookPoint::ResidPost(0),
                apply_b_dec_to_input: false,
                normalize_activations: NormalizeActivations::None,
            },
            w_enc: Tensor::zeros((d_in, d_sae), DType::F32, &device).unwrap(),
            w_dec: w_dec.clone(),
            b_enc: Tensor::zeros(d_sae, DType::F32, &device).unwrap(),
            b_dec: Tensor::zeros(d_in, DType::F32, &device).unwrap(),
            threshold: None,
        };

        let vec0 = sae.decoder_vector(0).unwrap();
        assert_eq!(vec0.dims(), &[d_in]);

        // Out of range should error.
        assert!(sae.decoder_vector(d_sae).is_err());
    }

    #[test]
    fn prepare_injection_basic() {
        let d_in = 4;
        let d_sae = 8;
        let device = Device::Cpu;

        let sae = SparseAutoencoder {
            config: SaeConfig {
                d_in,
                d_sae,
                architecture: SaeArchitecture::ReLU,
                hook_name: "blocks.0.hook_resid_post".into(),
                hook_point: HookPoint::ResidPost(0),
                apply_b_dec_to_input: false,
                normalize_activations: NormalizeActivations::None,
            },
            w_enc: Tensor::zeros((d_in, d_sae), DType::F32, &device).unwrap(),
            w_dec: Tensor::ones((d_sae, d_in), DType::F32, &device).unwrap(),
            b_enc: Tensor::zeros(d_sae, DType::F32, &device).unwrap(),
            b_dec: Tensor::zeros(d_in, DType::F32, &device).unwrap(),
            threshold: None,
        };

        let features = vec![(0_usize, 1.0_f32), (1, 0.5)];
        let hooks = sae
            .prepare_hook_injection(&features, 2, 5, &device)
            .unwrap();
        assert!(!hooks.is_empty());
    }
}
