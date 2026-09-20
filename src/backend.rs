// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core backend trait and model wrapper.
//!
//! [`MIBackend`] is the trait that every model backend implements.
//! [`MIModel`] wraps a backend with device metadata and convenience methods.

use candle_core::{DType, Device, Tensor};

use crate::error::{MIError, Result};
use crate::hooks::{HookCache, HookSpec};
use crate::tokenizer::MITokenizer;

// ---------------------------------------------------------------------------
// MIBackend trait
// ---------------------------------------------------------------------------

/// Unified interface for model backends with hook-aware forward passes.
///
/// Implementing this trait is the only requirement for adding a new model
/// to candle-mi.  The single [`forward`](Self::forward) method replaces
/// plip-rs's (frozen predecessor project, v1.4.0) proliferation of `forward_with_*` variants: the caller
/// specifies captures and interventions via [`HookSpec`], and the backend
/// returns a [`HookCache`] containing the output plus any requested
/// activations.
///
/// Optional capabilities (chat template, embedding access) have default
/// implementations that return `None` or an error.
pub trait MIBackend: Send + Sync {
    // --- Metadata --------------------------------------------------------

    /// Number of layers (transformer blocks or RWKV blocks).
    fn num_layers(&self) -> usize;

    /// Hidden dimension (`d_model`).
    fn hidden_size(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Number of attention heads (or RWKV heads).
    fn num_heads(&self) -> usize;

    // --- Core forward pass -----------------------------------------------

    /// Unified forward pass with optional hook capture and interventions.
    ///
    /// When `hooks` is empty, this must be equivalent to a plain forward
    /// pass with **zero extra allocations** (see `design/hook-overhead.md`).
    ///
    /// The returned [`HookCache`] always contains the output tensor
    /// (logits or hidden states, depending on the backend) and any
    /// activations requested via [`HookSpec::capture`].
    ///
    /// # Shapes
    /// - `input_ids`: `[batch, seq]` -- token IDs
    /// - returns: [`HookCache`] containing `logits` at `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on tensor operation failures and
    /// [`MIError::Intervention`] if an intervention is invalid for
    /// the current model dimensions.
    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>;

    // --- Logit projection ------------------------------------------------

    /// Project a hidden-state tensor to vocabulary logits.
    ///
    /// Applies the model's final layer norm before the unembedding projection,
    /// matching the standard logit lens technique (nostalgebraist, 2020).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, hidden_size]` or `[batch, seq, hidden_size]` -- hidden
    ///   states (pre-norm)
    /// - returns: the same leading dimensions with `hidden_size` replaced by
    ///   `vocab_size`, i.e. `[batch, vocab_size]` or `[batch, seq, vocab_size]`
    ///
    /// Implementations **must preserve rank**.  Both forms are supported: a logit
    /// lens reads a single position (`[batch, hidden_size]`), while a probe that
    /// has captured a whole residual stream projects every position at once
    /// (`[batch, seq, hidden_size]`).  Project against a rank-2 unembedding weight
    /// with `broadcast_matmul` rather than `matmul`, since candle's `matmul`
    /// requires both operands to have equal rank.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on shape mismatch or tensor operation failure.
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;

    // --- Optional capabilities -------------------------------------------

    /// Format a prompt with the model's chat template, if any.
    ///
    /// Returns `None` for base (non-instruct) models.
    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        None
    }

    /// Return the raw embedding vector for a single token.
    ///
    /// For models with tied embeddings this is also the unembedding direction.
    ///
    /// # Shapes
    /// - returns: `[hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the backend does not support this.
    fn embedding_vector(&self, _token_id: u32) -> Result<Tensor> {
        Err(MIError::Hook(
            "embedding_vector not supported for this backend".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// MIModel
// ---------------------------------------------------------------------------

/// High-level model wrapper combining a backend with device metadata.
///
/// `MIModel` delegates to the wrapped [`MIBackend`] and adds convenience
/// methods including `from_pretrained` (requires one of the `transformer`,
/// `rwkv` or `diffusion` features) for one-line model loading from
/// `HuggingFace`.
pub struct MIModel {
    /// The underlying model backend.
    // TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch
    backend: Box<dyn MIBackend>,
    /// The device this model lives on.
    device: Device,
    /// Tokenizer loaded alongside the model (present when loaded via `from_pretrained`).
    tokenizer: Option<MITokenizer>,
}

/// Shared parts resolved for a random-model control: `(config, tokenizer,
/// cached repo files, device)`. Returned by `resolve_transformer_control`.
#[cfg(feature = "transformer")]
type TransformerControlParts = (
    crate::config::TransformerConfig,
    Option<MITokenizer>,
    std::collections::HashMap<String, std::path::PathBuf>,
    Device,
);

impl MIModel {
    /// Load a model from a `HuggingFace` model ID or local path.
    ///
    /// Checks local `HuggingFace` cache first, then downloads if necessary.
    /// Automatically selects the appropriate backend based on `model_type`
    /// in the model's `config.json`.
    ///
    /// # `DType` selection
    ///
    /// Always uses `F32` for research-grade precision — numerically identical
    /// to Python/PyTorch F32 on both CPU and CUDA.  Models up to ~7B fit in
    /// 16 GB VRAM at F32.  For larger models or when speed matters more than
    /// precision, use the backend-specific `load()` API with `DType::BF16`.
    ///
    /// # Quantized checkpoints
    ///
    /// When the model's `config.json` carries a `quantization_config` block and
    /// candle-mi is built with the `quantized` feature, the weights are
    /// transparently dequantized to `BF16` in memory (bitsandbytes `NF4`/`FP4`/
    /// `INT8`, `AWQ`, `GPTQ`, auto-detected via `anamnesis`) before the forward
    /// pass — no separate API call is needed.  Without the `quantized` feature,
    /// such a checkpoint returns a clear [`MIError::Config`] telling you to enable
    /// it.  Single-file safetensors only for now (sharded quantized checkpoints
    /// are not yet supported).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the model type is unsupported, or
    /// [`MIError::Model`] if weight loading fails.
    #[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
    pub fn from_pretrained(model_id: &str) -> Result<Self> {
        // --- Device and dtype ---
        let device = Self::select_device()?;
        // F32 everywhere: research-grade precision, matching Python/PyTorch.
        let dtype = DType::F32;

        // --- Download / resolve local files ---
        // hf-fetch-model 0.9.x requires explicit opt-in to HF_TOKEN; go through
        // the shared builder so gated models (Llama/Mistral/Gemma/Qwen) work.
        let fetch_config = crate::download::fetch_config_builder()
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;
        let files =
            hf_fetch_model::download_files_with_config_blocking(model_id.to_owned(), &fetch_config)
                .map(hf_fetch_model::DownloadOutcome::into_inner)
                .map_err(|e| MIError::Download(e.to_string()))?;

        let config_path = files
            .get("config.json")
            .ok_or_else(|| MIError::Config("config.json not found in downloaded files".into()))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| MIError::Config(format!("read config.json: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| MIError::Config(format!("parse config.json: {e}")))?;

        // --- Dispatch on model_type ---
        let model_type = json
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| MIError::Config("missing 'model_type' field".into()))?;

        // --- Load tokenizer (best-effort: present for HF models) ---
        let tokenizer = files
            .get("tokenizer.json")
            .and_then(|p| MITokenizer::from_hf_path(p).ok());

        let (weights_paths, weight_format) = resolve_weight_paths(&files)?;
        // A `quantization_config` block means the weights are stored quantized
        // (bitsandbytes / AWQ / GPTQ).  Route those through anamnesis to
        // dequantize to BF16 before building the VarBuilder; standard
        // checkpoints take the direct candle path.
        let vb = if json.get("quantization_config").is_some() {
            load_quantized_var_builder(&weights_paths, dtype, &device)?
        } else {
            create_var_builder(&weights_paths, weight_format, dtype, &device)?
        };

        match model_type {
            #[cfg(feature = "transformer")]
            mt if crate::config::SUPPORTED_MODEL_TYPES.contains(&mt) => {
                use crate::config::TransformerConfig;
                use crate::transformer::GenericTransformer;

                let config = TransformerConfig::from_hf_config(&json)?;
                let transformer = GenericTransformer::load(config, &device, dtype, vb)?;
                Ok(Self::with_tokenizer(
                    Box::new(transformer),
                    device,
                    tokenizer,
                ))
            }
            #[cfg(feature = "rwkv")]
            mt if crate::rwkv::SUPPORTED_RWKV_MODEL_TYPES.contains(&mt) => {
                use crate::rwkv::{GenericRwkv, RwkvConfig};

                let config = RwkvConfig::from_hf_config(&json)?;
                let rwkv = GenericRwkv::load(config, &device, dtype, vb)?;
                Ok(Self::with_tokenizer(Box::new(rwkv), device, tokenizer))
            }
            #[cfg(feature = "diffusion")]
            mt if crate::diffusion::SUPPORTED_DIFFUSION_MODEL_TYPES.contains(&mt) => {
                use crate::diffusion::{GenericMdlm, MdlmConfig};

                let config = MdlmConfig::from_hf_config(&json)?;
                let mdlm = GenericMdlm::load(config, &device, dtype, vb)?;
                Ok(Self::with_tokenizer(Box::new(mdlm), device, tokenizer))
            }
            #[cfg(feature = "transformer")]
            _unknown => {
                use crate::config::TransformerConfig;
                use crate::transformer::GenericTransformer;

                // Extract tensor names for auto-config inference
                let tensor_names = extract_tensor_names(&files)?;

                // Preflight: check compatibility before attempting to load
                TransformerConfig::check_auto_compatibility(&json, &tensor_names).into_result()?;

                let config = TransformerConfig::from_hf_config_auto(&json, &tensor_names)?;
                let transformer = GenericTransformer::load(config, &device, dtype, vb)?;
                Ok(Self::with_tokenizer(
                    Box::new(transformer),
                    device,
                    tokenizer,
                ))
            }
            #[cfg(not(feature = "transformer"))]
            other => Err(MIError::Config(format!(
                "unsupported model_type: '{other}' (enable the `transformer` feature for auto-config)"
            ))),
        }
    }

    /// Load `model_id`'s architecture and tokenizer but fill every weight with
    /// seeded Gaussian noise — a "dead-salmon" random-model control (the
    /// interpretability-illusion baseline: an analysis pipeline must not
    /// manufacture structure on a randomly initialized network).
    ///
    /// The config and tokenizer are read from the (cached) repo and the
    /// checkpoint's **tensor names are read from the safetensors header** so the
    /// random model is architecturally identical to the real one (e.g. tied
    /// embeddings stay tied). **No weight values are read** — every tensor the
    /// loader requests is generated as `N(0, std)` from a `seed`-seeded RNG, in
    /// the deterministic order the loader requests them, so the same
    /// `(seed, std)` reproduces the same random model. `std = 0.02` matches the
    /// usual transformer init scale. Only transformer model types are supported.
    ///
    /// See [`from_pretrained_shuffled`](Self::from_pretrained_shuffled) for the
    /// stricter norm-preserving variant.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the repo cannot be resolved,
    /// [`MIError::Config`] if the config is missing or is not a supported
    /// transformer type, and [`MIError::Model`] if model construction fails.
    #[cfg(feature = "transformer")]
    pub fn from_pretrained_random_init(model_id: &str, seed: u64, std: f64) -> Result<Self> {
        use crate::transformer::GenericTransformer;

        let dtype = DType::F32;
        let (config, tokenizer, files, device) = Self::resolve_transformer_control(model_id)?;

        // Real tensor name set (header-only read) so the random model keeps the
        // real architecture — notably tied embeddings via `contains_tensor`.
        let names: std::collections::HashSet<String> =
            extract_tensor_names(&files)?.into_iter().collect();

        let backend = RandnBackend::new(seed, std, names);
        let vb = candle_nn::VarBuilder::from_backend(Box::new(backend), dtype, device.clone());
        let transformer = GenericTransformer::load(config, &device, dtype, vb)?;
        Ok(Self::with_tokenizer(
            Box::new(transformer),
            device,
            tokenizer,
        ))
    }

    /// Load `model_id` but **permute the elements of every weight tensor**
    /// (seeded) — a stricter dead-salmon control than fresh random init.
    ///
    /// Each tensor keeps its exact value multiset (and hence its norm and scale
    /// statistics) while its learned structure is destroyed, so this rules out
    /// the objection that the published effect is "just the weight scales" that a
    /// fresh Gaussian init changes. The permutation is independent per tensor and
    /// reproducible from `seed` (paths and names are visited in sorted order).
    /// Only transformer model types with `safetensors` weights are supported.
    ///
    /// # Memory
    ///
    /// Loads the checkpoint into CPU RAM one `safetensors` shard at a time,
    /// up-casting each to `F32`, and accumulates the shuffled weights as an
    /// `F32` CPU map before the model is built on `device`. Peak: roughly the
    /// `F32` model size on CPU (~10 GB for `Gemma 2 2B`) plus one shard, then
    /// the model again on the device.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the repo cannot be resolved,
    /// [`MIError::Config`] if the config is not a supported transformer type or
    /// the weights are not `safetensors`, and [`MIError::Model`] if a weight
    /// file cannot be loaded or model construction fails.
    #[cfg(feature = "transformer")]
    pub fn from_pretrained_shuffled(model_id: &str, seed: u64) -> Result<Self> {
        use crate::transformer::GenericTransformer;

        let dtype = DType::F32;
        let (config, tokenizer, files, device) = Self::resolve_transformer_control(model_id)?;

        let (weights_paths, format) = resolve_weight_paths(&files)?;
        if format != WeightFormat::Safetensors {
            return Err(MIError::Config(
                "from_pretrained_shuffled requires safetensors weights".into(),
            ));
        }
        let shuffled = shuffle_checkpoint_tensors(&weights_paths, seed, dtype)?;
        let vb = candle_nn::VarBuilder::from_tensors(shuffled, dtype, &device);
        let transformer = GenericTransformer::load(config, &device, dtype, vb)?;
        Ok(Self::with_tokenizer(
            Box::new(transformer),
            device,
            tokenizer,
        ))
    }

    /// Shared resolution for the random-model controls: device, cached repo
    /// files, transformer config, and tokenizer. Errors unless the repo is a
    /// supported transformer type.
    #[cfg(feature = "transformer")]
    fn resolve_transformer_control(model_id: &str) -> Result<TransformerControlParts> {
        use crate::config::TransformerConfig;

        let device = Self::select_device()?;
        let fetch_config = crate::download::fetch_config_builder()
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;
        let files =
            hf_fetch_model::download_files_with_config_blocking(model_id.to_owned(), &fetch_config)
                .map(hf_fetch_model::DownloadOutcome::into_inner)
                .map_err(|e| MIError::Download(e.to_string()))?;

        let config_path = files
            .get("config.json")
            .ok_or_else(|| MIError::Config("config.json not found in downloaded files".into()))?;
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| MIError::Config(format!("read config.json: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| MIError::Config(format!("parse config.json: {e}")))?;

        let model_type = json
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| MIError::Config("missing 'model_type' field".into()))?;
        if !crate::config::SUPPORTED_MODEL_TYPES.contains(&model_type) {
            return Err(MIError::Config(format!(
                "random-model controls support transformer model types only, got '{model_type}'"
            )));
        }

        let tokenizer = files
            .get("tokenizer.json")
            .and_then(|p| MITokenizer::from_hf_path(p).ok());
        let config = TransformerConfig::from_hf_config(&json)?;
        Ok((config, tokenizer, files, device))
    }

    /// Select the best available device (CUDA GPU 0, or CPU fallback).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] on device detection failure.
    #[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
    fn select_device() -> Result<Device> {
        match Device::cuda_if_available(0) {
            Ok(dev) => Ok(dev),
            Err(e) => Err(MIError::Model(e)),
        }
    }

    /// Wrap an existing backend (no tokenizer).
    // TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch
    #[must_use]
    pub fn new(backend: Box<dyn MIBackend>, device: Device) -> Self {
        Self {
            backend,
            device,
            tokenizer: None,
        }
    }

    /// Wrap an existing backend with an optional tokenizer.
    // TRAIT_OBJECT: heterogeneous model backends require dynamic dispatch
    #[must_use]
    pub fn with_tokenizer(
        backend: Box<dyn MIBackend>,
        device: Device,
        tokenizer: Option<MITokenizer>,
    ) -> Self {
        Self {
            backend,
            device,
            tokenizer,
        }
    }

    /// The device this model lives on.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// The tokenizer loaded alongside the model, if available.
    ///
    /// Present when the model was loaded via `from_pretrained` (requires one of
    /// the `transformer`, `rwkv` or `diffusion` features) and a
    /// `tokenizer.json` was found in the downloaded files.
    #[must_use]
    pub const fn tokenizer(&self) -> Option<&MITokenizer> {
        self.tokenizer.as_ref()
    }

    /// Number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.backend.num_layers()
    }

    /// Hidden dimension.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.backend.hidden_size()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.backend.vocab_size()
    }

    /// Number of attention heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.backend.num_heads()
    }

    /// Run a forward pass with the given hook specification.
    ///
    /// # Shapes
    /// - `input_ids`: `[batch, seq]` -- token IDs
    /// - returns: [`HookCache`] containing `logits` at `[batch, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
        self.backend.forward(input_ids, hooks)
    }

    /// Project hidden states to vocabulary logits.
    ///
    /// Applies the model's final layer norm before the unembedding projection,
    /// matching the standard logit lens technique (nostalgebraist, 2020).
    ///
    /// # Shapes
    /// - `hidden`: `[batch, hidden_size]` -- hidden states (pre-norm)
    /// - returns: `[batch, vocab_size]`
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying backend.
    pub fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor> {
        self.backend.project_to_vocab(hidden)
    }

    /// Access the underlying backend (e.g., for backend-specific methods).
    // TRAIT_OBJECT: caller needs dynamic dispatch for backend-specific methods
    #[must_use]
    pub fn backend(&self) -> &dyn MIBackend {
        &*self.backend
    }

    /// Run a forward pass from text, returning both MI outputs and token
    /// position mapping.
    ///
    /// Combines [`MITokenizer::encode_with_offsets`](crate::MITokenizer::encode_with_offsets)
    /// + tensor creation + [`forward`](Self::forward) in a single call.
    ///
    /// The returned [`TextForwardResult`] carries the [`HookCache`]
    /// alongside the [`EncodingWithOffsets`](crate::EncodingWithOffsets),
    /// eliminating the need for separate encoding and manual offset
    /// tracking.
    ///
    /// # Shapes
    /// - input: text string
    /// - returns: [`TextForwardResult`] with logits at `[1, seq, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if no tokenizer is available.
    /// Returns [`MIError::Tokenizer`] on encoding failure.
    /// Propagates errors from the underlying backend.
    pub fn forward_text(&self, text: &str, hooks: &HookSpec) -> Result<TextForwardResult> {
        let tokenizer = self
            .tokenizer()
            .ok_or_else(|| MIError::Config("forward_text requires a tokenizer".into()))?;
        let encoding = tokenizer.encode_with_offsets(text)?;
        let input = Tensor::new(&encoding.ids[..], &self.device)?.unsqueeze(0)?;
        let cache = self.forward(&input, hooks)?;
        Ok(TextForwardResult { cache, encoding })
    }
}

// ---------------------------------------------------------------------------
// TextForwardResult
// ---------------------------------------------------------------------------

/// Result of a text-based forward pass, bundling MI outputs with token
/// position mapping.
///
/// Returned by [`MIModel::forward_text`]. Provides access to the
/// [`HookCache`] (logits + captured activations) alongside the
/// [`EncodingWithOffsets`](crate::EncodingWithOffsets) (token strings,
/// IDs, and byte offset ranges for mapping between source text positions
/// and token indices).
#[derive(Debug)]
pub struct TextForwardResult {
    /// Hook cache containing output logits and any captured activations.
    cache: HookCache,
    /// Token encoding with character offset mapping.
    encoding: crate::util::positioning::EncodingWithOffsets,
}

impl TextForwardResult {
    /// Access the hook cache (logits + captured activations).
    #[must_use]
    pub const fn cache(&self) -> &HookCache {
        &self.cache
    }

    /// Consume the result and return the hook cache.
    #[must_use]
    pub fn into_cache(self) -> HookCache {
        self.cache
    }

    /// Access the token encoding with character offset mapping.
    #[must_use]
    pub const fn encoding(&self) -> &crate::util::positioning::EncodingWithOffsets {
        &self.encoding
    }

    /// The output tensor from the forward pass (typically logits).
    ///
    /// Shortcut for `self.cache().output()`.
    ///
    /// # Shapes
    /// - returns: `[1, seq, vocab_size]`
    #[must_use]
    pub const fn output(&self) -> &Tensor {
        self.cache.output()
    }

    /// Retrieve a captured tensor by hook point, returning an error if
    /// not found.
    ///
    /// Shortcut for `self.cache().require(hook)`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if the hook point was not captured.
    pub fn require(&self, hook: &crate::hooks::HookPoint) -> Result<&Tensor> {
        self.cache.require(hook)
    }

    /// Retrieve a captured tensor by hook point.
    ///
    /// Shortcut for `self.cache().get(hook)`.
    #[must_use]
    pub fn get(&self, hook: &crate::hooks::HookPoint) -> Option<&Tensor> {
        self.cache.get(hook)
    }

    /// The raw BPE token strings (with space-prefix markers like `Ġ`).
    ///
    /// Shortcut for `self.encoding().tokens`.
    #[must_use]
    pub fn tokens(&self) -> &[String] {
        &self.encoding.tokens
    }

    /// Number of tokens in the encoded sequence.
    #[must_use]
    pub const fn seq_len(&self) -> usize {
        self.encoding.len()
    }
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Sample a token from logits using the given temperature.
///
/// When `temperature <= 0.0`, performs greedy (argmax) decoding.
///
/// # Shapes
/// - `logits`: `[vocab_size]` -- logit scores for each vocabulary token
///
/// # Errors
///
/// Returns [`MIError::Model`] if the logits tensor is empty or
/// cannot be converted to `f32`.
pub fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32> {
    if temperature <= 0.0 {
        argmax(logits)
    } else {
        sample_with_temperature(logits, temperature)
    }
}

/// Greedy (argmax) sampling.
fn argmax(logits: &Tensor) -> Result<u32> {
    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    let (max_idx, _) = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| MIError::Model(candle_core::Error::Msg("empty logits".into())))?;

    // CAST: usize → u32, vocab size fits in u32 (max ~250K tokens)
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    Ok(max_idx as u32)
}

/// Temperature-scaled softmax sampling.
fn sample_with_temperature(logits: &Tensor, temperature: f32) -> Result<u32> {
    use rand::Rng;

    let logits_f32 = logits.to_dtype(DType::F32)?;
    let logits_vec: Vec<f32> = logits_f32.flatten_all()?.to_vec1()?;

    if logits_vec.is_empty() {
        return Err(MIError::Model(candle_core::Error::Msg(
            "empty logits".into(),
        )));
    }

    // Scale by temperature.
    let scaled: Vec<f32> = logits_vec.iter().map(|x| x / temperature).collect();

    // Numerically stable softmax.
    let max_val = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|x| x / sum).collect();

    // Sample from the categorical distribution.
    let mut rng = rand::thread_rng();
    let r: f32 = rng.r#gen();
    let mut cumsum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            // CAST: usize → u32, vocab index fits in u32
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            return Ok(idx as u32);
        }
    }

    // Fallback to last token (floating-point rounding edge case).
    // CAST: usize → u32, vocab index fits in u32
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    Ok((probs.len() - 1) as u32)
}

/// Extract the probability of a specific token from a logit tensor.
///
/// Applies softmax to the last sequence position and returns the probability
/// at `token_id`. Useful for measuring steering effectiveness.
///
/// # Shapes
/// - `logits`: `[1, seq_len, vocab]` or `[seq_len, vocab]` or `[vocab]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on shape mismatch or tensor operation failure.
pub fn extract_token_prob(logits: &Tensor, token_id: u32) -> Result<f32> {
    use candle_core::IndexOp;

    let logits_f32 = logits.to_dtype(DType::F32)?;

    // Get last-position logits as a 1-D vector.
    let last_logits = match logits_f32.dims().len() {
        1 => logits_f32,
        2 => {
            let seq_len = logits_f32.dim(0)?;
            logits_f32.i(seq_len - 1)?
        }
        3 => {
            let seq_len = logits_f32.dim(1)?;
            logits_f32.i((0, seq_len - 1))?
        }
        n => {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "extract_token_prob: expected 1-3 dims, got {n}"
            ))));
        }
    };

    // Terminal read-out (probability extraction): no gradient ever wanted, so
    // the fused kernel is used directly rather than `crate::nn_ops`.
    let probs = candle_nn::ops::softmax_last_dim(&last_logits)?;
    // CAST: u32 → usize, token ID used as tensor index
    #[allow(clippy::as_conversions)]
    let prob = probs.i(token_id as usize)?.to_scalar::<f32>()?;
    Ok(prob)
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Output of a text generation run with token-level details.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Original prompt text.
    pub prompt: String,
    /// Full output (prompt + generated).
    pub full_text: String,
    /// Only the generated portion.
    pub generated_text: String,
    /// Token IDs from the prompt.
    pub prompt_tokens: Vec<u32>,
    /// Token IDs that were generated.
    pub generated_tokens: Vec<u32>,
    /// Total token count (prompt + generated).
    pub total_tokens: usize,
}

impl GenerationResult {
    /// Assemble a generation result, deriving `total_tokens`.
    ///
    /// The type is `#[non_exhaustive]`, so a struct expression is rejected
    /// outside this crate; this is the construction path for callers running
    /// their own decode loop that want the crate's result shape.
    ///
    /// `total_tokens` is **computed**, not accepted, so it cannot disagree with
    /// the two token vectors it summarises.
    ///
    /// The three `String` parameters are adjacent and a transposition would
    /// compile, so they are ordered exactly as the fields are declared:
    /// `prompt`, then the full text, then the generated portion alone.
    #[must_use]
    pub const fn new(
        prompt: String,
        full_text: String,
        generated_text: String,
        prompt_tokens: Vec<u32>,
        generated_tokens: Vec<u32>,
    ) -> Self {
        let total_tokens = prompt_tokens.len() + generated_tokens.len();
        Self {
            prompt,
            full_text,
            generated_text,
            prompt_tokens,
            generated_tokens,
            total_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Weight loading helpers (used by from_pretrained)
// ---------------------------------------------------------------------------

/// Index structure for sharded safetensors models.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    /// Maps weight name → shard filename.
    weight_map: std::collections::HashMap<String, String>,
}

/// Extract tensor names from a downloaded file map for auto-config inference.
///
/// Tries `model.safetensors.index.json` first (sharded models), falls back
/// to reading the header of `model.safetensors` (single-file models).
#[cfg(feature = "transformer")]
fn extract_tensor_names(
    files: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Result<Vec<String>> {
    if let Some(index_path) = files.get("model.safetensors.index.json") {
        return crate::config::tensor_names_from_index(index_path);
    }
    if let Some(st_path) = files.get("model.safetensors") {
        return crate::config::tensor_names_from_safetensors(st_path);
    }
    Err(MIError::Config(
        "no safetensors files found for tensor name extraction".into(),
    ))
}

/// A [`candle_nn::var_builder::SimpleBackend`] that fabricates every requested
/// tensor as seeded `N(0, std)` Gaussian noise — the weight source for
/// [`MIModel::from_pretrained_random_init`]'s random-model (dead-salmon)
/// control.
///
/// `contains_tensor` reflects the *real* checkpoint's tensor set (read from the
/// safetensors header) so the architecture is preserved (e.g. tied embeddings);
/// `get` ignores the value and returns noise of the requested shape.
#[cfg(feature = "transformer")]
struct RandnBackend {
    /// Frozen seeded RNG behind a `Mutex` (the loader drives `get` sequentially;
    /// the `Mutex` keeps the backend `Send + Sync` while giving deterministic
    /// draws for a fixed request order).
    rng: std::sync::Mutex<rand_chacha::ChaCha8Rng>,
    /// Standard deviation of the `N(0, std)` weight noise.
    std: f64,
    /// The real checkpoint's tensor names, so `contains_tensor` preserves the
    /// architecture (e.g. tied embeddings).
    names: std::collections::HashSet<String>,
}

#[cfg(feature = "transformer")]
impl RandnBackend {
    /// Build a backend seeded by `seed`, drawing `N(0, std)` weights, reporting
    /// `names` as the present tensor set.
    fn new(seed: u64, std: f64, names: std::collections::HashSet<String>) -> Self {
        Self {
            rng: std::sync::Mutex::new(crate::util::rng::seeded(seed)),
            std,
            names,
        }
    }
}

#[cfg(feature = "transformer")]
impl candle_nn::var_builder::SimpleBackend for RandnBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        _name: &str,
        _h: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let n = s.elem_count();
        let data = {
            let mut rng = self
                .rng
                .lock()
                .map_err(|_| candle_core::Error::Msg("RandnBackend RNG poisoned".into()))?;
            crate::util::randn::randn_f32(&mut rng, n, self.std)
        };
        // Build on CPU, then move to the target device and dtype.
        Tensor::from_vec(data, s, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        _dtype: DType,
        _dev: &Device,
    ) -> candle_core::Result<Tensor> {
        Err(candle_core::Error::Msg(format!(
            "RandnBackend: get_unchecked('{name}') is unsupported (shape unknown)"
        )))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

/// Load every safetensors weight (CPU, up-cast to F32) and permute the elements
/// of each tensor independently with a `seed`-seeded RNG, preserving shapes and
/// per-tensor value multisets. Files and tensor names are visited in sorted
/// order so the shuffle is reproducible. Returns a `name -> shuffled` map at
/// `dtype`.
#[cfg(feature = "transformer")]
fn shuffle_checkpoint_tensors(
    weights_paths: &[std::path::PathBuf],
    seed: u64,
    dtype: DType,
) -> Result<std::collections::HashMap<String, Tensor>> {
    use rand::Rng;

    let mut rng = crate::util::rng::seeded(seed);
    let mut out: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

    let mut paths: Vec<&std::path::PathBuf> = weights_paths.iter().collect();
    paths.sort();
    for path in paths {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        // Sort (name, tensor) pairs so the seeded shuffle order is reproducible.
        let mut entries: Vec<(String, Tensor)> = tensors.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, tensor) in entries {
            let shape = tensor.shape().clone();
            // PROMOTE: shuffle in F32 so BF16/F16 checkpoints round-trip through a
            // single canonical element type before casting to `dtype`.
            let mut data = tensor
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // Fisher-Yates in place: destroys structure, preserves the multiset.
            for i in (1..data.len()).rev() {
                let j = rng.gen_range(0..=i);
                data.swap(i, j);
            }
            let shuffled = Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(dtype)?;
            out.insert(name, shuffled);
        }
    }
    Ok(out)
}

/// Weight storage format detected among the downloaded files.
///
/// Private internal enum, matched exhaustively within this module.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightFormat {
    /// `.safetensors` (single `model.safetensors` or sharded via index).
    Safetensors,
    /// Single-file `pytorch_model.bin` pickle, loaded via
    /// `VarBuilder::from_pth`.
    Pytorch,
}

/// Resolve weight file paths and their format from a downloaded file map.
///
/// Prefers `.safetensors` (the safe, mmap-able format) over a
/// `pytorch_model.bin` pickle.  The pickle fallback lets `from_pretrained`
/// load repositories that ship weights only as `pytorch_model.bin` (e.g.
/// `DeepSeek-Coder`), which the safetensors-only path could not.
///
/// # Errors
///
/// Returns [`MIError::Config`] when no recognized weight file is present, or
/// when the only weights are a sharded `pytorch_model.bin.index.json`
/// (sharded pickles are unsupported — convert to `safetensors`).
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
fn resolve_weight_paths(
    files: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Result<(Vec<std::path::PathBuf>, WeightFormat)> {
    if files.contains_key("model.safetensors.index.json") || files.contains_key("model.safetensors")
    {
        return Ok((resolve_safetensors_paths(files)?, WeightFormat::Safetensors));
    }
    if let Some(path) = files.get("pytorch_model.bin") {
        // BORROW: explicit .clone() — PathBuf from HashMap value
        return Ok((vec![path.clone()], WeightFormat::Pytorch));
    }
    if files.contains_key("pytorch_model.bin.index.json") {
        return Err(MIError::Config(
            "sharded pytorch_model.bin (pickle) weights are unsupported (convert to safetensors)"
                .into(),
        ));
    }
    Err(MIError::Config(
        "no weight files found (expected model.safetensors or pytorch_model.bin)".into(),
    ))
}

/// Resolve safetensors file paths from a downloaded file map.
///
/// Tries `model.safetensors.index.json` first (sharded), falls back to
/// single `model.safetensors`.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
fn resolve_safetensors_paths(
    files: &std::collections::HashMap<String, std::path::PathBuf>,
) -> Result<Vec<std::path::PathBuf>> {
    // Try sharded first
    if let Some(index_path) = files.get("model.safetensors.index.json") {
        let index_str = std::fs::read_to_string(index_path)
            .map_err(|e| MIError::Model(candle_core::Error::Msg(format!("read index: {e}"))))?;
        let index: SafetensorsIndex = serde_json::from_str(&index_str)
            .map_err(|e| MIError::Config(format!("parse index: {e}")))?;

        // Collect unique shard filenames
        let mut shard_names: Vec<String> = index.weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();

        let mut paths = Vec::with_capacity(shard_names.len());
        for shard_name in &shard_names {
            // BORROW: explicit .as_str() — &str from String for HashMap lookup
            let path = files.get(shard_name.as_str()).ok_or_else(|| {
                MIError::Model(candle_core::Error::Msg(format!(
                    "shard {shard_name} not found in downloaded files"
                )))
            })?;
            // BORROW: explicit .clone() — PathBuf from HashMap value
            paths.push(path.clone());
        }
        return Ok(paths);
    }

    // Single file
    let path = files.get("model.safetensors").ok_or_else(|| {
        MIError::Model(candle_core::Error::Msg(
            "model.safetensors not found in downloaded files".into(),
        ))
    })?;
    // BORROW: explicit .clone() — PathBuf from HashMap value
    Ok(vec![path.clone()])
}

/// Create a `VarBuilder` from resolved weight file paths.
///
/// For [`WeightFormat::Safetensors`], uses buffered (safe) loading by
/// default, or memory-mapped loading with the `mmap` feature.  For
/// [`WeightFormat::Pytorch`], loads the single `pytorch_model.bin` pickle via
/// `VarBuilder::from_pth`.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
fn create_var_builder(
    paths: &[std::path::PathBuf],
    format: WeightFormat,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    match format {
        WeightFormat::Safetensors => {
            #[cfg(feature = "mmap")]
            {
                mmap_var_builder(paths, dtype, device)
            }
            #[cfg(not(feature = "mmap"))]
            {
                buffered_var_builder(paths, dtype, device)
            }
        }
        WeightFormat::Pytorch => pth_var_builder(paths, dtype, device),
    }
}

/// Load a single quantized safetensors file by dequantizing to BF16 in memory
/// via anamnesis, then building a `VarBuilder` from the resulting standard
/// safetensors bytes.
///
/// anamnesis auto-detects the quantization scheme (bitsandbytes `NF4`/`FP4`/
/// `INT8`, AWQ, GPTQ) and emits standard BF16 safetensors bytes;
/// [`candle_nn::VarBuilder::from_buffered_safetensors`] then up-casts to `dtype`
/// (F32 by default).  Single-file only for now — sharded quantized checkpoints
/// error rather than silently load a partial model.
///
/// Peak memory is ~1x the model's BF16 size: anamnesis buffers the whole
/// serialized output before returning.  Comfortable for the `<=~7 B` quantized
/// models this targets.
#[cfg(all(
    any(feature = "transformer", feature = "rwkv", feature = "diffusion"),
    feature = "quantized"
))]
fn load_quantized_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    if paths.len() != 1 {
        return Err(MIError::Config(format!(
            "quantized checkpoints are supported only as a single safetensors file for now \
             (found {} shards); a sharded-dequant path is not yet implemented",
            paths.len()
        )));
    }
    let path = paths.first().ok_or_else(|| {
        MIError::Model(candle_core::Error::Msg(
            "no quantized safetensors path".into(),
        ))
    })?;
    let parsed = anamnesis::parse(path)
        .map_err(|e| MIError::Config(format!("anamnesis parse {}: {e}", path.display())))?;
    // anamnesis dequantizes to BF16; `from_buffered_safetensors` then up-casts
    // to `dtype` (F32) on load.
    let bytes = parsed
        .remember_to_bytes(anamnesis::TargetDtype::BF16)
        .map_err(|e| MIError::Config(format!("anamnesis dequantize {}: {e}", path.display())))?;
    let vb = candle_nn::VarBuilder::from_buffered_safetensors(bytes, dtype, device)?;
    Ok(vb)
}

/// Fallback when the `quantized` feature is disabled: a clear, actionable error
/// instead of a cryptic tensor-not-found failure on quantized weights.
#[cfg(all(
    any(feature = "transformer", feature = "rwkv", feature = "diffusion"),
    not(feature = "quantized")
))]
fn load_quantized_var_builder(
    _paths: &[std::path::PathBuf],
    _dtype: DType,
    _device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    Err(MIError::Config(
        "this checkpoint is quantized (`quantization_config` present in config.json); \
         rebuild candle-mi with the `quantized` feature to load it (it dequantizes to \
         BF16 in memory via anamnesis)"
            .into(),
    ))
}

/// Load weights from a single `pytorch_model.bin` pickle via `from_pth`.
///
/// `from_pth` reads the entire pickle into memory and materializes every
/// tensor (there is no memory-mapped pickle path), so peak memory is ~1x the
/// model size in `dtype`.  Adequate for the small (<=~3 B) families that ship
/// only `.bin`; larger models should be converted to `safetensors`.
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
fn pth_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    let path = paths.first().ok_or_else(|| {
        MIError::Model(candle_core::Error::Msg("no pytorch_model.bin path".into()))
    })?;
    let vb = candle_nn::VarBuilder::from_pth(path, dtype, device)?;
    Ok(vb)
}

/// Load weights via buffered (safe) reading — reads all data into RAM.
///
/// Only supports single-file models. For sharded models (7B+), enable
/// the `mmap` feature.
#[cfg(all(
    any(feature = "transformer", feature = "rwkv", feature = "diffusion"),
    not(feature = "mmap")
))]
fn buffered_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    if paths.len() > 1 {
        return Err(MIError::Config(format!(
            "this model is sharded across {} files and requires the `mmap` feature.\n  \
             Library:  candle-mi = {{ features = [\"mmap\"] }}\n  \
             Example:  cargo run --features mmap --example <name>",
            paths.len()
        )));
    }
    let path = paths
        .first()
        .ok_or_else(|| MIError::Model(candle_core::Error::Msg("no safetensors files".into())))?;
    let data = std::fs::read(path).map_err(|e| {
        MIError::Model(candle_core::Error::Msg(format!(
            "read {}: {e}",
            path.display()
        )))
    })?;
    let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, dtype, device)?;
    Ok(vb)
}

/// Load weights via memory-mapped files — minimal RAM overhead for large models.
///
/// # Safety
///
/// The safetensors files must not be modified while the model is loaded.
/// This is the standard invariant for memory-mapped files.
#[cfg(all(
    any(feature = "transformer", feature = "rwkv", feature = "diffusion"),
    feature = "mmap"
))]
#[allow(unsafe_code)]
fn mmap_var_builder(
    paths: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::VarBuilder<'static>> {
    // SAFETY: safetensors files must not be modified while loaded.
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
    Ok(vb)
}

#[cfg(all(
    test,
    any(feature = "transformer", feature = "rwkv", feature = "diffusion")
))]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{WeightFormat, resolve_weight_paths};

    fn file_map(names: &[&str]) -> HashMap<String, PathBuf> {
        names
            .iter()
            .map(|n| ((*n).to_owned(), PathBuf::from(format!("/cache/{n}"))))
            .collect()
    }

    #[test]
    fn resolve_prefers_single_safetensors() {
        let files = file_map(&["config.json", "model.safetensors", "tokenizer.json"]);
        let (paths, format) = resolve_weight_paths(&files).unwrap();
        assert_eq!(format, WeightFormat::Safetensors);
        assert_eq!(paths, vec![PathBuf::from("/cache/model.safetensors")]);
    }

    #[test]
    fn resolve_falls_back_to_pytorch_bin() {
        // DeepSeek-Coder ships only pytorch_model.bin — must resolve to it.
        let files = file_map(&["config.json", "pytorch_model.bin", "tokenizer.json"]);
        let (paths, format) = resolve_weight_paths(&files).unwrap();
        assert_eq!(format, WeightFormat::Pytorch);
        assert_eq!(paths, vec![PathBuf::from("/cache/pytorch_model.bin")]);
    }

    #[test]
    fn resolve_prefers_safetensors_over_bin_when_both_present() {
        let files = file_map(&["model.safetensors", "pytorch_model.bin"]);
        let (_, format) = resolve_weight_paths(&files).unwrap();
        assert_eq!(format, WeightFormat::Safetensors);
    }

    #[test]
    fn resolve_sharded_pickle_is_rejected() {
        let files = file_map(&["config.json", "pytorch_model.bin.index.json"]);
        assert!(resolve_weight_paths(&files).is_err());
    }

    #[test]
    fn resolve_no_weights_errors() {
        let files = file_map(&["config.json", "tokenizer.json"]);
        assert!(resolve_weight_paths(&files).is_err());
    }
}
