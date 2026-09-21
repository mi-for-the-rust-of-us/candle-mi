// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-Layer Transcoder (CLT) support.
//!
//! Loads pre-trained CLT weights from `HuggingFace` (circuit-tracer format),
//! encodes residual stream activations into sparse feature activations,
//! injects decoder vectors into the residual stream for steering, and
//! scores features by decoder projection for attribution graph construction.
//!
//! Memory-efficient: uses stream-and-free for encoders (~75 MB/layer on GPU)
//! and a micro-cache for steering vectors (~450 KB for 50 features).
//! Decoder scoring operates entirely on CPU (one file at a time, up to ~2 GB).
//!
//! # CLT Architecture
//!
//! A cross-layer transcoder at layer `l` implements:
//! ```text
//! Encode:  features = ReLU(W_enc[l] @ residual_mid[l] + b_enc[l])
//! Decode:  For each downstream layer l' >= l:
//!            mlp_out_hat[l'] += W_dec[l, l'] @ features + b_dec[l']
//! Inject:  residual[pos] += strength × W_dec[l, target_layer, feature_idx, :]
//! ```
//!
//! # Weight File Layout (circuit-tracer format)
//!
//! Each encoder file `W_enc_{l}.safetensors` contains:
//! - `W_enc_{l}`: shape `[n_features, d_model]` (BF16) — encoder weight matrix
//! - `b_enc_{l}`: shape `[n_features]` (BF16) — encoder bias
//! - `b_dec_{l}`: shape `[d_model]` (BF16) — decoder bias for target layer l
//!
//! Each decoder file `W_dec_{l}.safetensors` contains:
//! - `W_dec_{l}`: shape `[n_features, n_target_layers, d_model]` (BF16)
//!   where `n_target_layers = n_layers - l` (layer l writes to layers l..n_layers-1)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};
use safetensors::tensor::SafeTensors;
use tracing::info;

use crate::error::{MIError, Result};

pub mod gemmascope;

use crate::clt::gemmascope::GEMMASCOPE_WEIGHTS_REPO;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Identifies a single CLT feature by its source layer and index within that layer.
// EXHAUSTIVE: a two-field value identity, not a growing record. It is `Copy`,
// `Ord`, `Hash` and `Serialize`, and callers construct it by literal all over
// the examples; `#[non_exhaustive]` would shut that for no benefit, since the
// pair (layer, index) is what a CLT feature *is* and cannot gain a field.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CltFeatureId {
    /// Source layer where this feature's encoder lives (`0..n_layers`).
    pub layer: usize,
    /// Feature index within the layer (`0..n_features_per_layer`).
    pub index: usize,
}

impl std::fmt::Display for CltFeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}:{}", self.layer, self.index)
    }
}

use crate::sparse::{FeatureId, SparseActivations};

impl FeatureId for CltFeatureId {}

/// A single edge in a CLT attribution graph.
///
/// Represents a feature's decoder projection score onto a target direction
/// at a specific downstream layer. Positive scores indicate alignment,
/// negative scores indicate opposition.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct AttributionEdge {
    /// The CLT feature contributing this edge.
    pub feature: CltFeatureId,
    /// Decoder projection score (dot product or cosine similarity).
    pub score: f32,
}

/// Attribution graph for CLT circuit analysis.
///
/// Represents a set of CLT features scored by how strongly their decoder
/// vectors project along a target direction at a specific layer. Built by
/// [`CrossLayerTranscoder::build_attribution_graph()`] or
/// [`CrossLayerTranscoder::build_attribution_graph_batch()`].
///
/// Edges are always sorted by score in descending order.
///
/// # Pruning
///
/// - [`top_k()`](Self::top_k): keep only the k highest-scoring features
/// - [`threshold()`](Self::threshold): keep features with |score| above a minimum
#[derive(Debug, Clone)]
pub struct AttributionGraph {
    /// Target layer these scores were computed for.
    target_layer: usize,
    /// Edges sorted by score descending.
    edges: Vec<AttributionEdge>,
}

impl AttributionGraph {
    /// Target layer this graph was scored against.
    #[must_use]
    pub const fn target_layer(&self) -> usize {
        self.target_layer
    }

    /// All edges, sorted by score descending.
    #[must_use]
    pub fn edges(&self) -> &[AttributionEdge] {
        &self.edges
    }

    /// Number of edges in the graph.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether the graph has no edges.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Return a new graph with only the top-k highest-scoring edges.
    #[must_use]
    pub fn top_k(&self, k: usize) -> Self {
        Self {
            target_layer: self.target_layer,
            edges: self.edges.iter().take(k).cloned().collect(),
        }
    }

    /// Return a new graph keeping only edges whose absolute score meets
    /// or exceeds `min_score`.
    #[must_use]
    pub fn threshold(&self, min_score: f32) -> Self {
        Self {
            target_layer: self.target_layer,
            edges: self
                .edges
                .iter()
                .filter(|e| e.score.abs() >= min_score)
                .cloned()
                .collect(),
        }
    }

    /// Extract the feature IDs from all edges in score order.
    #[must_use]
    pub fn features(&self) -> Vec<CltFeatureId> {
        self.edges.iter().map(|e| e.feature).collect()
    }

    /// Consume the graph and return its edges.
    #[must_use]
    pub fn into_edges(self) -> Vec<AttributionEdge> {
        self.edges
    }
}

/// On-disk layout of a transcoder repository.
///
/// Determines how weight files are named, which tensors they contain, and
/// whether `W_dec` is rank-2 (per-layer) or rank-3 (cross-layer). Auto-detected
/// from the repo file listing at [`CrossLayerTranscoder::open`] time —
/// see the schema-detection rules documented on each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranscoderSchema {
    /// Cross-Layer Transcoder (`mntss/clt-*`): two files per layer,
    /// `W_enc_{l}.safetensors` + `W_dec_{l}.safetensors`, layer-suffixed tensor
    /// names (`W_enc_{l}`, `W_dec_{l}`). `W_dec` is rank-3
    /// `[n_features, n_target_layers, d_model]` — writes to multiple downstream
    /// layers.
    CltSplit,
    /// Per-Layer Transcoder bundle (`mntss/transcoder-*`, `mwhanna/qwen3-*-transcoders*`):
    /// one file per layer, `layer_{l}.safetensors`, un-suffixed tensor names
    /// (`W_enc`, `W_dec`, `W_skip`, `b_enc`, `b_dec`). `W_dec` is rank-2
    /// `[n_features, d_model]` — writes only to layer `l`. The `W_skip` linear
    /// path is loaded but not used by encode/inject/suppress.
    PltBundle,
    /// `GemmaScope` NPZ transcoder (`google/gemma-scope-2b-pt-transcoders`,
    /// pointed to by `mntss/gemma-scope-transcoders/config.yaml`). Per-layer
    /// NPZ file at `layer_N/width_16k/average_l0_X/params.npz`. Tensors:
    /// `W_enc [d_model, n_features]` (transposed vs `PltBundle`), `W_dec`,
    /// `b_enc`, `b_dec`, and `threshold [n_features]` for `JumpReLU` gating.
    /// No `W_skip`.
    ///
    /// Loaded via the two-repo flow when the `sae` feature is enabled
    /// (NPZ parsing is provided by `anamnesis/npz`). Without `sae`,
    /// [`CrossLayerTranscoder::open`] returns [`MIError::Config`]
    /// instructing the caller to enable the feature.
    GemmaScopeNpz,
    /// `BlueLightAI` Cross-Layer Transcoder with `JumpReLU` activation
    /// (`bluelightai/clt-qwen3-{0.6b,1.7b}-base-20k`). Same file layout
    /// as [`CltSplit`](Self::CltSplit) (per-layer `W_enc_{l}.safetensors` +
    /// `W_dec_{l}.safetensors`, layer-suffixed tensor names, rank-3
    /// cross-layer `W_dec`) but the encoder file additionally carries
    /// a per-feature `threshold_{l} [n_features]` tensor that gates the
    /// pre-activation via `pre * (pre > threshold)` — same activation
    /// semantics as [`GemmaScopeNpz`](Self::GemmaScopeNpz).
    ///
    /// Encoder-file tensors: `W_enc_{l}`, `b_enc_{l}`, `b_dec_{l}`,
    /// `threshold_{l}`. The `b_dec_{l}` tensor is `BlueLightAI`-specific
    /// (mntss `CltSplit` keeps `b_dec` with `W_dec`); candle-mi does not
    /// load `b_dec` for any schema, so this is a documentation-only
    /// quirk with no load-site consequences.
    ///
    /// Distinguished from plain `CltSplit` at classification time by the
    /// presence of a `features/index.json.gz` sidecar (circuit-tracer
    /// metadata that mntss `CltSplit` repos do not ship). The classifier
    /// is filename-only (no HTTP I/O); the sidecar is the cheapest
    /// reliable signal for the `JumpReLU` semantics.
    CltSplitJumpReLU,
}

impl TranscoderSchema {
    /// Whether this schema writes to multiple downstream layers per feature
    /// (`CltSplit`, `CltSplitJumpReLU`) or only to its own layer
    /// (`PltBundle`, `GemmaScopeNpz`).
    #[must_use]
    pub const fn is_cross_layer(self) -> bool {
        matches!(self, Self::CltSplit | Self::CltSplitJumpReLU)
    }

    /// Whether this schema uses `JumpReLU` gating with a per-feature `threshold`
    /// tensor (`GemmaScopeNpz`, `CltSplitJumpReLU`) rather than plain `ReLU`.
    #[must_use]
    pub const fn is_jump_relu(self) -> bool {
        matches!(self, Self::GemmaScopeNpz | Self::CltSplitJumpReLU)
    }
}

/// CLT configuration auto-detected from tensor shapes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CltConfig {
    /// Number of layers in the base model (26 for Gemma 2 2B).
    pub n_layers: usize,
    /// Hidden dimension of the base model (2304 for Gemma 2 2B).
    pub d_model: usize,
    /// Number of features per encoder layer (16384 for CLT-426K).
    pub n_features_per_layer: usize,
    /// Total feature count across all layers.
    pub n_features_total: usize,
    /// Base model name from config.yaml.
    pub model_name: String,
    /// Detected on-disk schema.
    pub schema: TranscoderSchema,
    /// Per-layer NPZ paths for `GemmaScopeNpz` repos routed via
    /// `mntss/gemma-scope-transcoders/config.yaml`. Populated by `open()`
    /// for `GemmaScopeNpz` schemas; empty for `CltSplit` / `PltBundle`.
    pub gemmascope_npz_paths: Vec<String>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Currently loaded encoder weights on GPU.
struct LoadedEncoder {
    /// Layer index this encoder corresponds to.
    layer: usize,
    /// Encoder weight matrix.
    ///
    /// Stored in the `[n_features, d_model]` orientation regardless of
    /// schema: `CltSplit` and `PltBundle` files store `W_enc` directly in
    /// this layout, while `GemmaScopeNpz` files store the transpose
    /// `[d_model, n_features]` on disk and the loader applies `.t()` to
    /// canonicalise the orientation here.
    ///
    /// # Shapes
    /// - `w_enc`: `[n_features, d_model]`
    w_enc: Tensor,
    /// Encoder bias vector.
    ///
    /// # Shapes
    /// - `b_enc`: `[n_features]`
    b_enc: Tensor,
    /// Per-feature `JumpReLU` threshold, present only for the `GemmaScopeNpz`
    /// schema. `None` for `CltSplit` and `PltBundle`, which use plain `ReLU`.
    /// When `Some`, [`encode`](CrossLayerTranscoder::encode) gates the
    /// pre-activation by `pre > threshold` element-wise instead of clamping
    /// at zero.
    ///
    /// # Shapes
    /// - `threshold`: `[n_features]` (when `Some`)
    threshold: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// CrossLayerTranscoder
// ---------------------------------------------------------------------------

/// Cross-Layer Transcoder.
///
/// Loads CLT encoder/decoder weights on-demand from `HuggingFace` safetensors,
/// with memory-efficient streaming (only one encoder on GPU at a time)
/// and a micro-cache for steering vectors.
///
/// Downloads are lazy: [`open()`](Self::open) only fetches config and the first
/// encoder for dimension detection. Subsequent files are downloaded as needed by
/// [`load_encoder()`](Self::load_encoder), [`decoder_vector()`](Self::decoder_vector),
/// and [`cache_steering_vectors()`](Self::cache_steering_vectors).
///
/// # Example
///
/// ```no_run
/// # fn main() -> candle_mi::Result<()> {
/// use candle_mi::clt::CrossLayerTranscoder;
/// use candle_core::Device;
///
/// let mut clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k")?;
/// println!("CLT: {} layers, d_model={}", clt.config().n_layers, clt.config().d_model);
///
/// // Load encoder for layer 10
/// let device = Device::Cpu;
/// clt.load_encoder(10, &device)?;
/// # Ok(())
/// # }
/// ```
pub struct CrossLayerTranscoder {
    /// `HuggingFace` repository ID for on-demand downloads.
    repo_id: String,
    /// Fetch configuration for `hf-fetch-model` downloads.
    fetch_config: hf_fetch_model::FetchConfig,
    /// Local paths to already-downloaded encoder files (None = not yet downloaded).
    encoder_paths: Vec<Option<PathBuf>>,
    /// Local paths to already-downloaded decoder files (None = not yet downloaded).
    decoder_paths: Vec<Option<PathBuf>>,
    /// Auto-detected configuration.
    config: CltConfig,
    /// Currently loaded encoder (stream-and-free: only one at a time).
    loaded_encoder: Option<LoadedEncoder>,
    /// Micro-cache: pre-extracted steering vectors pinned on device.
    /// Key: (`feature_id`, `target_layer`), Value: decoder vector `[d_model]` on device.
    steering_cache: HashMap<(CltFeatureId, usize), Tensor>,
}

impl CrossLayerTranscoder {
    /// Open a transcoder from `HuggingFace` and detect its configuration.
    ///
    /// Classifies the repository into a [`TranscoderSchema`] from its file
    /// listing (no downloads), then fetches `config.yaml` (if present) and
    /// the first encoder file to probe dimensions:
    ///
    /// - `CltSplit`: downloads `W_enc_0.safetensors` (~75 MB, CLT-426K)
    ///   and reads the tensor `W_enc_0`.
    /// - `PltBundle`: downloads `layer_0.safetensors` (~1 GiB — bundle of
    ///   `W_enc`/`W_dec`/`W_skip`/`b_enc`/`b_dec`) and reads un-suffixed `W_enc`.
    /// - `GemmaScopeNpz`: requires the `sae` feature (NPZ parsing via
    ///   `anamnesis/npz`). With `sae` enabled, the two-repo flow fetches
    ///   `mntss/gemma-scope-transcoders/config.yaml` (~2.5 KiB) for the
    ///   per-layer `.npz` curation, then downloads the layer-0 `.npz`
    ///   (~288 MiB FP32) from `google/gemma-scope-2b-pt-transcoders` to
    ///   probe dimensions. Without `sae`, returns a `MIError::Config`
    ///   instructing the caller to enable the feature.
    ///
    /// All other encoder/decoder files are downloaded lazily on first use.
    ///
    /// # Arguments
    /// * `clt_repo` — `HuggingFace` repository ID (e.g., `"mntss/clt-gemma-2-2b-426k"`
    ///   for CLT, `"mntss/transcoder-Llama-3.2-1B"` for PLT,
    ///   `"mntss/gemma-scope-transcoders"` for `GemmaScope` PLT — note that
    ///   the user-facing arg names the curation repo, not the underlying
    ///   `google/*` weights repo)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the repository is inaccessible, if the
    /// HTTP client cannot be built, or if a file cannot be fetched.
    /// Returns [`MIError::Config`] if the repository layout is unrecognised,
    /// if no encoder files are found, if the `GemmaScopeNpz` schema is
    /// detected without the `sae` feature, if a weight file (safetensors or
    /// `.npz`) cannot be deserialised, if the expected encoder tensor is
    /// missing, or if its shape is not 2D.
    // Schema-branched open() is a deliberately flat sequence (detect → reject
    // GemmaScope → count layers → parse config → probe dimensions → build
    // CltConfig). Extracting helpers would scatter the control flow for no
    // gain in reuse, so we suppress the pedantic length lint here.
    #[allow(clippy::too_many_lines)]
    pub fn open(clt_repo: &str) -> Result<Self> {
        // Route through the shared builder so HF_TOKEN is picked up for any
        // gated transcoder repository (hf-fetch-model 0.9.x requires explicit
        // opt-in).
        let fetch_config = crate::download::fetch_config_builder()
            .on_progress(|event| {
                tracing::info!(
                    filename = %event.filename,
                    percent = event.percent,
                    bytes_downloaded = event.bytes_downloaded,
                    bytes_total = event.bytes_total,
                    "CLT download progress",
                );
            })
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;

        // List repo files (no downloads needed) for schema detection and layer counting.
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| MIError::Download(format!("failed to create tokio runtime: {e}")))?;
        // hf-fetch-model 0.9.6 takes a shared reqwest client for the metadata
        // listing. Thread HF_TOKEN through so gated transcoder repos work
        // (currently mntss/* are public, but the call site must stay uniform
        // with the rest of the auth handling).
        // BORROW: std::env::var returns an owned String; .as_deref() hands the
        // inner &str to build_client without cloning.
        let hf_token = std::env::var("HF_TOKEN").ok();
        let http_client = hf_fetch_model::build_client(hf_token.as_deref())
            .map_err(|e| MIError::Download(format!("failed to build HTTP client: {e}")))?;
        let repo_files = rt
            .block_on(hf_fetch_model::repo::list_repo_files_with_metadata(
                clt_repo,
                None,
                None,
                &http_client,
            ))
            .map_err(|e| MIError::Download(format!("failed to list repo files: {e}")))?;

        // Collect filenames once so schema classification is a pure function over a slice.
        // BORROW: explicit .as_str() — &str view into each RepoFile.filename
        let filenames: Vec<&str> = repo_files.iter().map(|f| f.filename.as_str()).collect();
        let schema = classify_transcoder_schema(&filenames).map_err(|_| {
            MIError::Config(format!(
                "unrecognised transcoder repo layout for {clt_repo}"
            ))
        })?;
        info!("Transcoder schema detected for {clt_repo}: {schema:?}");

        // GemmaScope dispatches into a dedicated implementation: it needs the
        // two-repo flow (curation YAML on mntss, weights on google/) and
        // NPZ parsing via anamnesis. Without the `sae` feature, surface a
        // clear feature-gate error instead of a cryptic deserialisation failure.
        if matches!(schema, TranscoderSchema::GemmaScopeNpz) {
            #[cfg(feature = "sae")]
            {
                return Self::open_gemmascope(clt_repo, fetch_config);
            }
            #[cfg(not(feature = "sae"))]
            {
                return Err(MIError::Config(
                    "GemmaScope loading requires the 'sae' feature \
                     (NPZ parsing via anamnesis/npz); add 'sae' to your \
                     candle-mi features list"
                        .into(),
                ));
            }
        }

        // Count layers per schema. GemmaScopeNpz dispatches above so the
        // arm here is dead at runtime — kept to satisfy exhaustive matching
        // and to fail loudly if the dispatch is ever bypassed.
        let n_layers = match schema {
            // CltSplit and CltSplitJumpReLU share the W_enc_*.safetensors
            // per-layer file convention; layer count derives from the same
            // filename pattern.
            TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => repo_files
                .iter()
                .filter(|f| {
                    f.filename.starts_with("W_enc_") && f.filename.ends_with(".safetensors")
                })
                .count(),
            TranscoderSchema::PltBundle => repo_files
                .iter()
                .filter(|f| {
                    f.filename.starts_with("layer_") && f.filename.ends_with(".safetensors")
                })
                .count(),
            TranscoderSchema::GemmaScopeNpz => {
                return Err(MIError::Config(
                    "internal: GemmaScope schema reached the n_layers match \
                     (open_gemmascope dispatch should have fired earlier)"
                        .into(),
                ));
            }
        };
        if n_layers == 0 {
            return Err(MIError::Config(format!(
                "no encoder files found in {clt_repo} (schema={schema:?})"
            )));
        }

        // Parse config.yaml for model_name (simple line-by-line, no serde_yaml dep).
        let model_name = match hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            "config.yaml",
            &fetch_config,
        ) {
            Ok(outcome) => {
                let path = outcome.into_inner();
                let text = std::fs::read_to_string(&path)?;
                parse_yaml_value(&text, "model_name").unwrap_or_else(|| "unknown".to_owned())
            }
            Err(_) => "unknown".to_owned(),
        };

        // Resolve the first-layer encoder filename and its `W_enc` tensor name
        // via the schema-aware helper. `gemmascope_npz_paths` is not yet
        // populated at open() time; the helper is only reached here with
        // CltSplit / PltBundle (GemmaScope was rejected above), so passing an
        // empty slice is safe.
        let (enc0_filename, enc_tensor_name, _) = encoder_file_and_tensor_names(schema, 0, &[])?;
        // BORROW: explicit .to_owned() — hf_fetch_model download_file_blocking requires an owned repo ID.
        let enc0_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            &enc0_filename,
            &fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {enc0_filename}: {e}")))?
        .into_inner();

        let data = std::fs::read(&enc0_path)?;
        let tensors = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize {enc_tensor_name}: {e}"))
        })?;
        let w_enc_view = tensors
            .tensor(&enc_tensor_name)
            .map_err(|e| MIError::Config(format!("tensor '{enc_tensor_name}' not found: {e}")))?;
        let shape = w_enc_view.shape();
        if shape.len() != 2 {
            return Err(MIError::Config(format!(
                "expected 2D encoder weight, got shape {shape:?}"
            )));
        }
        let n_features_per_layer = *shape
            .first()
            .ok_or_else(|| MIError::Config("encoder weight shape is empty".into()))?;
        let d_model = *shape.get(1).ok_or_else(|| {
            MIError::Config("encoder weight shape has fewer than 2 dimensions".into())
        })?;

        // Initialise paths: only first encoder known, rest downloaded lazily.
        let mut encoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];
        if let Some(slot) = encoder_paths.first_mut() {
            *slot = Some(enc0_path);
        }
        let decoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];

        let config = CltConfig {
            n_layers,
            d_model,
            n_features_per_layer,
            n_features_total: n_layers * n_features_per_layer,
            model_name,
            schema,
            gemmascope_npz_paths: Vec::new(),
        };
        info!(
            "CLT config: {} layers, d_model={}, features_per_layer={}, total={}, schema={:?}",
            config.n_layers,
            config.d_model,
            config.n_features_per_layer,
            config.n_features_total,
            config.schema,
        );

        Ok(Self {
            repo_id: clt_repo.to_owned(),
            fetch_config,
            encoder_paths,
            decoder_paths,
            config,
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        })
    }

    /// `GemmaScope`-specific [`open`](Self::open) implementation, gated
    /// behind the `sae` feature for `anamnesis/npz` parsing.
    ///
    /// Implements the two-repo flow:
    /// 1. Fetches `mntss/gemma-scope-transcoders/config.yaml` (~2.5 KiB)
    ///    from `clt_repo` (the curation entry-point).
    /// 2. Parses the curation `YAML` via [`gemmascope::parse_gemmascope_config`]
    ///    to obtain the per-layer `.npz` paths inside
    ///    `google/gemma-scope-2b-pt-transcoders` (the weights repo).
    /// 3. Downloads the layer-0 `.npz` (~288 MiB FP32) from the weights
    ///    repo to probe `(d_model, n_features_per_layer)` from `W_enc`.
    ///    The on-disk shape is `[d_model, n_features]` — transposed vs
    ///    the `[n_features, d_model]` convention used by the other
    ///    schemas.
    /// 4. Builds [`CltConfig`] and returns the constructed transcoder.
    ///    All other layer NPZs are downloaded lazily.
    ///
    /// Caller (`open()`) has already classified the schema and built
    /// `fetch_config`.
    #[cfg(feature = "sae")]
    fn open_gemmascope(clt_repo: &str, fetch_config: hf_fetch_model::FetchConfig) -> Result<Self> {
        use crate::clt::gemmascope::{GEMMASCOPE_WEIGHTS_REPO, parse_gemmascope_config};

        info!(
            "Opening GemmaScope transcoder via two-repo flow \
             (curation: {clt_repo}, weights: {GEMMASCOPE_WEIGHTS_REPO})"
        );

        // Step 1: fetch the curation YAML from the mntss repo.
        // BORROW: explicit .to_owned() — hf_fetch_model takes ownership of the repo ID.
        let yaml_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            "config.yaml",
            &fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {clt_repo}/config.yaml: {e}")))?
        .into_inner();
        let yaml_text = std::fs::read_to_string(&yaml_path)?;

        // Step 2: parse YAML to discover per-layer NPZ paths and model_name.
        let model_name =
            parse_yaml_value(&yaml_text, "model_name").unwrap_or_else(|| "unknown".to_owned());
        let gemmascope_npz_paths = parse_gemmascope_config(&yaml_text)?;
        let n_layers = gemmascope_npz_paths.len();

        // Step 3: probe layer-0 NPZ to discover dimensions.
        let first_npz_relpath = gemmascope_npz_paths.first().ok_or_else(|| {
            MIError::Config(
                "parse_gemmascope_config returned empty paths despite passing validation".into(),
            )
        })?;
        info!(
            "Downloading first GemmaScope NPZ for dimension probe: \
             {GEMMASCOPE_WEIGHTS_REPO}/{first_npz_relpath} (~288 MiB)"
        );
        // TODO(hf-fetch-model): this downloads ~288 MiB to read two integers.
        // Both halves of the fix now exist publicly, so the blocker this TODO
        // used to name is gone — but the replacement is NOT the two-liner an
        // earlier version of this comment sketched. Corrected 2026-09-21:
        //
        //   * There is no `hf_fetch_model::range_reader(repo, file)`. That name
        //     was invented. The real entry point is
        //     `HttpRangeReader::open(repo_id, revision: Option<&str>,
        //     filename, token: Option<&str>)`, re-exported at the crate root
        //     of `hf-fetch-model` >= 0.12.1 (already our floor).
        //   * It is `async`, and its own docs require it to be called from
        //     inside a `tokio` runtime and then handed to a blocking context
        //     (`spawn_blocking`), because its `Read`/`Seek` impls drive async
        //     requests through a captured runtime handle. `open()` below is
        //     blocking and already spawns a runtime for the HF API, so there
        //     is one to borrow — but this is an integration, not a swap.
        //   * `token` is load-bearing: `GemmaScope` is gated and `HF_TOKEN` is
        //     not read automatically, so it must come from
        //     `crate::download::fetch_config_builder()`, never `None`.
        //
        // The `.npz` format question is settled, and not by us:
        // `HttpRangeReader::open_with_limits`' own doc lists `.npz` among the
        // archive-header paths its default budgets are tuned for ("a handful
        // of requests, well under 1 MiB"), which independently corroborates
        // the ~7-requests estimate. Caveat: those budgets are hard limits, and
        // exceeding them surfaces as a misleading "pathological archive
        // layout" error — so keep the full download below as a fallback
        // rather than replacing it outright.
        //
        // Pairs with `anamnesis::inspect_npz_from_reader<R: Read + Seek>`.
        // Expected win: `open()` cold start ~30 s on a 100 Mbps link -> <1 s.
        // BORROW: explicit .to_owned() — hf_fetch_model takes ownership of the repo ID.
        let first_npz_path = hf_fetch_model::download_file_blocking(
            GEMMASCOPE_WEIGHTS_REPO.to_owned(),
            first_npz_relpath,
            &fetch_config,
        )
        .map_err(|e| {
            MIError::Download(format!(
                "failed to download {GEMMASCOPE_WEIGHTS_REPO}/{first_npz_relpath}: {e}"
            ))
        })?
        .into_inner();
        let (n_features_per_layer, d_model) = read_gemmascope_npz_shape(&first_npz_path)?;

        // Step 4: assemble the path cache with the layer-0 path pre-populated.
        let mut encoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];
        if let Some(slot) = encoder_paths.first_mut() {
            *slot = Some(first_npz_path);
        }
        let decoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];

        let config = CltConfig {
            n_layers,
            d_model,
            n_features_per_layer,
            n_features_total: n_layers * n_features_per_layer,
            model_name,
            schema: TranscoderSchema::GemmaScopeNpz,
            gemmascope_npz_paths,
        };
        info!(
            "GemmaScope config: {} layers, d_model={}, features_per_layer={}, total={}",
            config.n_layers, config.d_model, config.n_features_per_layer, config.n_features_total,
        );

        Ok(Self {
            // BORROW: explicit .to_owned() — store an owned repo ID for lazy downloads.
            repo_id: clt_repo.to_owned(),
            fetch_config,
            encoder_paths,
            decoder_paths,
            config,
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        })
    }

    /// Access the auto-detected CLT configuration.
    #[must_use]
    pub const fn config(&self) -> &CltConfig {
        &self.config
    }

    /// Check whether an encoder is currently loaded and for which layer.
    #[must_use]
    pub fn loaded_encoder_layer(&self) -> Option<usize> {
        self.loaded_encoder.as_ref().map(|e| e.layer)
    }

    // --- Lazy download helpers ---

    /// `HuggingFace` repo to fetch weight files from for the current schema.
    ///
    /// For `CltSplit` and `PltBundle` this is the repo passed to `open()`.
    /// For `GemmaScopeNpz` it is `google/gemma-scope-2b-pt-transcoders` —
    /// the `mntss/*` repo passed by the caller is only the curation
    /// entry-point (it holds the `config.yaml`, not the actual weights).
    /// See [`gemmascope::GEMMASCOPE_WEIGHTS_REPO`].
    fn download_repo(&self) -> String {
        match self.config.schema {
            TranscoderSchema::CltSplit
            | TranscoderSchema::PltBundle
            | TranscoderSchema::CltSplitJumpReLU => {
                // BORROW: explicit .clone() — hf_fetch_model takes ownership of the repo ID
                self.repo_id.clone()
            }
            TranscoderSchema::GemmaScopeNpz => {
                // BORROW: explicit .to_owned() — promote &'static str to owned String
                GEMMASCOPE_WEIGHTS_REPO.to_owned()
            }
        }
    }

    /// Ensure the encoder file for a given layer is downloaded. Returns the path.
    fn ensure_encoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(path) = self
            .encoder_paths
            .get(layer)
            .and_then(std::option::Option::as_ref)
        {
            // BORROW: explicit .clone() — PathBuf from Vec
            return Ok(path.clone());
        }
        let (filename, _, _) = encoder_file_and_tensor_names(
            self.config.schema,
            layer,
            &self.config.gemmascope_npz_paths,
        )?;
        let repo = self.download_repo();
        info!("Downloading {filename} from {repo}");
        let path = hf_fetch_model::download_file_blocking(repo, &filename, &self.fetch_config)
            .map_err(|e| MIError::Download(format!("failed to download {filename}: {e}")))?
            .into_inner();
        if let Some(slot) = self.encoder_paths.get_mut(layer) {
            // BORROW: explicit .clone() — store PathBuf in cache
            *slot = Some(path.clone());
        }
        Ok(path)
    }

    /// Ensure the decoder file for a given layer is downloaded. Returns the path.
    ///
    /// For non-split schemas (`PltBundle`, `GemmaScopeNpz`), the encoder and
    /// decoder live in the same bundle file; this method delegates to
    /// [`ensure_encoder_path`](Self::ensure_encoder_path) to reuse the shared
    /// path cache and avoid double-downloading the same file. Both
    /// `CltSplit` and `CltSplitJumpReLU` ship encoder and decoder in
    /// separate `W_enc_{l}.safetensors` / `W_dec_{l}.safetensors` files
    /// and take the full download path.
    fn ensure_decoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if !matches!(
            self.config.schema,
            TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU,
        ) {
            // Bundle schemas: encoder and decoder share the same file.
            return self.ensure_encoder_path(layer);
        }
        if let Some(path) = self
            .decoder_paths
            .get(layer)
            .and_then(std::option::Option::as_ref)
        {
            // BORROW: explicit .clone() — PathBuf from Vec
            return Ok(path.clone());
        }
        let (filename, _) = decoder_file_and_tensor_name(
            self.config.schema,
            layer,
            &self.config.gemmascope_npz_paths,
        )?;
        let repo = self.download_repo();
        info!("Downloading {filename} from {repo}");
        let path = hf_fetch_model::download_file_blocking(repo, &filename, &self.fetch_config)
            .map_err(|e| MIError::Download(format!("failed to download {filename}: {e}")))?
            .into_inner();
        if let Some(slot) = self.decoder_paths.get_mut(layer) {
            // BORROW: explicit .clone() — store PathBuf in cache
            *slot = Some(path.clone());
        }
        Ok(path)
    }

    // --- Encoder loading (stream-and-free) ---

    /// Load a single encoder's weights to the specified device.
    ///
    /// Frees any previously loaded encoder first (stream-and-free pattern).
    /// Peak GPU overhead: ~75 MB for CLT-426K, ~450 MB for CLT-2.5M, ~144 MB
    /// for `GemmaScope`-2b at `width_16k` (one of `W_enc` / `W_dec`).
    ///
    /// File format and tensor layout depend on [`TranscoderSchema`]:
    /// - `CltSplit` / `PltBundle`: safetensors with `W_enc` already in
    ///   `[n_features, d_model]` orientation; `threshold` absent.
    /// - `GemmaScopeNpz`: `.npz` archive with `W_enc` stored as
    ///   `[d_model, n_features]` (transposed) plus a `threshold [n_features]`
    ///   tensor for `JumpReLU`. The loader transposes `W_enc` so the
    ///   in-memory layout matches the other schemas.
    ///
    /// # Arguments
    /// * `layer` — Layer index (`0..n_layers`)
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the layer is out of range, if the
    /// `GemmaScopeNpz` schema is encountered without the `sae` feature,
    /// or if a required tensor (`W_enc`, `b_enc`, or `threshold` for
    /// `GemmaScopeNpz`) is missing from the file.
    /// Returns [`MIError::Download`] if the encoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor deserialization failure.
    pub fn load_encoder(&mut self, layer: usize, device: &Device) -> Result<()> {
        if layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "layer {layer} out of range (CLT has {} layers)",
                self.config.n_layers
            )));
        }

        // Skip if already loaded.
        if let Some(ref enc) = self.loaded_encoder
            && enc.layer == layer
        {
            return Ok(());
        }

        // Drop previous encoder (frees GPU memory).
        self.loaded_encoder = None;

        info!("Loading CLT encoder for layer {layer}");

        let enc_path = self.ensure_encoder_path(layer)?;

        let loaded = match self.config.schema {
            // All safetensors-backed schemas route through the same
            // loader. `load_encoder_safetensors` conditionally reads
            // `threshold_{layer}` when `schema.is_jump_relu()` is true
            // (currently only `CltSplitJumpReLU` on the safetensors path).
            TranscoderSchema::CltSplit
            | TranscoderSchema::PltBundle
            | TranscoderSchema::CltSplitJumpReLU => {
                self.load_encoder_safetensors(&enc_path, layer, device)?
            }
            TranscoderSchema::GemmaScopeNpz => {
                #[cfg(feature = "sae")]
                {
                    Self::load_encoder_npz(&enc_path, layer, device)?
                }
                #[cfg(not(feature = "sae"))]
                {
                    return Err(MIError::Config(
                        "GemmaScope encoder loading requires the 'sae' feature \
                         (NPZ parsing via anamnesis/npz)"
                            .into(),
                    ));
                }
            }
        };

        self.loaded_encoder = Some(loaded);

        Ok(())
    }

    /// Load an encoder from a safetensors bundle (`CltSplit`, `PltBundle`,
    /// or `CltSplitJumpReLU`).
    ///
    /// Reads the file once into a `Vec<u8>`, deserialises with `safetensors`,
    /// extracts `W_enc` and `b_enc` by their schema-specific tensor names.
    /// For `is_jump_relu()` schemas (currently only `CltSplitJumpReLU` on
    /// this safetensors path; `GemmaScopeNpz` goes through
    /// [`load_encoder_npz`](Self::load_encoder_npz)) a per-feature
    /// `threshold` tensor is read from the same file. For plain-`ReLU`
    /// schemas (`CltSplit`, `PltBundle`) `threshold` is `None`.
    fn load_encoder_safetensors(
        &self,
        path: &Path,
        layer: usize,
        device: &Device,
    ) -> Result<LoadedEncoder> {
        let data = std::fs::read(path)?;
        let st = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize encoder layer {layer}: {e}"))
        })?;

        let (_, w_enc_name, b_enc_name) = encoder_file_and_tensor_names(
            self.config.schema,
            layer,
            &self.config.gemmascope_npz_paths,
        )?;

        let w_enc = crate::util::safetensors_view::tensor_from_view(
            &st.tensor(&w_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{w_enc_name}' not found: {e}")))?,
            device,
            "CLT",
        )?;
        let b_enc = crate::util::safetensors_view::tensor_from_view(
            &st.tensor(&b_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{b_enc_name}' not found: {e}")))?,
            device,
            "CLT",
        )?;

        let threshold = if self.config.schema.is_jump_relu() {
            // CltSplitJumpReLU stores `threshold_{layer}` in the same
            // file as `W_enc_{layer}` and `b_enc_{layer}` (layer-suffixed
            // naming). PROMOTE: BF16 on disk → F32 for the `pre > threshold`
            // gate, matching the F32-everywhere precision strategy.
            let threshold_name = format!("threshold_{layer}");
            if let Ok(view) = st.tensor(&threshold_name) {
                let t = crate::util::safetensors_view::tensor_from_view(&view, device, "CLT")?;
                // PROMOTE: threshold may be BF16 on disk; encode-path gate
                // needs F32 to match the F32 pre-activation tensor.
                Some(t.to_dtype(DType::F32)?)
            } else {
                // The schema was flagged CltSplitJumpReLU by the
                // `features/index.json.gz` sidecar heuristic, but this repo
                // ships a plain-ReLU encoder with no `threshold_{l}` tensor
                // (e.g. mntss/clt-gemma-2-2b-*). Treat as ReLU (no gate) rather
                // than failing the load; the encode path falls back to ReLU
                // when the threshold is `None`.
                tracing::warn!(
                    "encoder file for layer {layer} has no '{threshold_name}'; \
                     treating this CltSplitJumpReLU repo as plain ReLU"
                );
                None
            }
        } else {
            None
        };

        Ok(LoadedEncoder {
            layer,
            w_enc,
            b_enc,
            threshold,
        })
    }

    /// Load an encoder from a `GemmaScope` `.npz` bundle (`GemmaScopeNpz`).
    ///
    /// Delegates to [`crate::sae::npz::load_npz_selective`] for the NPZ →
    /// candle tensor conversion, requesting only the three tensors the
    /// encoder actually needs (`W_enc`, `b_enc`, `threshold`). Then:
    /// - Transposes `W_enc` from on-disk `[d_model, n_features]` to the
    ///   canonical `[n_features, d_model]` orientation. The transpose
    ///   produces non-unit strides, so `.contiguous()` is applied
    ///   immediately to keep the encode-path matmul efficient.
    /// - Loads the per-feature `threshold` tensor required by `JumpReLU`.
    ///
    /// The decoder-side tensors (`W_dec`, `b_dec`) are still parsed into
    /// raw byte buffers by `anamnesis::parse_npz` (a true selective read
    /// would need cross-crate work; see `anamnesis/ROADMAP.md`), but the
    /// `byte → F32 candle Tensor` conversion is skipped — saving the
    /// ~144 MiB `F32` allocation for `W_dec` per layer load.
    #[cfg(feature = "sae")]
    fn load_encoder_npz(path: &Path, layer: usize, device: &Device) -> Result<LoadedEncoder> {
        let npz_map =
            crate::sae::npz::load_npz_selective(path, &["W_enc", "b_enc", "threshold"], device)?;

        let w_enc_disk = npz_map.get("W_enc").ok_or_else(|| {
            MIError::Config(format!(
                "tensor 'W_enc' not found in {} for layer {layer}",
                path.display()
            ))
        })?;
        // CONTIGUOUS: t() yields non-unit strides; the encode-path matmul
        // requires a contiguous W_enc on the n_features-major axis
        let w_enc = w_enc_disk.t()?.contiguous()?;

        // BORROW: explicit .clone() — Tensor is Arc-backed; this hands the
        // LoadedEncoder its own handle without copying the underlying buffer.
        let b_enc = npz_map
            .get("b_enc")
            .ok_or_else(|| {
                MIError::Config(format!(
                    "tensor 'b_enc' not found in {} for layer {layer}",
                    path.display()
                ))
            })?
            .clone();

        // BORROW: explicit .clone() — same Arc-handle pattern as b_enc.
        let threshold = npz_map
            .get("threshold")
            .ok_or_else(|| {
                MIError::Config(format!(
                    "tensor 'threshold' not found in {} for layer {layer} \
                     (GemmaScope requires a JumpReLU threshold)",
                    path.display()
                ))
            })?
            .clone();

        Ok(LoadedEncoder {
            layer,
            w_enc,
            b_enc,
            threshold: Some(threshold),
        })
    }

    /// Load the `W_skip` matrix for a `PltBundle` layer and return it as a
    /// dense `F32` tensor on `device`.
    ///
    /// `W_skip` is the linear skip path that per-layer transcoders (Llama
    /// mntss/transcoder-*) apply in parallel to the sparse
    /// `ReLU(W_enc @ x + b_enc)` branch: the reconstruction is
    /// `W_skip @ x + W_dec @ sparse_features + b_dec`. Step B instrumentation
    /// projects `W_skip @ x` at the spike position onto the unembedding
    /// direction to decompose the apparent planning signal into
    /// sparse-feature vs linear-skip contributions (V3 Step 1.7).
    ///
    /// # Shapes
    /// - returns: `[d_model, d_model]` dense `F32` tensor.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the transcoder schema is not `PltBundle`
    /// (`CltSplit` files have no skip path; `GemmaScopeNpz` is unsupported and
    /// is itself gated at `open()`).
    /// Returns [`MIError::Config`] if `layer >= n_layers`.
    /// Returns [`MIError::Download`] if the bundle file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor deserialization failure.
    pub fn load_skip_matrix(&mut self, layer: usize, device: &Device) -> Result<Tensor> {
        if !matches!(self.config.schema, TranscoderSchema::PltBundle) {
            return Err(MIError::Config(format!(
                "load_skip_matrix: W_skip is only present in PltBundle schema \
                 (current schema: {:?})",
                self.config.schema,
            )));
        }
        if layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "layer {layer} out of range (transcoder has {} layers)",
                self.config.n_layers
            )));
        }
        info!("Loading W_skip for PltBundle layer {layer}");

        let path = self.ensure_encoder_path(layer)?;
        let data = std::fs::read(&path)?;
        let st = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize bundle layer {layer}: {e}"))
        })?;
        let view = st.tensor("W_skip").map_err(|e| {
            MIError::Config(format!("tensor 'W_skip' not found in layer {layer}: {e}"))
        })?;
        let w_skip = crate::util::safetensors_view::tensor_from_view(&view, device, "CLT")?;
        // PROMOTE: W_skip is BF16 on disk in mntss/transcoder-*; F32 for matmul precision
        let w_skip_f32 = w_skip.to_dtype(DType::F32)?;
        Ok(w_skip_f32)
    }

    // --- Encoding ---

    /// Encode a residual stream activation into sparse CLT features.
    ///
    /// The residual should be the "residual mid" activation at the given layer
    /// (after attention, before MLP).
    ///
    /// Returns all features that pass the activation threshold, sorted by
    /// activation magnitude in descending order. Activation depends on
    /// schema:
    /// - `CltSplit` / `PltBundle`: plain `ReLU` — features with
    ///   `pre > 0` are kept, others zeroed.
    /// - `GemmaScopeNpz` / `CltSplitJumpReLU`: `JumpReLU` — features
    ///   with `pre > threshold[i]` are kept (gated by per-feature
    ///   threshold), others zeroed.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations<CltFeatureId>`] with `(CltFeatureId, f32)` pairs
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if no encoder is loaded, the wrong layer is
    /// loaded, or the schema uses `JumpReLU` (`GemmaScopeNpz`,
    /// `CltSplitJumpReLU`) but the loaded encoder lacks a `threshold` tensor
    /// (an internal load-path mismatch).
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode(
        &self,
        residual: &Tensor,
        layer: usize,
    ) -> Result<SparseActivations<CltFeatureId>> {
        let mut features = self.compute_active_features(residual, layer)?;
        // Full sort by activation magnitude (descending). `f32::total_cmp`
        // gives a strict total ordering — handles NaN deterministically
        // (instead of the silent `Ordering::Equal` fallback that previously
        // masked numerical bugs).
        features.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(SparseActivations { features })
    }

    /// Apply the schema-specific activation, sparsify, and return the
    /// **unsorted** list of `(feature_id, activation)` pairs.
    ///
    /// Shared workhorse for [`encode`](Self::encode) (which then full-sorts)
    /// and [`top_k`](Self::top_k) (which does a partial sort instead). The
    /// activation matches the schema convention: plain `ReLU` for `CltSplit`
    /// and `PltBundle`, `JumpReLU(threshold)` for `GemmaScopeNpz` and
    /// `CltSplitJumpReLU`.
    fn compute_active_features(
        &self,
        residual: &Tensor,
        layer: usize,
    ) -> Result<Vec<(CltFeatureId, f32)>> {
        let pre_acts = self.encode_pre_activation_impl(residual, layer)?;

        // Schema-specific activation. The CltSplit/PltBundle path keeps the
        // existing `encode == relu ∘ encode_pre_activation` invariant
        // (covered by the `encode_pre_activation_matches_encode_postrelu`
        // test); the JumpReLU schemas (`GemmaScopeNpz`,
        // `CltSplitJumpReLU`) gate by per-feature threshold.
        let acts = match self.config.schema {
            TranscoderSchema::CltSplit | TranscoderSchema::PltBundle => pre_acts.relu()?,
            TranscoderSchema::GemmaScopeNpz | TranscoderSchema::CltSplitJumpReLU => {
                // Re-borrow the encoder so we can reach into `threshold`.
                // encode_pre_activation_impl already validated that an
                // encoder is loaded for this `layer`.
                let enc = self.loaded_encoder.as_ref().ok_or_else(|| {
                    MIError::Hook(
                        "encoder dropped between pre-activation and activation \
                         (internal logic error)"
                            .into(),
                    )
                })?;
                if let Some(threshold) = enc.threshold.as_ref() {
                    // JumpReLU: select pre-activation where mask is non-zero,
                    // 0 otherwise. `where_cond` fuses the mask-select into one
                    // op — avoids the U8 → F32 dtype cast and the explicit
                    // elementwise multiply the older `pre * mask` form needed.
                    let mask = pre_acts.gt(threshold)?;
                    let zeros = pre_acts.zeros_like()?;
                    mask.where_cond(&pre_acts, &zeros)?
                } else {
                    // GemmaScope must always carry a threshold; its absence is a
                    // genuine load-path mismatch (asserted by tests).
                    // CltSplitJumpReLU, by contrast, can be a plain-ReLU mntss
                    // repo mis-flagged JumpReLU by the `features/index.json.gz`
                    // sidecar heuristic (no `threshold_{l}` tensor on disk) — fall
                    // back to ReLU rather than failing.
                    if matches!(self.config.schema, TranscoderSchema::GemmaScopeNpz) {
                        return Err(MIError::Hook(
                            "JumpReLU encode requires a threshold tensor; \
                             the loaded encoder has none (load path mismatch?)"
                                .into(),
                        ));
                    }
                    pre_acts.relu()?
                }
            }
        };

        // Transfer to CPU for sparse extraction.
        let acts_vec: Vec<f32> = acts.to_vec1()?;

        let features: Vec<(CltFeatureId, f32)> = acts_vec
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v > 0.0)
            .map(|(i, v)| (CltFeatureId { layer, index: i }, *v))
            .collect();
        Ok(features)
    }

    /// Encode a residual into the dense `W_enc @ x + b_enc` pre-activation
    /// vector **before** the `ReLU` (or `JumpReLU`) sparsifier runs.
    ///
    /// Step B instrumentation histograms these pre-activation values at the
    /// spike layer and its two neighbours to discriminate activation-regime
    /// explanations across transcoders (V3 Step 1.7 (D)). Callers that only
    /// need the sparse post-activation features should use
    /// [`encode`](Self::encode).
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position.
    /// - returns: `[n_features_per_layer]` dense `F32` on the same device as
    ///   `residual` (no sparsification, includes negative values).
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if no encoder is loaded or the wrong layer is loaded.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode_pre_activation(&self, residual: &Tensor, layer: usize) -> Result<Tensor> {
        self.encode_pre_activation_impl(residual, layer)
    }

    /// Internal workhorse shared by [`encode`](Self::encode) and
    /// [`encode_pre_activation`](Self::encode_pre_activation). Returns the
    /// dense `F32` `[n_features]` pre-activation tensor; callers apply their
    /// own downstream ops (`ReLU` + sparsify, or raw histograms).
    fn encode_pre_activation_impl(&self, residual: &Tensor, layer: usize) -> Result<Tensor> {
        let enc = self.loaded_encoder.as_ref().ok_or_else(|| {
            MIError::Hook(format!(
                "no encoder loaded — call load_encoder({layer}) first"
            ))
        })?;
        if enc.layer != layer {
            return Err(MIError::Hook(format!(
                "loaded encoder is for layer {}, but layer {layer} was requested",
                enc.layer
            )));
        }

        // Compute pre-activations in F32 for numerical stability.
        // W_enc: [n_features, d_model], residual: [d_model]
        // pre_acts = W_enc @ residual + b_enc → [n_features]
        let residual_f32 = residual.flatten_all()?;
        // PROMOTE: matmul and bias add require F32 for numerical stability
        let residual_f32 = residual_f32.to_dtype(DType::F32)?;
        let w_enc_f32 = enc.w_enc.to_dtype(DType::F32)?;
        let b_enc_f32 = enc.b_enc.to_dtype(DType::F32)?;

        let pre_acts = w_enc_f32.matmul(&residual_f32.unsqueeze(1)?)?.squeeze(1)?;
        let pre_acts = (&pre_acts + &b_enc_f32)?;
        Ok(pre_acts)
    }

    /// Encode and return only the top-k most active features.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations<CltFeatureId>`] truncated to at most `k` entries
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Same as [`encode()`](Self::encode).
    pub fn top_k(
        &self,
        residual: &Tensor,
        layer: usize,
        k: usize,
    ) -> Result<SparseActivations<CltFeatureId>> {
        let mut features = self.compute_active_features(residual, layer)?;
        if k == 0 {
            return Ok(SparseActivations {
                features: Vec::new(),
            });
        }
        if features.len() <= k {
            // Already small enough — full sort is cheap and matches
            // `encode()` semantics for the result.
            features.sort_by(|a, b| b.1.total_cmp(&a.1));
            return Ok(SparseActivations { features });
        }
        // Partial sort: O(N) average via `select_nth_unstable_by` puts the
        // top-k partition at indices 0..k (unordered), then a small O(k log k)
        // sort orders just that partition for the descending-by-magnitude
        // contract. Beats `encode().truncate(k)` (O(N log N)) by a wide
        // margin for dense layers — e.g. for N≈10 000 active features and
        // k=10, ~10× fewer comparisons (~10 K vs ~133 K).
        features.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        features.truncate(k);
        features.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(SparseActivations { features })
    }

    // --- Decoder access ---

    /// Extract a single feature's decoder vector for a target downstream layer.
    ///
    /// Loads from safetensors on demand. Checks the steering cache first
    /// to avoid redundant file reads.
    ///
    /// # Shapes
    /// - returns: `[d_model]` — decoder vector on `device`
    ///
    /// # Arguments
    /// * `feature` — The CLT feature to extract the decoder for
    /// * `target_layer` — The downstream layer to decode to (must be >= feature.layer)
    /// * `device` — Device to place the resulting tensor on
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if layer indices are out of range.
    /// Returns [`MIError::Download`] if the decoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn decoder_vector(
        &mut self,
        feature: &CltFeatureId,
        target_layer: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if feature.layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "feature source layer {} out of range (CLT has {} layers)",
                feature.layer, self.config.n_layers
            )));
        }
        if target_layer < feature.layer || target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} must be >= source layer {} and < {}",
                feature.layer, self.config.n_layers
            )));
        }
        if feature.index >= self.config.n_features_per_layer {
            return Err(MIError::Config(format!(
                "feature index {} out of range (max {})",
                feature.index, self.config.n_features_per_layer
            )));
        }

        // Check steering cache first.
        let cache_key = (*feature, target_layer);
        if let Some(cached) = self.steering_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // W_dec_l has shape [n_features, n_layers - l, d_model]
        // target_offset = target_layer - feature.layer
        let target_offset = target_layer - feature.layer;

        let dec_path = self.ensure_decoder_path(feature.layer)?;
        let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, feature.layer)?;

        // Shape depends on schema: CltSplit rank-3, PltBundle/GemmaScopeNpz rank-2.
        let column = decoder_row(&w_dec, feature.index, target_offset, self.config.schema)?;

        // Transfer to target device.
        let column = column.to_device(device)?;

        Ok(column)
    }

    // --- Micro-cache ---

    /// Pre-load decoder vectors into the steering micro-cache.
    ///
    /// Each entry is a `(CltFeatureId, target_layer)` pair. Vectors are
    /// loaded to the specified device and kept pinned for repeated injection.
    ///
    /// Uses an OOM-safe pattern: loads each decoder file to CPU, extracts needed
    /// columns as independent F32 tensors, drops the large file, then moves
    /// small tensors to the target device.
    ///
    /// Memory: 50 features × 2304 × 4 bytes = ~450 KB (negligible).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn cache_steering_vectors(
        &mut self,
        features: &[(CltFeatureId, usize)],
        device: &Device,
    ) -> Result<()> {
        // Group by source layer to batch decoder file reads.
        let mut by_source: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (fid, target_layer) in features {
            by_source
                .entry(fid.layer)
                .or_default()
                .push((fid.index, *target_layer));
        }

        let mut loaded = 0_usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, entries)) in by_source.iter().enumerate() {
            info!(
                "cache_steering_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );

            // Group by target_layer to identify needed offsets.
            let mut by_target: HashMap<usize, Vec<usize>> = HashMap::new();
            for &(index, target_layer) in entries {
                by_target.entry(target_layer).or_default().push(index);
            }

            // Load decoder file, extract needed columns as independent CPU
            // tensors, then drop the large file data BEFORE any GPU transfer.
            // This prevents OOM when early-layer decoders can be >1.6 GB each.
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, *source_layer)?;
                info!(
                    "cache_steering_vectors: loaded {} MB for layer {}",
                    (w_dec.elem_count() * w_dec.dtype().size_in_bytes()) / (1024 * 1024),
                    source_layer
                );

                for (target_layer, indices) in &by_target {
                    let target_offset = target_layer - source_layer;
                    for &index in indices {
                        let fid = CltFeatureId {
                            layer: *source_layer,
                            index,
                        };
                        let cache_key = (fid, *target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            // Extract as independent F32 tensor: to_dtype +
                            // to_vec1 copies data OUT of candle's Arc storage,
                            // so dropping w_dec truly frees the ~1.6 GB decoder.
                            let view =
                                decoder_row(&w_dec, index, target_offset, self.config.schema)?;
                            let dims = view.dims().to_vec();
                            // PROMOTE: F32 for numerical stability in accumulation
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, *target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Now move the small independent columns to the target device.
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let device_tensor = cpu_tensor.to_device(device)?;
                    e.insert(device_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Cache steering vectors for ALL downstream layers of each feature.
    ///
    /// For each feature at source layer `l`, caches decoder vectors for every
    /// downstream target layer `l..n_layers`. This enables multi-layer
    /// "clamping" injection where the steering signal propagates through all
    /// downstream transformer layers.
    ///
    /// Same OOM-safe pattern as [`cache_steering_vectors()`](Self::cache_steering_vectors).
    ///
    /// # Arguments
    /// * `features` — Feature IDs to cache (all downstream layers are cached automatically)
    /// * `device` — Device to store cached tensors on (typically GPU)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any feature layer is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn cache_steering_vectors_all_downstream(
        &mut self,
        features: &[CltFeatureId],
        device: &Device,
    ) -> Result<()> {
        let n_layers = self.config.n_layers;

        // Group by source layer to batch decoder file reads.
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            if fid.layer >= n_layers {
                return Err(MIError::Config(format!(
                    "feature source layer {} out of range (max {})",
                    fid.layer,
                    n_layers - 1
                )));
            }
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut loaded = 0_usize;
        let n_source_layers = by_source.len();
        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            // CltSplit and CltSplitJumpReLU write to every downstream layer;
            // PltBundle / GemmaScopeNpz only to their own source layer.
            // Keeping this schema-aware prevents the per-layer schemas from
            // caching the same decoder row under many spurious
            // (feature, target_layer) keys.
            let n_target_layers = if self.config.schema.is_cross_layer() {
                n_layers - source_layer
            } else {
                1
            };
            info!(
                "cache_steering_vectors_all_downstream: loading decoder for source layer {} \
                 ({}/{}, {} downstream layers)",
                source_layer,
                layer_idx + 1,
                n_source_layers,
                n_target_layers
            );

            // Load decoder file, extract ALL offsets as independent CPU tensors, then drop.
            let mut cpu_columns: Vec<(CltFeatureId, usize, Tensor)> = Vec::new();
            {
                let dec_path = self.ensure_decoder_path(*source_layer)?;
                let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, *source_layer)?;
                info!(
                    "cache_steering_vectors_all_downstream: loaded {} MB for layer {}",
                    (w_dec.elem_count() * w_dec.dtype().size_in_bytes()) / (1024 * 1024),
                    source_layer
                );

                for &index in indices {
                    let fid = CltFeatureId {
                        layer: *source_layer,
                        index,
                    };
                    for target_offset in 0..n_target_layers {
                        let target_layer = source_layer + target_offset;
                        let cache_key = (fid, target_layer);
                        if !self.steering_cache.contains_key(&cache_key) {
                            let view =
                                decoder_row(&w_dec, index, target_offset, self.config.schema)?;
                            let dims = view.dims().to_vec();
                            // PROMOTE: F32 for numerical stability in accumulation
                            let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                            let independent =
                                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                            cpu_columns.push((fid, target_layer, independent));
                        }
                    }
                }
                // data, st, w_dec all drop here — freeing the large decoder file
            }

            // Move small independent columns to the target device.
            for (fid, target_layer, cpu_tensor) in cpu_columns {
                let cache_key = (fid, target_layer);
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.steering_cache.entry(cache_key)
                {
                    let device_tensor = cpu_tensor.to_device(device)?;
                    e.insert(device_tensor);
                    loaded += 1;
                }
            }
        }

        info!(
            "Cached {loaded} new steering vectors across all downstream layers ({} total in cache)",
            self.steering_cache.len()
        );
        Ok(())
    }

    /// Clear all cached steering vectors, freeing device memory.
    pub fn clear_steering_cache(&mut self) {
        let count = self.steering_cache.len();
        self.steering_cache.clear();
        if count > 0 {
            info!("Cleared {count} steering vectors from cache");
        }
    }

    /// Number of vectors currently in the steering cache.
    #[must_use]
    pub fn steering_cache_len(&self) -> usize {
        self.steering_cache.len()
    }

    /// Number of transcoder layers (source layers `0..n_layers`).
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.config.n_layers
    }

    /// Number of features per layer (feature indices `0..n_features_per_layer`).
    ///
    /// Useful for uniformly sampling a random feature index within a layer,
    /// e.g. for layer-matched random-feature controls.
    #[must_use]
    pub const fn n_features_per_layer(&self) -> usize {
        self.config.n_features_per_layer
    }

    /// Residual-stream width the transcoder decodes into.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.config.d_model
    }

    // --- Injection ---

    /// Build a [`crate::HookSpec`] that injects CLT decoder vectors into the residual stream.
    ///
    /// Groups cached steering vectors by target layer, accumulates them per layer,
    /// scales by `strength`, and creates [`crate::Intervention::Add`] entries on
    /// [`crate::HookPoint::ResidPost`] for each target layer. The resulting `HookSpec`
    /// can be passed directly to [`MIModel::forward()`](crate::MIModel::forward).
    ///
    /// # Shapes
    /// - Internally constructs `[1, seq_len, d_model]` tensors with the steering
    ///   vector placed at `position` and zeros elsewhere.
    ///
    /// # Arguments
    /// * `features` — List of `(feature_id, target_layer)` pairs (must be cached)
    /// * `position` — Token position in the sequence to inject at
    /// * `seq_len` — Total sequence length (needed to construct position-specific tensors)
    /// * `strength` — Scalar multiplier for the accumulated steering vectors
    /// * `device` — Device to construct injection tensors on
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if any feature is not in the steering cache.
    /// Returns [`MIError::Model`] on tensor construction failure.
    pub fn prepare_hook_injection(
        &self,
        features: &[(CltFeatureId, usize)],
        position: usize,
        seq_len: usize,
        strength: f32,
        device: &Device,
    ) -> Result<crate::hooks::HookSpec> {
        use crate::hooks::{HookPoint, HookSpec, Intervention};

        // Group features by target layer and accumulate their decoder vectors.
        let mut per_layer: HashMap<usize, Tensor> = HashMap::new();
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                MIError::Hook(format!(
                    "feature {feature} for target layer {target_layer} not in steering cache \
                     — call cache_steering_vectors() first"
                ))
            })?;
            // PROMOTE: accumulate in F32 for numerical stability
            let cached_f32 = cached.to_dtype(DType::F32)?;
            if let Some(acc) = per_layer.get_mut(target_layer) {
                let acc_ref: &Tensor = acc;
                *acc = (acc_ref + &cached_f32)?;
            } else {
                per_layer.insert(*target_layer, cached_f32);
            }
        }

        // Build HookSpec with Intervention::Add at each target layer.
        let mut hooks = HookSpec::new();

        for (target_layer, accumulated) in &per_layer {
            // Scale by strength, on the caller's device. The payload takes its
            // device from this tensor, so moving it here keeps the documented
            // contract that the injection lands on `device`. Previously the
            // zeros were built on `device` and `Tensor::cat` would have errored
            // on a mismatch; this is the same guarantee, stated positively.
            let scaled = (accumulated * f64::from(strength))?.to_device(device)?;

            // [1, seq_len, d_model] carrying the vector at `position`. The
            // shared helper also bounds-checks `position`, which the former
            // inline narrow/cat did not: at `position == seq_len` it silently
            // produced a `[1, seq_len + 1, d_model]` payload.
            let injection = crate::util::inject::position_delta(&scaled, position, seq_len)?;

            hooks.intervene(
                HookPoint::ResidPost(*target_layer),
                Intervention::Add(injection),
            );
        }

        Ok(hooks)
    }

    /// Inject cached steering vectors directly into a residual stream tensor.
    ///
    /// Convenience method for use outside the forward pass (e.g., in analysis
    /// scripts). Returns a new tensor with the injection applied:
    /// `residual[:, position, :] += strength × Σ decoder_vectors`
    ///
    /// # Shapes
    /// - `residual`: `[batch, seq_len, d_model]` — hidden states
    /// - returns: `[batch, seq_len, d_model]` — modified hidden states
    ///
    /// # Arguments
    /// * `residual` — Hidden states tensor
    /// * `features` — List of `(feature, target_layer)` pairs to inject (must be cached)
    /// * `position` — Token position in the sequence to inject at
    /// * `strength` — Scalar multiplier for the steering vectors
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if any feature is not in the steering cache.
    /// Returns [`MIError::Config`] if dimensions don't match.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn inject(
        &self,
        residual: &Tensor,
        features: &[(CltFeatureId, usize)],
        position: usize,
        strength: f32,
    ) -> Result<Tensor> {
        let (batch, seq_len, d_model) = residual.dims3()?;
        if position >= seq_len {
            return Err(MIError::Config(format!(
                "injection position {position} out of range (seq_len={seq_len})"
            )));
        }
        if d_model != self.config.d_model {
            return Err(MIError::Config(format!(
                "residual d_model={d_model} doesn't match CLT d_model={}",
                self.config.d_model
            )));
        }

        // Accumulate all steering vectors into one vector (F32 for stability).
        let mut accumulated = Tensor::zeros((d_model,), DType::F32, residual.device())?;
        for (feature, target_layer) in features {
            let cache_key = (*feature, *target_layer);
            let cached = self.steering_cache.get(&cache_key).ok_or_else(|| {
                MIError::Hook(format!(
                    "feature {feature} for target layer {target_layer} not in steering cache"
                ))
            })?;
            // PROMOTE: accumulate in F32 for numerical stability
            let cached_f32 = cached.to_dtype(DType::F32)?;
            accumulated = (&accumulated + &cached_f32)?;
        }

        // Scale by strength.
        let accumulated = (accumulated * f64::from(strength))?;

        // Convert to residual dtype.
        let accumulated = accumulated.to_dtype(residual.dtype())?;

        // Build steering tensor and inject at position.
        let pos_slice = residual.narrow(1, position, 1)?; // [batch, 1, d_model]
        let steering_expanded = accumulated
            .unsqueeze(0)?
            .unsqueeze(0)?
            .expand((batch, 1, d_model))?; // [batch, 1, d_model]
        let pos_updated = (&pos_slice + &steering_expanded)?;

        // Reassemble: before + updated_position + after.
        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        if position > 0 {
            parts.push(residual.narrow(1, 0, position)?);
        }
        parts.push(pos_updated);
        if position + 1 < seq_len {
            parts.push(residual.narrow(1, position + 1, seq_len - position - 1)?);
        }

        let result = Tensor::cat(&parts, 1)?;
        Ok(result)
    }

    // --- Attribution / decoder scoring ---

    /// Score all CLT features by how strongly their decoder vector at
    /// `target_layer` projects along a given direction vector.
    ///
    /// For each source layer `0..n_layers` where `source_layer <= target_layer`:
    /// loads the decoder file to CPU, extracts the target layer slice
    /// `[n_features, d_model]`, and computes `scores = slice @ direction`.
    ///
    /// When `cosine` is true, scores are normalized by both the direction
    /// vector norm and each decoder row norm (cosine similarity).
    ///
    /// # Shapes
    /// - `direction`: `[d_model]` — target direction vector (e.g., token embedding)
    /// - returns: top-k `(CltFeatureId, f32)` pairs, sorted by score descending
    ///
    /// # Arguments
    /// * `direction` — `[d_model]` direction vector to project decoders onto
    /// * `target_layer` — downstream layer to examine decoders at
    /// * `top_k` — number of top-scoring features to return
    /// * `cosine` — whether to use cosine similarity instead of dot product
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `direction` shape is wrong or `target_layer`
    /// is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Processes one decoder file at a time on CPU (up to ~2 GB for layer 0).
    /// No GPU memory required.
    pub fn score_features_by_decoder_projection(
        &mut self,
        direction: &Tensor,
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<(CltFeatureId, f32)>> {
        let d_model = self.config.d_model;
        if direction.dims() != [d_model] {
            return Err(MIError::Config(format!(
                "direction must have shape [{d_model}], got {:?}",
                direction.dims()
            )));
        }
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // PROMOTE: F32 for dot-product precision matching Python reference
        let direction_f32 = direction.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        // Optionally normalize direction to unit length for cosine similarity.
        let direction_norm = if cosine {
            let norm: f32 = direction_f32.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
            if norm > 1e-10 {
                direction_f32.broadcast_div(&Tensor::new(norm, &Device::Cpu)?)?
            } else {
                direction_f32
            }
        } else {
            direction_f32
        };

        let mut all_scores: Vec<(CltFeatureId, f32)> = Vec::new();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue; // This source layer cannot decode to target_layer.
            }
            // Per-layer schemas (PltBundle, GemmaScopeNpz) only decode to their
            // own layer. Skip any source layer that would require a non-zero
            // target_offset — otherwise decoder_layer_slice errors out below
            // when it would be silently dropping non-decoding candidates.
            if !self.config.schema.is_cross_layer() && source_layer != target_layer {
                continue;
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU.
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, source_layer)?;
            info!(
                "score_features_by_decoder_projection: loaded {} MB for layer {}",
                (w_dec.elem_count() * w_dec.dtype().size_in_bytes()) / (1024 * 1024),
                source_layer
            );
            // PROMOTE: decoder weights are BF16 on disk for safetensors schemas
            // (no-op for the F32 GemmaScope NPZ path); F32 for matmul precision.
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;

            // Extract target layer slice: [n_features, d_model]
            let dec_slice = decoder_layer_slice(&w_dec_f32, target_offset, self.config.schema)?;

            // raw_scores = dec_slice @ direction_norm → [n_features]
            let raw_scores = dec_slice
                .matmul(&direction_norm.unsqueeze(1)?)?
                .squeeze(1)?;

            let scores_vec: Vec<f32> = if cosine {
                // Divide by each decoder row's L2 norm → cosine similarity.
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?;
                let cosine_scores = raw_scores.broadcast_div(&dec_norms)?;
                cosine_scores.to_vec1()?
            } else {
                raw_scores.to_vec1()?
            };

            for (idx, &score) in scores_vec.iter().enumerate() {
                if score.is_finite() {
                    all_scores.push((
                        CltFeatureId {
                            layer: source_layer,
                            index: idx,
                        },
                        score,
                    ));
                }
            }

            info!(
                "Scored {} features at source layer {source_layer} (target layer {target_layer})",
                scores_vec.len()
            );
        }

        // Sort by score descending, take top-k.
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_scores.truncate(top_k);

        Ok(all_scores)
    }

    /// Batch version of [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection).
    ///
    /// Scores multiple direction vectors against all decoder files in a single
    /// pass. Each decoder file is loaded **once** for all directions, reducing
    /// I/O from `n_words × n_layers` file reads to just `n_layers`.
    ///
    /// # Shapes
    /// - `directions`: slice of `[d_model]` tensors (one per word/direction)
    /// - returns: one `Vec<(CltFeatureId, f32)>` per direction (top-k per word)
    ///
    /// # Arguments
    /// * `directions` — slice of `[d_model]` direction vectors
    /// * `target_layer` — downstream layer to examine decoders at
    /// * `top_k` — number of top-scoring features to return per direction
    /// * `cosine` — whether to use cosine similarity
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any direction has wrong shape, directions is
    /// empty, or `target_layer` is out of range.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Stacks directions to `[n_words, d_model]` on CPU. Each decoder file
    /// loaded one at a time (up to ~2 GB for layer 0). No GPU memory required.
    // Mirrors score_features_by_decoder_projection structure (validation →
    // per-source-layer loop → matmul → sort) at the n_words-batched cadence.
    // Extracting the per-layer body would fragment a naturally sequential
    // pipeline, so the pedantic length lint is suppressed.
    #[allow(clippy::too_many_lines)]
    pub fn score_features_by_decoder_projection_batch(
        &mut self,
        directions: &[Tensor],
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<Vec<(CltFeatureId, f32)>>> {
        let d_model = self.config.d_model;
        let n_words = directions.len();
        if n_words == 0 {
            return Err(MIError::Config(
                "at least one direction vector required".into(),
            ));
        }
        for (i, dir) in directions.iter().enumerate() {
            if dir.dims() != [d_model] {
                return Err(MIError::Config(format!(
                    "direction vector {i} must have shape [{d_model}], got {:?}",
                    dir.dims()
                )));
            }
        }
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // PROMOTE: directions may arrive as BF16; F32 for matmul precision
        let dirs_f32: Vec<Tensor> = directions
            .iter()
            .map(|d| d.to_dtype(DType::F32)?.to_device(&Device::Cpu))
            .collect::<std::result::Result<_, _>>()?;
        let stacked = Tensor::stack(&dirs_f32, 0)?; // [n_words, d_model]

        // For cosine: row-normalize direction vectors to unit length.
        let stacked_norm = if cosine {
            let norms = stacked.sqr()?.sum(1)?.sqrt()?; // [n_words]
            let ones = Tensor::ones_like(&norms)?;
            let safe_norms = norms.maximum(&(&ones * 1e-10f64)?)?; // [n_words]
            stacked.broadcast_div(&safe_norms.unsqueeze(1)?)?
        } else {
            stacked
        };
        let directions_t = stacked_norm.t()?; // [d_model, n_words]

        // Per-word score accumulators.
        let mut all_scores: Vec<Vec<(CltFeatureId, f32)>> =
            (0..n_words).map(|_| Vec::new()).collect();

        for source_layer in 0..self.config.n_layers {
            if target_layer < source_layer {
                continue;
            }
            // Per-layer schemas only decode to their own layer (same guard
            // as score_features_by_decoder_projection above).
            if !self.config.schema.is_cross_layer() && source_layer != target_layer {
                continue;
            }
            let target_offset = target_layer - source_layer;

            // Load decoder file ONCE for all words.
            let dec_path = self.ensure_decoder_path(source_layer)?;
            let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, source_layer)?;
            info!(
                "score_features_batch: loaded {} MB for layer {}",
                (w_dec.elem_count() * w_dec.dtype().size_in_bytes()) / (1024 * 1024),
                source_layer
            );
            // PROMOTE: decoder weights are BF16 on disk for safetensors schemas
            // (no-op for the F32 GemmaScope NPZ path); F32 for matmul precision.
            let w_dec_f32 = w_dec.to_dtype(DType::F32)?;
            // `[n_features, d_model]` — schema-aware slice at `target_offset`.
            let dec_slice = decoder_layer_slice(&w_dec_f32, target_offset, self.config.schema)?;

            // Batch matmul: [n_features, d_model] × [d_model, n_words] = [n_features, n_words]
            let raw_scores = dec_slice.matmul(&directions_t)?;

            // Transpose to [n_words, n_features] for easy extraction.
            let scores_2d: Vec<Vec<f32>> = if cosine {
                let dec_norms = dec_slice.sqr()?.sum(1)?.sqrt()?; // [n_features]
                let cosine_scores = raw_scores.broadcast_div(&dec_norms.unsqueeze(1)?)?;
                cosine_scores.t()?.to_vec2()?
            } else {
                raw_scores.t()?.to_vec2()?
            };

            for (w, word_scores) in scores_2d.iter().enumerate() {
                for (idx, &score) in word_scores.iter().enumerate() {
                    if score.is_finite()
                        && let Some(word_vec) = all_scores.get_mut(w)
                    {
                        word_vec.push((
                            CltFeatureId {
                                layer: source_layer,
                                index: idx,
                            },
                            score,
                        ));
                    }
                }
            }

            info!(
                "Batch scored {} words × {} features at source layer {} (target layer {})",
                n_words,
                scores_2d.first().map_or(0, Vec::len),
                source_layer,
                target_layer
            );
        }

        // Sort and truncate per word.
        for word_scores in &mut all_scores {
            word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            word_scores.truncate(top_k);
        }

        Ok(all_scores)
    }

    /// Extract decoder vectors for a set of features at a specific target layer.
    ///
    /// Groups features by source layer, loads each decoder file once, and
    /// extracts the decoder vector at the target layer offset as an independent
    /// F32 CPU tensor. Uses the OOM-safe `to_vec1` + `from_vec` pattern to
    /// ensure large decoder files are freed before processing the next layer.
    ///
    /// # Shapes
    /// - returns: `HashMap<CltFeatureId, Tensor>` where each tensor is `[d_model]` (F32, CPU)
    ///
    /// # Arguments
    /// * `features` — feature IDs to extract decoder vectors for
    /// * `target_layer` — downstream layer to extract decoders at
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if any feature layer or `target_layer` is out
    /// of range, or if `target_layer < feature.layer` for any feature.
    /// Returns [`MIError::Download`] if decoder files cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation failure.
    ///
    /// # Memory
    ///
    /// Loads each decoder to CPU (up to ~2 GB), extracts independent F32
    /// tensors, then drops the large file before processing the next layer.
    pub fn extract_decoder_vectors(
        &mut self,
        features: &[CltFeatureId],
        target_layer: usize,
    ) -> Result<HashMap<CltFeatureId, Tensor>> {
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }

        // Group by source layer.
        let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
        for fid in features {
            if fid.layer >= self.config.n_layers {
                return Err(MIError::Config(format!(
                    "feature source layer {} out of range (max {})",
                    fid.layer,
                    self.config.n_layers - 1
                )));
            }
            if target_layer < fid.layer {
                return Err(MIError::Config(format!(
                    "target layer {target_layer} must be >= source layer {}",
                    fid.layer
                )));
            }
            by_source.entry(fid.layer).or_default().push(fid.index);
        }

        let mut result: HashMap<CltFeatureId, Tensor> = HashMap::new();
        let n_source_layers = by_source.len();

        for (layer_idx, (source_layer, indices)) in by_source.iter().enumerate() {
            info!(
                "extract_decoder_vectors: loading decoder for source layer {} ({}/{})",
                source_layer,
                layer_idx + 1,
                n_source_layers
            );
            let target_offset = target_layer - source_layer;

            // Load decoder file to CPU, extract needed rows as independent tensors.
            let dec_path = self.ensure_decoder_path(*source_layer)?;
            let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, *source_layer)?;

            for &index in indices {
                let fid = CltFeatureId {
                    layer: *source_layer,
                    index,
                };
                if let std::collections::hash_map::Entry::Vacant(e) = result.entry(fid) {
                    // Extract as independent F32 tensor (OOM-safe copy).
                    let view = decoder_row(&w_dec, index, target_offset, self.config.schema)?;
                    let dims = view.dims().to_vec();
                    // PROMOTE: decoder weights are BF16 on disk for safetensors
                    // schemas (no-op for F32 GemmaScope NPZ); extract as F32.
                    let values = view.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    let independent = Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)?;
                    e.insert(independent);
                }
            }
            // w_dec drops here — freeing the large decoder tensor.
        }

        info!(
            "Extracted {} decoder vectors across {} source layers",
            result.len(),
            n_source_layers
        );

        Ok(result)
    }

    /// Extract the full decoder matrix for one `(source_layer, target_layer)`
    /// pair as a contiguous `[n_features_per_layer, d_model]` tensor on the
    /// requested device.
    ///
    /// Returns the same data as
    /// [`extract_decoder_vectors`](Self::extract_decoder_vectors) restricted
    /// to a single source layer, but packed into a single matmul-ready matrix
    /// instead of a `HashMap` of independent vectors.  Useful for batched
    /// cosine-similarity analysis against external directions
    /// (vocabulary scans, feature labelling, attribution across all features
    /// at once) where the per-feature loop in `extract_decoder_vectors`
    /// would force `n_features` small device transfers.
    ///
    /// Schema-aware: `CltSplit` / `CltSplitJumpReLU` return the
    /// `target_offset = target_layer - source_layer` slice of the rank-3
    /// decoder; `PltBundle` / `GemmaScopeNpz` accept only
    /// `target_layer == source_layer` (rank-2 decoder, single target).
    ///
    /// # Shapes
    /// - returns: `[n_features_per_layer, d_model]` `F32` on `device`
    ///
    /// # Arguments
    /// * `source_layer` — feature source layer (`0..n_layers`)
    /// * `target_layer` — downstream target layer (`source_layer..n_layers`
    ///   for cross-layer schemas; must equal `source_layer` for per-layer
    ///   schemas)
    /// * `device` — target device (`Cpu` or `Cuda`)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `source_layer`
    /// or `target_layer` is out of range, or if `target_layer < source_layer`
    /// (cross-layer schemas), or if `target_layer != source_layer` for
    /// per-layer schemas.
    /// Returns [`MIError::Download`] if the
    /// decoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor operation
    /// failure.
    ///
    /// # Memory
    ///
    /// Loads one decoder file to CPU (up to ~2 GiB for layer 0 of a 28-layer
    /// cross-layer CLT in `BF16`), slices to the requested `target_layer`,
    /// promotes to `F32`, and moves to `device`.  The large
    /// `BF16` source-tensor drops before this function returns.  Peak:
    /// ~1 decoder file + ~1 sliced + promoted matrix.
    pub fn decoder_matrix(
        &mut self,
        source_layer: usize,
        target_layer: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if source_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "source_layer {source_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }
        if target_layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "target_layer {target_layer} out of range (max {})",
                self.config.n_layers - 1
            )));
        }
        if target_layer < source_layer {
            return Err(MIError::Config(format!(
                "target_layer {target_layer} must be >= source_layer {source_layer}"
            )));
        }
        let target_offset = target_layer - source_layer;
        let dec_path = self.ensure_decoder_path(source_layer)?;
        let w_dec = load_decoder_w_dec(self.config.schema, &dec_path, source_layer)?;
        let slice = decoder_layer_slice(&w_dec, target_offset, self.config.schema)?;
        // PROMOTE: decoder weights are BF16 on disk for safetensors schemas
        // (and F32 for GemmaScopeNpz); the returned matrix is F32 to match the
        // F32-everywhere precision strategy and to be matmul-ready for
        // cosine analysis.
        let slice_f32 = slice.to_dtype(DType::F32)?;
        let slice_on_device = slice_f32.to_device(device)?;
        Ok(slice_on_device)
    }

    /// Build an attribution graph by scoring features against a direction.
    ///
    /// Convenience wrapper around
    /// [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection)
    /// that returns an [`AttributionGraph`] instead of a raw Vec.
    ///
    /// # Shapes
    /// - `direction`: `[d_model]`
    ///
    /// # Errors
    ///
    /// Same as [`score_features_by_decoder_projection`](Self::score_features_by_decoder_projection).
    pub fn build_attribution_graph(
        &mut self,
        direction: &Tensor,
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<AttributionGraph> {
        let scored =
            self.score_features_by_decoder_projection(direction, target_layer, top_k, cosine)?;
        Ok(AttributionGraph {
            target_layer,
            edges: scored
                .into_iter()
                .map(|(feature, score)| AttributionEdge { feature, score })
                .collect(),
        })
    }

    /// Build attribution graphs for multiple directions in a single pass.
    ///
    /// Convenience wrapper around
    /// [`score_features_by_decoder_projection_batch`](Self::score_features_by_decoder_projection_batch)
    /// that returns `Vec<AttributionGraph>`.
    ///
    /// # Shapes
    /// - `directions`: slice of `[d_model]` tensors
    ///
    /// # Errors
    ///
    /// Same as [`score_features_by_decoder_projection_batch`](Self::score_features_by_decoder_projection_batch).
    pub fn build_attribution_graph_batch(
        &mut self,
        directions: &[Tensor],
        target_layer: usize,
        top_k: usize,
        cosine: bool,
    ) -> Result<Vec<AttributionGraph>> {
        let batch = self.score_features_by_decoder_projection_batch(
            directions,
            target_layer,
            top_k,
            cosine,
        )?;
        Ok(batch
            .into_iter()
            .map(|scored| AttributionGraph {
                target_layer,
                edges: scored
                    .into_iter()
                    .map(|(feature, score)| AttributionEdge { feature, score })
                    .collect(),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Schema classification
// ---------------------------------------------------------------------------

/// Classify a transcoder repository's on-disk [`TranscoderSchema`] from its
/// file listing alone — no downloads, no network I/O. Pure function,
/// unit-testable.
///
/// Detection rules, checked in order:
///
/// 1. Any file matching `W_enc_*.safetensors` AND a
///    `features/index.json.gz` sidecar (the `BlueLightAI` `circuit-tracer`
///    metadata signature) → `CltSplitJumpReLU`.
/// 2. Any file matching `W_enc_*.safetensors` (without the `BlueLightAI`
///    sidecar) → `CltSplit`.
/// 3. Any file matching `layer_*.safetensors` (at repo root) → `PltBundle`.
/// 4. Any file matching `layer_N/width_Xk/average_l0_Y/params.npz` (direct
///    `google/gemma-scope-2b-pt-transcoders` layout) → `GemmaScopeNpz`.
/// 5. `config.yaml` present together with `features/layer_*.bin` and no
///    safetensors weight files (the `mntss/gemma-scope-transcoders` metadata
///    repo) → `GemmaScopeNpz`.
///
/// Rule 1 wins over Rule 2 when the `BlueLightAI` `features/index.json.gz`
/// sidecar is present — distinguishes the `JumpReLU` `BlueLightAI` variant
/// from the plain-`ReLU` mntss `CltSplit` repos which ship only
/// `W_{enc,dec}_*.safetensors` and no `features/` sidecar.
///
/// Rules 1 or 2 win over Rule 3 if both match, but no real repo carries
/// both. Rule 5 requires *no* safetensors at repo root so it doesn't
/// clash with an eventual mixed layout.
///
/// # Errors
///
/// Returns [`MIError::Config`] if none of the rules match.
fn classify_transcoder_schema(filenames: &[&str]) -> Result<TranscoderSchema> {
    let has_clt_split = filenames
        .iter()
        .any(|f| f.starts_with("W_enc_") && f.ends_with(".safetensors"));
    // `BlueLightAI` repos ship a `features/index.json.gz` sidecar
    // (circuit-tracer feature metadata); mntss CltSplit repos do not.
    // The sidecar is the cheapest reliable filename-only signal for
    // the JumpReLU variant — no HTTP-range header peek required.
    let has_bluelightai_sidecar = filenames.contains(&"features/index.json.gz");
    let has_plt_bundle = filenames
        .iter()
        .any(|f| f.starts_with("layer_") && f.ends_with(".safetensors"));
    let has_gemmascope_npz_direct = filenames.iter().any(|f| {
        f.starts_with("layer_")
            && f.contains("/width_")
            && f.contains("/average_l0_")
            && f.ends_with("/params.npz")
    });
    let has_config_yaml = filenames.contains(&"config.yaml");
    // Case-sensitive extension check — the mntss metadata repo always ships
    // lowercase `.bin`; case-insensitive would admit bogus matches.
    let has_gemmascope_bin_metadata = filenames.iter().any(|f| {
        f.starts_with("features/layer_")
            && std::path::Path::new(f)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
    });
    let has_gemmascope_metadata_repo =
        has_config_yaml && has_gemmascope_bin_metadata && !has_clt_split && !has_plt_bundle;

    if has_clt_split && has_bluelightai_sidecar {
        Ok(TranscoderSchema::CltSplitJumpReLU)
    } else if has_clt_split {
        Ok(TranscoderSchema::CltSplit)
    } else if has_plt_bundle {
        Ok(TranscoderSchema::PltBundle)
    } else if has_gemmascope_npz_direct || has_gemmascope_metadata_repo {
        Ok(TranscoderSchema::GemmaScopeNpz)
    } else {
        Err(MIError::Config(
            "unrecognised transcoder repo layout".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Schema-aware encoder/decoder helpers
// ---------------------------------------------------------------------------
//
// Introduced with the `PltBundle` schema to concentrate per-schema branching
// in four private free functions, rather than scattering `match schema { … }`
// across every encoder/decoder access site.
//
// - `encoder_file_and_tensor_names` — resolve (filename, `W_enc` name,
//   `b_enc` name) for a given layer. Used by `ensure_encoder_path`,
//   `load_encoder`, and `open()`'s first-layer dimension probe.
// - `decoder_file_and_tensor_name` — resolve (filename, `W_dec` name) for a
//   given source layer. Used by `ensure_decoder_path`.
// - `decoder_row` — extract a single `[d_model]` decoder vector for
//   `(feature_index, target_offset)`. Used by `decoder_vector`,
//   `cache_steering_vectors`, `cache_steering_vectors_all_downstream`, and
//   `extract_decoder_vectors`.
// - `decoder_layer_slice` — extract a `[n_features, d_model]` slice at a
//   given `target_offset`. Used by `score_features_by_decoder_projection`
//   and `score_features_by_decoder_projection_batch`.

/// Resolve the encoder file name and the tensor names (`W_enc`, `b_enc`)
/// inside it for a given source layer, schema-aware.
///
/// Return tuple is `(filename, W_enc_tensor_name, b_enc_tensor_name)`.
///
/// - `CltSplit`, `CltSplitJumpReLU`: `W_enc_{layer}.safetensors` +
///   `W_enc_{layer}` + `b_enc_{layer}`. `CltSplitJumpReLU` additionally
///   carries `threshold_{layer}` and `b_dec_{layer}` in the same file;
///   `threshold_{layer}` is read by the encoder loader for `JumpReLU`
///   gating, `b_dec_{layer}` is unused by candle-mi.
/// - `PltBundle`: `layer_{layer}.safetensors` + un-suffixed `W_enc` + `b_enc` —
///   bundled with `W_dec`/`W_skip`/`b_dec` in the same file.
/// - `GemmaScopeNpz`: `gemmascope_npz_paths[layer]` + un-suffixed `W_enc` +
///   `b_enc` — per-layer NPZ file, loaded via `load_encoder_npz`.
///
/// # Errors
///
/// Returns [`MIError::Config`] if `schema` is `GemmaScopeNpz` but
/// `gemmascope_npz_paths` lacks an entry for `layer`.
fn encoder_file_and_tensor_names(
    schema: TranscoderSchema,
    layer: usize,
    gemmascope_npz_paths: &[String],
) -> Result<(String, String, String)> {
    match schema {
        TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => Ok((
            format!("W_enc_{layer}.safetensors"),
            format!("W_enc_{layer}"),
            format!("b_enc_{layer}"),
        )),
        // BORROW: explicit .to_owned() — tensor names as owned Strings for uniform return type
        TranscoderSchema::PltBundle => Ok((
            format!("layer_{layer}.safetensors"),
            "W_enc".to_owned(),
            "b_enc".to_owned(),
        )),
        TranscoderSchema::GemmaScopeNpz => {
            let path = gemmascope_npz_paths.get(layer).ok_or_else(|| {
                MIError::Config(format!(
                    "GemmaScope NPZ path for layer {layer} missing (gemmascope_npz_paths has {} entries)",
                    gemmascope_npz_paths.len()
                ))
            })?;
            // BORROW: explicit .clone() — return owned path String
            // BORROW: explicit .to_owned() — tensor names as owned Strings
            Ok((path.clone(), "W_enc".to_owned(), "b_enc".to_owned()))
        }
    }
}

/// Resolve the decoder file name and the tensor name inside it for a given
/// source layer, schema-aware.
///
/// Return tuple is `(filename, W_dec_tensor_name)`.
///
/// - `CltSplit`, `CltSplitJumpReLU`: `W_dec_{layer}.safetensors` +
///   layer-suffixed `W_dec_{layer}` — encoder and decoder live in
///   separate files.
/// - `PltBundle`: `layer_{layer}.safetensors` + un-suffixed `W_dec` —
///   encoder and decoder share the bundle file.
/// - `GemmaScopeNpz`: `gemmascope_npz_paths[layer]` + un-suffixed `W_dec` —
///   per-layer NPZ file path routed via `mntss/gemma-scope-transcoders/config.yaml`.
///
/// # Errors
///
/// Returns [`MIError::Config`] if `schema` is `GemmaScopeNpz` but
/// `gemmascope_npz_paths` lacks an entry for `layer`.
fn decoder_file_and_tensor_name(
    schema: TranscoderSchema,
    layer: usize,
    gemmascope_npz_paths: &[String],
) -> Result<(String, String)> {
    match schema {
        TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => Ok((
            format!("W_dec_{layer}.safetensors"),
            format!("W_dec_{layer}"),
        )),
        // BORROW: explicit .to_owned() — tensor name as owned String for uniform return type
        TranscoderSchema::PltBundle => {
            Ok((format!("layer_{layer}.safetensors"), "W_dec".to_owned()))
        }
        TranscoderSchema::GemmaScopeNpz => {
            let path = gemmascope_npz_paths.get(layer).ok_or_else(|| {
                MIError::Config(format!(
                    "GemmaScope NPZ path for layer {layer} missing (gemmascope_npz_paths has {} entries)",
                    gemmascope_npz_paths.len()
                ))
            })?;
            // BORROW: explicit .clone() — return owned path String
            // BORROW: explicit .to_owned() — tensor name as owned String
            Ok((path.clone(), "W_dec".to_owned()))
        }
    }
}

/// Extract a single decoder row `[d_model]` for `(feature_index, target_offset)`,
/// schema-aware.
///
/// - `CltSplit`: rank-3 `W_dec` indexed as `(feature_index, target_offset)`.
/// - `PltBundle` / `GemmaScopeNpz`: rank-2 `W_dec` indexed as `feature_index`.
///   `target_offset` must be `0` — per-layer transcoders only write to their
///   own source layer.
///
/// # Shapes
/// - `w_dec`: rank-3 `[n_features, n_target_layers, d_model]` for `CltSplit`,
///   rank-2 `[n_features, d_model]` otherwise.
/// - returns: `[d_model]`
///
/// # Errors
///
/// Returns [`MIError::Config`] if `schema` is `PltBundle` or `GemmaScopeNpz`
/// and `target_offset != 0` (callers must pre-check that `target_layer == source_layer`
/// for per-layer schemas). Returns [`MIError::Model`] on candle tensor indexing
/// failure.
fn decoder_row(
    w_dec: &Tensor,
    feature_index: usize,
    target_offset: usize,
    schema: TranscoderSchema,
) -> Result<Tensor> {
    match schema {
        TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => {
            Ok(w_dec.i((feature_index, target_offset))?)
        }
        TranscoderSchema::PltBundle | TranscoderSchema::GemmaScopeNpz => {
            if target_offset != 0 {
                return Err(MIError::Config(format!(
                    "per-layer schema {schema:?} only writes to its own layer \
                     (target_offset must be 0, got {target_offset})"
                )));
            }
            Ok(w_dec.i(feature_index)?)
        }
    }
}

/// Extract a layer slice `[n_features, d_model]` from `W_dec` at a given
/// `target_offset`, schema-aware.
///
/// Used by the batch decoder-projection scorers to project all features at a
/// single target layer in one matmul.
///
/// - `CltSplit`: rank-3 `W_dec` sliced as `(.., target_offset, ..)` to drop
///   the target-layer dimension.
/// - `PltBundle` / `GemmaScopeNpz`: rank-2 `W_dec` is returned as-is (shallow
///   clone — candle tensors are Arc-backed); `target_offset` must be `0`.
///
/// # Shapes
/// - `w_dec`: rank-3 `[n_features, n_target_layers, d_model]` for `CltSplit`,
///   rank-2 `[n_features, d_model]` otherwise.
/// - returns: `[n_features, d_model]`
///
/// # Errors
///
/// Returns [`MIError::Config`] if `schema` is `PltBundle` or `GemmaScopeNpz`
/// and `target_offset != 0`. Returns [`MIError::Model`] on candle tensor
/// indexing failure.
fn decoder_layer_slice(
    w_dec: &Tensor,
    target_offset: usize,
    schema: TranscoderSchema,
) -> Result<Tensor> {
    match schema {
        TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => {
            Ok(w_dec.i((.., target_offset, ..))?)
        }
        TranscoderSchema::PltBundle | TranscoderSchema::GemmaScopeNpz => {
            if target_offset != 0 {
                return Err(MIError::Config(format!(
                    "per-layer schema {schema:?} only writes to its own layer \
                     (target_offset must be 0, got {target_offset})"
                )));
            }
            Ok(w_dec.clone())
        }
    }
}

/// Parse a value from a simple YAML file by key.
///
/// No `serde_yaml` dependency — uses line-by-line matching.
fn parse_yaml_value(yaml_text: &str, key: &str) -> Option<String> {
    for line in yaml_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key)
            && let Some(rest) = rest.strip_prefix(':')
        {
            let value = rest.trim().trim_matches('"');
            return Some(value.to_owned());
        }
    }
    None
}

/// Load the raw `W_dec` tensor for a given layer onto CPU, normalised across
/// schemas.
///
/// Reads either the safetensors decoder file (`CltSplit` / `PltBundle`) or
/// the `.npz` archive (`GemmaScopeNpz` — requires the `sae` feature) and
/// returns the on-disk `W_dec` tensor unchanged. Callers slice it via
/// [`decoder_row`] / [`decoder_layer_slice`], which already understand the
/// per-schema rank / orientation differences (rank-3
/// `[n_features, n_target_layers, d_model]` for `CltSplit`; rank-2
/// `[n_features, d_model]` for `PltBundle` and `GemmaScopeNpz`).
///
/// The dtype follows on-disk convention: `BF16` for safetensors decoders,
/// `F32` for `GemmaScope` NPZ. Callers that need `F32` for matmul precision
/// apply `.to_dtype(DType::F32)?` (a no-op for the NPZ path).
///
/// # Errors
///
/// Returns [`MIError::Config`] if the file cannot be deserialised or the
/// expected `W_dec` tensor is missing.
/// Returns [`MIError::Io`] if the file cannot be read.
/// For `GemmaScopeNpz` without the `sae` feature, returns
/// [`MIError::Config`] explaining the feature gate.
fn load_decoder_w_dec(schema: TranscoderSchema, path: &Path, layer: usize) -> Result<Tensor> {
    match schema {
        // CltSplit and CltSplitJumpReLU share the layer-suffixed
        // `W_dec_{layer}` tensor-name and per-layer safetensors file
        // convention; the JumpReLU vs plain-ReLU distinction lives in
        // the encoder, not the decoder.
        TranscoderSchema::CltSplit | TranscoderSchema::CltSplitJumpReLU => {
            load_w_dec_safetensors(path, &format!("W_dec_{layer}"), layer)
        }
        TranscoderSchema::PltBundle => load_w_dec_safetensors(path, "W_dec", layer),
        TranscoderSchema::GemmaScopeNpz => {
            #[cfg(feature = "sae")]
            {
                load_w_dec_npz(path, layer)
            }
            #[cfg(not(feature = "sae"))]
            {
                let _ = (path, layer);
                Err(MIError::Config(
                    "GemmaScope decoder access requires the 'sae' feature \
                     (NPZ parsing via anamnesis/npz)"
                        .into(),
                ))
            }
        }
    }
}

/// Helper for the safetensors branches of [`load_decoder_w_dec`].
fn load_w_dec_safetensors(path: &Path, tensor_name: &str, layer: usize) -> Result<Tensor> {
    let data = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&data).map_err(|e| {
        MIError::Config(format!("failed to deserialize decoder layer {layer}: {e}"))
    })?;
    crate::util::safetensors_view::tensor_from_view(
        &st.tensor(tensor_name)
            .map_err(|e| MIError::Config(format!("tensor '{tensor_name}' not found: {e}")))?,
        &Device::Cpu,
        "CLT",
    )
}

/// Helper for the `GemmaScope` branch of [`load_decoder_w_dec`].
///
/// Delegates to [`crate::sae::npz::load_npz_selective`] so only the
/// `W_dec` tensor is converted to a candle [`Tensor`] — the encoder-side
/// tensors and `b_dec` are read into raw byte buffers by `parse_npz` but
/// not promoted to `F32` candle tensors.
#[cfg(feature = "sae")]
fn load_w_dec_npz(path: &Path, layer: usize) -> Result<Tensor> {
    let npz_map = crate::sae::npz::load_npz_selective(path, &["W_dec"], &Device::Cpu)?;
    let w_dec = npz_map.get("W_dec").ok_or_else(|| {
        MIError::Config(format!(
            "tensor 'W_dec' not found in {} for layer {layer}",
            path.display()
        ))
    })?;
    // BORROW: explicit .clone() — Tensor is Arc-backed; cheap shared handle.
    Ok(w_dec.clone())
}

/// Read `(n_features_per_layer, d_model)` from a `GemmaScope` `.npz` file
/// by inspecting the `W_enc` tensor shape.
///
/// `GemmaScope`'s on-disk `W_enc` shape is `[d_model, n_features]` —
/// transposed vs the `[n_features, d_model]` convention used by the
/// `CltSplit` and `PltBundle` schemas. The first dimension is therefore
/// `d_model`, the second is `n_features_per_layer`. The encoder loader
/// re-applies the transpose at load time so downstream matmul code stays
/// schema-agnostic.
///
/// Uses `anamnesis::inspect_npz` rather than `parse_npz` so the call only
/// reads the `.npz` central directory (~kB) instead of materialising every
/// tensor (~288 MiB at `width_16k`). Open-time shape probe is metadata-only.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the `.npz` cannot be parsed, if `W_enc`
/// is missing from the archive, or if `W_enc` is not a 2D tensor.
#[cfg(feature = "sae")]
fn read_gemmascope_npz_shape(npz_path: &Path) -> Result<(usize, usize)> {
    let info = anamnesis::inspect_npz(npz_path)?;
    let w_enc = info
        .tensors
        .iter()
        .find(|t| t.name == "W_enc")
        .ok_or_else(|| {
            MIError::Config(format!(
                "tensor 'W_enc' not found in {}",
                npz_path.display()
            ))
        })?;
    if w_enc.shape.len() != 2 {
        return Err(MIError::Config(format!(
            "expected 2D W_enc, got shape {:?} in {}",
            w_enc.shape,
            npz_path.display()
        )));
    }
    let d_model = *w_enc
        .shape
        .first()
        .ok_or_else(|| MIError::Config("W_enc shape is empty".into()))?;
    let n_features_per_layer = *w_enc
        .shape
        .get(1)
        .ok_or_else(|| MIError::Config("W_enc shape has fewer than 2 dimensions".into()))?;
    Ok((n_features_per_layer, d_model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn clt_feature_id_display() {
        let fid = CltFeatureId {
            layer: 5,
            index: 42,
        };
        assert_eq!(fid.to_string(), "L5:42");
    }

    #[test]
    fn clt_feature_id_ordering() {
        let a = CltFeatureId {
            layer: 0,
            index: 10,
        };
        let b = CltFeatureId {
            layer: 0,
            index: 20,
        };
        let c = CltFeatureId { layer: 1, index: 0 };
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn sparse_activations_basics() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let sparse = SparseActivations { features };
        assert_eq!(sparse.len(), 3);
        assert!(!sparse.is_empty());
    }

    #[test]
    fn sparse_activations_truncate() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let mut sparse = SparseActivations { features };
        sparse.truncate(2);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.features[0].0.index, 5);
        assert_eq!(sparse.features[1].0.index, 2);
    }

    #[test]
    fn parse_yaml_value_basic() {
        let yaml = "model_name: \"google/gemma-2-2b\"\nmodel_kind: cross_layer_transcoder\n";
        assert_eq!(
            parse_yaml_value(yaml, "model_name"),
            Some("google/gemma-2-2b".to_owned())
        );
        assert_eq!(
            parse_yaml_value(yaml, "model_kind"),
            Some("cross_layer_transcoder".to_owned())
        );
        assert_eq!(parse_yaml_value(yaml, "missing_key"), None);
    }

    #[test]
    fn encode_synthetic() {
        // Create a small synthetic encoder: 4 features, d_model=8
        let device = Device::Cpu;
        let d_model = 8;
        let n_features = 4;

        // W_enc: [4, 8] — identity-like rows so we can predict output
        #[rustfmt::skip]
        let w_enc_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 0: picks up residual[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 1: picks up residual[1]
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 2: picks up residual[2]
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // feature 3: picks up residual[3]
        ];
        let w_enc = Tensor::from_vec(w_enc_data, (n_features, d_model), &device).unwrap();

        // b_enc: [4] — bias shifts to test ReLU
        let b_enc_data: Vec<f32> = vec![0.0, -0.5, 0.0, -2.0]; // feature 3 will need residual[3] > 2.0
        let b_enc = Tensor::from_vec(b_enc_data, (n_features,), &device).unwrap();

        // Residual: [8] — values: [1.5, 0.3, 0.0, 1.0, ...]
        let residual_data: Vec<f32> = vec![1.5, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let residual = Tensor::from_vec(residual_data, (d_model,), &device).unwrap();

        // Expected pre_acts = W_enc @ residual + b_enc
        // = [1.5, 0.3, 0.0, 1.0] + [0.0, -0.5, 0.0, -2.0]
        // = [1.5, -0.2, 0.0, -1.0]
        // After ReLU: [1.5, 0.0, 0.0, 0.0]
        // Only feature 0 is active with activation 1.5

        // Create a fake loaded encoder
        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
                threshold: None,
            }),
            steering_cache: HashMap::new(),
        };

        let sparse = clt.encode(&residual, 0).unwrap();
        assert_eq!(sparse.len(), 1, "only feature 0 should be active");
        assert_eq!(sparse.features[0].0.index, 0);
        assert!((sparse.features[0].1 - 1.5).abs() < 1e-5);
    }

    #[test]
    fn encode_wrong_layer_errors() {
        let device = Device::Cpu;
        let w_enc = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let b_enc = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let residual = Tensor::zeros((8,), DType::F32, &device).unwrap();

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![None; 2],
            config: CltConfig {
                n_layers: 2,
                d_model: 8,
                n_features_per_layer: 4,
                n_features_total: 8,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
                threshold: None,
            }),
            steering_cache: HashMap::new(),
        };

        // Requesting layer 1 when layer 0 is loaded should error.
        let result = clt.encode(&residual, 1);
        assert!(result.is_err());
    }

    #[test]
    fn encode_gemmascope_jump_relu_gates_below_threshold() {
        // Synthetic GemmaScopeNpz encoder: feature i picks up residual[i].
        // Per-feature thresholds force features 2 and 3 to be gated out.
        let device = Device::Cpu;
        let d_model = 8;
        let n_features = 4;

        // W_enc: [4, 8] — identity-like rows.
        #[rustfmt::skip]
        let w_enc_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 0: residual[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 1: residual[1]
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 2: residual[2]
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // feature 3: residual[3]
        ];
        let w_enc = Tensor::from_vec(w_enc_data, (n_features, d_model), &device).unwrap();
        let b_enc = Tensor::zeros((n_features,), DType::F32, &device).unwrap();
        // Per-feature JumpReLU thresholds.
        let threshold =
            Tensor::from_vec(vec![0.5_f32, 1.0, 1.5, 2.0], (n_features,), &device).unwrap();
        // Residual: features 0–3 see pre-activation 1.5 each.
        // Gating: pre > threshold → features 0 (1.5 > 0.5) and 1 (1.5 > 1.0) pass;
        //         feature 2 (1.5 > 1.5 is false) and feature 3 (1.5 > 2.0 false) fail.
        let residual_data: Vec<f32> = vec![1.5, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0];
        let residual = Tensor::from_vec(residual_data, (d_model,), &device).unwrap();

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 1],
            decoder_paths: vec![None; 1],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::GemmaScopeNpz,
                gemmascope_npz_paths: vec!["dummy/path/params.npz".to_owned()],
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
                threshold: Some(threshold),
            }),
            steering_cache: HashMap::new(),
        };

        let sparse = clt.encode(&residual, 0).unwrap();
        assert_eq!(
            sparse.len(),
            2,
            "only features above their per-feature threshold should pass"
        );
        let indices: Vec<usize> = sparse.features.iter().map(|(f, _)| f.index).collect();
        assert!(indices.contains(&0), "feature 0 (1.5 > 0.5) should pass");
        assert!(indices.contains(&1), "feature 1 (1.5 > 1.0) should pass");
        assert!(
            !indices.contains(&2),
            "feature 2 (1.5 not > 1.5) should be gated"
        );
        assert!(
            !indices.contains(&3),
            "feature 3 (1.5 not > 2.0) should be gated"
        );
    }

    #[test]
    fn encode_gemmascope_without_threshold_returns_clear_error() {
        // Defensive test: if a GemmaScopeNpz encoder somehow ends up with
        // threshold = None (load-path bug), encode() must fail with a
        // helpful message instead of producing silently wrong output.
        let device = Device::Cpu;
        let w_enc = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let b_enc = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let residual = Tensor::zeros((8,), DType::F32, &device).unwrap();

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 1],
            decoder_paths: vec![None; 1],
            config: CltConfig {
                n_layers: 1,
                d_model: 8,
                n_features_per_layer: 4,
                n_features_total: 4,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::GemmaScopeNpz,
                gemmascope_npz_paths: vec!["dummy".to_owned()],
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
                threshold: None, // intentional load-path mismatch
            }),
            steering_cache: HashMap::new(),
        };
        let err = clt.encode(&residual, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("threshold"),
            "error message must mention threshold; got: {msg}"
        );
    }

    #[test]
    fn inject_position() {
        let device = Device::Cpu;
        let d_model = 4;

        // Residual: [1, 3, 4] — batch=1, seq_len=3, d_model=4
        let residual = Tensor::ones((1, 3, d_model), DType::F32, &device).unwrap();

        // Create a CLT with a pre-cached steering vector.
        let fid = CltFeatureId { layer: 0, index: 0 };
        let target_layer = 1;
        let steering_vec =
            Tensor::from_vec(vec![10.0_f32, 20.0, 30.0, 40.0], (d_model,), &device).unwrap();

        let mut steering_cache = HashMap::new();
        steering_cache.insert((fid, target_layer), steering_vec);

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![None; 2],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: 1,
                n_features_total: 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache,
        };

        // Inject at position 1 with strength 1.0
        let result = clt
            .inject(&residual, &[(fid, target_layer)], 1, 1.0)
            .unwrap();

        // Position 0 should be unchanged (all 1.0)
        let pos0: Vec<f32> = result.i((0, 0)).unwrap().to_vec1().unwrap();
        assert_eq!(pos0, vec![1.0, 1.0, 1.0, 1.0]);

        // Position 1 should have the steering vector added (1 + [10, 20, 30, 40])
        let pos1: Vec<f32> = result.i((0, 1)).unwrap().to_vec1().unwrap();
        assert_eq!(pos1, vec![11.0, 21.0, 31.0, 41.0]);

        // Position 2 should be unchanged
        let pos2: Vec<f32> = result.i((0, 2)).unwrap().to_vec1().unwrap();
        assert_eq!(pos2, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn prepare_hook_injection_creates_correct_hooks() {
        use crate::hooks::HookPoint;

        let device = Device::Cpu;
        let d_model = 4;

        let fid = CltFeatureId { layer: 0, index: 0 };
        let target_layer = 5;
        let steering_vec =
            Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (d_model,), &device).unwrap();

        let mut steering_cache = HashMap::new();
        steering_cache.insert((fid, target_layer), steering_vec);

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 10],
            decoder_paths: vec![None; 10],
            config: CltConfig {
                n_layers: 10,
                d_model,
                n_features_per_layer: 1,
                n_features_total: 10,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache,
        };

        let hooks = clt
            .prepare_hook_injection(&[(fid, target_layer)], 2, 5, 1.0, &device)
            .unwrap();

        // Should have an intervention at ResidPost(5).
        assert!(hooks.has_intervention_at(&HookPoint::ResidPost(target_layer)));
        // Should NOT have interventions at other layers.
        assert!(!hooks.has_intervention_at(&HookPoint::ResidPost(0)));
        assert!(!hooks.has_intervention_at(&HookPoint::ResidPost(4)));
    }

    // ====================================================================
    // Attribution graph — pure type tests
    // ====================================================================

    #[test]
    fn attribution_edge_basics() {
        let edge = AttributionEdge {
            feature: CltFeatureId {
                layer: 3,
                index: 42,
            },
            score: 0.75,
        };
        assert_eq!(edge.feature.layer, 3);
        assert_eq!(edge.feature.index, 42);
        assert!((edge.score - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn attribution_graph_empty() {
        let graph = AttributionGraph {
            target_layer: 5,
            edges: Vec::new(),
        };
        assert_eq!(graph.target_layer(), 5);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(graph.features().is_empty());
        assert!(graph.into_edges().is_empty());
    }

    #[test]
    fn attribution_graph_top_k() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 0 },
                score: 5.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 1 },
                score: 3.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 0 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 1 },
                score: -1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 0 },
                score: -4.0,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 3,
            edges,
        };

        assert_eq!(graph.len(), 5);

        let top3 = graph.top_k(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3.target_layer(), 3);
        assert!((top3.edges()[0].score - 5.0).abs() < f32::EPSILON);
        assert!((top3.edges()[1].score - 3.0).abs() < f32::EPSILON);
        assert!((top3.edges()[2].score - 1.0).abs() < f32::EPSILON);

        // top_k larger than graph size returns all edges.
        let top10 = graph.top_k(10);
        assert_eq!(top10.len(), 5);
    }

    #[test]
    fn attribution_graph_threshold() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 0 },
                score: 5.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 1 },
                score: 3.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 0 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 1, index: 1 },
                score: -1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 0 },
                score: -4.0,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 3,
            edges,
        };

        // Threshold at 2.0 keeps |score| >= 2.0: 5.0, 3.0, -4.0
        let pruned = graph.threshold(2.0);
        assert_eq!(pruned.len(), 3);
        assert!((pruned.edges()[0].score - 5.0).abs() < f32::EPSILON);
        assert!((pruned.edges()[1].score - 3.0).abs() < f32::EPSILON);
        assert!((pruned.edges()[2].score - -4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn attribution_graph_features() {
        let edges = vec![
            AttributionEdge {
                feature: CltFeatureId { layer: 2, index: 7 },
                score: 1.0,
            },
            AttributionEdge {
                feature: CltFeatureId { layer: 0, index: 3 },
                score: 0.5,
            },
        ];
        let graph = AttributionGraph {
            target_layer: 5,
            edges,
        };

        let features = graph.features();
        assert_eq!(features.len(), 2);
        assert_eq!(features[0], CltFeatureId { layer: 2, index: 7 });
        assert_eq!(features[1], CltFeatureId { layer: 0, index: 3 });
    }

    // ====================================================================
    // Attribution graph — synthetic decoder file tests
    // ====================================================================

    /// Create a synthetic decoder safetensors file and return its path.
    fn create_synthetic_decoder(
        dir: &std::path::Path,
        layer: usize,
        n_features: usize,
        n_target_layers: usize,
        d_model: usize,
        values: &[f32],
    ) -> PathBuf {
        assert_eq!(values.len(), n_features * n_target_layers * d_model);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let name = format!("W_dec_{layer}");
        let shape = vec![n_features, n_target_layers, d_model];
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, &bytes).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(name, view);
        let serialized = safetensors::serialize(&tensors, &None).unwrap();
        let path = dir.join(format!("W_dec_{layer}.safetensors"));
        std::fs::write(&path, serialized).unwrap();
        path
    }

    #[test]
    fn score_decoder_projection_synthetic() {
        // 2 layers, 4 features/layer, d_model=4.
        // Layer 0 can decode to layers 0 and 1. Layer 1 can decode to layer 1.
        // Target layer = 1.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 4;

        // W_dec_0: [4 features, 2 target_layers, 4 d_model]
        // Feature 0, offset 1 (target layer 1): [1, 0, 0, 0]
        // Feature 1, offset 1: [0, 1, 0, 0]
        // Feature 2, offset 1: [0, 0, 1, 0]
        // Feature 3, offset 1: [0, 0, 0, 1]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            0.0, 0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0,
            // feature 1
            0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,
            // feature 2
            0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0,
            // feature 3
            0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        // W_dec_1: [4 features, 1 target_layer, 4 d_model]
        // Feature 0, offset 0 (target layer 1): [2, 0, 0, 0]  (strong on dim 0)
        // Feature 1: [0, 0, 0, 0]
        // Feature 2: [0, 0, 0, 0]
        // Feature 3: [0, 3, 0, 0]  (strong on dim 1)
        #[rustfmt::skip]
        let dec1_values: Vec<f32> = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
        ];
        let path1 = create_synthetic_decoder(dir.path(), 1, n_features, 1, d_model, &dec1_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![Some(path0), Some(path1)],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // Direction: [1, 0, 0, 0] — should pick up L0:0 (score=1) and L1:0 (score=2).
        let direction =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let scores = clt
            .score_features_by_decoder_projection(&direction, 1, 10, false)
            .unwrap();

        // Top scorer should be L1:0 (score=2), then L0:0 (score=1).
        assert!(scores.len() >= 2, "expected at least 2 non-zero scores");
        assert_eq!(scores[0].0, CltFeatureId { layer: 1, index: 0 });
        assert!((scores[0].1 - 2.0).abs() < 1e-5);
        assert_eq!(scores[1].0, CltFeatureId { layer: 0, index: 0 });
        assert!((scores[1].1 - 1.0).abs() < 1e-5);

        // Direction: [0, 1, 0, 0] — should pick up L1:3 (score=3) and L0:1 (score=1).
        let direction2 =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let scores2 = clt
            .score_features_by_decoder_projection(&direction2, 1, 10, false)
            .unwrap();

        assert_eq!(scores2[0].0, CltFeatureId { layer: 1, index: 3 });
        assert!((scores2[0].1 - 3.0).abs() < 1e-5);
        assert_eq!(scores2[1].0, CltFeatureId { layer: 0, index: 1 });
        assert!((scores2[1].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn score_decoder_projection_cosine_synthetic() {
        // Same setup: verify cosine normalization.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // W_dec_0: [2 features, 1 target_layer, 4 d_model]
        // Feature 0: [3, 0, 0, 0]  (length 3, aligned with [1,0,0,0])
        // Feature 1: [1, 1, 0, 0]  (length sqrt(2), partially aligned)
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            3.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let direction =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        // Dot product: feature 0 = 3.0, feature 1 = 1.0.
        let dot_scores = clt
            .score_features_by_decoder_projection(&direction, 0, 10, false)
            .unwrap();
        assert!((dot_scores[0].1 - 3.0).abs() < 1e-5);
        assert!((dot_scores[1].1 - 1.0).abs() < 1e-5);

        // Cosine: feature 0 = 1.0 (perfectly aligned), feature 1 = 1/sqrt(2) ≈ 0.707.
        let cos_scores = clt
            .score_features_by_decoder_projection(&direction, 0, 10, true)
            .unwrap();
        assert!(
            (cos_scores[0].1 - 1.0).abs() < 1e-4,
            "expected ~1.0, got {}",
            cos_scores[0].1
        );
        let expected_cos = 1.0 / 2.0_f32.sqrt();
        assert!(
            (cos_scores[1].1 - expected_cos).abs() < 1e-4,
            "expected ~{expected_cos}, got {}",
            cos_scores[1].1
        );
    }

    #[test]
    fn score_decoder_projection_batch_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // W_dec_0: feature 0 = [1,0,0,0], feature 1 = [0,1,0,0]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // Two directions: [1,0,0,0] and [0,1,0,0].
        let dir0 =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();
        let dir1 =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let batch = clt
            .score_features_by_decoder_projection_batch(&[dir0, dir1], 0, 10, false)
            .unwrap();

        assert_eq!(batch.len(), 2);

        // Direction 0 should score feature 0 highest.
        assert_eq!(batch[0][0].0, CltFeatureId { layer: 0, index: 0 });
        assert!((batch[0][0].1 - 1.0).abs() < 1e-5);

        // Direction 1 should score feature 1 highest.
        assert_eq!(batch[1][0].0, CltFeatureId { layer: 0, index: 1 });
        assert!((batch[1][0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn extract_decoder_vectors_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 3;

        // W_dec_0: [3 features, 2 target_layers, 4 d_model]
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            // feature 1
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
            // feature 2
            17.0, 18.0, 19.0, 20.0,  21.0, 22.0, 23.0, 24.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![Some(path0), None],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let features = vec![
            CltFeatureId { layer: 0, index: 0 },
            CltFeatureId { layer: 0, index: 2 },
        ];

        // Extract at target_layer=1 (offset 1 for source layer 0).
        let vectors = clt.extract_decoder_vectors(&features, 1).unwrap();
        assert_eq!(vectors.len(), 2);

        // Feature 0, offset 1: [5, 6, 7, 8]
        let v0: Vec<f32> = vectors[&CltFeatureId { layer: 0, index: 0 }]
            .to_vec1()
            .unwrap();
        assert_eq!(v0, vec![5.0, 6.0, 7.0, 8.0]);

        // Feature 2, offset 1: [21, 22, 23, 24]
        let v2: Vec<f32> = vectors[&CltFeatureId { layer: 0, index: 2 }]
            .to_vec1()
            .unwrap();
        assert_eq!(v2, vec![21.0, 22.0, 23.0, 24.0]);
    }

    #[test]
    fn decoder_matrix_clt_split_returns_contiguous_slice() {
        // Mirrors `extract_decoder_vectors_synthetic`, but checks the
        // contiguous-matrix accessor: the returned tensor should have shape
        // `[n_features, d_model]` and contain the full target-offset slice
        // for the given (source_layer, target_layer) pair, F32 on the
        // requested device.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 3;

        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
            // feature 1
            9.0, 10.0, 11.0, 12.0,  13.0, 14.0, 15.0, 16.0,
            // feature 2
            17.0, 18.0, 19.0, 20.0,  21.0, 22.0, 23.0, 24.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            decoder_paths: vec![Some(path0), None],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // source_layer=0, target_layer=1 → target_offset=1, which is
        // [feature, 1, d_model] = the second 4-element block of each feature.
        let matrix = clt.decoder_matrix(0, 1, &Device::Cpu).unwrap();
        assert_eq!(matrix.dims(), &[n_features, d_model]);
        assert_eq!(matrix.dtype(), DType::F32);

        let flat: Vec<f32> = matrix.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            flat,
            vec![
                5.0, 6.0, 7.0, 8.0, // feature 0, offset 1
                13.0, 14.0, 15.0, 16.0, // feature 1, offset 1
                21.0, 22.0, 23.0, 24.0, // feature 2, offset 1
            ]
        );

        // target_layer < source_layer is rejected.
        let err = clt.decoder_matrix(1, 0, &Device::Cpu);
        assert!(err.is_err(), "target_layer < source_layer must error");
    }

    #[test]
    fn build_attribution_graph_synthetic() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 1, d_model, &dec0_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![Some(path0)],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let direction =
            Tensor::from_vec(vec![0.0_f32, 1.0, 0.0, 0.0], (d_model,), &Device::Cpu).unwrap();

        let graph = clt
            .build_attribution_graph(&direction, 0, 10, false)
            .unwrap();

        assert_eq!(graph.target_layer(), 0);
        assert!(!graph.is_empty());
        // Feature 1 has score 2.0, feature 0 has score 0.0.
        assert_eq!(
            graph.edges()[0].feature,
            CltFeatureId { layer: 0, index: 1 }
        );
        assert!((graph.edges()[0].score - 2.0).abs() < 1e-5);

        // Pruning: threshold at 1.0 should keep only feature 1.
        let pruned = graph.threshold(1.0);
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned.features()[0], CltFeatureId { layer: 0, index: 1 });
    }

    // ====================================================================
    // Schema classification (v0.1.9)
    // ====================================================================

    #[test]
    fn classify_clt_split_layout() {
        let files = [
            "W_enc_0.safetensors",
            "W_enc_1.safetensors",
            "W_dec_0.safetensors",
            "W_dec_1.safetensors",
            "config.yaml",
        ];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::CltSplit);
        assert!(schema.is_cross_layer());
        assert!(!schema.is_jump_relu());
    }

    #[test]
    fn classify_plt_bundle_layout() {
        // mntss/transcoder-Llama-3.2-1B + mwhanna/qwen3-*-transcoders*: bundle per layer.
        let files = [
            "layer_0.safetensors",
            "layer_1.safetensors",
            "features/layer_0.bin", // feature-dashboard metadata, irrelevant for weights
            "features/layer_1.bin",
            "README.md",
        ];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::PltBundle);
        assert!(!schema.is_cross_layer());
        assert!(!schema.is_jump_relu());
    }

    #[test]
    fn classify_gemmascope_metadata_repo() {
        // mntss/gemma-scope-transcoders layout: config.yaml + features/layer_*.bin
        // and NO weight safetensors at repo root — weights live in the google repo.
        let files = [
            "config.yaml",
            "features/index.json.gz",
            "features/layer_0.bin",
            "features/layer_1.bin",
        ];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::GemmaScopeNpz);
        assert!(!schema.is_cross_layer());
        assert!(schema.is_jump_relu());
    }

    #[test]
    fn classify_gemmascope_npz_direct() {
        // google/gemma-scope-2b-pt-transcoders layout: per-layer NPZ files.
        let files = [
            "layer_0/width_16k/average_l0_100/params.npz",
            "layer_1/width_16k/average_l0_105/params.npz",
            "layer_2/width_16k/average_l0_108/params.npz",
        ];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::GemmaScopeNpz);
    }

    #[test]
    fn classify_unrecognised_layout_errors() {
        let files = ["random_file.txt", "README.md"];
        let err = classify_transcoder_schema(&files).unwrap_err();
        match err {
            MIError::Config(msg) => assert!(
                msg.contains("unrecognised transcoder repo layout"),
                "unexpected error message: {msg}"
            ),
            other => panic!("expected MIError::Config, got {other:?}"),
        }
    }

    #[test]
    fn classify_empty_listing_errors() {
        let files: [&str; 0] = [];
        let err = classify_transcoder_schema(&files).unwrap_err();
        assert!(matches!(err, MIError::Config(_)));
    }

    #[test]
    fn classify_prefers_clt_split_over_plt_bundle() {
        // If somehow both signatures coexist, CltSplit wins (not observed in the wild,
        // but the detection rules give it priority).
        let files = ["W_enc_0.safetensors", "layer_0.safetensors"];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::CltSplit);
    }

    #[test]
    fn classify_bin_files_alone_are_not_enough() {
        // Without config.yaml, bare features/layer_*.bin files should not trigger
        // the GemmaScope metadata-repo path.
        let files = ["features/layer_0.bin", "features/layer_1.bin"];
        assert!(classify_transcoder_schema(&files).is_err());
    }

    #[test]
    fn classify_clt_split_jump_relu_layout() {
        // bluelightai/clt-qwen3-1.7b-base-20k layout: CltSplit-style
        // per-layer safetensors PLUS a `features/index.json.gz`
        // circuit-tracer sidecar that mntss CltSplit repos don't ship.
        let files = [
            "W_enc_0.safetensors",
            "W_enc_1.safetensors",
            "W_dec_0.safetensors",
            "W_dec_1.safetensors",
            "config.yaml",
            "features/index.json.gz",
            "features/layer_0.bin",
            "features/layer_1.bin",
        ];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::CltSplitJumpReLU);
        assert!(schema.is_cross_layer());
        assert!(schema.is_jump_relu());
    }

    #[test]
    fn classify_clt_split_without_sidecar_stays_plain() {
        // mntss/clt-* CltSplit repos ship W_enc_/W_dec_ safetensors with
        // no circuit-tracer sidecar; classification must still resolve
        // to plain CltSplit (plain ReLU activation) and NOT be mistaken
        // for the JumpReLU variant.
        let files = ["W_enc_0.safetensors", "W_dec_0.safetensors", "config.yaml"];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::CltSplit);
        assert!(!schema.is_jump_relu());
    }

    #[test]
    fn classify_prefers_jump_relu_over_plain_clt_split_when_sidecar_present() {
        // If both signatures match (W_enc_*.safetensors AND the
        // features/index.json.gz sidecar), the JumpReLU variant wins —
        // this is the `BlueLightAI` case. Rule ordering, not file count.
        let files = ["W_enc_0.safetensors", "features/index.json.gz"];
        let schema = classify_transcoder_schema(&files).unwrap();
        assert_eq!(schema, TranscoderSchema::CltSplitJumpReLU);
    }

    // ====================================================================
    // Schema-aware helper direct tests
    // ====================================================================

    #[test]
    fn encoder_file_and_tensor_names_clt_split() {
        let (filename, w_enc_name, b_enc_name) =
            encoder_file_and_tensor_names(TranscoderSchema::CltSplit, 7, &[]).unwrap();
        assert_eq!(filename, "W_enc_7.safetensors");
        assert_eq!(w_enc_name, "W_enc_7");
        assert_eq!(b_enc_name, "b_enc_7");
    }

    #[test]
    fn encoder_file_and_tensor_names_plt_bundle() {
        let (filename, w_enc_name, b_enc_name) =
            encoder_file_and_tensor_names(TranscoderSchema::PltBundle, 3, &[]).unwrap();
        assert_eq!(filename, "layer_3.safetensors");
        assert_eq!(w_enc_name, "W_enc");
        assert_eq!(b_enc_name, "b_enc");
    }

    #[test]
    fn encoder_file_and_tensor_names_clt_split_jump_relu() {
        // CltSplitJumpReLU shares the layer-suffixed file/tensor naming
        // with plain CltSplit; the helper must return the same triple.
        // The additional `threshold_{layer}` tensor lives in the same
        // file but is not surfaced by this helper (it's read directly
        // by `load_encoder_safetensors`).
        let (filename, w_enc_name, b_enc_name) =
            encoder_file_and_tensor_names(TranscoderSchema::CltSplitJumpReLU, 13, &[]).unwrap();
        assert_eq!(filename, "W_enc_13.safetensors");
        assert_eq!(w_enc_name, "W_enc_13");
        assert_eq!(b_enc_name, "b_enc_13");
    }

    #[test]
    fn decoder_file_and_tensor_name_clt_split_jump_relu() {
        // CltSplitJumpReLU decoder lives in a separate file with
        // layer-suffixed tensor name, matching plain CltSplit.
        let (filename, tname) =
            decoder_file_and_tensor_name(TranscoderSchema::CltSplitJumpReLU, 13, &[]).unwrap();
        assert_eq!(filename, "W_dec_13.safetensors");
        assert_eq!(tname, "W_dec_13");
    }

    #[test]
    fn encoder_file_and_tensor_names_gemmascope_needs_paths() {
        // Without populated gemmascope_npz_paths, GemmaScopeNpz fails.
        assert!(encoder_file_and_tensor_names(TranscoderSchema::GemmaScopeNpz, 0, &[]).is_err());

        let paths = vec![
            "layer_0/width_16k/average_l0_100/params.npz".to_owned(),
            "layer_1/width_16k/average_l0_105/params.npz".to_owned(),
        ];
        let (filename, w_enc_name, b_enc_name) =
            encoder_file_and_tensor_names(TranscoderSchema::GemmaScopeNpz, 1, &paths).unwrap();
        assert_eq!(filename, "layer_1/width_16k/average_l0_105/params.npz");
        assert_eq!(w_enc_name, "W_enc");
        assert_eq!(b_enc_name, "b_enc");
    }

    #[test]
    fn decoder_file_and_tensor_name_all_schemas() {
        let (filename, tname) =
            decoder_file_and_tensor_name(TranscoderSchema::CltSplit, 5, &[]).unwrap();
        assert_eq!(filename, "W_dec_5.safetensors");
        assert_eq!(tname, "W_dec_5");

        let (filename, tname) =
            decoder_file_and_tensor_name(TranscoderSchema::PltBundle, 5, &[]).unwrap();
        assert_eq!(filename, "layer_5.safetensors");
        assert_eq!(tname, "W_dec");

        // GemmaScopeNpz needs populated paths.
        assert!(decoder_file_and_tensor_name(TranscoderSchema::GemmaScopeNpz, 0, &[]).is_err());
    }

    #[test]
    fn decoder_row_clt_split_indexes_rank3() {
        // 2 features × 3 target_offsets × 4 d_model.
        // Feature 0, offset 1 → [5, 6, 7, 8].
        #[rustfmt::skip]
        let values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1, offset 2
            0.0, 0.0, 0.0, 0.0,   5.0, 6.0, 7.0, 8.0,   0.0, 0.0, 0.0, 0.0,
            // feature 1: offset 0, offset 1, offset 2
            0.0, 0.0, 0.0, 0.0,   9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let w_dec = Tensor::from_vec(values, (2, 3, 4), &Device::Cpu).unwrap();

        let row = decoder_row(&w_dec, 0, 1, TranscoderSchema::CltSplit).unwrap();
        let got: Vec<f32> = row.to_vec1().unwrap();
        assert_eq!(got, vec![5.0, 6.0, 7.0, 8.0]);

        let row = decoder_row(&w_dec, 1, 1, TranscoderSchema::CltSplit).unwrap();
        let got: Vec<f32> = row.to_vec1().unwrap();
        assert_eq!(got, vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn decoder_row_plt_bundle_indexes_rank2_at_offset_zero() {
        // 2 features × 4 d_model (rank-2 for PltBundle).
        #[rustfmt::skip]
        let values: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,   // feature 0
            5.0, 6.0, 7.0, 8.0,   // feature 1
        ];
        let w_dec = Tensor::from_vec(values, (2, 4), &Device::Cpu).unwrap();

        let row = decoder_row(&w_dec, 1, 0, TranscoderSchema::PltBundle).unwrap();
        let got: Vec<f32> = row.to_vec1().unwrap();
        assert_eq!(got, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn decoder_row_plt_bundle_rejects_nonzero_offset() {
        // target_offset != 0 for a per-layer schema is a caller bug — must error
        // loudly in release builds (was a debug_assert previously; see aa23c90).
        let w_dec = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        let err = decoder_row(&w_dec, 0, 1, TranscoderSchema::PltBundle).unwrap_err();
        match err {
            MIError::Config(msg) => {
                assert!(msg.contains("target_offset must be 0"), "got: {msg}");
            }
            other => panic!("expected MIError::Config, got {other:?}"),
        }
    }

    #[test]
    fn decoder_layer_slice_plt_bundle_rejects_nonzero_offset() {
        let w_dec = Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(decoder_layer_slice(&w_dec, 2, TranscoderSchema::PltBundle).is_err());
    }

    #[test]
    fn decoder_layer_slice_plt_bundle_returns_full_rank2_at_offset_zero() {
        #[rustfmt::skip]
        let values: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let w_dec = Tensor::from_vec(values.clone(), (2, 4), &Device::Cpu).unwrap();
        let slice = decoder_layer_slice(&w_dec, 0, TranscoderSchema::PltBundle).unwrap();
        assert_eq!(slice.dims(), &[2, 4]);
        let got: Vec<Vec<f32>> = slice.to_vec2().unwrap();
        assert_eq!(got[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(got[1], vec![5.0, 6.0, 7.0, 8.0]);
    }

    // ====================================================================
    // PltBundle round-trip: cache_steering_vectors_all_downstream
    // ====================================================================

    /// Write a synthetic PltBundle `layer_{layer}.safetensors` file containing
    /// all five un-suffixed tensors (`W_enc`, `W_dec`, `W_skip`, `b_enc`, `b_dec`).
    /// All byte buffers are kept alive via ownership until `serialize` returns.
    fn create_synthetic_plt_bundle(
        dir: &std::path::Path,
        layer: usize,
        n_features: usize,
        d_model: usize,
        w_enc: &[f32],
        w_dec: &[f32],
        w_skip: &[f32],
        b_enc: &[f32],
        b_dec: &[f32],
    ) -> PathBuf {
        assert_eq!(w_enc.len(), n_features * d_model);
        assert_eq!(
            w_dec.len(),
            n_features * d_model,
            "PltBundle W_dec is rank-2"
        );
        assert_eq!(w_skip.len(), d_model * d_model);
        assert_eq!(b_enc.len(), n_features);
        assert_eq!(b_dec.len(), d_model);

        // Five byte buffers, all alive for the duration of `tensors` below.
        let w_enc_bytes: Vec<u8> = w_enc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_dec_bytes: Vec<u8> = w_dec.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_skip_bytes: Vec<u8> = w_skip.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_enc_bytes: Vec<u8> = b_enc.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_dec_bytes: Vec<u8> = b_dec.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        // BORROW: explicit .to_owned() — safetensors HashMap key as owned String
        tensors.insert(
            "W_enc".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![n_features, d_model],
                &w_enc_bytes,
            )
            .unwrap(),
        );
        tensors.insert(
            "W_dec".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![n_features, d_model],
                &w_dec_bytes,
            )
            .unwrap(),
        );
        tensors.insert(
            "W_skip".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![d_model, d_model],
                &w_skip_bytes,
            )
            .unwrap(),
        );
        tensors.insert(
            "b_enc".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![n_features],
                &b_enc_bytes,
            )
            .unwrap(),
        );
        tensors.insert(
            "b_dec".to_owned(),
            safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![d_model],
                &b_dec_bytes,
            )
            .unwrap(),
        );

        let serialized = safetensors::serialize(&tensors, &None).unwrap();
        let path = dir.join(format!("layer_{layer}.safetensors"));
        std::fs::write(&path, serialized).unwrap();
        path
    }

    #[test]
    fn plt_bundle_cache_steering_all_downstream_is_single_entry_per_feature() {
        // Regression test for aa23c90: `cache_steering_vectors_all_downstream`
        // must use `n_target_layers = 1` for PltBundle (not n_layers - source_layer),
        // otherwise the same rank-2 W_dec row gets cached under every downstream
        // (feature, target_layer) key.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // Layer 0 bundle: W_dec row 0 = [1, 2, 3, 4], row 1 = [5, 6, 7, 8].
        #[rustfmt::skip]
        let w_dec_0: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];
        let path0 = create_synthetic_plt_bundle(
            dir.path(),
            0,
            n_features,
            d_model,
            &vec![0.0; n_features * d_model],
            &w_dec_0,
            &vec![0.0; d_model * d_model],
            &vec![0.0; n_features],
            &vec![0.0; d_model],
        );

        // Layer 1 bundle (weights don't matter — only layer 0 feature is requested).
        let path1 = create_synthetic_plt_bundle(
            dir.path(),
            1,
            n_features,
            d_model,
            &vec![0.0; n_features * d_model],
            &vec![99.0; n_features * d_model],
            &vec![0.0; d_model * d_model],
            &vec![0.0; n_features],
            &vec![0.0; d_model],
        );

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            // PltBundle: encoder_paths doubles as decoder_paths (shared file).
            encoder_paths: vec![Some(path0), Some(path1)],
            decoder_paths: vec![None, None],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::PltBundle,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let features = vec![CltFeatureId { layer: 0, index: 0 }];
        clt.cache_steering_vectors_all_downstream(&features, &Device::Cpu)
            .unwrap();

        // PltBundle → exactly ONE cache entry, keyed at the source layer.
        assert_eq!(
            clt.steering_cache_len(),
            1,
            "PltBundle must cache exactly 1 entry per feature (not n_layers)"
        );
        let cached = clt
            .steering_cache
            .get(&(CltFeatureId { layer: 0, index: 0 }, 0))
            .expect("entry at (feature, source_layer)");
        let values: Vec<f32> = cached.to_vec1().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

        // And there must be NO entry at (feature, layer 1) — that would indicate
        // the pre-aa23c90 bug.
        assert!(
            !clt.steering_cache
                .contains_key(&(CltFeatureId { layer: 0, index: 0 }, 1)),
            "PltBundle must not cache downstream entries"
        );
    }

    #[test]
    fn clt_split_cache_steering_all_downstream_caches_all_targets() {
        // Companion regression test: CltSplit must still cache n_target_layers entries
        // per feature (existing behaviour preserved by commit aa23c90).
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 2;

        // W_dec_0: rank-3 [2 features × 2 target_offsets × 4 d_model].
        #[rustfmt::skip]
        let dec0_values: Vec<f32> = vec![
            // feature 0: offset 0, offset 1
            1.0, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0,
            // feature 1
            0.0, 0.0, 1.0, 0.0,   0.0, 0.0, 0.0, 1.0,
        ];
        let path0 = create_synthetic_decoder(dir.path(), 0, n_features, 2, d_model, &dec0_values);

        // W_dec_1: rank-3 [2 × 1 × 4].
        #[rustfmt::skip]
        let dec1_values: Vec<f32> = vec![
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
        ];
        let path1 = create_synthetic_decoder(dir.path(), 1, n_features, 1, d_model, &dec1_values);

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None, None],
            decoder_paths: vec![Some(path0), Some(path1)],
            config: CltConfig {
                n_layers: 2,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features * 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        // Feature at layer 0 should produce 2 cache entries (layer 0 writes to 0, 1).
        let features = vec![CltFeatureId { layer: 0, index: 0 }];
        clt.cache_steering_vectors_all_downstream(&features, &Device::Cpu)
            .unwrap();

        assert_eq!(
            clt.steering_cache_len(),
            2,
            "CltSplit: layer 0 writes to 2 downstream layers"
        );
        assert!(
            clt.steering_cache
                .contains_key(&(CltFeatureId { layer: 0, index: 0 }, 0))
        );
        assert!(
            clt.steering_cache
                .contains_key(&(CltFeatureId { layer: 0, index: 0 }, 1))
        );
    }

    #[test]
    fn encode_pre_activation_matches_encode_postrelu() {
        // Asserts the invariant that encode()'s sparse output is exactly the
        // post-ReLU view of encode_pre_activation()'s dense output — so Step B
        // histograms over the dense pre-activation are numerically coherent
        // with the sparse features that intervene in the causal tests.
        let dir = tempfile::tempdir().unwrap();
        let d_model = 4;
        let n_features = 5;

        // Seed W_enc so that some pre-activations land positive and others
        // negative (so ReLU actually sparsifies). b_enc shifts the split point.
        #[rustfmt::skip]
        let w_enc: Vec<f32> = vec![
             1.0,  0.0,  0.0,  0.0, // feature 0: picks up residual[0]
             0.0,  1.0,  0.0,  0.0, // feature 1: picks up residual[1]
             0.0,  0.0, -1.0,  0.0, // feature 2: flips sign of residual[2]
             0.5,  0.5,  0.5,  0.5, // feature 3: mean of residual
            -1.0, -1.0, -1.0, -1.0, // feature 4: negated sum (will fire only with negative sum)
        ];
        let b_enc: Vec<f32> = vec![0.0, 0.0, 0.0, -1.0, 2.0];
        let path0 = create_synthetic_plt_bundle(
            dir.path(),
            0,
            n_features,
            d_model,
            &w_enc,
            &vec![0.0; n_features * d_model],
            &vec![0.0; d_model * d_model],
            &b_enc,
            &vec![0.0; d_model],
        );

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![Some(path0)],
            decoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::PltBundle,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        clt.load_encoder(0, &Device::Cpu).unwrap();
        // residual chosen so feature 0, 1 fire positive (ReLU passes), feature 2
        // fires positive (negates -0.5 residual), feature 3 hits b_enc=-1 (barely
        // positive), feature 4 gets negative pre-activation (ReLU clips to 0).
        let residual =
            Tensor::from_vec(vec![1.0_f32, 2.0, -0.5, 1.5], (d_model,), &Device::Cpu).unwrap();

        let pre_acts_tensor = clt.encode_pre_activation(&residual, 0).unwrap();
        let pre_acts: Vec<f32> = pre_acts_tensor.to_vec1().unwrap();
        // Dense shape check.
        assert_eq!(pre_acts.len(), n_features);

        let sparse = clt.encode(&residual, 0).unwrap();

        // Every sparse feature must match max(0, pre_activation).
        for (fid, act) in &sparse.features {
            let pre = pre_acts[fid.index];
            assert!(
                pre > 0.0,
                "sparse feature {fid:?} must have positive pre-act"
            );
            assert!(
                (pre - act).abs() < 1e-6,
                "feature {fid:?}: sparse={act}, pre={pre}"
            );
        }
        // Every dense pre-activation that is <= 0 must be absent from sparse.
        let sparse_indices: std::collections::HashSet<usize> =
            sparse.features.iter().map(|(f, _)| f.index).collect();
        for (i, &pre) in pre_acts.iter().enumerate() {
            if pre <= 0.0 {
                assert!(
                    !sparse_indices.contains(&i),
                    "feature {i} pre-act {pre} <= 0 but appears in sparse output"
                );
            }
        }
    }

    #[test]
    fn load_skip_matrix_round_trip_plt_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let d_model = 3;
        let n_features = 2;

        // Seed W_skip with distinct values so we can verify row/column order.
        #[rustfmt::skip]
        let w_skip: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let path0 = create_synthetic_plt_bundle(
            dir.path(),
            0,
            n_features,
            d_model,
            &vec![0.0; n_features * d_model],
            &vec![0.0; n_features * d_model],
            &w_skip,
            &vec![0.0; n_features],
            &vec![0.0; d_model],
        );

        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![Some(path0)],
            decoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::PltBundle,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let loaded = clt.load_skip_matrix(0, &Device::Cpu).unwrap();
        assert_eq!(loaded.dims(), &[d_model, d_model]);
        let values: Vec<Vec<f32>> = loaded.to_vec2().unwrap();
        assert_eq!(values[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(values[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(values[2], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn load_skip_matrix_rejects_clt_split_schema() {
        // CltSplit transcoders (mntss/clt-*) have no W_skip — attempting to
        // load one must fail with MIError::Config pointing at the schema.
        let mut clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            decoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model: 4,
                n_features_per_layer: 2,
                n_features_total: 2,
                model_name: "test".to_owned(),
                schema: TranscoderSchema::CltSplit,
                gemmascope_npz_paths: Vec::new(),
            },
            loaded_encoder: None,
            steering_cache: HashMap::new(),
        };

        let err = clt.load_skip_matrix(0, &Device::Cpu).unwrap_err();
        match err {
            MIError::Config(msg) => {
                assert!(
                    msg.contains("PltBundle"),
                    "error message should mention PltBundle schema: {msg}"
                );
            }
            other => panic!("expected MIError::Config, got {other:?}"),
        }
    }
}
