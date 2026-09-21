// SPDX-License-Identifier: MIT OR Apache-2.0

//! # candle-mi
//!
//! Mechanistic interpretability for language models in Rust, built on
//! [candle](https://github.com/huggingface/candle).
//!
//! candle-mi re-implements model forward passes with built-in hook points
//! (following the [`TransformerLens`](https://github.com/TransformerLensOrg/TransformerLens)
//! design), enabling activation capture, attention knockout, steering, logit
//! lens, and sparse-feature analysis (CLTs and SAEs) — all in pure Rust with
//! GPU acceleration.
//!
//! ## Requirements
//!
//! **Rust 1.91 or newer**, edition 2024. On an older toolchain cargo does not
//! report an error: because its resolver is MSRV-aware, it silently picks the
//! newest release your compiler can build and `cargo update` never moves you
//! past it. Two such ceilings exist — Rust 1.87 or older stops at **0.1.4**
//! (0.1.5 raised the floor to 1.88, forced by `libloading 0.9.0`), and Rust
//! 1.88 to 1.90 stops at **0.1.22** (0.1.23 raised it to 1.91, forced by
//! `hf-fetch-model 0.11.3` adopting `hf-hub` 1.0 and its mandatory
//! `hf-xet`). 0.1.4 in particular predates RWKV, the masked-diffusion
//! backends, trainable backbones and quantized loading. If `cargo add
//! candle-mi` gave you either, run `rustup update && cargo update`.
//!
//! candle-mi targets a single consumer GPU: models up to ~7B fit in 16 GB at
//! F32. CPU-only works for small models and tokenizer-only workflows.
//!
//! ## Supported backends
//!
//! | Backend | Models | Feature flag |
//! |---------|--------|-------------|
//! | `GenericTransformer` | `LLaMA`, `Qwen2`, `Qwen3`, Gemma, Gemma 2, `Phi-3`, `StarCoder2`, Mistral; bidirectional masked-diffusion decoders (Dream, `a2d-qwen2`, `a2d-qwen3`); + auto-config for unknown families | `transformer` |
//! | `GenericRwkv` | RWKV-6 (Finch), RWKV-7 (Goose) | `rwkv` |
//! | `GenericMdlm` | MDLM masked-diffusion `DiT` (bidirectional) | `diffusion` |
//! | `OthelloGpt` | Plain GPT-2-style backbone (learned absolute positions, full `LayerNorm`, exact-`GELU`); the `OthelloMDLM` world model | `diffusion` |
//! | `StoicheiaRnn` / `StoicheiaTransformer` | `AlgZoo` `ReLU` RNN, attention-only transformer (8–1,408 params) | `stoicheia` |
//!
//! See [`BACKENDS.md`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/main/BACKENDS.md)
//! for how to add a new model architecture.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `transformer` | yes | Generic transformer backend (decoder-only) |
//! | `cuda` | yes | CUDA GPU acceleration |
//! | `rwkv` | no | RWKV-6/7 linear RNN backend |
//! | `rwkv-tokenizer` | no | RWKV world tokenizer (required for RWKV inference) |
//! | `diffusion` | no | MDLM masked-diffusion backend (bidirectional `DiT`; standalone) |
//! | `clt` | no | Cross-Layer Transcoder support |
//! | `sae` | no | Sparse Autoencoder support (NPZ via `anamnesis`) |
//! | `quantized` | no | Load quantized checkpoints (bitsandbytes `NF4`/`FP4`/`INT8`, `AWQ`, `GPTQ`) by dequantizing to `BF16` via `anamnesis` |
//! | `mmap` | no | Memory-mapped weight loading (required for sharded models) |
//! | `memory` | no | RAM/VRAM reporting (delegated to the `hypomnesis` crate) |
//! | `memory-debug` | no | Raw GPU-backend measurement values on stderr (via `hypomnesis`; implies `memory`) |
//! | `stoicheia` | no | `AlgZoo` tiny-model backends + MI analysis tools; agnostic `.safetensors`/`.pth` loading via `anamnesis` |
//! | `probing` | no | Linear probing via linfa (experimental) |
//! | `metal` | no | Apple Metal GPU acceleration |
//!
//! ## Quick start
//!
//! Load a model, run a forward pass, and inspect the output:
//!
//! ```no_run
//! use candle_mi::{HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! let tokenizer = model.tokenizer().unwrap();
//!
//! let tokens = tokenizer.encode("The capital of France is")?;
//! let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//!
//! let cache = model.forward(&input, &HookSpec::new())?;
//! let logits = cache.output();  // [1, seq, vocab]
//!
//! let last_logits = logits.get(0)?.get(tokens.len() - 1)?;
//! let token_id = candle_mi::sample_token(&last_logits, 0.0)?;  // greedy
//! println!("{}", tokenizer.decode(&[token_id])?);  // " Paris"
//! # Ok(())
//! # }
//! ```
//!
//! ## Activation capture
//!
//! Use [`HookSpec::capture`] to snapshot tensors at any
//! [`HookPoint`] during the forward pass:
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! let mut hooks = HookSpec::new();
//! hooks.capture(HookPoint::AttnPattern(5))       // post-softmax attention
//!      .capture(HookPoint::ResidPost(10));        // residual stream at layer 10
//!
//! let cache = model.forward(&input, &hooks)?;
//!
//! let attn = cache.require(&HookPoint::AttnPattern(5))?;   // [1, heads, seq, seq]
//! let resid = cache.require(&HookPoint::ResidPost(10))?;    // [1, seq, hidden]
//! # Ok(())
//! # }
//! ```
//!
//! ## Interventions
//!
//! Use [`HookSpec::intervene`] to modify activations mid-forward-pass.
//! Six intervention types are available: [`Intervention::Replace`],
//! [`Intervention::PatchAt`], [`Intervention::Add`],
//! [`Intervention::Knockout`], [`Intervention::Scale`], and
//! [`Intervention::Zero`].
//!
//! [`Intervention::PatchAt`] is the activation-patching primitive: it
//! overwrites one sequence position and leaves the rest of the activation in
//! flight, so a causal trace needs no captures from the recipient pass. It is
//! accepted only at hook points whose activation is `[batch, seq_len, hidden]`
//! (see [`HookPoint::accepts_positional_patch`]); at an attention hook point,
//! where dim 1 is a head rather than a position, it is an error rather than a
//! silent write to the wrong axis.
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, Intervention, KnockoutSpec, create_knockout_mask};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let seq_len = tokens.len();
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! // Knock out the attention edge: last token cannot attend to position 0
//! let spec = KnockoutSpec::new().layer(8).edge(seq_len - 1, 0);
//! let mask = create_knockout_mask(
//!     &spec, model.num_heads(), seq_len, model.device(), candle_core::DType::F32,
//! )?;
//!
//! let mut hooks = HookSpec::new();
//! hooks.intervene(HookPoint::AttnScores(8), Intervention::Knockout(mask));
//!
//! let ablated = model.forward(&input, &hooks)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Logit lens
//!
//! Project intermediate residual streams to vocabulary space using
//! [`MIModel::project_to_vocab`]:
//!
//! ```no_run
//! use candle_mi::{HookPoint, HookSpec, MIModel};
//!
//! # fn main() -> candle_mi::Result<()> {
//! # let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # let tokenizer = model.tokenizer().unwrap();
//! # let tokens = tokenizer.encode("The capital of France is")?;
//! # let seq_len = tokens.len();
//! # let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
//! let mut hooks = HookSpec::new();
//! hooks.capture_all((0..model.num_layers()).map(HookPoint::ResidPost));
//! let cache = model.forward(&input, &hooks)?;
//!
//! for layer in 0..model.num_layers() {
//!     let resid = cache.require(&HookPoint::ResidPost(layer))?;
//!     let last = resid.get(0)?.get(seq_len - 1)?.unsqueeze(0)?;
//!     let logits = model.project_to_vocab(&last)?;
//!     let token_id = candle_mi::sample_token(&logits.flatten_all()?, 0.0)?;
//!     println!("Layer {layer:>2}: {}", tokenizer.decode(&[token_id])?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Training (differentiable backbones)
//!
//! Backbones carry gradients end-to-end when built over a `candle_nn::VarMap`
//! (since v0.1.20): the [`nn_ops`] wrappers dispatch on `Tensor::track_op`, so
//! an inference forward takes candle's fused kernels unchanged — byte-identical
//! to previous releases — while a tracked forward takes composed,
//! differentiable forms. Train a model and probe *the same weights on the same
//! forward pass*; `OthelloGpt::init` (`diffusion` feature) additionally gives a
//! seeded GPT-2-recipe initialization, reproducible from `(config, seed)`
//! alone — the generator is `ChaCha8` keyed in-crate, so a future `rand`
//! release cannot move it. `OthelloGpt::init_with_dtype` applies the same
//! recipe at any dtype, creating every parameter at it; pass `DType::BF16` to
//! halve activation bytes when the training batch-size ceiling is the binding
//! constraint rather than arithmetic throughput.
//!
//! ```ignore
//! // requires: --features diffusion
//! let varmap = candle_nn::VarMap::new();
//! let model = OthelloGpt::init(config, &varmap, &device, 42)?;
//! let mut adam = candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-3)?;
//! for _step in 0..steps {
//!     let cache = MIBackend::forward(&model, &batch, &HookSpec::new())?;
//!     let loss = candle_nn::loss::cross_entropy(&cache.output().flatten_to(1)?, &targets)?;
//!     candle_nn::Optimizer::backward_step(&mut adam, &loss)?; // all 29 params update
//! }
//! ```
//!
//! candle-mi deliberately ships **no training loop, schedule, or data loader** —
//! those are experiment-shaped and stay with the caller. See
//! `docs/dogfooding-feedbacks/trainable-backbones.md` for the design rationale.
//!
//! The one exception is state that a training loop cannot reconstruct for
//! itself. `candle_nn::AdamW` keeps its moments and step counter private, so a
//! run split across processes silently resets Adam's bias correction at every
//! boundary. `optim::AdamW` (default-off `training` feature) is candle's update
//! rule with that state reachable, so a staged run resumes exactly; its
//! trajectory is held to candle's own by `tests/validate_optim_parity.rs`.
//! It buys no throughput, only resumability.
//!
//! ## Fast downloads
//!
//! candle-mi uses [`hf-fetch-model`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model)
//! for high-throughput parallel downloads from the `HuggingFace` Hub:
//!
//! ```rust,no_run
//! # async fn example() -> candle_mi::Result<()> {
//! // Async: parallel chunked download with progress bars
//! let _path = candle_mi::download_model("meta-llama/Llama-3.2-1B".to_owned()).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ```no_run
//! # fn main() -> candle_mi::Result<()> {
//! // Sync: blocking variant (uses local HF cache if already downloaded)
//! candle_mi::download_model_blocking("meta-llama/Llama-3.2-1B".to_owned())?;
//! let model = candle_mi::MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Further reading
//!
//! - [`HOOKS.md`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/main/HOOKS.md) —
//!   complete hook point reference with shapes, intervention walkthrough, and
//!   worked examples.
//! - [`BACKENDS.md`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/main/BACKENDS.md) —
//!   how to add a new model architecture (auto-config, config parser, or
//!   custom `MIBackend`).
//! - [`examples/README.md`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/main/examples/README.md) —
//!   23 runnable examples covering inference, logit lens, attention patterns,
//!   knockout, steering, activation patching, `CounterFact` replication,
//!   CLT circuits, SAE encoding, RWKV inference, `AlgZoo` analysis, and more.

#![deny(warnings)]
// All warns → errors in CI
// Rule 5: safe by default. The only `unsafe` in the crate is the `mmap`
// safetensors loader, the CUDA pool-trim in `memory::sync_and_trim_gpu`
// (gated behind `cuda`), and the multi-tensor `AdamW` kernel launch in
// `optim` (gated behind `training` + `cuda`). Since the hypomnesis migration,
// the `memory` feature carries NO unsafe on its own — all measurement FFI
// lives in hypomnesis — so `memory` without `cuda` stays `forbid(unsafe_code)`.
#![cfg_attr(
    not(any(
        feature = "mmap",
        all(feature = "memory", feature = "cuda"),
        all(feature = "training", feature = "cuda")
    )),
    forbid(unsafe_code)
)]
// mmap, memory+cuda (the pool-trim FFI) or training+cuda (the optimizer
// launch): deny for scoped unsafe.
#![cfg_attr(
    any(
        feature = "mmap",
        all(feature = "memory", feature = "cuda"),
        all(feature = "training", feature = "cuda")
    ),
    deny(unsafe_code)
)]
// Test-code relaxations: the strict `unwrap_used`/`expect_used`/`indexing_slicing`/`panic`
// denies in `Cargo.toml` target *library* code (Rule 3). Inside `#[cfg(test)]` blocks,
// `unwrap()`, `expect("context")`, `assert_eq!(v[0], …)`, and `panic!` are the canonical
// Rust test idioms — failure-by-panic IS the test signal. Allow them only under
// `cfg(test)`; production builds remain strict.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::unreadable_literal,
        clippy::panic,
        clippy::wildcard_enum_match_arm,
        clippy::redundant_clone,
        clippy::needless_collect,
        clippy::format_push_string,
        clippy::doc_markdown,
        clippy::too_many_arguments,
        clippy::missing_const_for_fn,
    )
)]

pub mod backend;
pub mod cache;
#[cfg(feature = "clt")]
pub mod clt;
pub mod config;
#[cfg(feature = "diffusion")]
pub mod diffusion;
pub mod download;
pub mod error;
pub mod hooks;
pub mod interp;
#[cfg(feature = "memory")]
pub mod memory;
pub mod nn_ops;
#[cfg(feature = "training")]
pub mod optim;
#[cfg(feature = "rwkv")]
pub mod rwkv;
#[cfg(feature = "sae")]
pub mod sae;
pub mod sparse;
// Steering builders are only useful when a backend can apply their output; gate
// the module behind the same predicate as `hooks::apply_intervention` so builder
// and applier appear/disappear together. `sparse` stays ungated — it is shared
// data types (`FeatureId`/`SparseActivations`) consumed by the backend-independent
// `clt`/`sae` features, not a builder with a backend-gated applier.
// Gated on the same predicate as `hooks::apply_intervention`, so the
// intervention *builders* and the code that *applies* them appear and
// disappear together (diakrisis-intervention-dogfood.md, finding 2).
// `stoicheia` joined that predicate in v0.2.0 when both stoicheia backends
// began honouring interventions; this keeps the two in step.
#[cfg(any(
    feature = "transformer",
    feature = "rwkv",
    feature = "diffusion",
    feature = "stoicheia"
))]
pub mod steering;
#[cfg(feature = "stoicheia")]
pub mod stoicheia;
pub mod tokenizer;
#[cfg(feature = "transformer")]
pub mod transformer;
mod util;

/// Build-hygiene guard: asserts every `tests/*.rs` / `examples/*.rs` file is
/// registered in `Cargo.toml`. Test-only; see the module docs for rationale.
#[cfg(test)]
mod registration_guard;

// --- Public re-exports ---------------------------------------------------

// Backend
pub use backend::{
    GenerationResult, MIBackend, MIModel, TextForwardResult, extract_token_prob, sample_token,
};

// Config
pub use config::{
    Activation, CompatibilityReport, MlpLayout, NormType, QkvLayout, RopeScaling,
    SUPPORTED_MODEL_TYPES, TransformerConfig,
};

// Transformer backend
#[cfg(feature = "transformer")]
pub use transformer::GenericTransformer;

// Recurrent feedback (anacrousis)
#[cfg(feature = "transformer")]
pub use transformer::recurrent::{RecurrentFeedbackEntry, RecurrentPassSpec};

// RWKV backend
#[cfg(feature = "rwkv")]
pub use rwkv::{GenericRwkv, RwkvConfig, RwkvLoraDims, RwkvVersion};

// Diffusion backend (MDLM)
#[cfg(feature = "diffusion")]
pub use diffusion::{
    DiffusionSamplingConfig, GenericMdlm, MdlmConfig, OthelloGpt, OthelloGptConfig,
    SUPPORTED_DIFFUSION_MODEL_TYPES,
};

// Stoicheia (AlgZoo) backends — Phase A
#[cfg(feature = "stoicheia")]
pub use stoicheia::{
    StoicheiaArch, StoicheiaConfig, StoicheiaOutput, StoicheiaRnn, StoicheiaTask,
    StoicheiaTransformer,
};

// Stoicheia MI tooling — Phase B
#[cfg(feature = "stoicheia")]
pub use stoicheia::fast::RnnWeights;
#[cfg(feature = "stoicheia")]
pub use stoicheia::probing::NeuronRole;
#[cfg(feature = "stoicheia")]
pub use stoicheia::standardize::StandardizedRnn;

// Sparse feature types (shared by CLT and SAE)
pub use sparse::{FeatureId, SparseActivations};

// CLT (Cross-Layer Transcoder)
#[cfg(feature = "clt")]
pub use clt::{AttributionEdge, AttributionGraph, CltConfig, CltFeatureId, CrossLayerTranscoder};

// SAE (Sparse Autoencoder)
#[cfg(feature = "sae")]
pub use sae::{
    NormalizeActivations, SaeArchitecture, SaeConfig, SaeFeatureId, SparseAutoencoder, TopKStrategy,
};

// Cache
pub use cache::{ActivationCache, AttentionCache, FullActivationCache, KVCache};

// Error
pub use error::{MIError, Result};

// Hooks
pub use hooks::{HookCache, HookPoint, HookSpec, Intervention};

// Interpretability — intervention specs and results
pub use interp::intervention::{
    AblationResult, AttentionEdge, HeadSpec, InterventionType, KnockoutSpec, LayerSpec,
    StateAblationResult, StateKnockoutSpec, StateSteeringResult, StateSteeringSpec, SteeringResult,
    SteeringSpec, apply_steering, create_knockout_mask, kl_divergence,
    measure_attention_to_targets,
};

// Interpretability — logit lens
pub use interp::logit_lens::{LogitLensAnalysis, LogitLensResult, TokenPrediction};

// Interpretability — steering calibration
pub use interp::steering::{DoseResponseCurve, DoseResponsePoint, SteeringCalibration};

// Steering — contrastive activation steering (Maar et al. 2026)
// Gated with the `steering` module above (needs a backend to apply its output).
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
pub use steering::contrastive::{
    ContrastiveDirection, PositionStrategy, build_contrastive_direction, contrastive_intervention,
    position_delta,
};

// Utility — masks
pub use util::masks::{
    clear_mask_caches, create_bidirectional_mask, create_causal_mask, create_generation_mask,
};

// Utility — PCA
pub use util::pca::{PcaResult, pca_top_k};

// Utility — positioning
pub use util::positioning::{
    EncodingWithOffsets, PositionConversion, TokenWithOffset, convert_positions,
};

// Tokenizer
pub use tokenizer::MITokenizer;

// Memory reporting
#[cfg(feature = "memory")]
pub use memory::{MemoryReport, MemorySnapshot, sync_and_trim_gpu};

// Download
pub use download::{download_model, download_model_blocking, fetch_config_builder};
