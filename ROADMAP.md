# Roadmap: A Rust Crate for Mechanistic Interpretability

> *MI for the Rust of us*

> **Historical planning document (frozen ~v0.1.8, April 2026).** This roadmap
> captures the original Phase 0–6 plan and is retained for provenance. It is
> **not** kept in lockstep with releases — for authoritative, up-to-date status
> (features shipped, versions published) see [CHANGELOG.md](CHANGELOG.md), which
> is the source of truth from v0.1.9 onward.

**Date:** February 19, 2026 (last updated: April 13, 2026; superseded as a status source by CHANGELOG.md)
**Status:** Phase 0–5 complete. Published on [crates.io](https://crates.io/crates/candle-mi) as v0.1.8. Phase 6 partially complete (character count helix finalized, activation patching extended with full causal trace heatmap, steering convergence, NPZ parsing migrated to `anamnesis`, MSRV 1.88, `forward_text` API, prolepsis finding via Transluce replication and factual routing). Stoicheia Phase A + B complete (AlgZoo backends + MI tooling, agnostic `.pth`/`.safetensors` loading via anamnesis v0.4.1). Default dtype F32 for research-grade precision. PR submitted to candle repo ([#3406](https://github.com/huggingface/candle/pull/3406)).
**Context:** Building on plip-rs experience (7 model backends incl. Gemma 2, attention knockout, state knockout, effective attention, steering, logit lens, CLT encoding/injection). Two successful replications of Anthropic's "Planning in Poems" Figure 13 validate the approach: Gemma 2 2B with 426K CLTs (melometis branch) and Llama 3.2 1B with 524K CLTs (tragos branch). Target: a publishable, generic Rust MI crate endorsed by HuggingFace.

---

## Table of Contents

- [0. Naming & License](#0-naming--license)
- [1. What Exists Today (State of the Art)](#1-what-exists-today-state-of-the-art)
  - [1.1 Python MI Ecosystem](#11-python-mi-ecosystem)
  - [1.2 Rust MI Ecosystem](#12-rust-mi-ecosystem)
  - [1.3 candle-transformers](#13-candle-transformers)
  - [1.4 Open Interpretability Dictionaries (CLTs and SAEs)](#14-open-interpretability-dictionaries-clts-and-saes)
- [2. Architecture: What We Build](#2-architecture-what-we-build)
  - [2.1 Design Philosophy](#21-design-philosophy)
  - [2.2 What We Reuse from plip-rs](#22-what-we-reuse-from-plip-rs)
  - [2.3 What We Change from plip-rs](#23-what-we-change-from-plip-rs)
- [3. The Generic Transformer](#3-the-generic-transformer)
  - [3.1 Configuration Axes](#31-configuration-axes)
  - [3.2 Config Struct](#32-config-struct)
  - [3.3 Coverage](#33-coverage)
  - [3.4 What It Does NOT Cover](#34-what-it-does-not-cover)
  - [3.5 Config Parsing from HuggingFace `config.json`](#35-config-parsing-from-huggingface-configjson)
  - [3.6 Weight Name Mapping](#36-weight-name-mapping)
- [4. Generic RWKV / Linear RNN Support](#4-generic-rwkv--linear-rnn-support)
  - [4.1 The RWKV Family](#41-the-rwkv-family)
  - [4.2 Available Models (HuggingFace, safetensors)](#42-available-models-huggingface-safetensors)
  - [4.3 Architectural Comparison](#43-architectural-comparison)
  - [4.4 Generic RWKV Design](#44-generic-rwkv-design)
  - [4.5 Related Linear RNN / SSM Architectures](#45-related-linear-rnn--ssm-architectures)
- [5. MI Capabilities](#5-mi-capabilities)
  - [5.1 Core (from plip-rs, reusable)](#51-core-from-plip-rs-reusable)
  - [5.2 New (required for Melomētis, general-purpose)](#52-new-required-for-melomētis-general-purpose)
  - [5.3 Future (not required now)](#53-future-not-required-now)
- [6. Crate Structure](#6-crate-structure)
  - [6.1 Feature Gates](#61-feature-gates)
  - [6.2 Documentation](#62-documentation)
- [7. Phased Development Plan](#7-phased-development-plan)
  - [7.0 Git Workflow](#70-git-workflow)
  - [Phase 0: Foundation](#phase-0-foundation) ✅
  - [Phase 1: Generic Transformer](#phase-1-generic-transformer) ✅
  - [Phase 2: RWKV-6 + RWKV-7](#phase-2-rwkv-6--rwkv-7) ✅
  - [Phase 3: CLT Support](#phase-3-clt-support) ✅
  - [Phase 4: SAE Support](#phase-4-sae-support) ✅
  - [Phase 5: Polish + Publish + Auto-Config](#phase-5-polish--publish--auto-config) ✅
  - [Phase 6: Research Examples & Utilities](#phase-6-research-examples--utilities)
  - [Phase 7: Standard MI Analysis Stack](#phase-7-standard-mi-analysis-stack)
  - [Phase 8: Static Circuit Analysis](#phase-8-static-circuit-analysis)
  - [Phase 9: Model Coverage & Ecosystem](#phase-9-model-coverage--ecosystem)
  - [Phase 10+: Extensions (Future)](#phase-10-extensions-future)
- [8. Key Design Decisions](#8-key-design-decisions)
- [9. Relationship to Existing Projects](#9-relationship-to-existing-projects)
  - [9.1 plip-rs (AIware 2026)](#91-plip-rs-aiware-2026)
  - [9.2 Melomētis + Tragos — Planning in Poems](#92-melomētis--tragos--planning-in-poems-plip-rs-melometis-branch-v140)
  - [9.3 Deloson (Visualization)](#93-deloson-visualization)
  - [9.4 candle-transformers](#94-candle-transformers)
- [10. Risk Assessment](#10-risk-assessment)
- [References](#references)

---

## 0. Naming & License

### Name: `candle-mi`

The initial concern was that all `candle-*` crates on crates.io are published by HuggingFace, and using the prefix might imply official affiliation. This is now resolved: **Eric Buehler** (HuggingFace candle collaborator) [endorsed the `candle-mi` name](https://github.com/huggingface/candle/discussions/3368) and invited a pull request to add the crate to candle's "Useful External Resources" section.

The name is short, unambiguous, and signals both the candle foundation and the MI purpose. All type names follow: `MIBackend`, `MIModel`, `MITokenizer`, etc.

### License: MIT OR Apache-2.0

Dual-licensed under MIT and Apache-2.0, the standard for the Rust ecosystem and consistent with candle itself. Each source file carries the SPDX header:

```
// SPDX-License-Identifier: MIT OR Apache-2.0
```

`Cargo.toml`:
```toml
license = "MIT OR Apache-2.0"
```

---

## 1. What Exists Today (State of the Art)

### 1.1 Python MI Ecosystem

| Library | Approach | Model Support | Key Insight |
|---------|----------|---------------|-------------|
| **TransformerLens** | Re-implements models with hook points | 50+ architectures (canonical format) | Gold standard; `run_with_cache`, `run_with_hooks` |
| **nnsight** | Wraps existing HF models via tracing proxy | Any PyTorch model | No re-implementation; remote execution via NDIF |
| **nnterp** | Standardized names on top of nnsight | 50+ variants across 16 families | Best of both: wrap existing model, uniform naming |
| **pyvene** | Configuration-based interventions | Any PyTorch model (incl. RNNs, CNNs) | Trainable interventions; declarative API |
| **SAELens** | SAE training + analysis | GPT-2, Gemma 2, LLaMA 3 | HookedSAETransformer for integrated MI+SAE |
| **circuit-tracer** | CLT attribution graphs | Gemma 2 2B, LLaMA 3.2 1B, Qwen-3 | Anthropic's open-source tool for circuit discovery |
| **sparsify** | SAE + transcoder training | Any HF model | EleutherAI; TopK SAEs |

### 1.2 Rust MI Ecosystem

**No published crates.** plip-rs is the only Rust project performing MI operations on language models (attention knockout, state knockout, effective attention, steering, logit lens, CLT encoding/injection). It has produced two independent replications of Anthropic's "Planning in Poems" Figure 13 ([Lindsey et al. 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location)):

| Replication | Model | CLT | Branch | Key Result |
|------------|-------|-----|--------|------------|
| **Melomētis** | Gemma 2 2B | 426K features | `melometis` | 48.3% probability redirect at planning site; 70% of 136 suppress+inject pairs peak at planning site |
| **Tragos** | Llama 3.2 1B | 524K features | `tragos` | Second independent replication confirming the phenomenon generalises across architectures |

These results validate the plip-rs infrastructure that candle-mi will reuse and extend.

### 1.3 candle-transformers

Over 100 model implementations, all separate files, no unified hook wrapper, no MI features. Same architectural pattern as plip-rs (separate forward files per model). No generic/config-driven transformer exists.

### 1.4 Open Interpretability Dictionaries (CLTs and SAEs)

| Model | CLT Weights | SAE Weights |
|-------|-------------|-------------|
| **Gemma 2 2B** | circuit-tracer (426K, 2.5M features) | Gemma Scope (JumpReLU, every layer/sublayer) |
| **Gemma 3 270M/1B** | Gemma Scope 2 (transcoders) | Gemma Scope 2 |
| **LLaMA 3.2 1B** | circuit-tracer (524K features) | Llama Scope |
| **Qwen-3 (0.6B-14B)** | circuit-tracer | — |
| **GPT-2 Small** | openCLT (incomplete) | SAELens (MLP transcoders) |
| **Claude 3.5 Haiku** | Internal only (not released) | Internal only |

---

## 2. Architecture: What We Build

### 2.1 Design Philosophy

**TransformerLens approach** (re-implement models with hooks built in), NOT nnsight approach (wrap existing models). Rationale:
- Rust has no equivalent of PyTorch's `register_forward_hook` — we can't wrap arbitrary code
- Re-implementation gives us full control over hook placement and data extraction
- plip-rs already validates this approach across 7 architectures (incl. Gemma 2)
- The generic transformer (see §3) amortizes the re-implementation cost: one implementation covers ~80% of modern LLMs

**Owning the definitions extends to the whole life of the weights** (since
v0.1.20): backbones are differentiable end-to-end when built over a `VarMap`
(`nn_ops` `track_op` dispatch — inference stays byte-identical), with seeded
from-scratch initialization (`OthelloGpt::init`). A study that trains a model
and probes it runs *one* forward implementation for both, so a `GELU` variant
or `LayerNorm` bias can never become a train-vs-probe confound — the same
argument as re-implementing with hooks built in, carried to its conclusion.
Deliberately out of scope: optimizers, schedules, data loaders (see
`docs/dogfooding-feedbacks/trainable-backbones.md` §7; the SAE-trainer sibling
crate is the v0.2.0-era home for training *pipelines*).

**TransformerLens hook points as the reference API:**

| Hook Point | Location | Shape |
|------------|----------|-------|
| `hook_embed` | After token embedding | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_pre` | Before layer i | `[batch, seq, d_model]` |
| `blocks.{i}.attn.hook_q` | Query vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_k` | Key vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_v` | Value vectors | `[batch, seq, n_heads, d_head]` |
| `blocks.{i}.attn.hook_scores` | Pre-softmax attention | `[batch, n_heads, seq_q, seq_k]` |
| `blocks.{i}.attn.hook_pattern` | Post-softmax attention | `[batch, n_heads, seq_q, seq_k]` |
| `blocks.{i}.hook_attn_out` | Attention output | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_mid` | Between attention and MLP | `[batch, seq, d_model]` |
| `blocks.{i}.mlp.hook_pre` | MLP pre-activation | `[batch, seq, d_mlp]` |
| `blocks.{i}.mlp.hook_post` | MLP post-activation | `[batch, seq, d_mlp]` |
| `blocks.{i}.hook_mlp_out` | MLP output | `[batch, seq, d_model]` |
| `blocks.{i}.hook_resid_post` | After full layer | `[batch, seq, d_model]` |
| `hook_final_norm` | After final norm | `[batch, seq, d_model]` |

Not all hooks need to be captured on every forward pass. A `HookSpec` enum selects which hooks are active, so the default path has zero overhead.

### 2.2 What We Reuse from plip-rs

**Directly reusable (~3500 lines, ~30% of plip-rs):**

| Module | Lines | Status |
|--------|-------|--------|
| `model.rs` → `backend.rs` | ~700 | Rename `PlipBackend` → `MIBackend`; keep trait methods, `PlipModel` → `MIModel`, `PlipTokenizer` → `MITokenizer` |
| `intervention.rs` | ~1700 | Copy as-is; attention knockout/steering are 100% model-agnostic; state interventions stay as optional trait methods |
| `kv_cache.rs` | ~200 | Copy as-is; pure tensor storage |
| `cache.rs` | ~100 | Copy as-is; per-layer activation storage |
| `masks.rs` | ~150 | Copy as-is; causal/generation masks with caching |
| `logit_lens.rs` | ~200 | Copy as-is; architecture-agnostic hidden→vocab projection |
| `positioning.rs` | ~200 | Copy as-is; character↔token mapping |
| `steering.rs` | ~300 | Refactor: remove Python/Rust labels, keep calibration+dose-response logic |
| `tokenizer_rwkv.rs` | ~300 | Keep as optional feature (`feature = "rwkv-tokenizer"`) |

**Not reused directly (architecture-specific, refactored into generic backends, ~8000 lines):**

| Module | Reason |
|--------|--------|
| `forward.rs` (StarCoder2) | Replaced by generic transformer |
| `forward_qwen2.rs` (Qwen2) | Replaced by generic transformer |
| `forward_gemma.rs` (Gemma) | Replaced by generic transformer |
| `forward_gemma2.rs` (Gemma 2) | Replaced by generic transformer |
| `forward_llama.rs` (LLaMA) | Replaced by generic transformer |
| `forward_phi3.rs` (Phi-3) | Replaced by generic transformer |
| `forward_rwkv6.rs` (RWKV-6) | Replaced by generic RWKV (see §4) |
| `clt.rs` | Refactor into `interp/clt.rs`; validated against circuit-tracer Python (90/90 top-10 matches) |
| `corpus.rs` | PLIP-paper-specific corpus format |
| `experiment.rs` | PLIP-paper-specific experiment runner |
| `probe.rs` | Move to optional `probing` feature or separate crate |

### 2.3 What We Change from plip-rs

1. **Trait redesign.** `PlipBackend` becomes `MIBackend` with a single hook-aware forward method:
   - `forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>` — captures and intervenes via `HookSpec`; replaces plip-rs's `forward_with_cache`, `forward_with_attention`, `forward_with_kv_cache`
   - `project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>` — logit projection (used by logit lens)
   - Metadata: `num_layers()`, `hidden_size()`, `vocab_size()`, `num_heads()`
   - Optional: `chat_template()`, `embedding_vector()`

2. **Hook-based architecture.** Instead of plip-rs's separate methods for each extraction type, use a single `HookSpec` that declares what to capture:
   ```rust
   let mut hooks = HookSpec::new();
   hooks.capture(HookPoint::AttnPattern(5));
   hooks.capture(HookPoint::ResidPost(5));
   hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
   let cache = model.forward(&tokens, &hooks)?;
   let attn = cache.require(&HookPoint::AttnPattern(5))?;
   ```

3. **Config-driven model loading.** Replace `ModelArchitecture::from_model_id()` string matching with `config.json` parsing that reads `model_type` and `architectures` fields (like HuggingFace does).

---

## 3. The Generic Transformer

### 3.1 Configuration Axes

Analysis of plip-rs's 6 transformer backends (StarCoder2, Qwen2, Gemma, Gemma 2, LLaMA, Phi-3) reveals ~12 configuration axes:

| Axis | Variants | Models |
|------|----------|--------|
| **Normalization** | `RmsNorm` / `LayerNorm` / `GemmaRmsNorm` (+1 to weight) | Most use RmsNorm; Gemma has custom variant |
| **Activation** | `Silu` / `Gelu` / `GeluApprox` (PyTorch tanh approx) | LLaMA/Qwen/Phi=Silu; StarCoder2/Gemma 2=GeluApprox; Gemma=Gelu |
| **QKV layout** | Separate Q,K,V projections / Fused single QKV | All separate except Phi-3 (fused) |
| **MLP layout** | Gated (gate+up→down) / Fused gated / Plain (fc→proj) | LLaMA/Qwen/Gemma=gated separate; Phi-3=gated fused; StarCoder2=plain |
| **QKV bias** | Yes / No | Qwen2 on Q,K,V; StarCoder2 on all projections |
| **Bias granularity** | `o_proj_bias`, `mlp_bias` (separate from QKV) | StarCoder2 has bias everywhere; Qwen2 only on Q,K,V |
| **Embedding** | Standard / Scaled (`* sqrt(hidden)`) | Only Gemma scales |
| **LM head** | Tied to embeddings / Separate | Controlled by `tie_word_embeddings` |
| **Gemma 2 extensions** | Attention logit soft-capping, final logit soft-capping, 4-norm per layer, custom attention scalar, alternating sliding window | Gemma 2 only |
| **Sliding window** | Global / Fixed window / Alternating (per-layer) | Mistral=fixed; Gemma 2=alternating |

### 3.2 Config Struct

```rust
pub struct TransformerConfig {
    // Dimensions
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,              // usually hidden_size / num_attention_heads
    pub intermediate_size: usize,
    pub vocab_size: usize,

    // Architecture axes
    pub norm_type: NormType,           // RmsNorm | LayerNorm | GemmaRmsNorm
    pub norm_eps: f64,
    pub activation: Activation,        // Silu | Gelu | GeluApprox
    pub qkv_layout: QkvLayout,        // Separate | Fused
    pub mlp_layout: MlpLayout,        // GatedSeparate | GatedFused | Plain
    pub qkv_bias: bool,
    pub o_proj_bias: bool,             // StarCoder2
    pub mlp_bias: bool,                // StarCoder2
    pub embedding_scale: Option<f64>,  // None or Some(sqrt(hidden_size))
    pub tie_word_embeddings: bool,

    // Positional encoding
    pub rope_theta: f64,
    pub max_position_embeddings: usize,

    // Gemma 2 extensions
    pub attn_logit_softcapping: Option<f64>,   // Some(50.0) for Gemma 2
    pub final_logit_softcapping: Option<f64>,  // Some(30.0) for Gemma 2
    pub query_pre_attn_scalar: Option<f64>,    // Some(256.0) for Gemma 2
    pub use_post_norms: bool,                  // 4-norm per layer (Gemma 2)

    // Sliding window attention
    pub sliding_window: Option<usize>,         // Mistral, Gemma 2
    pub alternating_sliding_window: bool,      // Gemma 2
}
```

### 3.3 Coverage

With this config, **one implementation** covers:

The **Validated** column uses three tiers:
- **exact-parity** — top-10 next-token logits match a PyTorch oracle to `<1e-3`
  (CPU) via a frozen `scripts/*_validation.py` reference + `tests/validate_*.rs`.
  This is the only tier that catches subtle numerical bugs (e.g. a dropped
  `rope_scaling`).
- **smoke** — a plausibility check only (the expected token appears in the
  top-k); a subtly-wrong forward pass can still pass.
- **—** — not yet validated.

| Model Family | Config Notes | Validated |
|--------------|-------------|-----------|
| **LLaMA 1/2/3 / Code-LLaMA** | GQA, SiLU, RmsNorm, separate lm_head; llama3 `rope_scaling` | LLaMA 3.2 1B (exact-parity) |
| **Qwen 2 / 2.5** | GQA, SiLU, RmsNorm, QKV bias, conditional tied embeddings | Qwen2.5-Coder-3B (exact-parity) |
| **Gemma 1 / CodeGemma** | GQA, GELU, GemmaRmsNorm, sqrt embedding scale, tied lm_head | — |
| **Gemma 2** | + GeluApprox, soft-capping, 4-norm, custom attn scalar, alternating sliding window | Gemma 2 2B (exact-parity) |
| **Phi-3 / Phi-4** | GQA, SiLU, RmsNorm, fused QKV, fused MLP | Phi-3 Mini 4K (exact-parity) |
| **StarCoder2** | GQA, GeluApprox, LayerNorm, plain MLP, bias everywhere, tied lm_head | StarCoder2 3B (exact-parity) |
| **Mistral / Mixtral** (dense layers) | GQA, SiLU, RmsNorm, sliding window | Mistral 7B v0.1 (exact-parity, F32 CPU + GPU; also fast BF16 GPU) |
| **DeepSeek-Coder** (dense layers) | GQA, SiLU, RmsNorm, **linear `rope_scaling`** | DeepSeek-Coder 1.3B (exact-parity) |
| **Qwen 3** | GQA, SiLU, RmsNorm, QK-norm | Qwen3-1.7B-Base (exact-parity) |
| **Yi** | GQA, SiLU, RmsNorm (LLaMA-like) | — |
| **InternLM 2** | GQA, SiLU, RmsNorm; non-standard `wqkv`/`wo` weight names (won't load via the standard loader) | — |

**`rope_scaling` schemes.** Implemented and exact-parity-validated: `linear`
(DeepSeek-Coder), `llama3` (Llama 3.1/3.2), and `longrope` (Phi-3.5-mini,
Phi-3-medium-128k); `linear`/`llama3` are also read under the newer
`rope_parameters` key. `longrope` uses per-dimension `short_factor`/`long_factor`
arrays (the **short** factors for sequences ≤ `original_max_position_embeddings`
= 4096, **long** beyond) plus an `attention_factor` (mscale) on cos/sin, read
from config when set else derived as `sqrt(1 + ln(factor)/ln(orig_max))`;
**both regimes** are validated end-to-end against `microsoft/Phi-3.5-mini-instruct`
(exact, 5e-5; short + a >4096-token long case). Any other scheme (`yarn`,
`dynamic`, …) errors at config-parse time. A config-key coverage audit
(`TransformerConfig::audit_config_coverage`) flags unrecognized keys so a new
scheme cannot be silently dropped.

**VRAM budget (F32, RTX 5060 Ti 16 GB).** Since v0.0.3+, candle-mi defaults to F32 for research-grade precision (see §8, decision 14). This doubles the VRAM footprint vs BF16 but gives exact numerical parity with Python/PyTorch. Non-validated families that fit on 16 GB at F32:

| Model Family | Smallest Size | F32 VRAM est. |
|--------------|---------------|---------------|
| Gemma 1 / CodeGemma | 2B (~2.5 B actual) | ~12 GB |
| DeepSeek-Coder (dense) | 1.3B | ~7 GB |
| InternLM 2 | 1.8B | ~9 GB |

The 3B+ variants (Yi 6B, Phi-4-mini 3.8B, 7B models) exceed 16 GB at F32 and require BF16 or larger GPUs.

### 3.4 What It Does NOT Cover

| Architecture | Why | Solution |
|-------------|-----|----------|
| **Mixture-of-Experts** (Mixtral, DeepSeek-V2/V3) | Router + expert selection | Separate `MoETransformer` implementation |
| **RWKV / Linear RNN** | Fundamentally different (no attention matrix) | Separate generic RWKV backend (see §4) |
| **Mamba / SSM** | Selective state space; different recurrence | Separate backend |
| **Encoder-only** (BERT, etc.) | Bidirectional attention, [MASK] token | Different forward pass structure |
| **Encoder-decoder** (T5, etc.) | Cross-attention between encoder and decoder | Different forward pass structure |
| **Very old architectures** (GPT-2, GPT-J) | Absolute position embeddings, post-norm | Planned for Phase 9 (GPT-2 family + Pythia) |
| **Codestral** (Mistral) | Only available at 22 B; exceeds 16 GB MI GPU budgets | Would need ≥48 GB VRAM (A6000, etc.) |

### 3.5 Config Parsing from HuggingFace `config.json`

The generic transformer reads `model_type` from `config.json` and maps it to a `TransformerConfig`:

```rust
impl TransformerConfig {
    pub fn from_hf_config(config: &serde_json::Value) -> Result<Self> {
        let model_type = config["model_type"].as_str()
            .ok_or_else(|| MIError::Config("missing 'model_type'".into()))?;
        match model_type {
            "llama" => Self::parse_llama(config),
            "qwen2" => Self::parse_qwen2(config),
            "gemma" => Self::parse_gemma(config),
            "gemma2" => Self::parse_gemma2(config),
            "phi3" => Self::parse_phi3(config),
            "starcoder2" => Self::parse_starcoder2(config),
            "mistral" => Self::parse_mistral(config),
            // Phase 5: unknown model_type will fall back to from_hf_config_auto()
            other => Err(MIError::Config(format!("unsupported model_type: '{other}'"))),
        }
    }
}
```

Each `parse_*` function reads the relevant fields and sets the config knobs. Adding a new model family = adding one `parse_*` function (~30 lines), not a new forward pass implementation.

### 3.6 Weight Name Mapping

Different model families use different weight naming conventions. All 7 validated families happen to share the same `model.layers.{i}.self_attn.*` / `model.layers.{i}.mlp.*` naming scheme (LLaMA-style), so weight loading is handled directly in `GenericTransformer::load()` using `VarBuilder` path prefixes. If a future model family requires different weight names, a `WeightMap` trait can be introduced at that point.

---

## 4. Generic RWKV / Linear RNN Support

### 4.1 The RWKV Family

| Version | Name | Year | Key Innovation | State Shape |
|---------|------|------|----------------|-------------|
| **RWKV-4** | Dove | 2023 | Scalar WKV with normalization denominator | Vector (scalar per channel) |
| **RWKV-5** | Eagle | 2024 | Matrix-valued states, multi-head, gating | `[n_heads, head_dim, head_dim]` |
| **RWKV-6** | Finch | 2024 | Data-dependent decay + token shift (LoRA) | `[n_heads, head_dim, head_dim]` |
| **RWKV-7** | Goose | 2025 | Generalized delta rule (diag + rank-1 state transition) | `[n_heads, head_dim, head_dim]` |

**Status:** RWKV team has declared v4-v6 archived. v7 is the active version. plip-rs implements v6.

### 4.2 Available Models (HuggingFace, safetensors)

| Version | Sizes | HF Repos |
|---------|-------|----------|
| **RWKV-4** | 169M, 430M, 1.5B, 3B, 7B, 14B | `RWKV/rwkv-4-*-pile`, `RWKV/rwkv-raven-*` |
| **RWKV-5** | 7B | `RWKV/v5-Eagle-7B-HF` |
| **RWKV-6** | 1.6B, 3B, 7B, 14B | `RWKV/v6-Finch-*-HF` |
| **RWKV-7** | 0.1B, 0.4B, 1.5B, 2.9B | `RWKV/RWKV7-Goose-*-HF` (fla format) |

### 4.3 Architectural Comparison

All RWKV versions share the same block structure:

```
Input → Embedding → [Pre-LN (layer 0 only)]
  → for each layer:
      → LN1 → TimeMix (recurrence) → residual add
      → LN2 → ChannelMix (FFN) → residual add
  → Final LN → LM Head → logits
```

The differences are **inside TimeMix** (the recurrence formula) and **inside ChannelMix** (the FFN):

#### TimeMix Recurrence

| Component | RWKV-4 | RWKV-5 | RWKV-6 | RWKV-7 |
|-----------|--------|--------|--------|--------|
| **Token shift** | Static lerp (`mu * x + (1-mu) * prev`) | Static lerp | Data-dependent lerp (LoRA: `A[D,32] @ B[32,D]`) | Static lerp (reverted from v6) |
| **State transition** | `N/D` ratio (scalar state with denominator) | `diag(w) * S + k^T @ v` (matrix state) | `diag(w_t) * S + k^T @ v` (w_t data-dependent via LoRA) | `(diag(w_t) + a^T @ b) * S + v^T @ k` (diag + rank-1) |
| **Output** | `sigmoid(r) * N/D` | `r @ (diag(u) * k^T@v + S_{t-1})` | Same as v5 | `r @ S_t` (uses S_t not S_{t-1}) |
| **Gate** | None | SiLU gate | SiLU gate | LoRA-based gate |
| **GroupNorm** | None | After WKV output | After WKV output | After WKV output |
| **Bonus** | `time_first` (scalar) | `time_faaaa` (vector) | `time_faaaa` (vector) | `bonus` (vector) |

#### ChannelMix (FFN)

| Component | RWKV-4/5/6 | RWKV-7 |
|-----------|------------|--------|
| **Structure** | Receptance-gated: `sigmoid(r) * (W_v @ sqrelu(W_k @ x))` | Plain 2-layer MLP: `W_v @ sqrelu(W_k @ x)` |
| **Activation** | Squared ReLU | Squared ReLU |
| **Token shift** | Static lerp (v4/v5) or data-dependent (v6) | Static lerp |
| **Hidden ratio** | `(hidden * 7/2) / 32 * 32` (implicit) | `hidden * hidden_ratio` (explicit in config, typically 4.0) |

#### Config Fields

| Field | v4 | v5 | v6 | v7 |
|-------|----|----|----|----|
| `model_type` | `rwkv` | `rwkv5` | `rwkv6` | `rwkv7` |
| Head size | N/A | `head_size: 64` | `head_size: 64` + misleading `num_attention_heads` | `head_dim: 64` |
| Intermediate | Explicit | Null (computed) | Null (computed) | Explicit |
| LoRA dims | N/A | N/A | Hardcoded (32, 64) | Explicit in config (`a_low_rank_dim`, `decay_low_rank_dim`, etc.) |
| Norm epsilon | `layer_norm_epsilon` | `layer_norm_epsilon` | `layer_norm_epsilon` | `norm_eps` |
| Rescale | `rescale_every: 6` | `rescale_every: 6` | `rescale_every: 6` | Removed |

#### Weight Name Prefixes

| Component | v4 (HF) | v5 (HF) | v6 (HF) | v7 (fla) |
|-----------|---------|---------|---------|---------|
| Block prefix | `rwkv.blocks.{i}` | `blocks.{i}` | `rwkv.blocks.{i}` | `model.layers.{i}` |
| Attn module | `.attention` | `.attention` | `.attention` | `.attn` |
| FFN module | `.feed_forward` | `.feed_forward` | `.feed_forward` | `.ffn` |
| Projections | `key/value/receptance/output` | Same + `gate` | Same + `gate` | `r_proj/k_proj/v_proj/o_proj` |
| Embedding | `rwkv.embeddings` | `embeddings` | `rwkv.embeddings` | `model.embeddings` |
| Final norm | `rwkv.ln_out` | `ln_out` | `rwkv.ln_out` | `model.norm` |
| LM head | `head` | `head` | `head` | `model.lm_head` |

### 4.4 Generic RWKV Design

Despite the differences, a **partially generic implementation** is feasible:

```rust
pub struct RwkvConfig {
    pub version: RwkvVersion,          // V6 | V7 (v4/v5 archived; add if community demand)
    pub hidden_size: usize,
    pub num_layers: usize,
    pub head_dim: usize,               // 64 for all current models
    pub num_heads: usize,              // hidden_size / head_dim
    pub vocab_size: usize,
    pub norm_eps: f64,
    pub intermediate_size: usize,      // explicit for v7, computed for v6

    // Version-specific
    pub rescale_every: Option<usize>,  // v6: Some(6), v7: None
    pub head_size_divisor: Option<f64>, // v6: Some(8.0), v7: None
    pub lora_dims: Option<RwkvLoraDims>, // v6: hardcoded, v7: from config
    pub hidden_ratio: Option<f64>,     // v7: Some(4.0), v6: None
}

pub enum RwkvVersion { V6, V7 }
```

**Shared code (v6 + v7):**
- Block structure (LN → TimeMix → residual → LN → ChannelMix → residual)
- LayerNorm (with bias)
- GroupNorm after WKV output
- Embedding → blocks → final LN → LM head
- KV state encoding in cache
- Weight loading framework (version-aware name mapping)

**Version-specific code (must be separate):**
- WKV recurrence formula (the core ~50 lines per version)
- Token shift mechanism (v6: LoRA-based ddlerp; v7: static lerp)
- ChannelMix (v6: receptance-gated; v7: plain MLP)
- Config parsing (different field names per version)
- Weight name mapping

**Recommended approach:** A `GenericRwkv` struct with version-specific `WkvKernel` and `ChannelMixKernel` trait objects:

```rust
trait WkvKernel {
    fn step(&self, r: &Tensor, k: &Tensor, v: &Tensor, w: &Tensor,
            state: &mut Tensor, bonus: &Tensor) -> Result<Tensor>;
}

trait ChannelMixKernel {
    fn forward(&self, x: &Tensor, prev_x: &Tensor) -> Result<Tensor>;
}
```

Two implementations of `WkvKernel` (v6 diagonal decay, v7 diag+rank-1), two implementations of `ChannelMixKernel` (receptance-gated for v6, plain MLP for v7). The outer `GenericRwkv` struct handles the shared block structure and delegates to the kernels. V4/V5 can be added later as additional kernel implementations if community demand materializes.

### 4.5 Related Linear RNN / SSM Architectures

The `fla` (flash-linear-attention) library unifies RWKV-7, GLA, RetNet, HGRN2, DeltaNet, and Mamba-2 under a common framework in Python. All share the recurrence pattern:

```
S_t = A_t * S_{t-1} + B_t    (state transition + input injection)
y_t = C_t * S_t               (output extraction)
```

Where `A_t` (transition), `B_t` (input), and `C_t` (output) vary by architecture:

| Architecture | A_t (transition) | B_t (input) | C_t (output) |
|-------------|-----------------|-------------|-------------|
| **RetNet** | `gamma * I` (fixed scalar) | `k^T @ v` | `q` |
| **GLA** | `diag(sigmoid(g))` (full gate) | `k^T @ v` | `q` |
| **RWKV-5** | `diag(w)` (diagonal decay) | `k^T @ v` | `r` |
| **RWKV-6** | `diag(w_t)` (data-dependent diagonal) | `k^T @ v` | `r` |
| **RWKV-7** | `diag(w_t) + a^T @ b` (diag + rank-1) | `v^T @ k` | `r` |
| **DeltaNet** | `I - eta * k^T @ k` (delta rule) | `eta * v^T @ k` | `q` |
| **Mamba-2** | `alpha * I` (scalar, discretized SSM) | `B * x` | `C` |

**Implication for the crate:** A truly generic linear RNN backend is possible using the same `WkvKernel` trait pattern. Each architecture provides its own `step()` implementation. The outer block structure (norm → recurrence → residual → norm → FFN → residual) is shared.

**Prioritization:** RWKV-6 first (validated by plip-rs), then RWKV-7 (available on HF). Mamba/GLA/RetNet as future extensions.

---

## 5. MI Capabilities

### 5.1 Core (from plip-rs, reusable)

| Capability | Status | Notes |
|-----------|--------|-------|
| **Attention extraction** | ✅ Working (transformers) | Per-layer, per-head attention patterns via `HookPoint::AttnPattern` |
| **Attention knockout** | ✅ Spec types ready | Pre-softmax `-inf` masking via `Intervention::Knockout`; measures KL divergence |
| **Attention steering** | ✅ Spec types ready | Post-softmax scaling/setting + renormalization via `Intervention::Steer` |
| **Steering calibration** | ✅ Infrastructure ready | Baseline measurement, dose-response curves |
| **State knockout** (RWKV) | ✅ Working (Phase 2) | Zero specific head states via `HookSpec::set_state_knockout()`; skips kv write (`state = decay * state`). V6 + V7 |
| **State steering** (RWKV) | ✅ Working (Phase 2) | Scale kv write via `HookSpec::set_state_steering()` (`state = scale * kv + decay * state`). V6 + V7 |
| **Effective attention** (RWKV) | ✅ Working (Phase 2) | `[b,h,t,t]` attention-equivalent from WKV recurrence via `HookPoint::RwkvEffectiveAttn`. V6 (prefix-sum) + V7 (backward linear functional) |
| **Logit lens** | ✅ Infrastructure ready | Per-layer vocab projection via `project_to_vocab` |
| **Activation caching** | ✅ Working | Per-layer hidden state storage (`ActivationCache`, `FullActivationCache`) |
| **KV cache** | Not implemented | `generate` re-runs a full forward per step. `cache::kv::KVCache` exists but has no caller and is **not public** as of v0.2.0; its docs say so. Restoring it is one `pub use` once an incremental-decode path exists |
| **Position mapping** | ✅ Working | Character offset ↔ token index conversion (`PositionConversion`) |

### 5.2 New (required for Melomētis, general-purpose)

| Capability | Priority | Notes |
|-----------|----------|-------|
| **CLT loading + encoding** | ✅ Working (Phase 3) | `CrossLayerTranscoder` struct; per-file download via `hf-fetch-model`; `encode()` for full sparse activations, `top_k()` for strongest features; validated on Gemma 2 2B (8/8 top-1 match vs Python HF, <5% relative error) |
| **CLT feature injection** | ✅ Working (Phase 3) | `cache_steering_vectors_all_downstream()` + `prepare_hook_injection()` for multi-layer causal interventions; `Intervention::Add` at `ResidPost` with dtype coercion; melometis position-sweep reproduced (last-position L2 ranks #1 in both Rust and Python) |
| **Attribution graphs** | ✅ Working (Phase 3) | `AttributionEdge`, `AttributionGraph` with `top_k()`/`threshold()` pruning; `score_features_by_decoder_projection()`, `build_attribution_graph()` |
| **SAE loading + encoding** | ✅ Working (Phase 4) | `SparseAutoencoder` struct; Gemma Scope NPZ format; `encode()` for sparse activations; validated on Gemma 2 2B |
| **SAE feature injection** | ✅ Working (Phase 4) | Same as CLT but for SAEs; `Intervention::Add` at hook points |
| **Activation patching** | ✅ Working (Phase 5/v0.1.0+) | Causal tracing via position-specific patching (Meng et al.); `activation_patching.rs` example with full layer × token heatmap (v0.1.3); generic `activation_patch()` framework planned for Phase 7 |
| **Residual stream decomposition** | High (Phase 7) | Decompose residual stream into per-layer, per-component contributions |

### 5.3 Future (not required now)

| Capability | Notes |
|-----------|-------|
| **Probing** | Linear probes on activations (Phase 10+) |
| **Causal scrubbing** | Systematic causal intervention framework (subsumed by activation patching, Phase 7) |
| **Indirect object identification** | IOI-style circuit analysis (enabled by Phase 7 activation patching + Phase 10+ head detection) |
| **Induction head detection** | Automated induction head finding (Phase 10+) |
| **Feature visualization** | Export attention/activation data for visualization tools (Phase 10+); deloson ([live demo](https://PCfVW.github.io/deloson/)) already consumes plip-rs layer scan JSON — candle-mi should preserve this output format |

---

## 6. Crate Structure

```
candle-mi/
├── Cargo.toml
├── LICENSE-MIT
├── LICENSE-APACHE
├── README.md
├── ROADMAP.md
├── CHANGELOG.md
├── CONVENTIONS.md                — Code conventions (unsafe policy, annotations, etc.)
├── design/                       — Design proposals (one Markdown file per decision, see §8)
├── src/
│   ├── lib.rs                      — Public API, feature gates, re-exports
│   │
│   ├── backend.rs                  — MIBackend trait, MIModel wrapper, from_pretrained, sampling
│   ├── hooks.rs                    — HookPoint, HookSpec, HookCache, Intervention enum
│   ├── config.rs                   — TransformerConfig, enums, per-family config parsers
│   ├── error.rs                    — MIError enum (thiserror)
│   │
│   ├── transformer/                — Generic transformer (feature: "transformer") ✅
│   │   ├── mod.rs                  — GenericTransformer, MIBackend impl, load()
│   │   ├── attention.rs            — Multi-head attention (GQA/MHA/MQA, separate + fused QKV)
│   │   ├── mlp.rs                  — MLP variants (gated separate, gated fused, plain)
│   │   ├── norm.rs                 — RmsNorm, LayerNorm, GemmaRmsNorm
│   │   └── rope.rs                 — Rotary position embeddings (pre-computed cos/sin)
│   │
│   ├── rwkv/                       — Generic RWKV (feature: "rwkv") ✅ Phase 2
│   │   ├── mod.rs                  — GenericRwkv struct, version dispatch, WKV kernels
│   │   ├── config.rs               — RwkvConfig, version-aware parsing
│   │   └── norm.rs                 — LayerNorm (with bias, RWKV-style)
│   │
│   ├── clt/                        — Cross-layer transcoder (feature: "clt") ✅ Phase 3
│   │   └── mod.rs                  — CrossLayerTranscoder, CltConfig, CltFeatureId, SparseActivations, encode/top_k/inject
│   │
│   ├── sae/                        — Sparse autoencoder (feature: "sae") ✅ Phase 4
│   │   ├── mod.rs                  — SparseAutoencoder, SaeConfig, encode/inject
│   │   └── npz.rs                  — NPZ parser for Gemma Scope SAE weights
│   │
│   ├── interp/                     — Interpretability tools ✅ (core)
│   │   ├── mod.rs
│   │   ├── intervention.rs         — Knockout, steering, ablation spec types
│   │   ├── steering.rs             — Calibration, dose-response
│   │   └── logit_lens.rs           — Per-layer vocab projection
│   │
│   ├── cache/                      — Caching infrastructure ✅
│   │   ├── mod.rs
│   │   ├── activation.rs           — ActivationCache, FullActivationCache
│   │   ├── kv.rs                   — KVCache
│   │   └── attention.rs            — AttentionCache
│   │
│   ├── tokenizer/                  — Tokenizer abstraction ✅
│   │   ├── mod.rs                  — MITokenizer (HuggingFace tokenizers wrapper)
│   │   └── rwkv.rs                 — RWKV World tokenizer (feature: "rwkv-tokenizer")
│   │
│   ├── download.rs                 — download_model / download_model_blocking (hf-fetch-model) ✅
│   ├── memory.rs                   — MemorySnapshot, MemoryReport, sync_and_trim_gpu (feature: "memory") ✅
│   │
│   └── util/                       — Shared utilities ✅
│       ├── mod.rs
│       ├── masks.rs                — Causal/generation masks with caching
│       ├── pca.rs                  — PCA via power iteration with deflation ✅
│       └── positioning.rs          — Character ↔ token mapping
│
├── examples/                       — 23 examples covering MI techniques ✅
│   ├── quick_start_transformer.rs  — Load model, forward pass, print top tokens ✅
│   ├── quick_start_sae.rs          — Load SAE, encode activations, print top features ✅
│   ├── fast_download.rs            — Parallel multi-connection model download ✅
│   ├── auto_config_dogfood.rs      — Auto-config + compatibility check dogfooding ✅
│   ├── logit_lens.rs               — Per-layer prediction tracking ✅
│   ├── attention_patterns.rs       — Attention weight visualization ✅
│   ├── attention_knockout.rs       — Zero-out specific attention heads ✅
│   ├── activation_patching.rs      — Causal intervention via patching ✅
│   ├── counterfact_patching.rs     — CounterFact patching (Transluce protocol replication) ✅
│   ├── factual_routing.rs          — Attention-routing shifts from output to subject under patching ✅
│   ├── steering_dose_response.rs   — Attention steering with dose curves ✅
│   ├── steering_convergence.rs     — Does residual injection converge to natural computation? ✅
│   ├── figure13_planning_poems.rs  — CLT-based circuit analysis (suppress + inject sweep) ✅
│   ├── attention_routing.rs        — How CLT suppress+inject reshapes output-position attention ✅
│   ├── correction_test.rs          — Late-layer correction of a prolepsis commitment ✅
│   ├── clt_probe.rs                — CLT encoder/decoder probe for feature discovery ✅
│   ├── character_count_helix.rs    — Helix manifold replication (Gurnee et al.) ✅
│   ├── token_positions.rs          — Token ↔ character offset mapping ✅
│   ├── generate.rs                 — Autoregressive text generation ✅
│   ├── rwkv_inference.rs           — RWKV model inference ✅
│   ├── recurrent_feedback.rs       — Anacrousis recurrent CLT feedback ✅
│   ├── stoicheia_inference.rs      — AlgZoo tiny-model inference + ground-truth check ✅
│   └── README.md                   — Example descriptions and usage instructions ✅
│
├── scripts/                        — Validation scripts and reference data
│   ├── README.md                   — Validation script docs and regeneration instructions ✅
│   ├── rwkv7_validation.py         — Python RWKV-7 reference output generator ✅
│   ├── rwkv7_validation_comparison.md — Rust vs Python RWKV-7 comparison ✅
│   ├── clt_position_sweep_validation.py — Python CLT position-sweep reference (Gemma 2) ✅
│   ├── clt_position_sweep_validation_llama.py — Python CLT position-sweep reference (Llama 3.2) ✅
│   ├── clt_position_sweep_comparison.md — Rust vs Python CLT comparison ✅
│   ├── sae_validation.py           — Python SAE reference output generator ✅
│   ├── rwkv6_reference.json        — RWKV-6 reference logits ✅
│   ├── rwkv7_reference.json        — RWKV-7 reference logits ✅
│   ├── clt_position_sweep_reference.json — CLT reference activations ✅
│   ├── anacrousis_reference.json   — Anacrousis reference data ✅
│   └── sae_reference.json          — SAE reference activations ✅
│
└── tests/
    ├── validate_models.rs          — Per-family transformer validation (CPU + GPU) ✅
    ├── validate_rwkv6.rs           — RWKV-6 validation against plip-rs reference ✅
    ├── validate_rwkv7.rs           — RWKV-7 validation (CPU F32 + GPU F32 + GPU BF16) ✅
    ├── validate_clt.rs             — CLT encode/inject + melometis position-sweep ✅
    ├── validate_anacrousis.rs      — Anacrousis recurrent feedback validation ✅
    ├── validate_sae.rs             — SAE encode/inject validation ✅
    ├── bench_hook_overhead.rs      — Hook overhead benchmark ✅
    └── fast_download.rs            — Download integration test ✅
```

### 6.1 Feature Gates

```toml
[features]
default = ["cuda", "transformer"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
transformer = []           # Generic transformer backend
mmap = []                  # Memory-mapped weight loading (unsafe, for 7B+ models)
rwkv = []                  # RWKV v6-v7 backends
rwkv-tokenizer = []        # RWKV World tokenizer
clt = []                   # Cross-layer transcoder support
sae = []                   # Sparse autoencoder support
memory = ["dep:hypomnesis"]  # RAM + VRAM reporting (delegated to the hypomnesis crate)
memory-debug = ["memory", "hypomnesis/debug-output"]  # Raw GPU-backend debug values on stderr
probing = ["linfa", "linfa-logistic", "ndarray"]  # Linear probing
```

### 6.2 Documentation

| Deliverable | Format | Contents |
|-------------|--------|----------|
| **README.md** | Markdown | Quick start (load a model, run logit lens in 10 lines), feature overview, supported models table, hardware requirements, link to docs.rs |
| **Rustdoc** (`cargo doc`) | Inline | Crate-level overview in `lib.rs`; module-level docs for every public module; doc-tests on all public functions and key types |
| **BACKENDS.md** | Markdown | Step-by-step guide to adding a new model architecture: config parser, weight map, validation protocol |
| **HOOKS.md** | Markdown | Hook point reference table (mirroring §2.1), intervention API walkthrough, worked examples (capture attention, run knockout, steer residual stream) |
| **CHANGELOG.md** | Markdown | [Keep a Changelog](https://keepachangelog.com/) format from v0.0.1 onwards |
| **Examples** | Rust (`examples/`) | 23 examples covering model loading, logit lens, attention patterns, knockout, steering, activation patching, CounterFact patching, factual routing, CLT circuits, CLT feature probing, prolepsis correction tests, recurrent CLT feedback, SAE, RWKV, AlgZoo/Stoicheia inference + MI analysis, auto-config, tokenization, generation, fast download, character count helix, steering convergence, attention routing — each self-contained with inline comments ✅ |

**Rustdoc policy:** Every `pub` item must have a doc comment. Types include a one-line summary + "# Examples" section with a runnable doc-test. `#![warn(missing_docs)]` enforced at crate level.

---

## 7. Phased Development Plan

### 7.0 Git Workflow

**Code quality — zero tolerance for warnings from day one.** Every commit must satisfy all three checks:

1. `cargo build --release` — zero errors, zero warnings (set `#![deny(warnings)]` in `lib.rs`)
2. `cargo clippy -- -W clippy::pedantic` — zero warnings (suppress individual false positives with targeted `#[allow(...)]` + a comment explaining why, never blanket `#[allow(clippy::pedantic)]`)
3. `cargo fmt --check` — zero formatting diffs (run `cargo fmt` before every commit)

CI enforces the same three checks on every push. A red CI is treated as a blocking bug, not a "fix later" item. This policy applies from the very first commit (Phase 0 repo scaffold) and at every subsequent step — there is no grace period.

**Commit granularity:** One commit per logical unit — a single module port, a new config parser, a passing validation, a bug fix. Each commit must pass the three code-quality checks above. Avoid monolithic "implement phase N" commits.

**Push cadence:** Push at the end of each working session (backup) and always at the milestones marked with **`PUSH`** below. Every push to `main` must pass CI (build + clippy + fmt + tests).

**Branch strategy:** Work directly on `main` during solo development. Use short-lived feature branches only when a change spans multiple sessions and may leave `main` broken in between; merge back when green.

**Tag convention:** Tag at each phase completion: `v0.0.1-phase0`, `v0.0.2-phase1`, etc. Tag `v0.1.0` at publication (Phase 5). Post-v0.1.0: patch releases for incremental improvements within the current phase (`v0.1.1`, `v0.1.2`, `v0.1.3`, ...); minor bumps at phase boundaries: `v0.2.0` (Phase 6), `v0.3.0` (Phase 7), `v0.4.0` (Phase 8), `v0.5.0` (Phase 9). Tags matching `v*` trigger `publish.yml`, which runs full CI then `cargo publish` automatically. **Always wait for `ci.yml` green before tagging** — bump `Cargo.toml` version + commit `Cargo.lock` first.

### Phase 0: Foundation

**Goal:** Core trait, hook system, infrastructure — no model backends yet.

- [x] Create repository, Cargo.toml, CI, ROADMAP.md (this document), CHANGELOG.md (empty "Unreleased" section), LICENSE-MIT, LICENSE-APACHE, `design/` directory (design proposals from §8) — **commit: `init: scaffold repo with CI, roadmap, design docs, and dual license`** — **PUSH**
- [x] Port `backend.rs` (MIBackend trait) from plip-rs `model.rs` — redesign with hook-based API — **commit**
- [x] Implement `HookSpec` and `HookCache` types — **commit**
- [x] Port `intervention.rs` (knockout, steering, ablation spec types and trait method signatures — attention-specific implementations are model-agnostic and land here; RWKV-specific state intervention *implementations* land in Phase 2) — **commit**
- [x] Port `cache/` modules (activation, KV, attention) — **commit**
- [x] Port `util/` modules (masks, positioning) — **commit** — **PUSH** (mid-phase checkpoint)
- [x] Port `interp/logit_lens.rs` — **commit**
- [x] Port `interp/steering.rs` (calibration, dose-response) — **commit**
- [x] Port `tokenizer/` (MITokenizer enum + RWKV tokenizer) — **commit**
- [x] Write comprehensive tests for all ported modules — **commit**

**Deliverable:** Compiling crate with full MI infrastructure, no backends. — **PUSH + tag `v0.0.1-phase0`** ✅ Published to [crates.io](https://crates.io/crates/candle-mi) on 2026-02-23.

### Phase 1: Generic Transformer

**Goal:** One forward pass implementation that covers LLaMA, Qwen2, Gemma, Gemma 2, Phi-3, StarCoder2, Mistral.

- [x] Implement `TransformerConfig` with all 7+ axes (~12 config fields) — **commit**
- [x] Implement config parsers for `llama`, `qwen2`, `gemma`, `gemma2`, `phi3`, `starcoder2`, `mistral` — **commit**
- [x] Implement generic forward pass (one commit per component):
  - [x] Embedding (with optional scaling) — **commit**
  - [x] RoPE (via `candle_nn::rotary_emb::rope()`) — **commit**
  - [x] Multi-head attention (GQA/MHA/MQA via `num_kv_heads`) — **commit**
  - [x] QKV projection (separate and fused) — **commit**
  - [x] MLP (gated, fused gated, plain) — **commit**
  - [x] Normalization (RmsNorm, LayerNorm, GemmaRmsNorm) — **commit**
  - [x] LM head (tied, separate, conditional) — **commit**
- [x] Forward pass compiles end-to-end — **PUSH**
- [x] Integrate hook points at all TransformerLens-equivalent locations — **commit**
- [x] Implement `MIBackend` for `GenericTransformer` — **commit**
- [x] Validate incrementally, one model family at a time (each adds 1–2 config axes):
  1. [x] **LLaMA** 3.2 1B — "Paris" #1 (CPU/GPU, exact match with Python HF) — **commit** — **PUSH**
  2. [x] **Qwen2** 2.5-Coder-3B — "Paris" #1 (CPU/GPU) — **commit** — **PUSH**
  3. [x] **Gemma 2** 2B — "Paris" #8 (correct: logit softcapping flattens distribution) — **commit** — **PUSH**
  4. [x] **Phi-3** Mini 4K — "Paris" #1 (CPU/GPU) — **commit** — **PUSH**
  5. [x] **StarCoder2** 3B — "Hello" #1 (CPU/GPU) — **commit** — **PUSH**
  6. [x] **Mistral** 7B v0.1 — "Paris" #4 (CPU/GPU, exact match with Python HF) — **commit** — **PUSH**
- [x] Benchmark hook overhead on LLaMA 3.2 1B (16 layers × 12 hooks = 194 hook points):
  - GPU (CUDA F32): +11.5% overhead with full capture (originally measured at BF16; now runs at F32)
  - CPU (F32): within noise (zero overhead when inactive) — **commit**

**Deliverable:** `MIModel::from_pretrained("meta-llama/Llama-3.2-1B")` works with full MI support. — **PUSH + tag `v0.0.2-phase1`** ✅ Completed on 2026-02-25.

**Validation protocol:** For each model family, compare top-10 logits for 5 test prompts against Python HuggingFace outputs. Tolerance: abs < 1e-4 (F32). All tests now run at F32 on GPU (research-grade precision; see §8 decision 14). Validate in the incremental order above — if LLaMA works and Qwen2 breaks, the bug is in QKV bias or tied embeddings.

### Phase 2: RWKV-6 + RWKV-7

**Goal:** Port plip-rs RWKV-6 backend, add RWKV-7, validate both.

- [x] Implement `RwkvConfig` with version dispatch — **commit**
- [x] Implement shared RWKV block structure — **commit**
- [x] Implement version-specific token shift (static lerp + ddlerp) — **commit**
- [x] Implement version-specific channel mix (receptance-gated + plain) — **commit**
- [x] Implement RWKV weight name mapping per version — **commit**
- [x] Port RWKV-6 WKV kernel from plip-rs `forward_rwkv6.rs` — **commit**
- [x] Validate RWKV-6: compare against plip-rs reference outputs — **commit** — **PUSH** (RWKV-6 green)
- [x] Implement RWKV-7 WKV kernel (generalized delta rule) — **commit**
- [x] Validate RWKV-7: compare against fla Python reference — **commit** — **PUSH** (RWKV-7 green)
- [x] Port effective attention computation for RWKV-6 — **commit**
- [x] Derive effective attention for RWKV-7 (new: diag+rank-1 complicates the formula) — **commit**
- [x] Implement RWKV-specific state knockout + state steering (the `MIBackend` trait methods and spec types were ported in Phase 0; this provides the concrete implementations extracted from plip-rs `forward_rwkv6.rs`) — **commit**

**Deliverable:** `MIModel::from_pretrained("RWKV/RWKV7-Goose-World3-1.5B-HF")` works. ✅ — **PUSH + tag `v0.0.3`**

### Phase 3: CLT Support

**Goal:** Load and use pre-trained cross-layer transcoders. CLT infrastructure already validated in plip-rs (`src/clt.rs`, 1,640 lines) — port and generalise.

- [x] Single-file download API in `hf-fetch-model` (`download_file` / `download_file_blocking`) — replaces plip-rs's lazy `hf_hub::Api::repo().get()` pattern with chunked parallel transfer, checksum verification, and retry. See `design/hf-fetch-model-single-file-api.md`.
- [x] Port CLT weight loading from plip-rs (circuit-tracer format, per-file download via `hf-fetch-model`) — `CrossLayerTranscoder`, `CltConfig`, `CltFeatureId`, `SparseActivations` — **commit `e230a45`**
- [x] Port CLT encoding (activations → feature activations, sparse top-k) — `encode()`, `top_k()` — **commit `6e5b231`**
- [x] Port CLT feature injection (suppress+inject protocol from melometis/tragos) — `cache_steering_vectors_all_downstream()`, `prepare_hook_injection()`, `Intervention::Add` at `ResidPost` — **commit `39013d1`**
- [x] Validate: load Gemma 2 2B CLT, reproduce melometis position-sweep results — correlational (8/8 top-1 match, <5% relative error) + causal (last-position L2 ranks #1), cross-validated against Python HF reference — **commit `7d7bd96`** — **PUSH** (CLT pipeline green on Gemma 2)
- [x] Validate: load Llama 3.2 1B CLT (524K, `mntss/clt-llama-3.2-1b-524k`), reproduce tragos position-sweep results — second independent replication confirming the phenomenon generalises across architectures; config (16 layers, 2048 d_model, 32768 features/layer), encoding (5 layers), injection (L2=77.9), correlational sweep (8/11 unique top-1, Jaccard=0.000), causal sweep (last position #1, concentration 24.85x) — **commit** — **PUSH** (CLT pipeline green on Llama 3.2)
- [x] Implement attribution graph construction — `AttributionEdge`, `AttributionGraph` types with `top_k()`/`threshold()` pruning; `score_features_by_decoder_projection()` (single + batch), `extract_decoder_vectors()`, `build_attribution_graph()` convenience methods; 9 unit tests with synthetic safetensors files — **commit `a6fffd9`**
- [x] Implement recurrent feedback (anacrousis) — `RecurrentPassSpec` with prefill-only and sustained generation-time modes; re-run commitment layers with optional feedback injection at every autoregressive step; `forward_recurrent()` and `generate_recurrent()` — **commit**
- [x] Validate: replicate anacrousis results (28 conditions × 15 couplets; best 11/15 with unembed layers 8–15, scale 2.0) — `tests/validate_anacrousis.rs` (`#[ignore]`, requires CUDA + cached model) — **commit** — **PUSH** (anacrousis green)
- [x] Add `scripts/README.md` documenting validation scripts, reference data files, and regeneration instructions — **commit**

**Deliverable:** Full CLT pipeline on Gemma 2 2B and Llama 3.2 1B, plus anacrousis recurrent feedback experiment. — **PUSH + tag `v0.0.4-phase3`** ✅ Published to [crates.io](https://crates.io/crates/candle-mi) on 2026-03-05 — **commit `8acfa53`**.

### Phase 4: SAE Support

**Goal:** Load and use pre-trained sparse autoencoders.

- [x] Implement SAE weight loading (SAELens / Gemma Scope format) — `ee39a87`
- [x] Implement SAE encoding and feature injection — `ee39a87`
- [x] Validate: load Gemma Scope SAE, encode activations, verify reconstruction — `8adefae`, `fd07235`

**Deliverable:** SAE pipeline working alongside CLTs (`ee39a87`..`fa22d1b`). — **PUSH + tag `v0.0.5-phase4`** ✅

### Phase 5: Polish + Publish + Auto-Config

**Goal:** Auto-config for unknown model families, API polish, documentation, examples, crates.io v0.1.0.

- [x] Implement `from_hf_config_auto()` — generic config parser for unknown `model_type` values; reads `config.json` scalars (Tier 1–2) + safetensors tensor names (Tier 3: QKV/MLP layout, bias flags, norm type, post-norms) + `model_type` fixups (Tier 4: GemmaRmsNorm, embedding_scale, alternating_sliding_window). Two-tier dispatch: known families use existing parsers, unknown families use auto-parser. Includes `CompatibilityReport` preflight check that detects incompatible models (missing norms, projections, etc.) before weight loading. ~120 lines + ~20 lines tensor-name utilities + ~80 lines compatibility check. See `candle-mi-auto-config-brainstorming.md` for field-by-field derivation plan — **commit `9948bc0`**, **commit `ceee9ac`**
- [x] Validate auto-config against all 7 known families (must produce identical configs to manual parsers) — **commit `8037419`**
- [x] Audit public API surface (`pub` vs `pub(crate)`) — **commits `70649e9`, `2eedecf`, `8595a61`**
- [x] Write crate-level documentation with examples — **commit `c7cbf9e`**
- [x] Write `BACKENDS.md` — how to add a new model architecture — **commit `36b5208`**
- [x] Write `HOOKS.md` — hook point reference and intervention walkthrough — **commit `4384545`**
- [x] Write example programs — 23 examples covering forward pass, logit lens, attention patterns, knockout, steering, activation patching, CounterFact patching, factual routing, CLT circuits, CLT feature probing, prolepsis correction tests, recurrent CLT feedback, SAE, RWKV, AlgZoo/Stoicheia inference + MI analysis, auto-config, tokenization, generation, fast download, character count helix, steering convergence, attention routing — **multiple commits**
- [x] Improve auto-config error messaging — when `check_auto_compatibility()` fails for non-standard models (non-HF weight naming), provide actionable error messages listing which weight tensors were expected vs. found — **commit** `2685893`
- [x] Update CHANGELOG.md with Phase 5 changes — **commit `359a623`** — **PUSH** (release candidate)
- [x] **Release workflow** (publish v0.1.0 to crates.io — automated via `publish.yml`):
  1. ~~Ensure `main` is clean~~ — done
  2. ~~Bump version in `Cargo.toml` to `0.1.0` + update `Cargo.lock`~~ — done (commit `092431e`)
  3. ~~Wait for `ci.yml` to pass~~ — done
  4. ~~`git tag v0.1.0`~~ — done
  5. ~~`git push origin v0.1.0`~~ — done (triggers `publish.yml`)
  6. ~~`publish.yml` runs full CI checks then `cargo publish`~~ — done
  7. ~~Verify the published crate~~ — done; subsequently published v0.1.1, v0.1.2, v0.1.3
- [x] Submit PR to candle repo adding candle-mi to "Useful External Resources" — [huggingface/candle#3406](https://github.com/huggingface/candle/pull/3406) (submitted, awaiting merge)
- [ ] Announce (Rust ML community, MI community)

### Phase 6: Research Examples & Utilities

**Goal:** Finalize the character count helix example, strengthen PCA utilities, add 2–3 beginner MI examples from the [Nanda quickstart proposal](docs/quickstart-examples-proposal.md). Keep the phase small (~40% unplanned capacity reserved for bug fixes discovered by early users of v0.1.0).

- [x] Helix finalization (completed across v0.1.0–v0.1.3) — **multiple commits**
  - [x] PCA utility (`src/util/pca.rs`) — `pca_top_k()` via power iteration with deflation (v0.1.0)
  - [x] `character_count_helix.rs` example — `--scan-layers`, `--pca-layers`, `--text-dir`, `--max-tokens`, `--text`, `--output` (v0.1.0)
  - [x] Memory reporting API (`src/memory.rs`) — RAM + VRAM via DXGI/NVML/nvidia-smi, gated behind `features = ["memory"]` (v0.1.0–v0.1.2)
  - [x] `--sweep` mode with auto-resume from JSON output file (v0.1.2)
  - [x] Chunking for long sequences (v0.1.2)
  - [x] `--sweep N` and `--sweep all` for multi-layer sweeps (v0.1.3)
  - [x] VRAM-aware `max_tokens` auto-tuning (v0.1.3)
  - [x] Explicit GPU tensor cleanup with `sync_and_trim_gpu` (v0.1.3)
  - [x] Rotating helix GIF and experiment README (v0.1.3)
- [x] Activation patching extended — full causal trace heatmap (layer × token grid, Meng et al. Figure 1e), Mathematica plotting script (v0.1.3) — **commit `7f31d76`**
- [ ] PCA enhancements — `pca_n_omit` parameter in `pca_top_k()`, projection helpers (`project_to_pcs()`, `reconstruct_from_pcs()`), optional randomized SVD solver for large matrices — **commit**
- [ ] `inspect_weights.rs` example — load a model, print layer count / hidden size / head count / vocab size, per-layer weight matrix shapes and Frobenius norms, embedding stats. Lowest barrier to entry; no MI knowledge needed — **commit**
- [ ] `logit_attribution.rs` example — direct logit attribution: project each layer's residual contribution through unembedding, show per-layer contribution to target prediction. Core MI technique — **commit**
- [ ] Migrate `.npz` parsing logic from `candle-mi` to `anamnesis` — pure NPZ/NPY parsing (format transformation) lives in anamnesis (wraps `npyz` crate, adds bf16 layer); download integration (`download_and_parse_npz`) stays in hf-fetch-model as a thin wrapper over `anamnesis::parse_npz()`. Replace candle-mi's internal `npz.rs` with a dependency on `anamnesis/npz` — **commit**
- [ ] Add `exaone4` auto-config support (`parse_exaone4()` in `config.rs`) — LLaMA-like with alternating sliding window / full attention ("LLLG" pattern), similar to Gemma 2's alternating scheme. Driven by anamnesis Phase 1 FP8 test model (`LGAI-EXAONE/EXAONE-4.0-1.2B-FP8`). Both architectures are mainstream and worth supporting independently of anamnesis — **commit**
- [ ] Add `qwen3` auto-config support (`parse_qwen3()` in `config.rs`) — extends `qwen2` with QK LayerNorm (`q_norm` / `k_norm` per layer). Requires handling the extra norm tensors in the forward pass. Driven by anamnesis Phase 1 FP8 test model (`Qwen/Qwen3-1.7B-FP8`) — **commit**
- [ ] ~40% unplanned: bug fixes, documentation gaps, user-reported issues — **commits as needed** — **PUSH**

**Deliverable:** Polished examples, stronger PCA, bug fixes. — **PUSH + tag `v0.2.0`**

### Phase 7: Standard MI Analysis Stack

**Goal:** The core causal analysis toolkit — transforms candle-mi from "model loader with hooks" into "general-purpose MI framework." Forward-only methods; no autograd needed (tractable on 16 GB VRAM up to ~7B F32).

- [ ] Per-head hooks (`hook_z`, `hook_result`) — capture attention-weighted values and per-head output projected to `d_model` before summation. Requires `attention.rs` changes + re-validation of all 7 transformer families — **commit** — **PUSH**
- [ ] Residual stream decomposition — `accumulated_resid()` (stack residual streams at each layer) + `decompose_resid()` (per-component additive contributions: each attention output, each MLP output, embeddings) on `FullActivationCache` — **commit**
- [ ] Direct logit attribution — `logit_attrs()`: dot product of each component's residual contribution with unembedding direction of target tokens. Per-head, per-layer, per-MLP granularity — **commit**
- [ ] Activation patching framework — `activation_patch()` generic + pre-built variants (`resid_pre`, `attn_out`, `mlp_out`, per-head). Clean/corrupted forward passes with activation swaps at specified hook points. Note: basic activation patching already works as an example (`activation_patching.rs`, v0.1.0) with full causal trace heatmap (v0.1.3); this phase promotes it to a reusable library API — **commit** — **PUSH**

**Deliverable:** Causal tracing, component attribution, circuit localization. — **PUSH + tag `v0.3.0`**

### Phase 8: Static Circuit Analysis

**Goal:** Weight-level understanding without forward passes — the "Mathematical Framework for Transformer Circuits" (Elhage et al. 2021) in Rust. Focus on the core primitives; higher-level detection utilities can be built on top later.

- [ ] `FactoredMatrix` type — lazy `A @ B` representation with efficient SVD, eigenvalues, composition scores, corner inspection (no `[d_model, d_model]` materialization) — **commit**
- [ ] QK/OV circuit extraction — expose `W_Q.T @ W_K` and `W_V @ W_O` as `FactoredMatrix` per head — **commit**
- [ ] Weight folding — `fold_layer_norm()`, `center_writing_weights()`, `center_unembed()` for clean decomposition (without folding, LayerNorm entangles every component) — **commit**
- [ ] Composition scores — Q/K/V-composition between head pairs via `FactoredMatrix` — **commit** — **PUSH**

**Deliverable:** Static circuit analysis primitives. — **PUSH + tag `v0.4.0`**

*Deferred to Phase 10+:* Head detection (induction, previous-token, duplicate-token) and SVD interpretation (project singular vectors through unembedding) — useful but higher-level; build on Phase 8 primitives.

### Phase 9: Model Coverage & Ecosystem

**Goal:** Breadth — cover the most-studied MI models and improve ergonomics. Tensor name remapping unlocks non-HF-standard architectures. Keep scope focused on high-impact additions.

- [ ] Tensor name remapping — build rename map from safetensors tensor names, wrap `VarBuilder` to translate non-standard weight paths (e.g. GPT-2 `transformer.h.{i}.attn.c_attn` → `model.layers.{i}.self_attn.q_proj`). Prerequisite for GPT-2/Pythia — **commit**
- [ ] GPT-2 family — absolute positional embeddings (new config axis), post-norm architecture. The "fruit fly" of MI research — **commit** — **PUSH**
- [ ] Pythia family (14M–12B) — EleutherAI MI research models with training checkpoints. Shares GPT-2-like architecture — **commit**
- [ ] Additional hook points (`hook_rot_q`/`hook_rot_k`, `hook_q_input`/`hook_k_input`/`hook_v_input`, LayerNorm hooks) — driven by demand from Phase 7 analysis work — **commit**
- [ ] Richer tokenizer utilities (`to_str_tokens`, `test_prompt`, `tokens_to_residual_directions`) — **commit** — **PUSH**

**Deliverable:** GPT-2/Pythia coverage, improved ergonomics. — **PUSH + tag `v0.5.0`**

*Deferred to Phase 10+:* MoE transformer variant (Mixtral — router analysis, expert specialization hooks), evaluation suite (`sanity_check`, `induction_loss`).

### Phase 10+: Extensions (Future)

- [ ] Head detection — automated induction head, previous-token head, duplicate-token head identification (builds on Phase 8 primitives)
- [ ] SVD interpretation — project weight singular vectors through unembedding to token-space representations
- [ ] MoE transformer variant (Mixtral) — router analysis, expert specialization hooks
- [ ] Evaluation suite (`sanity_check`, `induction_loss`) — quick model validation
- [ ] RWKV-4/5 backends (if community demand)
- [ ] Mamba / Mamba-2 backend
- [ ] GLA, RetNet backends (via generic linear RNN trait)
- [ ] Probing module (optional feature)
- [ ] Feature visualization export (JSON for web UIs)
- [ ] Quantized model support (GGUF, GPTQ, AWQ)
- [ ] Backward hooks / gradient attribution (scaling optimization for multi-GPU; not needed at 16 GB single-GPU scale)

---

## 8. Key Design Decisions

Detailed design proposals live in the [`design/`](design/) directory. Summary:

1. **Crate name.** ~~Must be decided first.~~ **Decided: `candle-mi`** — endorsed by HuggingFace (see §0).
2. **Hook system** — enum primary with `Display`/`FromStr` string conversion. See [`design/hook-system.md`](design/hook-system.md).
3. **Hook overhead** — zero-cost when inactive (conditional clone at each hook point). See [`design/hook-overhead.md`](design/hook-overhead.md).
4. **Intervention API** — unified `forward(tokens, config)` (pyvene-style). See [`design/intervention-api.md`](design/intervention-api.md).
5. **Error handling** — typed `MIError` enum with `thiserror`. See [`design/error-handling.md`](design/error-handling.md).
6. **Candle version** — pin to `0.9`, update incrementally. See [`design/candle-version.md`](design/candle-version.md).
7. **RWKV-7 effective attention** — diag+rank-1 state transition via backward linear functional (implemented in Phase 2). See [`design/rwkv7-effective-attention.md`](design/rwkv7-effective-attention.md).
8. **Config-driven architecture** — one `TransformerConfig` struct with ~12 axes, parsed from HuggingFace `config.json` via `from_hf_config`. Adding a model family = one `parse_*` function (~30 lines), not a new forward pass. `SUPPORTED_MODEL_TYPES` enumerates accepted `model_type` strings.
9. **Lint single source of truth** — `Cargo.toml [lints]` is the sole authority for lint configuration. `src/lib.rs` retains only the 3 attributes that `Cargo.toml` cannot express (`deny(warnings)`, two `cfg_attr` for `unsafe_code`).
10. **`apply_intervention` consolidation** — all intervention application goes through a single `pub(crate) fn apply_intervention` in `crate::hooks`, shared by transformer and RWKV backends.
11. **GPU test serialization** — all GPU integration tests carry `#[serial]` (from `serial_test`) to prevent CUDA OOM from concurrent model loading on 16 GB GPUs.
12. **`#[must_use]` policy (Rule 17)** — all pure public functions and methods that return a value carry `#[must_use]`. Enforced by `clippy::must_use_candidate` at `warn` level (promoted to error by `#![deny(warnings)]`).
13. **Coding conventions** — `CONVENTIONS.md` codifies mandatory annotation patterns (PROMOTE, CONTIGUOUS, TRAIT_OBJECT, EXHAUSTIVE, EXPLICIT, BORROW), shape documentation (Rule 12), `#[non_exhaustive]` policy (Rule 11), hook purity contract (Rule 16), and `#[must_use]` policy (Rule 17). Follows [Amphigraphic Strict / Grit](https://github.com/PCfVW/Amphigraphic-Strict).
14. **F32 research-grade precision** — default GPU dtype changed from BF16 to F32 for exact numerical parity with Python/PyTorch. Evidence: RWKV-7 GPU logit error dropped from 0.027 (0.36% relative) under BF16 to 0.000002 (6 decimal places) under F32 — identical to CPU F32 and Python F32. Models up to ~7B fit in 16 GB VRAM at F32. The BF16 GPU test is retained as a regression test. Transformer attention mask dtype is derived from `hidden.dtype()` instead of being hardcoded.
15. **CLT as separate module** — CLT code lives in `src/clt/mod.rs` (feature: `"clt"`), not inside `interp/`. This keeps the interpretability module focused on model-agnostic spec types while CLT has its own weight loading, encoding, and injection logic. `CltFeatureId` encodes `(layer, feature_index)` as a newtype for type safety.
16. **Cross-implementation validation** — every backend (transformer, RWKV, CLT) is validated against a Python reference script (`scripts/`) that produces a JSON reference file, with a markdown comparison document recording the side-by-side results. This ensures reproducibility and documents the expected accuracy bounds for each dtype/backend combination.

---

## 9. Relationship to Existing Projects

### 9.1 plip-rs (AIware 2026)

**Frozen at v1.4.0** on the `melometis` branch (the `tragos` branch has been merged into `melometis`). The crate starts fresh but reuses ~3500 lines of infrastructure. plip-rs remains the supplementary material for the AIware paper.

### 9.2 Melomētis + Tragos — Planning in Poems (plip-rs `melometis` branch, v1.4.0)

Two independent replications of Anthropic's "Planning in Poems" [Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location), now unified on the `melometis` branch:

**Gemma 2 2B** (426K and 2.5M CLTs): suppress all `-out` rhyme-group features, inject a single "around" feature at L22, sweep injection position across 31 tokens — P("around") is flat at 4.836×10⁻⁸ then **jumps to 0.483 at the planning site** (≈10-million-fold ratio; 9,974,880× measured). 70% of 136 suppress+inject pairs peak at the planning site. The 2.5M CLT upgrade achieves word-level resolution (209 words, 52.2% best redirect, 3.78-trillion-fold spike ratio).

**Llama 3.2 1B** (524K CLT): second independent replication confirming the phenomenon generalises across architectures. Cross-model comparison reveals the **search vs. commitment distinction**: both models have phonological search (Jaccard 0.24), but Llama lacks the commitment circuit that sustains the rhyme signal through the final layers (82% of planning features crammed into L15, the last layer). Layer suppression on Gemma confirms L22–25 are causally necessary for rhyming (0/10 without them).

Documentation: `docs/planning-in-poems/` (Gemma 2 2B) and `docs/planning-circuit-hunt/` (cross-model investigation). Full pipeline reproducible in ~3 hours: `cargo run --release --example reproduce_pipeline`.

Future: becomes a **consumer of candle-mi** — a separate repository that depends on `candle-mi` for model loading, CLT injection, and intervention infrastructure. Melomētis-specific code (GTPyhop integration, poetry corpus, HTN planning) lives in its own repo.

### 9.3 Deloson (Visualization)

Interactive web visualizer for plip-rs layer scan results. Built with React 19, React Flow, and Recharts; deployed on [GitHub Pages](https://PCfVW.github.io/deloson/). Renders per-layer attention statistics (Cohen's d, Welch's t, p-values, Python/Rust ratios) as node-based flow diagrams and cross-model comparison charts across 6 code LLMs (StarCoder2, Qwen2.5-Coder 3B/7B, CodeGemma, Code-LLaMA, Phi-3). Supports drag-and-drop loading of new `layer_scan_universal_*.json` files at runtime.

Deloson demonstrates the visualization half of the MI pipeline: candle-mi produces the numerical results, deloson makes them explorable. The JSON output format used by plip-rs's `layer_scan_universal` example should be preserved (or extended) in candle-mi to maintain compatibility.

### 9.4 candle-transformers

The crate does NOT depend on candle-transformers. It provides its own model implementations with MI hooks built in. This is intentional: candle-transformers' models don't expose internals.

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Generic transformer has subtle bugs vs HF reference | High | High | Rigorous per-model validation protocol (§7 Phase 1) |
| RWKV-7 HF format changes (fla library is young) | Medium | Medium | Pin to specific HF model revisions; version-aware config |
| Hook system adds unacceptable overhead | Low | High | Benchmarked in Phase 1 (after LLaMA validates); zero-cost when inactive |
| No users | Medium | Low | HuggingFace endorsement + two published replications lower the barrier; the crate serves Melomētis regardless |
| candle breaking changes | Medium | Medium | Pin candle version; update incrementally |
| CLT/SAE weight format changes | Medium | Medium | Support multiple formats; version-aware loading |

---

## References

1. Elhage, N. et al. "A Mathematical Framework for Transformer Circuits." Anthropic, 2021.
2. Lindsey, J. et al. "On the Biology of a Large Language Model." Anthropic, March 2025.
3. Nanda, N. & Bloom, J. "TransformerLens." GitHub, 2022-2026.
4. Fiotto-Kaufman, J. et al. "nnsight: Democratizing Access to Neural Network Internals." 2024.
5. Wu, Z. et al. "pyvene: A Library for Understanding and Improving PyTorch Models via Interventions." NeurIPS 2024.
6. Peng, B. et al. "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence." ICLR 2025.
7. Peng, B. et al. "RWKV-7 Goose with Expressive Dynamic State Evolution." arXiv 2503.14456, March 2025.
8. Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.
9. Yang, S. et al. "Gated Linear Attention Transformers with Hardware-Efficient Training." ICML 2024.
10. Anthropic. "circuit-tracer." GitHub, May 2025.
11. Lieberum, T. et al. "Gemma Scope." Google DeepMind, 2024.
12. Bloom, J. et al. "SAELens." GitHub, 2024-2026.
