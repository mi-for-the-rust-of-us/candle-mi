# How to Add a New Model Architecture

> This guide explains how candle-mi supports new model families.  There
> are three paths — from easiest (auto-config) to most flexible (custom
> backend) — depending on how close the new model is to a standard
> decoder-only transformer.

---

## Table of Contents

- [Overview: Three Paths](#overview-three-paths)
- [Path 1: Auto-Config (Zero Code)](#path-1-auto-config-zero-code)
  - [What Auto-Config Detects](#what-auto-config-detects)
  - [Compatibility Check](#compatibility-check)
  - [Limitations](#limitations)
- [Path 2: Config Parser (Recommended for Known Families)](#path-2-config-parser-recommended-for-known-families)
  - [Step 1: Identify Configuration Axes](#step-1-identify-configuration-axes)
  - [Step 2: Write the Parser](#step-2-write-the-parser)
  - [Step 3: Register the Model Type](#step-3-register-the-model-type)
  - [Step 4: Validate Against Python](#step-4-validate-against-python)
- [Path 3: Custom MIBackend (Non-Transformer Architectures)](#path-3-custom-mibackend-non-transformer-architectures)
  - [The MIBackend Trait](#the-mibackend-trait)
  - [Implementing forward()](#implementing-forward)
  - [Hook Integration Checklist](#hook-integration-checklist)
  - [Registering with MIModel::from_pretrained()](#registering-with-mimodelfrom_pretrained)
- [TransformerConfig Reference](#transformerconfig-reference)
  - [Configuration Axes](#configuration-axes)
  - [Existing Parsers as Templates](#existing-parsers-as-templates)
- [Testing Checklist](#testing-checklist)
- [Weight Naming Convention](#weight-naming-convention)

---

## Overview: Three Paths

| Path | When to use | Effort | Hook support |
|------|-------------|--------|--------------|
| **Auto-config** | Model uses HuggingFace-standard weight naming and is a standard decoder-only transformer | None | Full (automatic) |
| **Config parser** | Model is a decoder-only transformer with quirks (special norm, bias pattern, etc.) | ~30 lines | Full (automatic) |
| **Custom `MIBackend`** | Model is not a decoder-only transformer (e.g., RWKV, Mamba, masked diffusion, encoder-decoder) | ~500+ lines | Manual (you place hooks) |

Most HuggingFace transformer models work out of the box via auto-config.
If yours doesn't, a config parser is usually enough.  A custom backend is
only needed for fundamentally different architectures.

---

## Path 1: Auto-Config (Zero Code)

When `MIModel::from_pretrained()` encounters an unknown `model_type` in
`config.json`, it attempts to infer a `TransformerConfig` automatically:

```rust
// Works for any model with standard HF weight naming
let model = MIModel::from_pretrained("some-org/novel-transformer-3B")?;
```

### What Auto-Config Detects

Auto-config reads configuration from two sources:

**Tier 1–2: `config.json` scalars** (same fields as known families):
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `vocab_size`
- `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`
- `tie_word_embeddings`, `attention_bias`

**Tier 3: safetensors tensor names** (structural inference):
- **QKV layout**: `qkv_proj` → `Fused`; `q_proj` + `k_proj` + `v_proj` → `Separate`
- **MLP layout**: `gate_up_proj` → `GatedFused`; `gate_proj` + `up_proj` → `GatedSeparate`; `fc1` + `fc2` → `Plain`
- **Bias flags**: presence of `.bias` tensors in attention/MLP projections
- **Norm type**: `input_layernorm.bias` → `LayerNorm`; otherwise `RmsNorm`

**Tier 4: `model_type` fixups** (architecture-specific overrides):
- Gemma/Gemma2 → `GemmaRmsNorm`, `embedding_scale`, `alternating_sliding_window`
- Gemma → `GeluApprox`, **overriding `hidden_act` when it says `"gelu"`**. The
  Gemma 1.0 checkpoints ship a `hidden_act` naming the exact erf form while the
  models use the tanh approximation, so the config is not authoritative here;
  `hidden_activation`, when present, already takes precedence. See
  [`docs/adding-a-model.md`](docs/adding-a-model.md) ("Trap 1 has a second face")
- Any model with `attn_logit_softcapping` → soft-capping enabled

For a visual overview of how these config fields map to transformer blocks, see Raschka's
[The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison).

### Compatibility Check

Before loading weights, `check_auto_compatibility()` runs a preflight
check that detects incompatible models:

```rust
use candle_mi::{CompatibilityReport, TransformerConfig};

let report = TransformerConfig::check_auto_compatibility(&config_json, &tensor_names);
if !report.compatible {
    for issue in &report.issues {
        eprintln!("  - {issue}");
    }
}
```

The check verifies:
- Required `config.json` fields are present
- Expected weight tensors exist (`embed_tokens`, `input_layernorm`, `lm_head`, etc.)
- When tensors are missing, it suggests the closest match ("did you mean?")
- Detects non-HF naming conventions (e.g., GGUF, custom prefixes)

### Limitations

Auto-config does **not** handle:
- Non-standard weight naming (e.g., `transformer.h.0.attn` instead of `model.layers.0.self_attn`)
- Architectures that aren't decoder-only transformers (RWKV, Mamba, masked diffusion, encoder-decoder)
- Novel attention mechanisms (linear attention, local-global hybrid)
- Custom activations not in the standard set (`SiLU`, `GELU`, `GELU tanh`)
- **Bidirectional attention** — decoder-style masked-diffusion LMs (Dream, `a2d-qwen2`) share the Qwen layout *exactly*, so auto-config cannot distinguish them from a causal model; they load as registered `model_type`s with `bidirectional: true` (Path 2). DiT-style MDLM checkpoints (`backbone.*`) instead get a targeted *"enable the `diffusion` feature"* error.

If auto-config fails, write a config parser (Path 2) or a custom backend
(Path 3).

### What failure looks like

The `auto_config_dogfood` example demonstrates both success and failure modes:

```bash
# Success — known family, uses manual parser
cargo run --release --features transformer --example auto_config_dogfood -- "meta-llama/Llama-3.2-1B"

# Failure — unsupported architecture (weight name mismatch)
cargo run --release --features transformer --example auto_config_dogfood -- "allenai/OLMo-1B-hf"

# Failure — actionable diagnostics (non-standard naming convention)
cargo run --release --features transformer --example auto_config_dogfood -- "EleutherAI/pythia-1.4b"
```

**OLMo-1B** fails the compatibility check because its weight names
(`model.layers.*.input_layernorm.weight`, `model.final_norm.weight`) do not
match the normalisation tensor patterns that `GenericTransformer` expects.

**Pythia 1.4B** uses the `gpt_neox.layers.{i}` weight prefix instead of the
HF-standard `model.layers.{i}`. The error message shows which tensors
*were* found for each expected category (embedding, norm, attention, MLP),
detects the GPT-NeoX / Pythia naming convention, and points to Phase 9
(tensor name remapping) for planned support. This is the diagnostic output
that tells contributors exactly where to look when adding a new model family.

---

## Path 2: Config Parser (Recommended for Known Families)

If the model is a decoder-only transformer but auto-config fails (or you
want guaranteed correctness), write a dedicated config parser.  This is
typically ~30 lines of code.

### Step 1: Identify Configuration Axes

Compare the new model to the closest existing family.  The axes to check:

| Axis | Question | Where to look |
|------|----------|---------------|
| Norm type | RmsNorm, LayerNorm, or GemmaRmsNorm? | `config.json` or paper |
| Activation | SiLU, GELU, or GELU tanh? | `hidden_act` field |
| QKV layout | Separate or fused? | Weight tensor names |
| MLP layout | GatedSeparate, GatedFused, or Plain? | Weight tensor names |
| QKV bias | Do Q, K, V have bias terms? | Weight tensor names |
| Use bias | Does every projection have bias? | Weight tensor names |
| Post-norms | 2 or 4 norms per layer? | Weight tensor names |
| Embedding scale | Multiply embeddings by `sqrt(hidden_size)`? | Paper or code |
| Soft-capping | Pre-softmax logit capping? | `attn_logit_softcapping` |
| Tied embeddings | `lm_head.weight` shared with `embed_tokens`? | `tie_word_embeddings` |
| Sliding window | Alternating full/sliding attention? | `sliding_window` |

### Step 2: Write the Parser

Add a `parse_*` function in `src/config.rs`.  Use an existing parser as a
template — `parse_llama` is the simplest baseline:

```rust
/// Parse a Llama-family `config.json`.
fn parse_llama(config: &Value) -> Result<TransformerConfig> {
    let base = parse_common_fields(config)?;
    Ok(TransformerConfig {
        norm_type: NormType::RmsNorm,
        activation: Activation::Silu,
        qkv_layout: QkvLayout::Separate,
        mlp_layout: MlpLayout::GatedSeparate,
        qkv_bias: false,
        use_bias: false,
        use_post_norms: false,
        embedding_scale: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: None,
        alternating_sliding_window: false,
        ..base
    })
}
```

For a model with quirks, override only what differs:

```rust
fn parse_qwen2(config: &Value) -> Result<TransformerConfig> {
    let base = parse_common_fields(config)?;
    // Qwen2: attention_bias applies to Q, K, V (NOT o_proj)
    let qkv_bias = config
        .get("attention_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(TransformerConfig {
        norm_type: NormType::RmsNorm,
        activation: Activation::Silu,
        qkv_layout: QkvLayout::Separate,
        mlp_layout: MlpLayout::GatedSeparate,
        qkv_bias,
        use_bias: false,
        use_post_norms: false,
        embedding_scale: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: None,
        alternating_sliding_window: false,
        ..base
    })
}
```

### Step 3: Register the Model Type

1. Add the `model_type` string to `SUPPORTED_MODEL_TYPES` in `src/config.rs`:

```rust
pub const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "gemma", "gemma2", "llama", "mistral",
    "my_new_model",  // ← add here
    "phi3", "qwen2", "qwen3", "starcoder2",
];
```

2. Add a match arm in `from_hf_config()`:

```rust
"my_new_model" => parse_my_new_model(config),
```

### Step 4: Validate Against Python

Run the model in both Python (HuggingFace Transformers) and Rust, and
compare logits:

```python
# Python reference
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("org/model", torch_dtype=torch.float32)
tokens = tokenizer("The capital of France is", return_tensors="pt")
logits = model(**tokens).logits[0, -1, :]  # last position
top5 = torch.topk(logits, 5)
```

```rust
// Rust — should match Python's top-5 exactly
let model = MIModel::from_pretrained("org/model")?;
let tokens = tokenizer.encode("The capital of France is")?;
// ... forward pass, extract last-position logits, compare top-5
```

With F32 on both sides, logits should match to ~6 decimal places.

---

## Path 3: Custom MIBackend (Non-Transformer Architectures)

For architectures that aren't decoder-only transformers (RWKV, Mamba,
masked diffusion, encoder-decoder, etc.), implement the `MIBackend` trait
directly.

### The MIBackend Trait

```rust
pub trait MIBackend: Send + Sync {
    // --- Required: metadata ---
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn num_heads(&self) -> usize;

    // --- Required: forward pass ---
    fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>;

    // --- Required: logit projection ---
    fn project_to_vocab(&self, hidden: &Tensor) -> Result<Tensor>;

    // --- Optional ---
    fn chat_template(&self, _prompt: &str, _system_prompt: Option<&str>) -> Option<String> {
        None
    }
    fn embedding_vector(&self, _token_id: u32) -> Result<Tensor> {
        Err(MIError::Hook("embedding_vector not supported".into()))
    }
}
```

**Contract:**
- `forward()` must return a `HookCache` with logits at shape `[batch, seq, vocab_size]`
- When `hooks` is empty, the forward pass must produce **zero extra allocations**
- `project_to_vocab()` applies the final norm + unembedding projection (for logit lens),
  and **must preserve rank**: `[batch, hidden_size]` gives `[batch, vocab_size]`, and
  `[batch, seq, hidden_size]` gives `[batch, seq, vocab_size]`.  A logit lens reads one
  position; a probe that has captured a whole residual stream projects every position at
  once.  Project against a rank-2 unembedding weight with `broadcast_matmul`, **not**
  `matmul`, which requires operands of equal rank
- The trait is `Send + Sync` because `MIModel` may be shared across threads

### Implementing forward()

The forward pass must integrate with the hook system.  Here is the
pattern:

```rust
fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache> {
    // 1. Create a placeholder HookCache
    let placeholder = Tensor::zeros((1,), DType::F32, input_ids.device())?;
    let mut cache = HookCache::new(placeholder);

    // 2. Embedding
    let mut hidden = self.embed(input_ids)?;

    // 3. Every hook point goes through one helper, which captures and then
    //    applies any registered interventions, in that order.
    hook_point(&mut hidden, HookPoint::Embed, hooks, &mut cache)?;

    // 4. Layer loop
    for i in 0..self.num_layers {
        hook_point(&mut hidden, HookPoint::ResidPre(i), hooks, &mut cache)?;

        // ... your layer logic, firing hook_point at each stage ...

        hook_point(&mut hidden, HookPoint::ResidPost(i), hooks, &mut cache)?;
    }

    // 5. Final logits
    let logits = self.compute_logits(&hidden)?;
    cache.set_output(logits);
    Ok(cache)
}
```

**Key points:**
- **Use `crate::hooks::hook_point`. Do not hand-write capture and intervention
  separately.** Until v0.2.0 every backend inlined the two halves, and three of
  six had drifted: `GenericRwkv` applied interventions at `Embed` only, and both
  `stoicheia` backends applied none at all. They captured correctly, so the gap
  was invisible: `hooks.intervene(...)` returned no error and the forward
  returned the untouched baseline, which in a causal experiment is indis-
  tinguishable from a real null result. One helper is what stops that recurring.
- **A hook point either honours interventions or refuses them. Never ignores
  them.** Where a point exposes a *diagnostic read-out* that nothing downstream
  consumes, use `hook_point_readonly`, which captures and returns
  `MIError::Intervention` for any intervention aimed at it, naming the supported
  alternative. RWKV's `RwkvState`, `RwkvDecay` and `RwkvEffectiveAttn` are of
  this kind: they are computed *after* the recurrence that produced them, so an
  edit there could never reach a computation. Accepting and discarding would
  merely relocate the silent no-op from "the backend forgot" to "the tensor was
  dead".
- **Fire the hook where the value is still live.** If a sublayer returns a
  tensor the caller only observes, the hook belongs *inside* that sublayer, not
  on its return value. `StoicheiaTransformer` had `AttnScores`/`AttnPattern`
  fired on values its attention had already consumed; they now fire inside
  `AttentionLayer::forward` so an intervention reaches the weighted sum.
- **Pass the hook point you are actually at.** `apply_intervention` takes it
  because dim 1 does not mean the same thing everywhere: the sequence in a
  `[batch, seq_len, hidden]` activation, a head in a
  `[batch, n_heads, seq_len, head_dim]` one. `Intervention::PatchAt` uses it to
  refuse a positional write where it would silently overwrite a head, so passing
  the wrong point defeats that guard. See `HookPoint::accepts_positional_patch`.
  `hook_point` passes it for you.
- Capture checks (`is_captured`) are cheap `HashSet` lookups — when false,
  the `.clone()` is skipped entirely.
- Intervention iteration (`interventions_at`) returns an empty iterator
  when no interventions target that hook point, so a forward with an empty
  `HookSpec` is unchanged.

### Keep the Forward Differentiable (`nn_ops`)

Never call `candle_nn`'s fused kernels directly in a forward path:
`ops::softmax_last_dim`, the fused `layer_norm` (with bias), fused
`rms_norm`, and `sdpa` are built with `apply_op*_no_bwd` — their output
records **no** backward op, so it is a graph *leaf*. A training loop over
your backend then "works" (loss decreases, checkpoint saves) while every
parameter upstream of the last fused op silently receives no gradient
(see `docs/dogfooding-feedbacks/trainable-backbones.md`).

Use the [`nn_ops`] wrappers instead:

```rust
let pattern = crate::nn_ops::softmax_last_dim(&scores)?;   // not candle_nn::ops::
let normed  = crate::nn_ops::layer_norm(&self.ln1, &xs)?;  // not self.ln1.forward()
let normed  = crate::nn_ops::rms_norm(&self.norm, &xs)?;   // not self.norm.forward()
```

They dispatch on `Tensor::track_op`: an inference forward (no `Var`
upstream) takes the fused kernel unchanged — byte-identical, so parity
baselines keep their meaning — and a forward under a `VarMap` takes the
composed form, which carries a backward. Direct fused calls are correct
only for terminal read-outs where no gradient can ever be wanted
(sampling, probability display), such as `backend.rs`'s `sample_token`.

### Hook Integration Checklist

For a new backend, decide which hook points to support.  At minimum:

| Hook Point | Priority | Purpose |
|------------|----------|---------|
| `Embed` | Required | Token embedding output |
| `ResidPre(i)` | Required | Residual stream before each layer |
| `ResidPost(i)` | Required | Residual stream after each layer (logit lens) |
| `FinalNorm` | Required | After final norm (logit lens) |
| `AttnPattern(i)` | Recommended | Attention visualization |
| `AttnScores(i)` | Recommended | Knockout interventions |
| `AttnQ/K/V(i)` | Optional | Q, K, V inspection |
| `MlpPre/Post/Out(i)` | Optional | MLP analysis |

Backend-specific hook points should use `HookPoint::Custom("name")` or
propose a new enum variant.

### Registering with MIModel::from_pretrained()

To make your backend discoverable via `from_pretrained()`, add a match arm
in `src/backend.rs` inside `MIModel::from_pretrained()`:

```rust
#[cfg(feature = "my_backend")]
"my_model_type" => {
    let config = MyConfig::from_hf_config(&json)?;
    let model = MyBackend::load(config, &device, dtype, vb)?;
    Ok(Self::with_tokenizer(Box::new(model), device, tokenizer))
}
```

The backend is gated behind a feature flag, so users only compile what they
need.

### Stoicheia: A Worked Example

The `stoicheia` module (`src/stoicheia/`) implements two custom `MIBackend`s
for ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo)
tiny models (8–1,408 parameters):

- **`StoicheiaRnn`** — single-layer ReLU RNN (continuous input). Uses
  `HookPoint::Custom("rnn.hook_hidden.{t}")` for per-timestep activation
  capture.
- **`StoicheiaTransformer`** — attention-only transformer (discrete input,
  no MLP, no causal mask). Reuses standard `HookPoint` variants (`Embed`,
  `AttnScores`, `AttnPattern`, `ResidPre/Post`).

These load from `.safetensors` or `.pth` files (auto-detected by extension),
use explicit `load(config, path, device)` constructors instead of
`from_pretrained()`, and demonstrate how a non-language-model backend fits
into the `MIBackend` trait. See `src/stoicheia/mod.rs` for the full
implementation (~500 lines for both backends).

### MDLM: A Bidirectional Masked-Diffusion Backend

The `diffusion` module (`src/diffusion/`) implements `GenericMdlm`, a custom
`MIBackend` for [MDLM](https://huggingface.co/kuleshov-group/mdlm-owt) — a
masked-diffusion **bidirectional DiT** (Sahoo et al., NeurIPS 2024).  It is the
template for a *non-causal* language model, and differs from the transformer
backend on every axis that matters:

- **Bidirectional attention** — an all-zeros attention mask instead of the
  causal `-inf` mask, so every position attends to every other.
- **adaLN conditioning** — each block modulates its norms by a conditioning
  vector (`modulate(x, shift, scale) = x * (1 + scale) + shift`), and gates its
  attention/MLP residual contributions.  For the released time-independent
  checkpoint that vector is a constant, computed once at load.
- **Non-HF weight layout** — `backbone.*` tensors: a raw-`Parameter` embedding
  (no `.weight` suffix), weight-only `LayerNorm`, fused `attn_qkv` / `attn_out`
  (no bias), a plain GELU-tanh MLP, and an untied `output_layer.linear`.  None
  of these fit `TransformerConfig`, which is why MDLM is a backend (Path 3)
  rather than a parser (Path 2).
- **Standard hook surface** — it reuses the usual `HookPoint` variants
  (`Embed`, `ResidPre/Mid/Post`, `AttnQ/K/V/Scores/Pattern`, `AttnOut`,
  `MlpPre/Post/Out`, `FinalNorm`), so every logit-lens / patching / steering
  tool works unchanged.  `AttnOut` / `MlpOut` capture the **gated** residual
  contribution (`g_msa * attn_out(...)`, `g_mlp * mlp(...)`), matching the
  TransformerLens "added to the residual stream" convention.

It is gated behind the `diffusion` feature, ships its own config (`MdlmConfig`,
parsed from `model_type = "mdlm"`) plus a `SUBS` ancestral sampler, and is
dispatched from `MIModel::from_pretrained()` via
`SUPPORTED_DIFFUSION_MODEL_TYPES` (the same mechanism as the RWKV backend).  The
forward pass is validated bit-faithfully against an fp32 oracle (max abs-diff
1.3e-5 on GPU).  See
[`docs/roadmaps/diffusion-lm-roadmap.md`](docs/roadmaps/diffusion-lm-roadmap.md)
for the DiT-vs-decoder split.  Decoder-style diffusion LMs (Dream, `a2d-qwen2`)
do **not** need a backend at all — they load as a bidirectional
`GenericTransformer` via a one-field config flag (Path 2, Stage 3); only LLaDA's
OLMo-style weight naming still needs a remap (deferred, Stage 3e).

### OthelloGpt: A Plain GPT-2-Style Backbone

The `diffusion` module also implements `OthelloGpt`, a custom `MIBackend` for the
`OthelloMDLM` masked-diffusion world model — a nanoGPT/minGPT-lineage GPT-2
backbone. It is the template for the **most common interpretability target after
LLaMA**: a classic GPT-2 recipe that fits *neither* generic backbone, because

- **learned absolute positions** (`nn.Embedding`, added at the input) — RoPE
  cannot express these, which is why `GenericTransformer` (RoPE-only) does not
  fit and a dedicated module beats adding a learned-pos branch to seven RoPE
  families' hot path; and
- **full `LayerNorm`, with-bias attention/MLP, exact-erf `GELU`, untied head** —
  the exact opposite of MDLM's weight-only `LayerNorm` / no-bias / GELU-tanh DiT.

It loads via `OthelloGpt::load` over the state dict **verbatim** (candle's
`VarBuilder` `pp`-paths match the PyTorch module paths, so no remap and no
transpose), populates the standard `HookPoint` surface (here `AttnOut`/`MlpOut`
are the *ungated* residual contributions), and was validated against an fp32
PyTorch oracle to 4.18e-5 (logits) / 2.59e-4 (per-layer residuals) on CPU.  It is
not dispatched from `from_pretrained` (the checkpoint is a bare `torch.save`
dict, not an HF repo) — construct it directly.

The recurring traps when porting any PyTorch backbone (the GELU variant, bias
presence, norm type, positional scheme, conditioning) are written up in
[`docs/adding-a-model.md`](docs/adding-a-model.md).

---

## TransformerConfig Reference

### Configuration Axes

The `GenericTransformer` is parameterized by these fields:

| Field | Type | Description |
|-------|------|-------------|
| `hidden_size` | `usize` | Model dimension (`d_model`) |
| `num_layers` | `usize` | Number of decoder blocks |
| `num_attention_heads` | `usize` | Number of query heads |
| `num_kv_heads` | `usize` | Number of key/value heads (GQA) |
| `head_dim` | `usize` | Per-head dimension |
| `intermediate_size` | `usize` | MLP hidden dimension |
| `vocab_size` | `usize` | Vocabulary size |
| `norm_type` | `NormType` | `RmsNorm`, `LayerNorm`, or `GemmaRmsNorm` |
| `norm_eps` | `f64` | Normalization epsilon |
| `activation` | `Activation` | `Silu`, `Gelu`, or `GeluApprox` |
| `qkv_layout` | `QkvLayout` | `Separate` or `Fused` |
| `mlp_layout` | `MlpLayout` | `GatedSeparate`, `GatedFused`, or `Plain` |
| `qkv_bias` | `bool` | Bias on Q, K, V projections |
| `use_bias` | `bool` | Bias on all projections |
| `use_post_norms` | `bool` | 4-norm layers (Gemma 2) |
| `embedding_scale` | `Option<f64>` | Multiply embeddings by this value |
| `attn_logit_softcapping` | `Option<f64>` | Pre-softmax logit soft-capping threshold |
| `query_pre_attn_scalar` | `Option<f64>` | Custom Q scaling (overrides `1/sqrt(head_dim)`) |
| `rope_theta` | `f64` | RoPE base frequency |
| `max_position_embeddings` | `usize` | Maximum sequence length |
| `tie_word_embeddings` | `bool` | Share `embed_tokens` and `lm_head` weights |
| `alternating_sliding_window` | `bool` | Alternating full/sliding attention (Gemma 2) |
| `sliding_window` | `Option<usize>` | Sliding window size |
| `bidirectional` | `bool` | Non-causal attention — decoder-style masked-diffusion LMs (Dream, `a2d-qwen2`) |

### Existing Parsers as Templates

| Family | Parser | Key differences from LLaMA baseline |
|--------|--------|-------------------------------------|
| LLaMA | `parse_llama` | Baseline: GQA, SiLU, RmsNorm, separate QKV, gated MLP |
| Qwen2 | `parse_qwen2` | + `qkv_bias: true` (Q, K, V only, not `o_proj`) |
| Gemma | `parse_gemma` | + `GemmaRmsNorm`, `embedding_scale`, `GeluApprox` |
| Gemma 2 | `parse_gemma2` | + post-norms, soft-capping, `query_pre_attn_scalar`, alternating sliding window |
| Phi-3 | `parse_phi3` | + fused QKV, fused gate+up MLP |
| StarCoder2 | `parse_starcoder2` | + `LayerNorm`, `Plain` MLP, `use_bias: true`, `GeluApprox` |
| Mistral | `parse_mistral` | + sliding window attention |
| Dream / a2d-qwen2 | `parse_qwen2` + flag | + `bidirectional: true` (non-causal masked-diffusion decoder) |
| a2d-qwen3 | `parse_qwen3` + flag | + `bidirectional: true` |

---

## Testing Checklist

When adding a new model family, verify:

- [ ] **Config parsing**: `TransformerConfig::from_hf_config(&json)` produces the correct config for a known model
- [ ] **Forward pass**: top-5 predictions match Python HuggingFace Transformers (F32 on both sides)
- [ ] **Hook capture**: all hook points produce tensors with the expected shapes
- [ ] **Intervention**: `Intervention::Zero` at `ResidPost(0)` changes the output (proves hooks are wired). **Write this as a real test, not a manual check.** Three backends failed it silently for releases because nothing asserted it; `src/stoicheia/mod.rs` has the pattern, using a synthetic model so it needs no download
- [ ] **Logit lens**: `project_to_vocab()` produces meaningful predictions at intermediate layers
- [ ] **Rank preservation**: `project_to_vocab()` accepts both `[batch, hidden_size]` and `[batch, seq, hidden_size]`, and each position of the rank-3 result equals the rank-2 projection of that position
- [ ] **No regression**: existing model tests still pass (`cargo test --features transformer`)

For GPU testing, run with `--test-threads=1` to avoid OOM on 3B+ models.

---

## Weight Naming Convention

The `GenericTransformer` expects HuggingFace-standard weight names under
the `model.` prefix:

```
model.embed_tokens.weight                         # [vocab, hidden]
model.layers.{i}.input_layernorm.weight           # [hidden]
model.layers.{i}.self_attn.q_proj.weight          # [n_heads * head_dim, hidden]
model.layers.{i}.self_attn.k_proj.weight          # [n_kv_heads * head_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight          # [n_kv_heads * head_dim, hidden]
model.layers.{i}.self_attn.o_proj.weight          # [hidden, n_heads * head_dim]
model.layers.{i}.post_attention_layernorm.weight  # [hidden]
model.layers.{i}.mlp.gate_proj.weight             # [intermediate, hidden]
model.layers.{i}.mlp.up_proj.weight               # [intermediate, hidden]
model.layers.{i}.mlp.down_proj.weight             # [hidden, intermediate]
model.norm.weight                                 # [hidden]
lm_head.weight                                    # [vocab, hidden]
```

Variants by family:

| Family | Difference |
|--------|-----------|
| Phi-3 | `qkv_proj` instead of `q/k/v_proj`; `gate_up_proj` instead of `gate_proj` + `up_proj` |
| StarCoder2 | `fc1`/`fc2` instead of `gate_proj`/`up_proj`/`down_proj`; `.bias` on all projections |
| Gemma 2 | + `pre_feedforward_layernorm`, `post_feedforward_layernorm`, `post_attention_layernorm` (4 norms) |
| Tied embeddings | No `lm_head.weight` — reuses `embed_tokens.weight` |

Models with non-standard naming (e.g., `transformer.h.{i}.attn` instead
of `model.layers.{i}.self_attn`) require a custom `MIBackend` (Path 3).
