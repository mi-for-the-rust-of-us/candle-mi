# candle-mi

[![CI](https://github.com/mi-for-the-rust-of-us/candle-mi/actions/workflows/ci.yml/badge.svg)](https://github.com/mi-for-the-rust-of-us/candle-mi/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/candle-mi)](https://crates.io/crates/candle-mi)
[![docs.rs](https://img.shields.io/docsrs/candle-mi)](https://docs.rs/candle-mi)
[![Rust 1.91+](https://img.shields.io/badge/rust-1.91%2B-orange)](https://www.rust-lang.org)
[![Edition 2024](https://img.shields.io/badge/edition-2024-orange)](https://doc.rust-lang.org/edition-guide/rust-2024/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)
[![GitHub last commit](https://img.shields.io/github/last-commit/mi-for-the-rust-of-us/candle-mi)](https://github.com/mi-for-the-rust-of-us/candle-mi/commits/main)

*Mechanistic Interpretability for the Rust of us.*

> **Note:** v0.1.24 — the API may change between minor versions. See the [CHANGELOG](CHANGELOG.md).

## Supported model families

| Architecture | Families | Validated models | Feature |
|---|---|---|---|
| Decoder-only transformer | LLaMA 1/2/3, Mistral, Qwen 2/2.5, Qwen 3, Phi-3/4, Gemma, Gemma 2, StarCoder2 | LLaMA 3.2 1B, Qwen2.5-Coder-3B, Qwen3-1.7B-Base, Gemma 2 2B, Phi-3 Mini, StarCoder2 3B, Mistral 7B | `transformer` |
| Linear RNN | RWKV-6 (Finch), RWKV-7 (Goose) | RWKV-7 1.6B | `rwkv` |
| Masked diffusion (bidirectional `DiT`) | MDLM | MDLM-owt (0.2B) | `diffusion` |
| Masked diffusion (plain GPT-2 backbone) | `OthelloGpt` (learned absolute positions, full `LayerNorm`, exact-`GELU`) | OthelloMDLM world model (25M) | `diffusion` |
| Masked diffusion (decoder-style) | Dream (←Qwen2.5), `a2d-qwen2`/`a2d-qwen3` | `a2d-qwen2` 0.5B (forward-parity oracle) | `transformer` |
| [AlgZoo](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) tiny models | Single-layer ReLU RNN, attention-only transformer (8–1,408 params) | M₂,₂ (10 params), M₁₆,₁₀ (432 params), transformer h4n4 (176 params) | `stoicheia` |

Most HuggingFace transformer models work out of the box via **auto-config** — no code changes needed. See [BACKENDS.md](BACKENDS.md) for details and how to add new architectures.

## Requirements

**Hardware:** candle-mi runs on a single consumer GPU (developed on an RTX 5060 Ti, 16 GB VRAM). Models up to ~7B fit in 16 GB at F32 precision — no H100 cluster required. CPU-only works for small models and tokenizer-only workflows.

**Toolchain: Rust 1.91 or newer**, edition 2024.

> ⚠️ **On an older toolchain, `cargo add candle-mi` does not fail — it silently gives you an old
> release and never moves you off it.**
>
> Cargo's resolver is MSRV-aware: rather than erroring, it picks the newest candle-mi your
> compiler can build, and `cargo update` will not advance past it. There are two such ceilings:
>
> | Your Rust | You silently get | Because |
> |---|---|---|
> | 1.87 or older | **`0.1.4`** | `0.1.5` moved the floor to 1.88, forced by `libloading 0.9.0` |
> | 1.88 – 1.90 | **`0.1.22`** | `0.1.23` moved the floor to 1.91, forced by `hf-fetch-model 0.11.3` |
>
> `0.1.4` predates RWKV, the masked-diffusion backends, trainable backbones, quantized weight
> loading, and most of the CLT work described below. `0.1.22` is merely a few releases behind.
> **If `cargo add candle-mi` gave you either, that is why:**
>
> ```sh
> rustup update && cargo update
> ```
>
> **Neither floor is ours to lower.** 1.88 came from six crates in the graph (`libloading`, `time`,
> `zip`, `cookie_store` and others). 1.91 arrives through `hf-fetch-model 0.11.3`, which adopted
> `hf-hub` 1.0 and with it a mandatory `hf-xet`. Worth knowing if you audit this yourself: cargo's
> MSRV resolution **under-reports** that floor as 1.89, because it reads declared `rust-version`
> fields and `xet-core-structures` declares none while calling `str::floor_char_boundary`, stable
> only since 1.91. The declared-metadata floor is a lower bound; only a real build proves it.

## Table of Contents

- [What is this?](#what-is-this)
- [What can you do with it?](#what-can-you-do-with-it)
- [See it in action](#see-it-in-action)
- [Quick start](#quick-start)
- [Design philosophy](#design-philosophy)
- [Paper replications](#paper-replications)
- [Feature flags](#feature-flags)
- [Documentation](#documentation)
- [License](#license)
- [Development](#development)

## What is this?

**Mechanistic interpretability** (MI) is the study of *how* a language model arrives at its predictions — not just what it outputs, but what happens inside. By inspecting and manipulating the model's internal activations (attention patterns, residual streams, MLP outputs), researchers can understand which components drive specific behaviors.

**candle-mi** is a Rust library that makes this possible. It re-implements model forward passes with built-in **hook points** — type-safe, named locations in the computation graph (e.g., `HookPoint::AttnPattern(5)` for the post-softmax attention at layer 5) where you can:

- **Capture** activations (e.g., "what does the attention pattern look like at layer 5?")
- **Intervene** on activations mid-forward-pass (e.g., "what happens if I knock out this attention edge?" or "what if I steer the residual stream toward a concept?")

This is the Rust equivalent of Python's [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), built on [candle](https://github.com/huggingface/candle) for GPU acceleration. The hook system is type-safe (typos are caught at compile time, not silently ignored at runtime) and zero-overhead (an empty hook spec adds no allocations or clones to the forward pass).

**Why Rust?** Running published MI experiments — such as Anthropic's [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/) or [*Planning in poems*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems) — quickly hits the limits of CPU-only Python. Cloud GPUs are always an option, but not a frugal one. With a consumer-grade GPU, memory and runtime become the real bottleneck. [candle](https://github.com/huggingface/candle) solves both: Rust's zero-cost abstractions minimize memory overhead, compiled code runs faster, and candle provides direct CUDA/Metal access without Python's runtime tax. That's how candle-mi started: let's bring MI to local hardware. (See the [Figure 13 replication](examples/README.md#example-output-figure13_planning_poems) for a concrete example — *Planning in poems* reproduced on a consumer GPU.)

## What can you do with it?

| Technique | What it does | Example |
|-----------|-------------|---------|
| **Logit lens** | See what the model "thinks" at each layer by projecting intermediate residual streams to vocabulary space | [`logit_lens`](examples/README.md#example-output-logit_lens) |
| **Attention knockout** | Block specific attention edges (e.g., "token 5 cannot attend to token 0") and measure how predictions change | [`attention_knockout`](examples/README.md#example-output-attention_knockout) |
| **Activation steering** | Add a direction vector to the residual stream to shift model behavior (e.g., make it more positive or more formal) | [`steering_dose_response`](examples/README.md#example-output-steering_dose_response) |
| **Activation patching** | Swap activations between a clean and corrupted run to identify which components causally drive a prediction | [`activation_patching`](examples/README.md#example-output-activation_patching) |
| **Attention patterns** | Visualize where each attention head attends across the sequence | [`attention_patterns`](examples/README.md#example-output-attention_patterns) |
| **RWKV state analysis** | Inspect and intervene on recurrent state — not just transformers | [`rwkv_inference`](examples/README.md#example-output-rwkv_inference) |
| **AlgZoo analysis** | Exhaustive MI on [AlgZoo](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) tiny models: weight standardization, piecewise-linear region enumeration, neuron ablation, functional probing, surprise accounting | [`stoicheia_analysis`](examples/README.md) |
| **Masked-diffusion MI** | Logit lens and decoding-order analysis across denoising steps — the `(k, ℓ, π)` generalization of the logit lens for bidirectional masked-diffusion models (MDLM) | [`diffusion_logit_lens`](examples/README.md) |
| **Train what you probe** *(new in v0.1.20, extended in v0.1.21 and v0.1.22)* | Backbones carry gradients end-to-end when built over a `VarMap` — train a model and probe *the same weights on the same forward pass*, no second implementation to keep in parity. Seeded from-scratch initialization (`OthelloGpt::init`) makes tiny reference models reproducible from `(config, seed)` alone, on a `ChaCha8` generator keyed in-crate so a `rand` upgrade cannot move it; `init_with_dtype` applies the same recipe at `BF16` when VRAM, not arithmetic, is the training ceiling. Inference stays byte-identical (dispatch on `Tensor::track_op`). candle-mi deliberately ships no training loop, schedule or data loader — those are experiment-shaped and stay with you — but it does ship the one piece a loop cannot reconstruct: `optim::AdamW` (`training` feature) keeps its moments and step counter reachable, so a run staged across processes resumes exactly instead of silently resetting Adam's bias correction at every boundary | [design rationale](docs/dogfooding-feedbacks/trainable-backbones.md) |

candle-mi is (to our knowledge) the only MI toolkit with hook points for recurrent architectures — `RwkvState`, `RwkvDecay`, and `RwkvEffectiveAttn` enable mechanistic analysis of RWKV-6/7 models, a frontier that most MI tooling ignores entirely.

## See it in action

### The logit lens — what does the model "think" at each layer?

```bash
cargo run --release --features transformer --example logit_lens -- "meta-llama/Llama-3.2-1B"
```

This loads in ~2 seconds and runs in ~112ms on an RTX 5060 Ti (16 GB VRAM) or ~3 seconds on CPU, revealing how factual recall emerges across layers. The same prompt (*"The capital of France is"*) tells three different stories:

- **Llama 3.2 1B**: "Paris" appears at layer 11 (69% depth) — typical early factual resolution.
- **Gemma 2 2B**: "Paris" appears at layer 25 (the very last layer, rank 8) — the model hedges until the end.
- **StarCoder2 3B**: "Paris" never appears as a single token — its BPE tokenizer splits it into " Par", which dominates from layer 22 at 33%→74%. The model *knows* the answer but its code-oriented vocabulary hides it.

See the [full comparison](examples/README.md#example-output-logit_lens) with per-layer tables.

### The flagship — Anthropic's circuit tracing on consumer hardware

This library was originally built to replicate Anthropic's [circuit-tracing work](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) on consumer hardware. Here is [Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location) from *"On the Biology of a Large Language Model"*, running on a single GPU:

```bash
cargo run --release --features clt,transformer --example figure13_planning_poems
```

This uses Llama 3.2 1B with a 524K-feature Cross-Layer Transcoder to suppress natural rhyme features and inject an alternative ("that" → P=0.69), sweeping injection position across all prompt tokens. The [newline experiments](docs/experiments/figure13-newline/findings.md) then push past replication to *locate* the behaviour: give a sub-2B open model a full line to compose, and the Figure-13 spike lands at emission, not the newline. Steering the newline captures the *next* token almost deterministically and has decayed to nothing by the rhyme six words later, so the effective site is emission-adjacent even when the intervention is applied a whole line upstream. See the [full output](examples/README.md#example-output-figure13_planning_poems) and the [examples](examples/README.md) covering logit lens, attention knockout, steering, activation patching, CLT circuits, SAE encoding, RWKV inference, AlgZoo analysis, masked-diffusion MI, and more.

## Quick start

```rust
use candle_mi::{HookSpec, MIModel};

fn main() -> candle_mi::Result<()> {
    // 1. Load a model (auto-detects architecture from HuggingFace config)
    let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
    let tokenizer = model.tokenizer().unwrap();

    // 2. Tokenize a prompt
    let tokens = tokenizer.encode("The capital of France is")?;
    let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

    // 3. Run a forward pass (HookSpec::new() = no hooks, zero overhead)
    let cache = model.forward(&input, &HookSpec::new())?;
    let logits = cache.output();  // [1, seq_len, vocab_size]

    // 4. Decode the top prediction
    let last_logits = logits.get(0)?.get(tokens.len() - 1)?;
    let token_id = candle_mi::sample_token(&last_logits, 0.0)?;  // greedy
    println!("{}", tokenizer.decode(&[token_id])?);  // " Paris"
    Ok(())
}
```

### With hooks — capture attention patterns

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

fn main() -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
    let tokenizer = model.tokenizer().unwrap();

    let tokens = tokenizer.encode("The capital of France is")?;
    let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

    // Capture the post-softmax attention pattern at layer 5
    let mut hooks = HookSpec::new();
    hooks.capture(HookPoint::AttnPattern(5));

    let cache = model.forward(&input, &hooks)?;

    // Retrieve the captured tensor — [1, heads, seq, seq]
    let attn = cache.require(&HookPoint::AttnPattern(5))?;
    println!("Layer 5 attention shape: {:?}", attn.shape());
    Ok(())
}
```

Here is what an end-to-end run looks like (auto-config loading LLaMA 3.2 1B — config detection, forward pass, and top-5 predictions):

<p align="center">
  <img src="examples/screenshots/auto_config_llama.png" alt="Auto-config loading LLaMA 3.2 1B" width="700">
</p>

## Design philosophy

candle-mi makes a deliberate trade-off: **full-sequence recompute at every generation step** (no KV cache). This is slower than production inference engines, but it means:

- **Maximum observability.** Hooks can re-observe how earlier positions change under intervention at every step. Interventions "just work" without KV cache invalidation.
- **Interventions compound.** When steering the residual stream during autoregressive generation, each new token is generated with the intervention re-applied across the full context. This is why candle-mi's [recurrent feedback](examples/README.md#example-output-recurrent_feedback) rescues +2 rhyming couplets (out of 15) where a KV-cached approach gets +1 — the intervention is observed at every step, not just once during prefill.

This is a research-first design: MI analyses need to see everything, and the performance cost is acceptable when the alternative is missing causal effects. candle-mi is not an inference engine — for production serving, see [candle-vllm](https://github.com/EricLBuehler/candle-vllm), [vllm.rs](https://github.com/guoqingbao/vllm.rs), or [vLLM](https://github.com/vllm-project/vllm) (Python). It is optimized for observability, not throughput.

## Paper replications

| Paper | What we replicate | Example |
|-------|------------------|---------|
| Anthropic, [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (2025) | **Figure 13 — the rhyme "planning site".** `figure13_planning_poems` reproduces the suppress+inject position sweep (flat baseline, single spike). The **newline experiments** then chase the *planning floor*: a correlational census (Exp 1) finds no plan-like features enriched at the newline, and composition-horizon steering (Exp 2) puts the spike at **emission** on every 0.6B–2B open model incl. word-level CLTs: across 36 runs and 8,640 composed lines the newline is inert *for the rhyme*, though not in general — it captures the next token in up to 98% of samples. `figure13_newline_patch` asks the same question without a transcoder, by patching the newline residual from a minimal-pair poem. A reconstruction-proven CLT-hook reconciliation (`ResidMid`) validates the encoder lens. | [`figure13_planning_poems`](examples/README.md#example-output-figure13_planning_poems), `figure13_newline_census`, `figure13_newline_steering`, `figure13_newline_patch`, [findings](docs/experiments/figure13-newline/findings.md) |
| Hanna & Ameisen, [*Latent Planning Emerges with Scale*](https://arxiv.org/abs/2604.12493) (ICLR 2026) | CLT vs PLT method-matched comparison on the Llama 3.2 1B rhyming-couplet planning site — both transcoder classes detect at comparable ΔP when ranked via decoder-topology-respecting methods. Llama arm complete; Gemma arm in v0.1.10; **Qwen-3 scale sweep complete in v0.1.11** (Qwen3-0.6B + 1.7B with `BlueLightAI` 20K + 16K-dev CLTs; within-family inverse scaling at 0.6B → 1.7B). | [`clt_vs_plt_planning_site`](docs/experiments/clt-vs-plt-planning-site/findings.md), [`figure13-qwen3-cross-size`](docs/experiments/figure13-qwen3-cross-size.md) |
| Maar, Paperno, McDougall, Nanda, [*What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering*](https://arxiv.org/abs/2601.20164) (ICLR 2026) | Contrastive activation steering on three open-weights models (Llama 3.2 3B / 1B, Gemma 2 2B) with Maar's verbatim prompts + exact protocol (raw mean-diff direction, generated-couplet last-word family-membership metric, 25 greedy tokens).  **REPRODUCES** Maar's published "smaller-models" claim on Llama 3.2 3B (60% → 30%, all 6 binary flips are HIT→MISS); strength-sweep surface shows the protocol's effect direction is family-dependent (Llama monotonic inhibition vs Gemma non-monotonic enhancement with peak at m=1.0), with `‖d‖` varying 10× across architectures — Maar's global m=1.5 is not cross-architecture-transferable. **New in v0.1.12.** | [`maar_contrastive_steering`](examples/README.md#example-output-maar_contrastive_steering), [`maar-replication/findings.md`](docs/experiments/maar-replication/findings.md) |
| Anthropic, [*When Models Manipulate Manifolds*](https://transformer-circuits.pub/2025/linebreaks/index.html) (2025) | Character count helix — residual stream encodes line position as a helical manifold; reproduced on Gemma 2 2B with 30 Dickens chapters | [`character_count_helix`](examples/README.md#example-output-character_count_helix) |
| Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262) (2022) | Causal tracing via position-specific activation patching | [`activation_patching`](examples/README.md#example-output-activation_patching) |
| Taufeeque et al., [*Recurrent Feedback*](https://arxiv.org/abs/2407.15421) (2024) | Anacrousis — recurrent steering passes for rhyme completion | [`recurrent_feedback`](examples/README.md#example-output-recurrent_feedback) |
| Li et al., [*Training LMs to Explain Their Own Computations*](https://arxiv.org/abs/2511.08579) (2025) | CounterFact activation patching protocol — contiguous layer-block patching with forced-choice prompts (Transluce) | [`counterfact_patching`](examples/README.md#example-output-counterfact_patching) |


## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `transformer` | yes | Generic transformer backend (decoder-only) |
| `cuda` | yes | CUDA GPU acceleration |
| `rwkv` | no | RWKV-6/7 linear RNN backend |
| `rwkv-tokenizer` | no | RWKV world tokenizer (required for RWKV inference) |
| `diffusion` | no | MDLM masked-diffusion backend (bidirectional DiT; standalone) |
| `clt` | no | Cross-Layer Transcoder support |
| `sae` | no | Sparse Autoencoder support (NPZ via `anamnesis`) |
| `stoicheia` | no | [AlgZoo](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) tiny-model backends + MI analysis tools; agnostic `.safetensors`/`.pth` loading via `anamnesis` |
| `mmap` | no | Memory-mapped weight loading (required for sharded models) |
| `memory` | no | RAM/VRAM reporting (delegated to the [`hypomnesis`](https://crates.io/crates/hypomnesis) crate) |
| `memory-debug` | no | Raw GPU-backend measurement values on stderr (via `hypomnesis`; implies `memory`) |
| `probing` | no | Linear probing via linfa (experimental) |
| `training` | no | Checkpointable `AdamW` (`optim::AdamW`) so a run staged across processes resumes with its moments and step counter intact. candle's update rule, with the state reachable; still no training loop, schedule or data loader |
| `metal` | no | Apple Metal GPU acceleration |

## Documentation

| Document | Description |
|----------|-------------|
| [API docs (docs.rs)](https://docs.rs/candle-mi) | Crate-level documentation with quick start and examples |
| [HOOKS.md](HOOKS.md) | Hook point reference, intervention API walkthrough, and worked examples |
| [BACKENDS.md](BACKENDS.md) | How to add a new model architecture (auto-config, config parser, custom backend) |
| [docs/adding-a-model.md](docs/adding-a-model.md) | Porting a PyTorch backbone: the five silent-divergence traps (GELU, bias, norm, positions, conditioning) and the parity-test recipe |
| [examples/README.md](examples/README.md) | Runnable examples covering inference, logit lens, knockout, steering, AlgZoo analysis, masked-diffusion MI, and more |
| [docs/experiments/README.md](docs/experiments/README.md) | Experiment findings index + motivation — the Figure-13 planning-floor line (as-is and with a composition horizon), plus the CLT/PLT, Maar, and prolepsis replications |
| [docs/roadmaps/diffusion-lm-roadmap.md](docs/roadmaps/diffusion-lm-roadmap.md) | Masked-diffusion-LM support: the DiT vs decoder-style split, MDLM (done), and the Stage 2/3 plan |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [ROADMAP.md](ROADMAP.md) | Development roadmap and architecture decisions |

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)

## Development

- Exclusively developed with [Claude Code](https://claude.com/product/claude-code)
- Git workflow managed with [Fork](https://fork.dev/)
- All code follows [CONVENTIONS.md](CONVENTIONS.md), derived from [Amphigraphic-Strict](https://github.com/PCfVW/Amphigraphic-Strict)'s [Grit](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit) — a strict Rust subset designed to improve AI coding accuracy.

