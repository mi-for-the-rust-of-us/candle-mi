# Examples

![CUDA](https://img.shields.io/badge/CUDA-13.1-76B900?logo=nvidia)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%20Ti%2016GB-76B900?logo=nvidia)
![RAM](https://img.shields.io/badge/RAM-64%20GB-blue)

Runnable examples demonstrating candle-mi features.
See also: [HOOKS.md](../HOOKS.md) for hook point reference, [BACKENDS.md](../BACKENDS.md) for adding new models.

## Table of Contents

- [Available Examples](#available-examples)
- [Running](#running)
- Example output:
  [logit_lens](#example-output-logit_lens) |
  [attention_knockout](#example-output-attention_knockout) |
  [steering_dose_response](#example-output-steering_dose_response) |
  [attention_patterns](#example-output-attention_patterns) |
  [activation_patching](#example-output-activation_patching) |
  [token_positions](#example-output-token_positions) |
  [rwkv_inference](#example-output-rwkv_inference) |
  [recurrent_feedback](#example-output-recurrent_feedback) |
  [character_count_helix](#example-output-character_count_helix) |
  [auto_config_dogfood](#example-output-auto_config_dogfood) |
  [figure13_planning_poems](#example-output-figure13_planning_poems) |
  [steering_convergence](#example-output-steering_convergence) |
  [attention_routing](#example-output-attention_routing)
- [Prerequisites](#prerequisites)

## Available Examples

| Example | Features | Description |
|---------|----------|-------------|
| `quick_start_transformer` | `transformer` | Discover cached transformers, run inference, print top-5 predictions |
| `fast_download` | *(default)* | Download a model from `HuggingFace` Hub with parallel chunked transfers |
| `quick_start_sae` | `sae`, `transformer` | Load an SAE, encode model activations, print top features and reconstruction error |
| `auto_config_dogfood` | `transformer` | Download a model and test auto-config loading with compatibility check |
| `generate` | `transformer` | Greedy autoregressive text generation on all cached models |
| `logit_lens` | `transformer` | Layer-by-layer prediction tracking via residual stream projection |
| `attention_knockout` | `transformer` | Knock out a specific attention edge (last→first token), measure KL divergence and top changed tokens |
| `steering_dose_response` | `transformer` | Sweep steering dose levels, build a dose-response curve, and interpolate target attention |
| `steering_convergence` | `transformer` | Steering convergence: inject contrastive steering vectors, measure cosine similarity to natural activations, identify absorption boundaries |
| `attention_patterns` | `transformer` | Capture and analyze per-head attention patterns at every layer |
| `activation_patching` | `transformer` | Causal tracing via position-specific activation patching (Meng et al., 2022) |
| `counterfact_patching` | `transformer` | CounterFact activation patching (Li et al., 2025 / Transluce protocol): contiguous layer-block patching with forced-choice prompts |
| `factual_routing` | `transformer` | Attention routing in factual recall: identify factual routing heads by measuring attention deltas during CounterFact patching (prolepsis) |
| `token_positions` | *(default)* | Character-to-token mapping with `EncodingWithOffsets` and `convert_positions` |
| `rwkv_inference` | `rwkv` | RWKV-7 linear RNN inference with state hook capture and state knockout |
| `recurrent_feedback` | `transformer` | Anacrousis / recurrent passes for rhyme completion (Taufeeque et al., 2024) |
| `character_count_helix` | `transformer` | Replicate the character count helix from [Gurnee et al. (2025)](https://transformer-circuits.pub/2025/linebreaks/index.html) via PCA on residual stream activations |
| `figure13_planning_poems` | `clt`, `transformer` | Replication of [Anthropic's Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location) (suppress + inject position sweep).  Since v0.1.11: `--strength-grid` for 2D position × strength sweeps, `--no-suppress` for inject-only ablations, and five `qwen3-*` presets (`Qwen3-{0.6B, 1.7B}-Base` × `BlueLightAI` 20K and 16K-dev CLTs) |
| `attention_routing` | `clt`, `transformer` | Measure how CLT suppress+inject changes attention routing from output position to planning site — identifies specific heads involved in rhyme planning |
| `clt_probe` | `clt`, `transformer` | Probe CLT feature activations at a token position — find suppress/inject candidates for `figure13_planning_poems` and `attention_routing` |
| `correction_test` | `clt`, `transformer` | Test whether downstream layers can reverse a prolepsis commitment — injects a contradictory feature at late layers and measures whether the output redirects |
| `vocab_scan` | `clt`, `transformer` | (v0.1.11) Anthropic Appendix B vocabulary scan: enumerate CLT features by decoder-cosine against the model embedding matrix, top-K tokens per feature.  Pair with `scripts/vocab_scan_cmudict_filter.py` for CMUdict-clustered rhyme-group discovery and `scripts/pick_inject_feature.py` for `figure13_planning_poems` preset construction |
| `figure13_newline_census` | `clt`, `transformer` | BlackboxNLP Exp 1 — newline feature census: do any CLT features carry anticipatory rhyme content at the poem newlines? Encodes all active features per position; pairs with `scripts/newline_census_classify.py` |
| `figure13_newline_steering` | `clt`, `transformer` | BlackboxNLP Exp 2 — composition-horizon steering: truncate after the line-3 newline, compose line 4, sweep steering position (m4). Does the newline shape the rhyme, or only emission? Pairs with `scripts/newline_steering_classify.py` |
| `figure13_newline_patch` | `transformer` | Newline **activation patching** (CLT-free): patch the line-3 newline residual from a minimal-pair poem whose line 3 ends on a different rhyme, then let the model compose line 4. Single-layer trace, an all-layer patch, an identity control, and a donor-vs-recipient row-divergence diagnostic. Needs no transcoder, so a null cannot be blamed on feature discovery |
| `clt_hook_reconcile` | `clt`, `transformer` | Encode a CLT feature under `ResidPre` / `ResidMid` / `ResidPost` at a position — identifies which residual reproduces documented activations |
| `clt_reconstruction_check` | `clt`, `transformer` | Decide which residual the CLT encoder was trained to read by reconstructing each layer's `MlpOut` from encode→decode (`ResidMid` vs `ResidPost`) |
| `clt_vs_plt_planning_site` | `clt`, `transformer` | CLT vs PLT method-matched comparison on the Llama 3.2 1B rhyme planning site (Hanna & Ameisen, *Latent Planning Emerges with Scale*) |
| `maar_contrastive_steering` | `transformer` | Maar et al. (2026) *What's the plan?* contrastive-activation-steering replication — raw mean-diff direction, generated-couplet last-word rhyme-family metric |
| `gridworld_prolepsis` | `clt`, `transformer` | Gridworld prolepsis — does early irrevocable commitment appear in a spatial planning task? |
| `means_ends_prolepsis` | `transformer` | Means-ends prolepsis — baseline feasibility (Step A) for the linguistic planning cell on base Gemma 2 2B |
| `means_ends_sweep` | `clt`, `transformer` | Means-ends prolepsis Step B — suppress+inject planning-site position sweep |
| `commitment_onset` | `clt`, `transformer` | Commitment-onset layer at the planning site (logit-lens + CLT activation) |
| `contrastive_patch` | `transformer` | Contrastive activation patching of the goal→action signal (CLT-free) |
| `action_dla` | `transformer` | MLP-vs-attention direct logit attribution of the goal→action signal (CLT-free) |
| `decision_trace` | `transformer` | Depth-axis irrevocability — the running action margin `logit(correct) − logit(alt)` across layers |
| `stoicheia_inference` | `stoicheia` | Run an AlgZoo RNN or transformer, compare predictions against ground-truth task function |
| `stoicheia_analysis` | `stoicheia` | Full Phase B MI pipeline: standardize weights, ablate neurons, probe roles, enumerate regions, run surprise accounting |
| `quick_start_mdlm` | `diffusion` | MDLM masked-diffusion fill-in-the-blank: mask `" Paris"`, run a bidirectional forward, apply `SUBS`, print the fill |
| `quick_start_othello` | `diffusion` | Quick start: `OthelloGpt` masked-diffusion fill-in (plain GPT-2 backbone, learned absolute positions) |
| `diffusion_logit_lens` | `diffusion` | Diffusion-time logit lens — the `(layer × denoising-step)` slice of the `(k, ℓ, π)` object at a masked target position; watch the prediction crystallize over denoising time |
| `diffusion_decoding_order` | `diffusion` | Decoding-order analysis — fill a masked completion under random / confidence / entropy unmasking orders; report per-order reveal-confidence and prediction-stability |

## Running

```bash
# Transformer inference on all cached models
cargo run --release --example quick_start_transformer

# Download a model (defaults to a tiny test repo)
cargo run --example fast_download -- meta-llama/Llama-3.2-1B

# SAE encoding on Gemma 2 2B
cargo run --release --features sae,transformer --example quick_start_sae

# Auto-config dogfooding — success (known model family, manual parser)
cargo run --release --features transformer --example auto_config_dogfood -- "meta-llama/Llama-3.2-1B"

# Auto-config dogfooding — failure (unsupported architecture)
cargo run --release --features transformer --example auto_config_dogfood -- "allenai/OLMo-1B-hf"

# Auto-config dogfooding — failure with actionable hints (non-standard naming)
cargo run --release --features transformer --example auto_config_dogfood -- "EleutherAI/pythia-1.4b"

# Greedy text generation — single model (recommended for 7B+ to avoid OOM)
cargo run --release --features transformer --example generate -- "meta-llama/Llama-3.2-1B"

# Greedy text generation — all cached models (add mmap for sharded weights)
cargo run --release --features transformer,mmap --example generate

# Logit lens — single model
cargo run --release --features transformer --example logit_lens -- "meta-llama/Llama-3.2-1B"

# Logit lens — with JSON output
cargo run --release --features transformer --example logit_lens -- "meta-llama/Llama-3.2-1B" --output examples/results/logit_lens/llama-3.2-1b.json

# Logit lens — all cached models
cargo run --release --features transformer,mmap --example logit_lens

# Attention knockout — single model
cargo run --release --features transformer --example attention_knockout -- "meta-llama/Llama-3.2-1B"

# Attention knockout — all cached models
cargo run --release --features transformer,mmap --example attention_knockout

# Steering dose-response — single model
cargo run --release --features transformer --example steering_dose_response -- "meta-llama/Llama-3.2-1B"

# Steering dose-response — all cached models
cargo run --release --features transformer,mmap --example steering_dose_response

# Steering convergence — contrastive steering + absorption boundary analysis
cargo run --release --features transformer,mmap --example steering_convergence -- "meta-llama/Llama-3.2-1B"

# Attention patterns — single model
cargo run --release --features transformer --example attention_patterns -- "meta-llama/Llama-3.2-1B"

# Attention patterns — all cached models
cargo run --release --features transformer,mmap --example attention_patterns

# Activation patching (causal tracing) — single model
cargo run --release --features transformer --example activation_patching -- "meta-llama/Llama-3.2-1B"

# Activation patching — all cached models
cargo run --release --features transformer,mmap --example activation_patching

# CounterFact patching (Li et al., 2025 / Transluce protocol)
cargo run --release --features transformer --example counterfact_patching

# CounterFact patching — quick test with 3 prompts
cargo run --release --features transformer --example counterfact_patching -- --limit 3

# CounterFact patching — with JSON output
cargo run --release --features transformer --example counterfact_patching -- --output examples/results/counterfact_patching/llama-3.2-1b.json

# Factual routing — identify factual routing heads (89 gold pairs)
cargo run --release --features transformer --example factual_routing

# Factual routing — with JSON output and memory reporting
cargo run --release --features transformer,memory --example factual_routing -- --output examples/results/factual_routing/llama-3.2-1b.json

# Token positions — single model (tokenizer only, no GPU)
cargo run --example token_positions -- "meta-llama/Llama-3.2-1B"

# Token positions — all cached models
cargo run --example token_positions

# RWKV inference — auto-discover cached RWKV models
cargo run --release --features rwkv --example rwkv_inference

# RWKV inference — specific model
cargo run --release --features rwkv --example rwkv_inference -- "RWKV/RWKV7-Goose-World3-1.5B-HF"

# RWKV inference — RWKV-6 model (requires rwkv-tokenizer feature)
cargo run --release --features rwkv,rwkv-tokenizer --example rwkv_inference -- "RWKV/v6-Finch-1B6-HF"

# Stoicheia inference — AlgZoo RNN (requires safetensors weights)
cargo run --features stoicheia --release --example stoicheia_inference -- --task 2nd-argmax --hidden-size 2 --seq-len 2 --weights tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors

# Stoicheia inference — AlgZoo transformer
cargo run --features stoicheia --release --example stoicheia_inference -- --task longest-cycle --hidden-size 4 --seq-len 4 --weights tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors

# Stoicheia analysis — full Phase B MI pipeline on M₂,₂
cargo run --features stoicheia --release --example stoicheia_analysis -- --weights tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors --hidden-size 2 --seq-len 2

# Stoicheia analysis — larger model (M₁₆,₁₀), 10K samples
cargo run --features stoicheia --release --example stoicheia_analysis -- --weights path/to/rnn_16_10.safetensors --hidden-size 16 --seq-len 10 --samples 10000

# MDLM masked-diffusion fill-in-the-blank (downloads mdlm-owt + gpt2 tokenizer on first run)
cargo run --release --features diffusion,mmap --example quick_start_mdlm

# Diffusion-time logit lens — (layer × denoising-step) slice at a masked target
cargo run --release --features diffusion,mmap --example diffusion_logit_lens

# Decoding-order analysis — random / confidence / entropy unmasking orders
cargo run --release --features diffusion,mmap --example diffusion_decoding_order

# Recurrent feedback — default (Llama 3.2 1B, unembed layers 8-15, strength 2.0)
cargo run --release --features transformer --example recurrent_feedback

# Recurrent feedback — with JSON output (prefill mode)
cargo run --release --features transformer --example recurrent_feedback -- --output examples/results/recurrent_feedback/prefill.json

# Recurrent feedback — sustained mode with JSON output
cargo run --release --features transformer --example recurrent_feedback -- --sustained --loop-start 14 --loop-end 15 --strength 1.0 --output examples/results/recurrent_feedback/sustained.json

# Recurrent feedback — custom layer range and couplet limit
cargo run --release --features transformer --example recurrent_feedback -- --loop-start 14 --loop-end 15 --max-couplets 5

# Character count helix — default model (Gemma 2 2B, requires mmap for sharded weights)
cargo run --release --features transformer,mmap --example character_count_helix

# Character count helix — with memory reporting (GPU name, per-process VRAM)
cargo run --release --features transformer,mmap,memory --example character_count_helix

# Character count helix — with memory debug output (raw GPU-backend values + per-chunk VRAM on stderr)
cargo run --release --features transformer,mmap,memory-debug --example character_count_helix

# Character count helix — with JSON output for Mathematica plotting
cargo run --release --features transformer,mmap --example character_count_helix -- --output examples/results/character_count_helix/helix_output.json

# Character count helix — quick variance scan across all layers
cargo run --release --features transformer,mmap --example character_count_helix -- --scan-layers all

# Character count helix — full PCA analysis on layers 10-12
cargo run --release --features transformer,mmap --example character_count_helix -- --pca-layers 10..13

# Character count helix — sweep mode: one layer per run, auto-resume from JSON
cargo run --release --features transformer,mmap,memory --example character_count_helix -- --sweep --output examples/results/character_count_helix/sweep.json

# Character count helix — sweep the next 5 layers in one run
cargo run --release --features transformer,mmap,memory --example character_count_helix -- --sweep 5 --output examples/results/character_count_helix/sweep.json

# Character count helix — sweep all remaining layers (overnight run)
cargo run --release --features transformer,mmap,memory --example character_count_helix -- --sweep all --output examples/results/character_count_helix/sweep.json

# Character count helix — sweep over Dickens chapters with per-process VRAM
cargo run --release --features transformer,mmap,memory --example character_count_helix -- --sweep --text-dir examples/results/character_count_helix/texts --output examples/results/character_count_helix/sweep.json

# Character count helix — use a bundled prose file (Gettysburg Address)
cargo run --release --features transformer,mmap --example character_count_helix -- --text examples/results/character_count_helix/texts/gettysburg.txt

# Character count helix — use a bundled prose file (Dickens)
cargo run --release --features transformer,mmap --example character_count_helix -- --text examples/results/character_count_helix/texts/dickens_two_cities.txt

# Character count helix — non-sharded model (mmap not needed)
cargo run --release --features transformer --example character_count_helix -- "meta-llama/Llama-3.2-1B"

# Figure 13 replication — Llama 3.2 1B (default)
cargo run --release --features clt,transformer --example figure13_planning_poems

# Figure 13 replication — Gemma 2 2B, 426K CLT (requires mmap for sharded weights)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-426k

# Figure 13 replication — Gemma 2 2B, 2.5M CLT (word-level features)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset gemma2-2b-2.5m

# Figure 13 — Qwen3 family (v0.1.11; one preset per (model, CLT width, rhyme group))
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-1.7b-20k-ation
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-1.7b-20k-teen
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-20k-ation
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-20k-teen
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-16k-ation

# Figure 13 — 2D position × strength grid sweep (v0.1.11; gives best (strength, position) per cell)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-20k-teen --strength-grid 0.5,1,2.5,5,10,25,50,100

# Figure 13 — inject-only ablation (v0.1.11; skip the suppress half to isolate the CLT decoder vector's effect)
cargo run --release --features clt,transformer,mmap --example figure13_planning_poems -- --preset qwen3-0.6b-20k-teen --no-suppress --strength-grid 0.5,1,2.5,5,10,25,50,100

# Vocab scan — enumerate CLT features by decoder cosine against the model embedding (v0.1.11)
cargo run --release --features clt,transformer,mmap --example vocab_scan -- --model Qwen/Qwen3-0.6B-Base --clt-repo bluelightai/clt-qwen3-0.6b-base-20k --output docs/experiments/figure13-qwen3-0.6b-20k/vocab_scan_qwen3_raw.json

# Attention routing — 426K CLT, suppress+inject (Figure 13 paradigm)
cargo run --release --features clt,transformer,mmap --example attention_routing -- --suppress L16:13725 --suppress L25:9385

# Attention routing — 2.5M CLT, suppress+inject
cargo run --release --features clt,transformer,mmap --example attention_routing -- --clt-repo mntss/clt-gemma-2-2b-2.5m --feature L25:82839 --suppress L25:57092 --suppress L23:49923 --suppress L20:77102

# Attention routing — inject only (for comparison, 13x weaker signal)
cargo run --release --features clt,transformer,mmap --example attention_routing
```

### Example output: `logit_lens`

Prompt: *"The capital of France is"* — tracking when "Paris" first enters the
top predictions across layers.

**Llama 3.2 1B** (16 layers): "Paris" first appears at layer 11 (rank 1).
Early layers predict generic tokens ("is", "was"); by layer 4 semantic concepts
emerge ("city", "capitals"). At layer 11, "Paris" surfaces alongside related
cities (Marseille, Bordeaux, Brussels). Convergence at ~69% depth — typical for
factual recall in small LLMs.

**Gemma 2 2B** (26 layers): "Paris" first appears at layer 25 (rank 8, the very
last layer). Through most of its depth, Gemma 2 strongly predicts continuation
patterns — " is" dominates layers 0-14 (often 99.9%), then " also" takes over
(layers 15-21), then " a" (layers 22-25). Factual resolution happens extremely
late; "Paris" only barely enters the top-10 at 0.001% probability.

**StarCoder2 3B** (30 layers): "Paris" as a complete token never reaches top-10.
However, the BPE subword " Par" (the first piece of " Paris") dominates from
layer 22 onward (33% → 74% by layer 26), alongside variants "Par", " par",
"PAR". The model clearly knows the answer but its code-oriented tokenizer splits
" Paris" across multiple tokens, so the `first_appearance` substring check
misses it.

### Example output: `attention_knockout`

Prompt: *"The capital of France is"* — knock out the attention edge from the
last token to position 0 (first token) across all heads at the middle layer.

| Model | Layer | Heads | KL div | "Paris" baseline | "Paris" ablated | Logit diff |
|-------|-------|-------|--------|-----------------|----------------|------------|
| Llama 3.2 1B | 8 | 32 | 0.056 | 39.3% | 26.0% | +0.55 |
| Gemma 2 2B | 13 | 8 | 0.017 | 3.9% | 6.7% | −0.33 |
| StarCoder2 3B | 15 | 24 | 0.029 | 40.9% (" Par") | 32.2% (" Par") | −1.08 |

**Llama 3.2 1B** shows the strongest effect: "Paris" drops from 39.3% to
26.0% when the last token can't attend to the first token at layer 8.
The model relies on early-position attention at mid-depth for factual recall.

**Gemma 2 2B** shows an inverted effect: "Paris" *increases* from 3.9% to
6.7%. The middle-layer attention edge carries inhibitory signal — hedging
tokens ("also", "not") drop when it's removed, consistent with Gemma 2's
late factual resolution (layers 22+).

**StarCoder2 3B** shows " Par" (BPE subword for "Paris") dropping from 40.9%
to 32.2%. Code tokens ("{", "{}") rise and competing capitals ("Mad",
"London") appear — the model partially reverts to its code-completion prior.

### Example output: `steering_dose_response`

Prompt: *"The capital of France is"* — steer the attention edge from the last
token to position 0 at the middle layer, sweeping 6 dose levels.

| Model | Layer | Baseline attn | Dose 0.5 KL | Dose 4.0 KL | Dose 6.0 KL |
|-------|-------|--------------|-------------|-------------|-------------|
| Llama 3.2 1B | 8 | 0.630 | 0.006 | 0.029 | 0.043 |
| Gemma 2 2B | 13 | 0.589 | 0.001 | 0.002 | 0.003 |
| StarCoder2 3B | 15 | 0.673 | 0.002 | 0.004 | 0.005 |

**Llama 3.2 1B** shows the strongest dose-response: KL divergence grows from
0.006 at half-dose to 0.043 at 6× dose, with "Paris" logit diff reaching
−0.31. The model's factual recall is sensitive to attention steering at
mid-depth.

**Gemma 2 2B** shows much weaker sensitivity: KL stays below 0.004 even at 6×
dose. With GQA (8 KV heads) and soft-capped logits, the prediction
distribution is robust to single-edge steering.

### Example output: `attention_patterns`

Prompt: *"The capital of France is"* — capture attention at every layer and
analyze what the last token attends to.

| Model | Peak layer (last→first) | Peak attention | Top-1 at most layers |
|-------|------------------------|----------------|---------------------|
| Llama 3.2 1B | 2 | 0.847 | `<\|begin_of_text\|>` (BOS) |
| Gemma 2 2B | 22 | 0.845 | `<bos>` (BOS) |
| StarCoder2 3B | 26 | 0.866 | `The` (first real token, no BOS) |

All three models show strong attention to the first token across most layers
(the "BOS sink" pattern). **StarCoder2 3B** lacks a BOS token so the first
real token ("The") serves as the attention sink. **Llama 3.2 1B** peaks early
(layer 2), while **Gemma 2 2B** peaks late (layer 22).

### Example output: `activation_patching`

Replicates the causal tracing technique from
[Meng et al. (2022)](https://arxiv.org/abs/2202.05262) "Locating and Editing
Factual Associations in GPT" (Section 2.1, Figure 1e). Two prompt pairs are
tested: "The capital of France is" → "Paris" and the paper's original
"The Space Needle is in downtown" → "Seattle".

**Subject-position sweep** (one column of the heatmap):

| Model | Subject pos | Corrupted KL | Best layer | Best recovery | Sharp cliff |
|-------|------------|-------------|------------|--------------|-------------|
| Llama 3.2 1B | 4 | 3.78 | 1 (100%) | Layers 0-8: >99% | Layer 9-15: 92%→0% |
| Gemma 2 2B | 4 | 0.50 | 1 (100%) | Layers 0-17: >89% | Layer 18-25: 74%→0% |
| StarCoder2 3B | 3 | 4.16 | 9 (99.9%) | Layers 0-20: >94% | Layer 21: 5% cliff |

**Full causal trace heatmap** (layer × token position) — Gemma 2 2B on the
Space Needle prompt, matching the paper's Figure 1(e):

![Causal Trace Heatmap — Gemma 2 2B, Space Needle](results/activation_patching/plots/google_gemma-2-2b_causal_trace_heatmap.png)

The heatmap shows two key patterns from the paper:

- **Late site** (layers 22–25, "downtown" row): bright red — attention at the
  last token copies factual information to the output, matching Meng et al.'s
  finding in GPT-2-XL (layers 25–45).
- **Subject tokens** ("Space", "Needle" rows): strong recovery across most
  layers. In GPT-2-XL, the "early site" was concentrated at the last subject
  token in mid-layers; in Gemma 2 2B the signal is more diffuse, likely due
  to GQA and logit softcapping distributing information more broadly.

**Llama 3.2 1B** shows a gradual decline: recovery drops from 100% at early
layers to 72% at layer 11, reaching 0% by the final layer. The factual
association "France → Paris" forms in the middle layers (8-13).

**Gemma 2 2B** maintains high recovery through layer 17 (89%), then drops
sharply. The factual lookup happens later in the network, consistent with
its deeper architecture.

**StarCoder2 3B** shows an abrupt cliff at layer 21: recovery drops from
94% to 5% in a single layer. As a code model, it stores factual knowledge
in a concentrated layer band.

Output JSON and Mathematica plotting script (heatmap, recovery curve) are in
[`examples/results/activation_patching/`](results/activation_patching/).

### Example output: `token_positions`

Text: *"The Eiffel Tower is located in Paris, France."* — mapping character
annotations to token positions across different tokenizers.

| Entity | Char range | Llama 3.2 1B tokens | Gemma 2 2B tokens | StarCoder2 3B tokens |
|--------|-----------|--------------------|--------------------|---------------------|
| "Eiffel Tower" | 4-16 | 4 tokens (E+iff+el+Tower) | 2 tokens (Eiffel+Tower) | 5 tokens (E+iff+el+T+ower) |
| "Paris" | 31-36 | 1 token | 1 token | 2 tokens (Par+is) |
| "France" | 38-44 | 1 token | 1 token | 1 token |

The example shows how the same character span maps to different numbers of
tokens across models. `char_range_to_tokens()` handles this automatically,
and `convert_positions()` provides exact-vs-fuzzy matching for positions
between or beyond token boundaries.

### Example output: `rwkv_inference`

Prompt: *"The capital of France is"* — RWKV-7 linear RNN inference with state
hooks and state knockout.

**RWKV-7 Goose 1.5B**: Top-1 prediction is "Paris" at high probability. The
example captures RWKV-specific hook points — `RwkvState` (recurrent state
matrix, shape `[1, heads, head_dim, head_dim]`), `RwkvDecay` (data-dependent
decay), and `ResidPost` (residual stream) — demonstrating the structural
differences between recurrent and attention-based architectures.

State knockout at position 0 (making the first token invisible to future tokens)
shows the impact on factual recall via KL divergence and top changed tokens.

### Example output: `recurrent_feedback`

15 canonical couplets from Taufeeque et al. (2024) — baseline generation vs.
recurrent feedback with averaged rhyme direction injection.

| Mode | Settings | Rhymes | Rescued |
|------|----------|--------|---------|
| Baseline | — | 9/15 | — |
| Recurrent (prefill) | unembed L8–15, s=2.0 | 11/15 | +2 |
| Recurrent (sustained) | unembed L14–15, s=1.0 | 9/15 | +0 |

Per-couplet breakdown (from golden JSON in `results/recurrent_feedback/`):

| id | target | baseline | prefill L8–15 s=2.0 | sustained L14–15 s=1.0 |
|----|--------|----------|---------------------|------------------------|
| 1 | light | light | light | light |
| 2 | play | talk X | laugh X | talk X |
| 3 | sound | flashes X | flashes X | flashes X |
| 4 | rain | falls X | ground X | ground X |
| 5 | time | time | time | time |
| 6 | air | air | air | air |
| 7 | gold | fair X | fair X | fair X |
| 8 | fire | embers X | **fire RESCUED** | embers X |
| 9 | stone | stone | stone | stone |
| 10 | dream | dream | dream | dream |
| 11 | strange | alone X | **strange RESCUED** | alone X |
| 12 | love | love | love | love |
| 13 | truth | truth | truth | truth |
| 14 | world | world | world | world |
| 15 | earth | earth | earth | earth |

**Prefill mode** (default) re-runs the recurrent block (double pass + feedback
injection) over the original prompt tokens before generation starts.
**Sustained mode** (`--sustained`) additionally re-runs the recurrent block at
the current last token during each autoregressive generation step. Prefill mode
with layers 8–15 and strength 2.0 shows the best improvement (+2 rescued
couplets: fire and strange, both producing the exact target word).

**Why candle-mi differs from plip-rs.** candle-mi recomputes the full sequence
at each generation step (no KV cache), so "prefill-only" mode already
re-applies the recurrent block over the entire prompt at every step — making it
functionally closer to plip-rs's sustained mode. In plip-rs (which uses KV
cache), prefill-only truly fires once and sustained mode was needed to get +1.
In candle-mi, prefill-only already achieves +2 because every generation step
benefits from the double pass. This is an MI-first design trade-off:
full-sequence recompute is slower but gives maximum observability — hooks can
re-observe how earlier positions change under intervention at every step, and
interventions "just work" without KV cache invalidation.

Resistant failures (2, 3, 4, 7) persist across both conditions, representing
couplets where no quality-preserving intervention redirects the generation
trajectory.

Output JSON and Mathematica plotting script are in
[`examples/results/recurrent_feedback/`](results/recurrent_feedback/).

**References:**
- Taufeeque et al., "Planning in a recurrent neural network that plays Sokoban", [arXiv:2407.15421](https://arxiv.org/abs/2407.15421v2), 2024
- Taufeeque et al., "Path Channels and Plan Extension Kernels: a Mechanistic Description of Planning in a Sokoban RNN", [arXiv:2506.10138](https://arxiv.org/abs/2506.10138), 2025 — reverse-engineers the planning circuits discovered in the 2024 paper
- Lindsey et al., ["On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), 2025
- Eric Jacopin, ["Replicating 'Planning in Poems' with Open Tools"](https://github.com/PCfVW/plip-rs/tree/melometis/docs/planning-in-poems) (plip-rs melometis branch)

### Example output: `character_count_helix`

Replicates the core finding from [Gurnee et al. (2025)](https://transformer-circuits.pub/2025/linebreaks/index.html)
"When Models Manipulate Manifolds" (Transformer Circuits). The model's residual stream represents line
character count (characters since the last `\n`) as a helical 1D manifold in a
low-dimensional subspace.

The example wraps prose at 14 different line widths (20-150 chars), runs forward
passes capturing `ResidPost` at an early layer, averages residual vectors by
character count, and performs PCA on the resulting mean vectors. Expected results:

- **Helix geometry**: The top 6 PCs capture ~95% of variance. Projecting the
  150 mean vectors into PC1-3 reveals a helical curve.
- **Ringing pattern**: The cosine similarity matrix shows off-diagonal
  oscillation — nearby character counts are positively correlated, those further
  apart are negatively correlated, then positive again (Gibbs-phenomenon-like
  ringing from projecting a high-curvature curve into low dimensions).

The `--text` flag lets you supply your own prose file to test whether the helix
generalises across different text content. Add `--features memory` for
per-process VRAM reporting (via the `hypomnesis` crate — DXGI on Windows,
NVML on Linux) and GPU adapter identification (e.g.,
`[NVIDIA GeForce RTX 5060 Ti]`). Use `--features memory-debug` to additionally
print the raw backend values to stderr.

**CLI flags — "what to analyse" vs "how to iterate":**

`--scan-layers` and `--pca-layers` select *what* to analyse:
- `--scan-layers all` runs a lightweight variance scan (top-6 explained
  variance only, no JSON) — useful for finding which layers carry the
  strongest helix signal.
- `--pca-layers 10..13` runs the full analysis (PCA projections, cosine
  similarity matrix, ringing summary, optional JSON output).

`--sweep` controls *how* to iterate: it runs the same full analysis as
`--pca-layers`, auto-resuming from the output JSON file. Accepts
`--sweep` (1 layer), `--sweep N` (next N layers), or `--sweep all`
(all remaining layers). Results are appended to a JSON array, so
repeated runs walk through layers 0, 1, 2, ... automatically.
Requires `--output`.

Typical workflow: `--scan-layers all` first to find the interesting layers,
then either `--pca-layers 10..13` to analyse a few at once, or `--sweep all`
for an overnight run.

**Layer 12 helix** — Gemma 2 2B residual stream, 30 Dickens chapters, 150
character counts projected onto PC1–PC3 (98.5% top-6 variance). The spiral
is the model's internal representation of "position within a line":

![Character Count Helix — Layer 12 rotating](results/character_count_helix/plots/L12_helix_rotating.gif)

Output JSON and Mathematica plotting script (3D helix, cosine heatmap, variance
bars) are in [`examples/results/character_count_helix/`](results/character_count_helix/).

**Reference:** Gurnee et al., ["When Models Manipulate Manifolds"](https://transformer-circuits.pub/2025/linebreaks/index.html), Transformer Circuits, October 2025.

### Example output: `auto_config_dogfood`

**Success** on Llama 3.2 1B (known family, uses manual parser):

![auto_config_dogfood success on Llama-3.2-1B](screenshots/auto_config_llama.png)

**Failure** on OLMo-1B (unsupported architecture):

![auto_config_dogfood failure on OLMo-1B-hf](screenshots/auto_config_olmo.png)

OLMo-1B fails the compatibility check because its weight names
(`model.layers.*.input_layernorm.weight`, `model.final_norm.weight`) do not
match the normalisation tensor patterns that `GenericTransformer` expects.
candle-mi currently supports 7 model families: LLaMA, Qwen2, Gemma, Gemma 2,
Phi-3, Mistral, and StarCoder2.

**Failure with actionable diagnostics** on Pythia 1.4B (non-standard naming):

![auto_config_dogfood failure on Pythia 1.4B with actionable hints](screenshots/auto_config_pythia.png)

Pythia uses the `gpt_neox.layers.{i}` weight prefix instead of the
HF-standard `model.layers.{i}`. The error message now shows which tensors
*were* found for each expected category (embedding, norm, attention, MLP),
detects the GPT-NeoX / Pythia naming convention, and points to Phase 9
(tensor name remapping) for planned support.

### Example output: `figure13_planning_poems`

Replicates [Anthropic's Figure 13](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poem-location)
from "On the Biology of a Large Language Model": suppress natural rhyme
features and inject an alternative, sweeping injection position across all
tokens.  Three presets are available: `llama3.2-1b-524k` (Llama 3.2 1B),
`gemma2-2b-426k` (Gemma 2 2B, 426K CLT), and `gemma2-2b-2.5m` (Gemma 2 2B,
2.5M CLT with word-level feature granularity).

**Gemma 2 2B, 426K CLT** — suppress "out" features + inject "around", sweep
injection position across all tokens:

![Figure 13 — Gemma 2 2B, suppress "out" + inject "around" (log scale)](figure13/gemma_log.png)

**How to read this chart:** each bar shows the probability of the word "around"
at a given token position when the injected CLT feature fires there. The y-axis
is log-scale, so the tall red bar at "passage" means P("around") jumps from
near-zero (~10⁻⁸ at most positions) to ~0.7 — a seven-order-of-magnitude
spike. This is the CLT's "planning site": the model decides at "passage" what
word will end the line, even though the rhyming word is still several tokens
away. The flat gray baseline everywhere else confirms the effect is
position-specific, not a global bias — the feature only influences the output
when injected at the exact position where the model plans its rhyme.

Output JSON and Mathematica plotting script are in
[`examples/figure13/`](figure13/).

### Example output: `steering_convergence`

Measures **attractor dynamics** in the residual stream: when you inject a
steering vector, does the model converge back to its natural computation
or take a different internal path?

**Factual recall (France → Paris):** strong attractor with a hard boundary
at ~1.2× the contrastive distance. Below the threshold, perturbations are
absorbed within 1-2 layers; above it, the model diverges permanently.

**Rhyme planning (20 rhyme groups):** contrastive steering produces no
measurable effect on rhyme predictions at the last token. This negative result
confirmed that planning operates through attention routing, not residual
stream perturbation — leading to the `attention_routing` experiment.

Full results, batch data, convergence matrix heatmaps, and analysis in
[`examples/results/steering_convergence/`](results/steering_convergence/).

### Example output: `attention_routing`

Measures how CLT suppress+inject changes **attention patterns** from the
output position to the planning site — the mechanism that carries planning
decisions through the model.

**Key finding:** L21:H5 is the dominant planning routing head in Gemma 2 2B,
with the H5 family spanning layers 17-25. This fills a specific gap identified
by [Anthropic](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-poems):
*"One crucial interaction (seems) to be mediated by changing where attention
heads attend... This is invisible to our current approach."*

![Top 10 routing heads](results/attention_routing/plots/top10_routing_heads.png)

![Strength sweep — planning attractor boundary](results/attention_routing/plots/strength_sweep_top_head.png)

The planning attractor has a **soft boundary** (gradual saturation at ~15×
strength) — fundamentally different from factual recall's hard threshold.

**Cross-model comparison (N=4):** Running the same experiment on Llama 3.2 1B
across 4 prompts from 3 rhyme groups reveals that planning routing is
prompt-specific — each prompt recruits a different dominant head. The
recurring heads (2+ prompts) are concentrated in mid-layers (mean 7.4),
while Gemma's L21:H5 sits in the late layers. All 4 Llama prompts exceed
Gemma in total routing shift despite having no dominant head — the signal
is distributed rather than concentrated.

![Cross-model planning routing](results/attention_routing/plots/cross_model_top_head.png)

Combined with the `factual_routing` experiment (which found L15:H8 as the
dominant factual routing head on the same model, with zero overlap between
recurring planning heads and the factual top-10), this establishes
**prolepsis** — early irrevocable commitment propagated through attention
routing at task-dependent network depths — as a structural motif in
transformers, across tasks, models, and scales.

Full results, cross-model comparison, and prolepsis analysis in
[`examples/results/attention_routing/`](results/attention_routing/).

## Prerequisites

- **quick_start_transformer** and **quick_start_sae** require models cached
  in `~/.cache/huggingface/hub/`. Download them first with `fast_download`
  or via Python (`huggingface_hub.snapshot_download()`).
- **quick_start_sae** downloads the Gemma Scope SAE (`google/gemma-scope-2b-pt-res`)
  automatically via `hf-fetch-model`.
- **figure13_planning_poems** — model and CLT weights download automatically
  on first run (~2.5 GB + CLT weights). The Llama preset requires `HF_TOKEN`
  and Meta license acceptance; Gemma 2 2B preset requires `--features mmap`.
- **attention_routing** — same prerequisites as `figure13_planning_poems`
  (Gemma 2 2B + CLT weights). Requires `--features clt,transformer,mmap`.
- **rwkv_inference** requires an RWKV model cached locally. RWKV-7 models
  include `tokenizer.json`; RWKV-6 models require `--features rwkv-tokenizer`.
- **recurrent_feedback** requires `meta-llama/Llama-3.2-1B` (default) cached
  locally.
- **character_count_helix** defaults to `google/gemma-2-2b` (~8 GB VRAM at F32,
  requires `--features mmap` for sharded weights). Two bundled prose files
  (Gettysburg Address, Dickens) are in `results/character_count_helix/texts/`.
  The Dickens chapters are from *A Tale of Two Cities*
  ([Project Gutenberg #98](https://www.gutenberg.org/ebooks/98)), public domain
  — no licensing restrictions for testing or research.
  Use `--text` to supply any plain-text file.
- **stoicheia_inference** and **stoicheia_analysis** accept both `.safetensors`
  and `.pth` files (format detected automatically from extension; `.pth` files
  are converted in memory via [anamnesis](https://crates.io/crates/anamnesis)'
  pickle VM — no manual conversion step). The M₂,₂ and transformer h4n4
  fixtures are committed in `tests/fixtures/stoicheia/` and work out of the box.
  For other models, download `.pth` files from [AlgZoo's GCS
  bucket](https://github.com/alignment-research-center/alg-zoo) and point
  directly at them.
- **stoicheia cross-validation**: the `validate_stoicheia` test suite verifies
  that candle-mi's RNN and transformer forward passes match Python/PyTorch
  reference outputs to 1e-4 (RNN) and 1e-2 (transformer) tolerance. The
  `stoicheia_analysis` integration test exercises all six Phase B MI modules
  (fast, standardize, piecewise, ablation, probing, surprise) on the M₂,₂
  fixture.
- **`memory` feature** enables per-process VRAM reporting, GPU adapter
  identification, and — on the NVML path with an R510+ driver — the
  driver/firmware **reserved** carve-out (a subset of the device total, matching
  `nvidia-smi -q -d MEMORY`'s `Reserved` line), delegated to the
  [`hypomnesis`](https://crates.io/crates/hypomnesis) crate. On Windows it uses
  DXGI (`IDXGIAdapter3::QueryVideoMemoryInfo`) — the
  only reliable per-process method under WDDM (NVML returns `NOT_AVAILABLE`); on
  Linux, NVML per-process queries; with an `nvidia-smi` (device-wide) fallback,
  and Apple Metal on macOS. The `memory-debug` feature (implies `memory`)
  forwards `hypomnesis/debug-output`, printing the raw backend values to stderr.
- **GPU recommended** for models larger than 1B parameters. candle-mi is
  developed on an RTX 5060 Ti (16 GB VRAM) with 64 GB RAM and CUDA 13.1.
