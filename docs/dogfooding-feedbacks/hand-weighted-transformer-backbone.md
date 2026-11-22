> **Status: IMPLEMENTED on `main`, unreleased** (2026-11-22, commits `22d56b0` to `edd6964`), from
> the causal-links line of askesis (canvas leg). Design: `design/plain-transformer.md`, reviewed three
> times by the askesis side before code. Ask 1 shipped as **the second option the report offered, a
> separate backbone**: `PlainTransformer` behind a new `plain` feature, with every component
> explicit (norm none / `LayerNorm` / `RmsNorm`, activation including the new `Activation::Relu`,
> `mlp_hidden`, `head_dim` independent of `d_model`, plus an `f64` `attn_scale`, a causal flag,
> optional positions and biases, and an `output_size` separate from the input vocabulary), so
> `OthelloGpt` and its trained checkpoints are untouched rather than bit-identical by contract.
> Ask 2 shipped **for the new backbone only** (the maintainer's call): `PlainTransformer::from_tensors`,
> strict about names, shapes and dtype; `StoicheiaTransformer` is unchanged. Ask 3 holds by
> construction (logits at every position). Beyond the asks: `HookPoint::MlpAct`, the MLP neurons,
> which no backbone exposed (`MlpPost` is the MLP output in candle-mi, not the post-activation).
> Validated four ways, all in CI: seven `Tracr`-compiled programs reproduced within the agreed
> relative tolerance `1e-5 * max(1, max |x|)` (`JAX` against candle; zero difference was measured
> on Windows, which is an observation, not a guarantee), bit-identity with `OthelloGpt` configured
> as it (the only bit-identity check, candle against candle), an independent `f64` reference over 96 option combinations,
> and a hand construction of the "establish" step with an exact 0.5 / 0.5 tie (also the example
> `plain_construction`). **Limitation to know before the exhaustive check:** there is no padding
> mask, so with bidirectional attention batch only partial plans of the same length.
> The original report follows unchanged.

# A hand-weighted transformer has no backbone to land on: `StoicheiaTransformer` is norm-free but MLP-less, single-head and last-position-only, `OthelloGpt` has MLPs but forces `LayerNorm` and `GELU`

**Date:** November 22, 2026
**Source:** askesis canvas leg, causal-links line: "attention-SNLP by construction" (a transformer
whose heads implement causal-link establishment and threat detection by design), before its design note
**Affected area:** `src/diffusion/othello.rs` (`OthelloGpt`, `OthelloGptConfig`),
`src/stoicheia/mod.rs` (`StoicheiaTransformer`), `src/config.rs` (`Activation`, `NormType`)
**Severity:** Feature — **not a defect, nothing blocked.** The construction can be done today in
Python/JAX with Tracr; the gap only decides whether it is done in candle-mi (the dogfooding choice)
**Measured against:** candle-mi 0.2.2 (`b85631c`), `candle-core`/`candle-nn` 0.11; nothing built
yet, so every claim below is READ FROM SOURCE (a HYPOTHESIS until the construction runs)

---

## The use case

askesis asks whether a trained model plans in the sense of classical planning: whether attention
corresponds to causal links (producer, condition, consumer). Twenty trained seeds of a 6-layer
causal transformer on blocksworld now show a head that attends from a consumer to the producer of
`clear(x)` on 12 of 20 seeds (registered RESULTs Me44-Me48; exact 95 % interval 36-81 %). That is
an observation. The next step is the converse, an **existence proof**: write one round of SNLP
(McAllester and Rosenblitt 1991) as attention, with weights set by construction, and check it
exhaustively against SNLP on small blocksworld partial plans:

1. **establish:** a head whose query is an open condition `p` and whose keys are step effects, so
   that it attends to an adder of `p` (that is a link);
2. **detect threats:** a head whose query is a link and whose keys are steps that delete `p` and
   could fall between producer and consumer;
3. **resolve:** an output per position (an ordering token, promotion or demotion, or a new step).

The construction follows Tracr's discipline (Lindner et al. 2023): the residual stream is a sum
of named subspaces, attention reads and writes chosen subspaces, MLPs compute small lookups with
`ReLU`, and **there is no layer norm**, because a norm rescales every subspace at once. The
trained models' instruments (writer and keeper tests, attention redirects) would then be run on
the constructed model, so the reference and the observed heads are read by the same code.

## 1. The verification half already works

Every backbone fires `AttnScores(i)` and `AttnPattern(i)`, and `AttnScores` is fired pre-softmax,
so a knockout added there is seen by the softmax (`src/stoicheia/mod.rs:431-433`). `OthelloGpt`
folds the scale into `q` and documents that `AttnScores` still holds the post-scale scores
(`src/diffusion/othello.rs`, the "THE SCALE IS FOLDED INTO Q" comment). Reading a constructed head
and patching it needs nothing new.

## 2. `OthelloGpt` cannot host a norm-free, `ReLU` construction

READ FROM SOURCE: each block is pre-`LayerNorm` (weight and bias) before attention and before the
MLP, plus `ln_f` before the head (`othello.rs:235-236, 551`); the MLP is exact `GELU`
(`othello.rs:370`); its width is `mlp_ratio * n_embd`; `head_dim` is derived as `n_embd / n_head`.
None of these is optional. `Activation` has `Silu`, `Gelu`, `GeluApprox`, no `Relu`
(`src/config.rs:89`); `NormType` has no identity (`src/config.rs:66`).

HYPOTHESIS (to settle when the construction runs): a construction can be made `LayerNorm`-robust,
for instance by keeping the residual norm constant, and `GELU` approaches `ReLU` with large
pre-activations. Both are tricks the existence proof should not depend on: the point of building
by hand is that every number is explained.

Loading is not the problem: `OthelloGpt::load(config, vb)` takes any `VarBuilder`, and
`VarBuilder::from_tensors` builds one from a `HashMap` in memory (`othello.rs:507, 942`).

## 3. `StoicheiaTransformer` is norm-free, but too narrow

READ FROM SOURCE: no layer norm and learned positions, exactly right (`stoicheia/mod.rs:467-473`).
But it is attention-only (no MLP), single-head (`num_heads = 1` for the transformer arch,
`stoicheia/config.rs:133-137`; one `[batch, 1, seq, seq]` pattern), unembeds the last position only
(`stoicheia/mod.rs:607-610`), has no bias and no causal mask, and loads from a file path only
(`stoicheia/mod.rs:518`). Threat detection needs several heads per layer, an MLP for the ordering
test, and an output at every position.

## Ask

Aimed at the next minor release (v0.2.3 if it fits, else v0.3.0). In order of how much each helps:

1. **A plain block that can be norm-free and `ReLU`** (the one that matters). Either options on
   `OthelloGptConfig`:

   ```rust
   pub norm: Option<NormType>,     // None = identity at ln1, ln2 and ln_f
   pub activation: Activation,     // + Activation::Relu
   pub mlp_hidden: usize,          // replaces mlp_ratio * n_embd when set
   pub head_dim: usize,            // explicit; n_head * head_dim need not equal n_embd
   ```

   with the contract that the default config gives a forward pass bit-identical to today's, so
   every trained model, hook and intervention is untouched; or a separate small backbone for
   constructed models. Which of the two is the crate's call.
2. **A uniform in-memory constructor** on every backbone,
   `load_from_tensors(config, HashMap<String, Tensor>, device)`, so a constructed model never
   round-trips through a file (`StoicheiaTransformer` lacks it).
3. **Per-position outputs** wherever a backbone unembeds the last position only (or a hook on the
   pre-unembed residual at every position, which `FinalNorm` already is on `OthelloGpt`).

## Not asked for

- A RASP compiler or a Tracr port in candle-mi. The program is written by hand on the askesis
  side; candle-mi only has to run the weights faithfully.
- Hard (argmax) attention. A large softmax scale set in the weights is enough, and is itself part
  of what the construction must show.
- Training. The constructed model is never trained.

## Why it is a crate change and not an askesis one

askesis could keep a private forward pass, but then the constructed reference and the trained
models would be read by two different implementations, the very confound the house method avoids.
The point is one hook surface for both.

## What worked, and is worth keeping

- The hook surface (`AttnScores` pre-softmax, `AttnPattern` post-softmax) on every backbone.
- `VarBuilder` injection into `OthelloGpt::load`: weights from memory, no file.
- The documented scale fold: a reader knows exactly what `AttnScores` holds.

## How to reproduce

Read, at `b85631c`: `src/diffusion/othello.rs` lines 49-77 (config), 186-245 (block), 300-330
(attention), 370 (MLP), 507-552 (load); `src/stoicheia/mod.rs` lines 400-470 and 518-610;
`src/stoicheia/config.rs` 128-137; `src/config.rs` 66-99.
