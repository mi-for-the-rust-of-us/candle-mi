# `PlainTransformer`: a backbone for hand-weighted (constructed) models

**Status:** implemented on `main` (2026-11-22, commits `22d56b0` to `edd6964`), after three reviews (§9). Answers
[`docs/dogfooding-feedbacks/hand-weighted-transformer-backbone.md`](../docs/dogfooding-feedbacks/hand-weighted-transformer-backbone.md).
**Feature:** `plain` (default off). **Module:** `src/plain/`.

## 1. Purpose and scope

askesis wants to write one round of SNLP as attention, with every weight set
by construction, then read the constructed model with the same instruments it
uses on trained ones. That needs a backbone where every number is explained:
no hidden norm, no fixed activation, no derived dimension. `OthelloGpt` forces
`LayerNorm`, exact `GELU`, `head_dim = n_embd / n_head` and `mlp_ratio`;
`StoicheiaTransformer` is norm-free but single-head, MLP-less and unembeds the
last position only. Decisions taken with the maintainer (2026-11-22):

- a **separate** backbone, not new options on `OthelloGpt` (its trained
  checkpoints and published numbers stay untouched by construction);
- named `PlainTransformer` behind a new `plain` feature;
- a new typed hook `HookPoint::MlpAct(i)` for the MLP neurons;
- an in-memory constructor on this backbone only (report ask 2, partially);
- a Tracr oracle in this work.

Out of scope, as the report says: a RASP compiler, hard attention, training.

## 2. Architecture

Pre-norm residual blocks, each **attention then MLP**, as in GPT-2 and Tracr:

```text
x      = tok_emb[ids] + pos_emb[0..seq]          (pos_emb optional)
for each layer i:
  a    = attention_i(norm1_i(x))                 (norm optional: identity if none)
  x    = x + a
  m    = W_out_i · act(W_in_i · norm2_i(x) + b_in_i) + b_out_i
  x    = x + m
logits = unembed · norm_f(x)                     (every position)
```

Attention, per layer, `n_head` heads of explicit `head_dim`:

```text
q, k, v = x·W_qᵀ + b_q, x·W_kᵀ + b_k, x·W_vᵀ + b_v      [batch, seq, n_head*head_dim]
scores  = (q · scale) · kᵀ   (+ causal mask)             [batch, n_head, seq, seq]
pattern = softmax(scores)
out     = merge_heads(pattern · v) · W_oᵀ + b_o          [batch, seq, d_model]
```

`n_head * head_dim` need not equal `d_model` (Tracr pads every head to the
largest key size). `scale` is explicit, default `1 / sqrt(head_dim)` (Tracr's
and GPT-2's), so a construction may also sharpen softmax through the scale
instead of the weights.

## 3. Configuration

`PlainTransformerConfig`, `#[non_exhaustive]`, built with `new` plus `with_*`
setters (the crate idiom):

| Field | Meaning | Default from `new` |
|---|---|---|
| `vocab_size` | input token rows of `tok_emb` | required |
| `output_size` | rows of `unembed` (logit width) | `vocab_size` |
| `max_positions` | rows of `pos_emb`; `None` = no positional embedding | `None` |
| `n_layer`, `d_model`, `n_head`, `head_dim` | explicit dimensions | required |
| `mlp_hidden` | MLP width, explicit (no ratio) | required |
| `activation` | `Activation::Relu`, `Gelu`, `GeluApprox`, `Silu` | `Relu` |
| `norm` | `None` (identity at norm1, norm2, norm_f), `Some(LayerNorm)` or `Some(RmsNorm)` | `None` |
| `norm_eps` | epsilon when `norm` is set | `1e-5` |
| `causal` | causal attention mask | `false` (Tracr's default; a partial plan looks both ways) |
| `attn_scale` | multiplier applied to `q` before `qkᵀ`, stored as `f64` | `1.0 / (head_dim as f64).sqrt()` |
| `bias` | biases on q, k, v, o and both MLP linears | `false` |

`attn_scale` is an `f64`, computed and applied exactly as `OthelloGpt` does
(`1.0 / (head_dim as f64).sqrt()`, then `q * scale`). An `f32` scale, or one
computed as `1.0f32 / sqrt`, can differ in the last bit of every scaled query,
which the bit-identical parity test (§7.2) must be able to see.

`output_size` exists because a Tracr model unembeds onto its *output* space,
whose size differs from the input vocabulary. `MIBackend::vocab_size` returns
`output_size`, consistent with the trait's own contract (`forward` returns
logits `[batch, seq, vocab_size]`); `embedding_vector` indexes the input table.
**Trap, documented on the type:** code that uses `vocab_size()` as an *input*
id bound is wrong for this backbone. No library code does (checked
2026-11-22); four examples do (`vocab_size() - 1` as the `[MASK]` id in
`diffusion_decoding_order`, `diffusion_logit_lens`, `quick_start_mdlm` and
`quick_start_othello`), all specific to MDLM or `OthelloGpt`. The input size is
`config().vocab_size`. `Activation::Relu` is a new variant of the shared enum, and
`GenericTransformer` gains it too (HF `hidden_act: "relu"`), so the variant is
not a dead arm there.

## 4. Weights

Key names follow the `PyTorch` `nn.Linear` convention (`weight` is
`[out, in]`), like every other backbone here:

| Key | Shape | When |
|---|---|---|
| `tok_emb.weight` | `[vocab_size, d_model]` | always |
| `pos_emb.weight` | `[max_positions, d_model]` | `max_positions` set |
| `blocks.{i}.attn.{q,k,v}.weight` | `[n_head*head_dim, d_model]` | always |
| `blocks.{i}.attn.o.weight` | `[d_model, n_head*head_dim]` | always |
| `blocks.{i}.attn.{q,k,v,o}.bias` | out dim | `bias` |
| `blocks.{i}.mlp.fc.weight` | `[mlp_hidden, d_model]` | always |
| `blocks.{i}.mlp.proj.weight` | `[d_model, mlp_hidden]` | always |
| `blocks.{i}.mlp.{fc,proj}.bias` | out dim | `bias` |
| `blocks.{i}.{ln1,ln2}.weight`, `ln_f.weight` | `[d_model]` | `norm` set |
| `blocks.{i}.{ln1,ln2}.bias`, `ln_f.bias` | `[d_model]` | `norm = LayerNorm` |
| `unembed.weight` | `[output_size, d_model]` | always (no bias) |

Head `h` owns rows `h*head_dim .. (h+1)*head_dim` of `q`, `k` and `v`, and the
same columns of `o`. `PlainTransformerConfig::weight_shapes()` is public and
returns this table, so a construction can allocate exactly what is expected.

**Q, K and V are concatenated once at load** into one `[3*n_head*head_dim,
d_model]` projection, then split after the matmul. A construction still writes
three named tensors; the forward runs one matmul instead of three, and runs
exactly the operations `OthelloGpt` runs (§7.2).

## 5. Construction and loading

- `PlainTransformer::from_tensors(config, HashMap<String, Tensor>, &Device)`:
  the constructed-model path (report ask 2).
- `PlainTransformer::load(config, VarBuilder)`: the file path, for symmetry
  with the other backbones.

`from_tensors` is **strict**, because a typo in a hand-written key must never
silently zero a head: every expected key must be present, every present key
must be expected, every shape must match the table, and every tensor must be
`F32` (constructions are exact `F32` numbers; other dtypes can come later if
asked). Each failure is a `MIError::Config` naming the key, the expected and the
actual value. `load` gets the extra-key check as far as a `VarBuilder` allows
(it cannot enumerate its keys), and is documented as such.

`forward` rejects `seq > max_positions` and any token id `>= vocab_size`. The
id check is one `max_all` and a scalar read; it matters because candle's CUDA
`index_select` does not bounds-check (the same reason `OthelloGpt` checks its
self-conditioning ids).

**No padding mask.** The only mask is the optional causal one: there is no
key-padding mask and no other mask input. With `causal = false`, real tokens
attend to padding, so **padding is not harmless**. Batch only sequences of the
same length, or run one at a time. A padding mask can be added later if a
consumer needs it; it is deliberately not assumed away here.

## 6. Hooks

The surface every backbone shares, fired at the same places as `OthelloGpt`:
`Embed`, `ResidPre`, `AttnQ`, `AttnK`, `AttnV` (`[batch, n_head, seq,
head_dim]`, before scaling), `AttnScores` (post-scale, post-mask, pre-softmax),
`AttnPattern`, `AttnOut`, `ResidMid`, `MlpPre` (MLP input, after norm2),
`MlpPost` and `MlpOut` (MLP output, equal here), `ResidPost`, `FinalNorm`.

**New: `HookPoint::MlpAct(i)`**, `[batch, seq, mlp_hidden]`, after the
activation and before the down projection: the neurons. Capturable and
intervenable (zero or patch one neuron). It accepts a positional patch (dim 1
is the position). Display name `blocks.{i}.mlp.hook_act`. Existing backbones do
not fire it yet; adopting it there is a separate change.

Naming caveat, documented on both variants: `MlpPost`'s display name is
`blocks.{i}.mlp.hook_post`, which in `TransformerLens` is the post-activation,
but in candle-mi `MlpPost` has always been the MLP **output**. Its doc line
("MLP post-activation") is corrected to say so; its string is kept, since
changing it would break existing captures. The neurons are `MlpAct`.

## 7. Validation

Three independent checks in Rust, and one against Tracr. None depends on
another.

### 7.1 Independent reference forward

A forward written in the test with plain `f32` loops (no candle), compared on
seeded random weights across every option combination: norm none / LayerNorm /
RmsNorm, each activation, causal or not, bias or not, positions or not,
`n_head*head_dim != d_model`, `output_size != vocab_size`, plus a non-default
`attn_scale` (`2.0`: moderate on purpose, since softmax multiplies a score
difference by the scale; a large scale is the subject of §7.3, not of this
sweep). Tolerance, per compared tensor `x` (the reference's): `max |Δ| <= 1e-5 *
max(1, max |x|)`, the same relative form as §7.4, so a large activation (a
`RmsNorm` output, a scaled score) cannot fail on correct `F32` rounding. This is
the only in-Rust check with a tolerance, because it is not candle on both sides.

### 7.2 Bit-identical parity with `OthelloGpt`

`PlainTransformer` configured as `OthelloGpt` (LayerNorm, exact `GELU`,
`mlp_hidden = mlp_ratio * d_model`, learned positions, biases, default scale,
`n_head*head_dim = d_model`), loaded with `OthelloGpt`'s weights (`qkv` split
into `q`, `k`, `v`; the loader re-concatenates them, so the projection tensor
is the same bits), must reproduce `OthelloGpt` **bit for bit**: both are candle
running the same operations in the same order. The test compares every common
hook in forward order and the logits, and on a mismatch names **the first hook
point where a bit differs** (and the max difference there) instead of passing
on a tolerance. Requires `plain` and `diffusion`; the `fused-attn` path is off
(a different kernel, not the same operations), and `OthelloGpt` runs with
`self_conditioning = false` (its optional `self_cond_emb` is added before the
first hook, an operation `PlainTransformer` does not have).

It runs on four configurations, so neither a scale nor a mask difference can
hide:

- `head_dim = 64` (`n_embd = 512, n_head = 8`, `OthelloGpt`'s own width:
  scale `0.125`, exact in every precision) and `head_dim = 96` (`n_embd = 768,
  n_head = 8`, since `OthelloGpt` derives `head_dim = n_embd / n_head`). Real
  widths rather than toy ones: wider matrices take more of the matmul's
  blocking paths, which makes bit-identity a stricter test, and two layers at
  these widths still run in well under a second on CPU;

  **Why 96, not the 48 first planned (found while implementing).** candle
  multiplies an `F32` tensor by a scalar after casting the scalar to `f32`
  (`T::from_f64` in the CPU `affine` kernel), so `OthelloGpt`'s `f64` scale is
  applied as an `f32` anyway. A scale computed in `f32` instead of `f64` is
  therefore visible only where the two round to *different* `f32` values: true
  for 54 of the head sizes 1 to 256, `96` among them, but **not** for `48` (nor
  `64`). Planting exactly that mutation, the test passed unseen at `48`; at
  `96` it fails at `blocks.0.attn.hook_scores` (max `|Δ|` `7e-7`) and names
  it. The irrationality of the scale was necessary but not sufficient;
- each with `causal = false` and `causal = true`. The causal path reuses
  `crate::util::masks::create_causal_mask`, the function `OthelloGpt` calls,
  so the comparison is of the same code, not of equivalent code.

**The bit-identity guarantee is on CPU**, where CI runs it. On CUDA the same
shapes should select the same cuBLAS algorithms, but that is not promised: the
test also runs on CUDA when a device is present and reports the outcome there
separately from the CPU result.

### 7.3 A hand construction with exactly known outputs

A small construction written in the test, in the spirit of the SNLP
"establish" step: one head whose query is a condition and whose keys are step
effects, with a large `attn_scale`, and an MLP lookup on top. Its outputs are
known on paper; the test asserts them, and `MlpAct` shows the expected neuron
firing. Two cases:

- **one establisher**: exactly one key matches, so the pattern is one-hot to
  `F32` precision;
- **a tie**: two steps both add the condition (a POCL node can have two
  candidate establishers). A large scale then gives a **0.5 / 0.5** pattern,
  not a one-hot. The test asserts that pattern and the neuron reading that
  follows from it, so the behaviour is documented rather than discovered.

**The tie is built exact, and checked exact before it is used.** Under a large
`attn_scale` any difference between the two scores, however small, becomes a
one-hot: a tie is a knife-edge, not a robust state. Near-orthogonality is not
enough (a residue of `1e-9` times a large scale still picks a winner). The two
scores are equal to the bit only if the two keys differ solely in components
that contribute exact zeros to the query-key product, so that the nonzero
products are the same terms, in the same positions and summation order. The
test therefore runs the tie with **positions off**, and the two establisher
rows identical on everything the key projection reads; it asserts **bitwise
equality of the two scores at `AttnScores` first**, and only then the 0.5 / 0.5
pattern. The same fragility applies to any construction that relies on a tie:
it has to be exact by design, not by accident (or be broken deliberately).

If that equality check ever fails, **suspect the kernel before the
construction**: two identical key rows normally give bit-identical keys, but a
matmul could treat a row in a remainder tile differently from one in a full
tile. The tie test keeps `seq <= 4`, so both rows sit in one tile.

**Why the asserted values are exact, not approximate.** Softmax computes
`exp(s - max) / sum`. The winning (or tied) entries are `exp(0) = 1` exactly,
so the result is exact if the other keys add nothing to the `F32` sum. candle
promises no summation order, so the leftover `sum_i exp(-g_i)` must stay below
half an ulp of `1.0` (`2^-24`), the smallest partial sum the small terms might
be added to: each non-matching score must sit at least
`g >= 24 ln 2 + ln(seq) ~ 16.7 + ln(seq)` below the winner after scaling. The
construction uses a gap far above that. Then:

- **tie:** the sum rounds to exactly `2.0`, and the two tied entries are exactly
  `0.5`;
- **single establisher:** the sum rounds to exactly `1.0`, so the winner is
  exactly `1.0`. The losers are `exp(-g)`, tiny but **not zero**, unless `g`
  passes the point where `F32` `exp` underflows to `0` (about `104`). "One-hot
  to `F32` precision" therefore means: winner exactly `1.0`, every loser below
  `2^-24`. Exact zeros are asserted only if the construction's gap exceeds
  `104`.

The same construction is a runnable example (`examples/plain_construction.rs`).

### 7.4 Tracr oracle (JAX against candle)

`scripts/plain_tracr_validation.py` compiles a few RASP programs from
`tracr.compiler.lib` (non-causal and causal, categorical and numerical, more
than one head per layer), exports weights, inputs and Tracr's per-layer
residuals, and `tests/validate_plain_tracr.rs` loads each into
`PlainTransformer` and requires the following, with the tolerance fixed here,
before any run:

- **residual stream, every layer** (`Embed`, and `ResidMid` / `ResidPost`
  against Tracr's `residuals`): `max |Δ| <= 1e-5 * max(1, max |x_layer|)`,
  where `x_layer` is Tracr's residual at that layer. Relative, because Tracr's
  numerical (discretising) MLPs use a large-number hyperparameter
  (`mlp_exactness`), so residual values far above 1 are normal and an absolute
  `1e-5` could fail on correct `F32` rounding. The measured maximum is
  reported per layer.
- **output, categorical programs:** the **same decoded tokens** (argmax per
  position, excluding the BOS position, which Tracr's decoder special-cases).
- **output, numerical programs** (no argmax): the output value, with the same
  relative tolerance.

The tolerance is not loosened after seeing a failure; a failure is
investigated.

Bit-identity is not asked of this comparison: JAX and candle are different
`F32` implementations.

Export choices: **`bias = true`**, loading Tracr's zero biases. Tracr creates
every linear with a bias (zeroed at assembly), so this is the faithful copy and
exercises the bias path; with `bias = false` the strict loader would rightly
reject those tensors as unexpected keys. Fixtures are generated **without
padding**: Tracr's `where(mask, ...)` also masks padding tokens, which
`PlainTransformer` does not (§5).

Tracr facts the export relies on, read from its source (commit `9ce2b8c`,
2024-02-05): no layer norm, `ReLU`, `causal=False` by default
(`compile_rasp_to_model(..., causal=...)`), logits `/ sqrt(key_size)` then
`where(mask, logits, -1e30)`, all linears with (zero) biases, `haiku` weights
stored `[in, out]` (transposed at export), position 0's embedding an all-zero
row, unembed `x @ res_to_out` then argmax. Masking: Tracr writes `-1e30`,
candle-mi adds `-inf`; both give an exact zero after softmax, so the residuals
are comparable. Fixtures are compiled with the same `causal` setting the test
gives `PlainTransformer`.

Environment: Python 3.11 venv (uv-managed), `tracr` at `9ce2b8c` (jax 0.10.2,
dm-haiku 0.0.17, numpy 2.4.6). Pinning only the top-level versions would let
jax's and haiku's own dependencies drift, so the whole environment is locked:
`scripts/plain_tracr_requirements.txt` (every package, with hashes, from
`uv pip compile --universal --generate-hashes`; Tracr pinned by commit, since a
git source cannot carry a hash). The environment itself stays out of the
repository. Verified 2026-11-22 by following only the script header's steps in
a fresh folder: it regenerates the committed fixture byte for byte.

**The oracle runs in CI**, not as an `#[ignore]`d resurrection entry: the
fixture is in `tests/fixtures/plain_tracr/` (585 KB of weights and expected
tensors plus an 8.6 KB manifest), needs no model download and runs in
milliseconds on CPU. Python is needed only to regenerate it, and regeneration
is byte-identical (checked by hash).

**Result (2026-11-22).** Seven programs: `length` (bidirectional and causal),
`reverse` (4 layers), `hist`, `sort_unique`, `frac_prevs` (causal, numerical
output) and `shuffle_dyck` (6 layers, two heads in a layer, a discretising
numerical MLP of width 146). All load through the strict loader and reproduce
`Tracr` with a measured difference of **exactly zero**, at every layer and in
every output, against the `1e-5` relative bound. That is not vacuous: the
residuals hold non-integer values, and a planted error of `1e-3` in the token
embedding, in a first-layer MLP bias, or a `0.1 %` change to layer 3's value
weights each fails at exactly the layer it enters. Exact agreement is plausible
here (weights are mostly `0` / `1`, attention is sharp enough that losers
underflow, the rest is the same few `F32` operations), but it is a measured
outcome, not a promise: the criterion stays the relative bound.

## 8. Wiring

New lane `plain` in `ci.yml`, `publish.yml`, `scripts/preflight.ps1` and the
docs.rs list (lane parity enforces all four), with the `OthelloGpt` parity test
in a `plain,diffusion` step. Rows in the README and `src/lib.rs` feature and
backend tables; `BACKENDS.md` and `HOOKS.md` sections (including `MlpAct`);
CHANGELOG `[Unreleased]`. No release.

## 9. Review changes (2026-11-22)

An external review against `66abe8e` found no architectural problem and asked
for the following, all taken:

1. `attn_scale` stored and computed as `f64`, exactly as `OthelloGpt`; parity
   also at `head_dim = 48`, where the scale is irrational (§3, §7.2).
2. Tracr tolerance made relative, `1e-5 * max(1, max |x_layer|)`, fixed before
   the run; numerical outputs compared by value, tokens only for categorical
   outputs (§7.4).
3. Tracr exported with `bias = true` and its zero biases (§7.4).
4. No padding mask, stated as a limitation; fixtures without padding (§5,
   §7.4).
5. A tie case (0.5 / 0.5) in the hand construction (§7.3).
6. Parity with `self_conditioning = false`, with `causal = true` as well as
   `false`, reusing `create_causal_mask`; bit-identity guaranteed on CPU,
   CUDA reported separately (§7.2).
7. `vocab_size()` callers checked: none in the library uses it as an input
   bound; four MDLM and `OthelloGpt` examples do, and the trap is documented
   (§3).

A second pass, also taken:

8. The tie is built exact (positions off, keys identical on what the key
   projection reads) and its bitwise score equality asserted at `AttnScores`
   before the 0.5 / 0.5 pattern; the fragility is stated (§7.3).
9. The two parity widths spelled out: `n_embd = 512, n_head = 8` and
   `n_embd = 384, n_head = 8` (§7.2).
10. §7.1 uses the same relative tolerance as §7.4, with a moderate
    non-default `attn_scale` (§7.1).

A third pass, also taken, with one tightening:

11. Why the asserted softmax values are exact: the gap condition, stated with
    the order-independent bound `g >= 24 ln 2 + ln(seq)` (half an ulp of `1.0`,
    not of `2.0`, since candle promises no summation order); and the
    single-establisher case corrected to "winner exactly `1.0`, losers below
    `2^-24`", since losers are exactly zero only past `F32` `exp` underflow
    (§7.3).
12. A failed tie-equality check points at the kernel before the construction;
    the tie test keeps `seq <= 4` (§7.3).

Found while implementing:

13. The second parity width is `n_embd = 768` (`head_dim = 96`), not `384`
    (`48`): candle casts a scalar multiplier to `f32` before applying it, so an
    `f32`-computed scale differs from the `f64` one only where they round to
    different `f32` values, which `48` does not and `96` does. Shown by
    planting the mutation: unseen at `48`, caught and named at `96` (§7.2).
