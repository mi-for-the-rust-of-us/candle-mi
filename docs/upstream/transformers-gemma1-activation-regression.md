# Gemma 1 silently uses exact GELU since v4.48.0

**Target:** `huggingface/transformers`
**Kind:** bug report (issue)
**Status: RESOLVED.** Filed 2026-09-23 as
[transformers#49051](https://github.com/huggingface/transformers/issues/49051); fixed by our
[#49084](https://github.com/huggingface/transformers/pull/49084), **merged 2026-09-25 19:14 UTC**
as `27166ea03f`, which auto-closed the issue one second later. Fifty-four hours from filing to
merge. See the filing sheet for the round-by-round record.
Two earlier candidate PRs verified and reported the same day (see
[Verification of the two candidate PRs](#verification-of-the-two-candidate-prs));
the `## Why this matters` section was corrected the same day, here and in the issue
body together, because the BF16 result falsified its heading.
Verified 2026-09-22; prior art re-checked 2026-09-23.

**Title for the issue:** Gemma 1 checkpoints silently use exact GELU instead of
`gelu_pytorch_tanh` since v4.48.0 (#35235 removed the `hidden_activation` guard)

---

## Summary

Since v4.48.0, `transformers` computes the **exact erf GELU** for the original
Gemma 1.0 checkpoints, instead of the tanh approximation those models were
trained with. There is no warning. The affected repos are:

- [`google/gemma-2b`](https://huggingface.co/google/gemma-2b)
- [`google/gemma-2b-it`](https://huggingface.co/google/gemma-2b-it)
- [`google/gemma-7b`](https://huggingface.co/google/gemma-7b)
- [`google/gemma-7b-it`](https://huggingface.co/google/gemma-7b-it)
- [`google/codegemma-2b`](https://huggingface.co/google/codegemma-2b)

Gemma 1.1, CodeGemma 7B-it and Gemma 2 are unaffected.

**Workaround**, for anyone who finds this before it is fixed:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("google/gemma-2b")
config.hidden_act = "gelu_pytorch_tanh"
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", config=config)
```

## Root cause

`GemmaMLP` used to guard the legacy config value. PR [#35235](https://github.com/huggingface/transformers/issues/35235)
("All attention refactor", commit [`2c47618c`](https://github.com/huggingface/transformers/commit/2c47618c), 2024-12-18)
replaced the guard with a direct
lookup and deleted the warning in the same diff:

```diff
-        if config.hidden_activation is None:
-            logger.warning_once(
-                "`config.hidden_act` is ignored, you should use `config.hidden_activation` instead.\n"
-            config.hidden_activation = "gelu_pytorch_tanh"
-        hidden_activation = config.hidden_activation
-        self.act_fn = ACT2FN[hidden_activation]
+        self.act_fn = ACT2FN[config.hidden_act]
```

applied to both `modeling_gemma.py` and `modular_gemma.py`. In the same commit,
`gemma2/modeling_gemma2.py` kept `ACT2FN[config.hidden_activation]`, so only
Gemma 1 lost its fallback.

That guard existed precisely because the Gemma 1.0 configs are wrong. They
shipped in February 2024 with `"hidden_act": "gelu"`, which `ACT2FN` maps to
`GELUActivation`, the exact erf form. PR [#29402](https://github.com/huggingface/transformers/issues/29402) added
`hidden_activation` to override it, and [#29995](https://github.com/huggingface/transformers/issues/29995) refined the
warning. Gemma 1.1 and later shipped
configs carrying the corrected value, but the 1.0 repos were never updated and
still depend on the removed guard. Their `config.json` still carries that string today, and the file's last
commit on `main` is 2024-09-27.

`hidden_activation` was then removed from `GemmaConfig` in v5.0.0, so the
override is no longer reachable at all.

The library now contradicts itself: `GemmaConfig.hidden_act` still **defaults**
to `"gelu_pytorch_tanh"`, while the docstring described it as "the legacy activation
function. It is overwritten by the `hidden_activation`" -- a field that no
longer exists -- as late as v5.1.0. That wording was dropped by v5.17.0; the
behaviour was not changed.

## Reproduction

```python
from transformers import AutoConfig
from transformers.models.gemma.modeling_gemma import GemmaMLP

config = AutoConfig.from_pretrained("google/gemma-2b")
print(config.hidden_act)                      # 'gelu'
print(type(GemmaMLP(config).act_fn).__name__) # 'GELUActivation'  (expected: GELUTanh)
```

No warning is emitted. Checked with `warnings.catch_warnings(record=True)` and
`transformers.utils.logging.set_verbosity_debug()` with a handler attached to
the `transformers` logger: zero Python warnings, and of the 34 log lines
produced, the only one mentioning the activation is the config dump echoing
`"hidden_act": "gelu"` back.

The selection rule is exactly whether `hidden_act` is present in `config.json`:

| repo | `hidden_act` | `hidden_activation` | resolved |
|---|---|---|---|
| `google/gemma-2b` | `"gelu"` | absent | `GELUActivation` (wrong) |
| `google/gemma-2b-it` | `"gelu"` | absent | `GELUActivation` (wrong) |
| `google/gemma-7b` | `"gelu"` | absent | `GELUActivation` (wrong) |
| `google/gemma-7b-it` | `"gelu"` | absent | `GELUActivation` (wrong) |
| `google/codegemma-2b` | `"gelu"` | absent | `GELUActivation` (wrong) |
| `google/gemma-1.1-2b-it` | absent | `"gelu_pytorch_tanh"` | default applies, correct |
| `google/codegemma-7b-it` | absent | `"gelu_pytorch_tanh"` | default applies, correct |
| `google/gemma-2-2b` | `"gelu_pytorch_tanh"` | `"gelu_pytorch_tanh"` | correct |

Gemma 1.1 is saved by *omitting* the field, not by correcting it.

Thirteen Gemma-family repos were checked against the live Hub. Beyond the five
listed as affected, these resolve correctly and are shown for completeness:
`gemma-1.1-2b-it`, `gemma-1.1-7b-it`, `codegemma-7b`, `codegemma-7b-it`,
`codegemma-1.1-2b` and `codegemma-1.1-7b-it` (the 1.1 line either omits
`hidden_act` or sets it to `gelu_pytorch_tanh`), `gemma-2-2b`, and
`paligemma-3b-pt-224` (its nested `text_config` is `model_type: gemma` and sets
neither field, so the default applies). `recurrentgemma-2b` is a different
`model_type` and a different MLP.

**Derivatives inherit the string**, which widens this well beyond the five
repos. A model fine-tuned from one of them and saved with `save_pretrained`
writes out the config it was loaded with, legacy value included. Spot-checking
the most-downloaded full-weight derivatives of `google/gemma-2b`:
`yerevann/chemma-2b`, `NexaAI/Octopus-v2` and
`Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0` all carry
`"hidden_act": "gelu"`. The Hub lists 868 public derivatives of
`google/gemma-2b-it`, and both `google/gemma-2b` and `google/gemma-7b` exceed
the 1000-result listing cap. So fixing only the five Google repos would leave
the long tail wrong, which is the main argument for fixing this in the library
rather than on the Hub.

## Measured effect

Entirely within `transformers`, on one loaded `google/gemma-2b` (F32, CPU,
eager attention), swapping only `mlp.act_fn` between the two variants and
comparing top-10 next-token logits:

| prompt | top-1, `gelu` | top-1, `gelu_pytorch_tanh` | max abs-diff over top-10 |
|---|---|---|---|
| "The capital of France is" | -16.5290 | -16.5282 | 4.219e-3 |
| "Two plus two equals" | -12.3041 | -12.3138 | **1.978e-2** |
| "Once upon a time, there was a" | -16.3700 | -16.3675 | 3.235e-3 |

**Max abs-diff over all 30 top-10 logits: 1.978e-2. Top-10 index mismatches:
0 of 30.** That combination is the reason this is easy to miss: the numbers are
wrong, the rankings are not.

Reproducible with one model load:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN

REPO = "google/gemma-2b"
PROMPTS = ["The capital of France is", "Two plus two equals",
           "Once upon a time, there was a"]

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(
    REPO, dtype=torch.float32, device_map="cpu", attn_implementation="eager").eval()
print("as-loaded act_fn =", type(model.model.layers[0].mlp.act_fn).__name__)

def run(act_key):
    for layer in model.model.layers:
        layer.mlp.act_fn = ACT2FN[act_key]
    with torch.no_grad():
        return {p: model(input_ids=tok(p, return_tensors="pt").input_ids,
                         use_cache=False).logits[0, -1, :].float().topk(10)
                for p in PROMPTS}

a, b = run("gelu"), run("gelu_pytorch_tanh")
print("max abs-diff:", max(float((a[p].values - b[p].values).abs().max()) for p in PROMPTS))
print("index mismatches:", sum(int((a[p].indices != b[p].indices).sum()) for p in PROMPTS))
```

Independently cross-checked against
[candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi), a Rust
reimplementation that applies the tanh approximation, which is how the bug
surfaced. Against the current `transformers` output its max abs-diff is
`1.977e-2` on CUDA; against the `gelu_pytorch_tanh` output it is `1.7e-5` on
CUDA and `3.4e-5` on CPU, i.e. ordinary F32 accumulation noise. The agreement
between that `1.977e-2` and the `1.978e-2` measured inside PyTorch above is
what establishes the activation as the whole of the difference.

## Why this matters, given that the F32 rankings do not move

> **Edited 2026-09-24:** this section originally read "given that the rankings do
> not move" and argued from 0 of 30 index mismatches, which is an F32-only result;
> re-measured in BF16 it is 7 of 30, so the heading and argument 2 below have been
> corrected. Measurement in [this comment](https://github.com/huggingface/transformers/issues/49051#issuecomment-5811228231), and in
> [Verification of the two candidate PRs](#verification-of-the-two-candidate-prs)
> below.

The 0-of-30 figure above is the honest measurement in F32, and it is also the
obvious reason to file this as cosmetic. Three arguments against that reading.

**1. The question has already been answered here.** On
[`google/gemma-2b-it` discussion #39](https://huggingface.co/google/gemma-2b-it/discussions/39)
(April 2024), a user proposed settling the then-current warning by switching
these configs to the legacy `gelu`. A HuggingFace maintainer answered that the
model was designed with the approximate GELU and that `gelu_pytorch_tanh` is
what it should be run with, and the remedy recommended was to set
`hidden_activation` explicitly in `config.json`.

That is the field [#35235](https://github.com/huggingface/transformers/issues/35235) stopped reading, and that v5.0.0
removed. So anyone who followed the advice given on the Hub in April 2024 is
being silently ignored today.

**2. In BF16 the rankings do move, and the F32 measurement is single-token
anyway.** Re-running the same three prompts gives 0 of 30 top-10 index mismatches
in F32 on both CPU and CUDA, and **7 of 30 in BF16**, which is the precision these
checkpoints are actually served at (3 of 10, 4 of 10 and 0 of 10 across the three
prompts). A determinism control confirms that two unchanged forwards are
bit-identical on that device, so the reordering is the activation and not
run-to-run noise.

Beyond that, every number here is one next-token distribution from a short prompt,
and nothing in it measures a continuation. A 2e-2 logit shift is invisible when the
top-1 margin is wide and decisive when it is narrow, and decoding is sequential, so
one flipped comparison diverges the whole remaining sample. Even the F32 "0 of 30"
is evidence about three first tokens, not about generated text.

**3. It is a train/serve mismatch, not a precision budget.** The affected
checkpoints were trained with one function and are being served with another.
That is different in kind from accumulation noise, which is why the cross-check
above separates them: `1.7e-5` against the correct activation is noise, and
`1.977e-2` against the current one is not.

## Why this has gone unnoticed for 21 months

Worth stating, because it bears on how to fix it.

1. **The warning was deleted in the same commit that broke it.** The one signal
   a user could have received is gone.
2. **Rankings do not move.** All 30 top-10 token indices matched across three
   prompts even with the wrong activation. The shift is about 6e-4 relative. No
   benchmark, eval or generation smoke test would register it.
3. **It rode inside a large attention refactor.** An activation guard in an MLP
   constructor is not what a reviewer of an attention PR is looking for.
4. **Gemma 1.0 is superseded** by 1.1, 2 and 3, so these repos see little
   scrutiny.

Catching it requires exact-logit parity against an independent implementation,
on Gemma 1.0 specifically.

## Suggested fix

Options, roughly in order of preference:

1. **Restore the legacy mapping in `GemmaConfig.__init__`**: when
   `hidden_act == "gelu"`, set it to `"gelu_pytorch_tanh"` and warn once. This
   fixes every affected checkpoint without touching the Hub, and it keeps the
   correction in one place rather than in `GemmaMLP`.
2. **Update the five `config.json` files on the Hub** to
   `"hidden_act": "gelu_pytorch_tanh"`. Correct at the source, but it does
   nothing for already-downloaded caches, and nothing for the thousands of
   public derivatives that copied the legacy value into their own configs.
3. At minimum, **warn** when a Gemma 1 config resolves to exact GELU, so the
   silent case stops being silent.

Option 1 is the one that reaches the derivatives, which is why it is first.
Option 2 complements it at the source; doing both would be ideal.

Happy to open a PR for whichever shape is preferred.

## System info

- `transformers` 5.1.0, `torch` 2.10.0+cu130, Python 3.14, Windows 11
- Version bisect on the guard, by release tag: present through v4.47.1, gone in
  v4.48.0
- `google/gemma-2b` at revision `9cf48e52b224239de00d483ec8eb84fb8d0f3a3a`
- Cross-checked against [candle-mi](https://github.com/mi-for-the-rust-of-us/candle-mi) 0.2.1, an independent Rust
  implementation of the forward pass, whose
  [`tests/validate_gemma_forward.rs`](https://github.com/mi-for-the-rust-of-us/candle-mi/blob/main/tests/validate_gemma_forward.rs)
  is how this surfaced

## Prior art checked

Searched `huggingface/transformers` issues and PRs for `hidden_activation gemma`,
`gemma gelu_pytorch_tanh` and `gemma approximate gelu`. Every Gemma-activation item is from
March to April 2024 ([#29402](https://github.com/huggingface/transformers/issues/29402) added `hidden_activation`,
[#29995](https://github.com/huggingface/transformers/issues/29995) refined its warning), both closed, both part of the original correction. Nothing after
2024-12-18 refers to this. Re-checked 2026-09-23: still nothing.

cc @danielhanchen, who diagnosed the exact-versus-approximate GELU problem for
Gemma in [#29402](https://github.com/huggingface/transformers/issues/29402) and whose correction is what went missing here.

---

## Verification of the two candidate PRs

Posted 2026-09-24 as
[comment-5811228231](https://github.com/huggingface/transformers/issues/49051#issuecomment-5811228231).

Two PRs arrived within the hour of our 2026-09-23 reply, so no PR of ours was needed.
What we contributed instead was a measurement. The harness is
[`verify_gemma_activation.py`](verify_gemma_activation.py) in this directory: one script,
run unchanged against four trees (`5.1.0`, `main` at `4b28d51`,
[#49061](https://github.com/huggingface/transformers/pull/49061),
[#49063](https://github.com/huggingface/transformers/pull/49063)) on CPU and CUDA in F32
and BF16, twelve runs in total.

| | `5.1.0` | main `4b28d51` | #49061 | #49063 |
|---|---|---|---|---|
| `config.hidden_act` | `gelu` | `gelu` | **`gelu_pytorch_tanh`** | `gelu` |
| `save_pretrained` writes | `gelu` | `gelu` | **`gelu_pytorch_tanh`** | `gelu` |
| `ACT2FN[config.hidden_act]` | `GELUActivation` | `GELUActivation` | **`GELUTanh`** | `GELUActivation` |
| as-loaded `mlp.act_fn` | `GELUActivation` | `GELUActivation` | `GELUTanh` | `GELUTanh` |
| bit-identical to reference | no | no | **yes** | **yes** |
| repo-wide `check_modular_conversion` | n/a | pass | **pass** | **fail** |

Four findings worth keeping.

**1. The version gap was a non-issue, and now it is measured rather than argued.** The
issue's System Info says `5.1.0` while `main` is `5.18.0.dev0`, seventeen minors apart.
Both trees agree in every digit, and the originally filed per-prompt table reproduces
exactly on both (`4.219e-3`, `1.978e-2`, `3.235e-3`, 0 of 30).

**2. BF16 moves the rankings, and this falsified our own filed claim.** 0 of 30 top-10
index mismatches in F32 on CPU and CUDA; **7 of 30 in BF16** (3, 4 and 0 across the three
prompts). The `## Why this matters` section above was corrected on 2026-09-24 as a result,
in the local document and in the issue body together.

**3. The discriminator between the PRs is the config, not the numerics.** Both are
`torch.equal` to the reference. Only #49061 corrects `config.hidden_act`, what
`save_pretrained` writes, and therefore what any consumer reading `config.json` resolves
to. That last is candle-mi's own position and how the bug surfaced.

**4. #49063 fails repo-wide modular conversion, across 50 models.** Its guard sits in
`GemmaMLP.__init__` in `modular_gemma.py`, and `GemmaMLP` is the modular base for
`Qwen3MLP`, `OlmoeMLP`, `NomicBertMLP`, `BitNetMLP` and 46 more. Those families default to
`silu`, `relu2` or `gelu_pytorch_tanh`, so it does not fire for them by default; it is
CI-blocking structurally and behaviourally live for any of their checkpoints that sets
`hidden_act: "gelu"`.

Two method notes, both recorded because each cost a wrong intermediate conclusion:

- **The narrow invocation hides finding 4.** `check_modular_conversion --files
  src/transformers/models/gemma/modular_gemma.py --check_all` passes on both PRs. Only
  `--check_all` with no `--files` surfaces the propagation. The first pass here used the
  narrow form and concluded "both consistent".
- **`warning_once` is `@functools.lru_cache(None)`.** An earlier probe in the same process
  silently suppresses the warning at the point being measured, which made #49063 look as
  if it never warned. The harness clears the cache between measurements.

Controls, without which none of the above is readable: the two activation functions agree
to `4.734992980957031e-4` in all twelve runs, and two unchanged forwards are bit-identical
in all twelve. Stack: Windows 11, Python 3.14.0, torch 2.10.0+cu130, RTX 5060 Ti 16311 MiB
(driver 610.88, cc 12.0), ruff 0.14.10 matching the `setup.py` pin.
