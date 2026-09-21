# `OthelloGpt` needs a second input: the model's own previous prediction (self-conditioning)

> **Status: IMPLEMENTED in v0.2.0** (2026-09-21, commit `9d3502f`), unreleased at time of
> writing. The channel ships with all three guarantees the Ask asked for, and **under a different
> name**: `self_conditioning`, not `carry`. Both of the report's own corrections were confirmed
> against the crate and both are honoured. Nine implementation hazards the Ask did not cover were
> found during validation; all nine are resolved or answered. **See the closure section at the end
> of this file**, which also records the two questions the Ask left to the crate and what the crate
> decided.
>
> **Status: ASKED** (2026-09-20), from the canvas leg of askesis. Registration:
> `askesis/reference/canvas/docs/open-measurements.md`, *"Measurement O — the self-conditioned
> MDLM"*; the reading that motivates it is devlog `Ph121` (2026-09-20). Written for
> implementation in a separate session; nothing in candle-mi has been changed for it yet.

## What canvas found, in one paragraph

The canvas MDLM (candle-mi's `OthelloGpt`, trained over a `VarMap` by canvas's own loop) plans
blocksworld at 97-100 % in distribution and 0/41 on `PlanBench`'s all-on-table corner. Six
weeks of decode-side and objective-side levers moved the mechanism, not the number. The last
decode-only reading settled where the residue lives: decoded LEFT TO RIGHT, so that every slot
is chosen with its whole prefix visible, every checkpoint tracks the plan's state almost
perfectly (held 98.6 / 100.0 / 99.8 on three models, the mid-plan failure class gone to zero
on one) — yet the corner gets WORSE under that order, and worse still on the arm without a
register. Confidence order commits the opening last, after wrong cells that override it; left
to right commits it first, blind. Both are irrevocable, and out of distribution no order of
irrevocable commits can be trusted. What the corner needs is a prediction the model can SEE
and REVISE before anything is final. That is self-conditioning (Chen, Zhang & Hinton 2022), the
mechanism the flow-reasoning papers of 2026 measure as the active ingredient on Sudoku-Extreme
(13 % -> 33 % from self-conditioning alone, -> 99 % with fixed-point forcing). It needs one
change in the model, and the model is candle-mi's by identity — canvas deliberately owns no
copy of it (`train.rs` header).

## The Ask

### 1. A second input to `forward`: `carry_ids`

Today (`src/diffusion/othello.rs:647`):

```rust
fn forward(&self, input_ids: &Tensor, hooks: &HookSpec) -> Result<HookCache>
```

Asked: the same forward with an optional second token grid of the SAME shape, `[batch, seq]`
`U32`, holding the model's previous clean prediction per position, or a reserved `NONE` id where
there is none:

```rust
fn forward_with_carry(&self, input_ids: &Tensor, carry_ids: Option<&Tensor>, hooks: &HookSpec)
    -> Result<HookCache>
```

with `forward(input_ids, hooks)` == `forward_with_carry(input_ids, None, hooks)` so every
existing caller, hook, capture and intervention is untouched. Whether this is a new trait method
with a default, or a field on `HookSpec`, is the crate's call; canvas needs only that the two
spellings above are one path.

### 2. Its embedding: `carry_emb`, `vocab_size + 1` rows, ZERO-initialised

At the embedding site (`othello.rs:657-660`):

```rust
let mut hidden = self.tok_emb.forward(input_ids)?;
// asked:
if let Some(carry) = carry_ids {
    hidden = hidden.add(&self.carry_emb.forward(carry)?)?;
}
hidden = hidden.broadcast_add(&pos)?;
```

- Row `vocab_size` is `NONE`; every row is zero at creation. **Zero, not the crate's seeded
  init**: with `carry_ids` all `NONE`, or absent, the forward must be bit-identical to today's,
  which is what lets canvas's V9 parity oracle pass unchanged and lets an anchor checkpoint
  trained without the table keep its numbers to the last digit. A small `carry_emb` init would
  silently move every published reading.
- **Loading a checkpoint that has no `carry_emb.weight` must succeed**, the table created as
  zeros — the plain-GPT loader already does this shape of thing for a missing bias
  (`othello-mdlm-plain-gpt-loader.md`). Saving writes it beside `tok_emb.weight`.
- The `NONE` row stays zero forever if the crate wants it so (canvas is indifferent: a trained
  `NONE` row is a learned "no carry" bias, harmless); the other rows are trainable like
  `tok_emb`'s, in the plain (no weight decay) group canvas already puts embeddings in.

### 3. Config: one flag, off by default

`OthelloGptConfig` gains `carry: bool` (default `false`), serialised as `"carry"` in the
companion `config.json` and read back by `load`; absent means `false`. With `carry: false` the
table is not created and `forward_with_carry(.., Some(..), ..)` is an `MIError::Model`, not a
silent ignore.

### 4. Tests the crate should own

- Switch-off parity: `forward_with_carry(x, None)`, `forward_with_carry(x, Some(all NONE))` and
  `forward(x)` produce the same logits, bit for bit on CPU, on a seeded model with `carry: true`.
- A non-`NONE` carry changes the logits (the table is wired in, not dead).
- Round-trip: save with `carry: true`, load, same forward; load a `carry: false` checkpoint
  into a `carry: true` config, the table is zeros, same forward as before.
- The carry is DETACHED by construction in canvas (it is a token grid, an argmax), so nothing
  is asked of autograd; but a test that `backward()` through `forward_with_carry` updates
  `carry_emb` rows that were used and not the `NONE` row would pin the trainability contract
  the way `trainable-backbones.md` did for the backbone.

## What canvas will do with it (so the crate knows the callers)

- **Training** (canvas `train.rs`): on half the batches, drawn by the frozen generator, run one
  forward with `None`, take the argmax per position as `carry_ids` (no gradient), then the
  scored forward with that carry; on the other half, `None`. The loss is unchanged. Two forwards
  on carry batches, one backward — the crate's fused backward (`training-throughput-ceiling.md`)
  is untouched.
- **Decode** (canvas `decode.rs`): round 0 with `None`; every later round with the previous
  round's argmax over the whole row as the carry, committed cells included (the model's own
  reading of them). Budget, schedule and commit rank as today. A new trace column counts how
  many committed cells the new prediction disagrees with, per round: the revision rate.
- **Fixed-point forcing** (the registered follow-up): the training carry from a multi-step
  rollout of the model's own decode rather than one pass. Same channel, no further crate change.

## Not asked for

- No change to the blocks, attention, hooks, captures or interventions: the carry is an
  embedding-level addition and the residual stream after it is what every hook already sees.
- No continuous / one-hot flow input (the FRM/FLM formulation): canvas stays a masked model on
  purpose, so the reading isolates the carry.
- No recurrent hidden state across steps (looped flows): a later ask if this one reads well.

## Why it is a crate change and not a canvas one

canvas's contract is that the model trained and the model probed are the same object, so a
divergence between a training copy and a probing copy cannot become a confound. A carry channel
bolted on in canvas would be exactly that copy. The change is small (one table, one `add`, one
flag, four tests) and it is where the model lives.

## Consistency pass (2026-09-20, same day), against the crate as it is

**Two corrections to the Ask above.**

1. **Where the add goes: BEFORE `HookPoint::Embed`.** `forward` fires `hook_point(&mut hidden,
   HookPoint::Embed, ..)` right after the positional add (`othello.rs:663`). The carry must be
   added before that line, so that `Embed` — and every `ResidPre`/`ResidPost` after it — sees
   the residual stream the model actually runs on. A carry added after the hook would make the
   logit lens and every capture read a stream the logits did not come from, which is the
   confound canvas's instruments (`l_rung`, `capture_at_commit`) exist to rule out.
2. **The loader precedent is not established.** The Ask cites `othello-mdlm-plain-gpt-loader.md`
   for "a missing tensor created as zeros"; that report does not record such a rule. The
   requirement stands on its own: a checkpoint saved without `carry_emb.weight` must load into a
   `carry: true` model with a zero table. If the `VarBuilder` path cannot express "absent means
   zeros", the crate should say so and canvas will write the zero table into old checkpoints at
   load time instead — but one of the two must exist, or the anchor pair is lost.

**Additions the Ask left implicit.**

- **dtype and device:** the table is created through the same `VarBuilder` as `tok_emb`
  (`vb.pp("carry_emb")`), so it follows the model's dtype (`F32` in production, `BF16` under
  canvas's Measurement A) and device without a special case.
- **Shape and id checks:** `carry_ids` must be `U32`, `[batch, seq]` equal to `input_ids`'s
  dims, with every id `<= vocab_size` (the `NONE` row). A mismatch is `MIError::Model`, as the
  `seq_len > block_size` case already is; never a broadcast.
- **The config key is read by name.** `OthelloGptConfig` is parsed from a `serde_json::Value`
  by key (`othello.rs:1009-1027`), so `"carry"` absent reads as `false` naturally; the
  companion `config.json` that canvas writes (`train::write_companion_config`) must gain the key
  — a canvas-side change, noted here so the two do not drift.
- **The carry may repeat the input.** At decode the carry holds the previous round's argmax
  over the WHOLE row, committed cells included, so at a committed position the model sees the
  same token through both tables. Intended: the carry is the model's own reading, and "what I
  predicted here last round" is information even where the cell is fixed. The alternative —
  carry only at masked positions, `NONE` elsewhere — is a canvas decode-time choice, to be
  pinned in canvas's substep O4 and readable with the same channel. Nothing in the crate need
  decide it.
- **EMA and checkpoint state (canvas side):** the new table is one more named `Var`, so it
  joins the `EMA` shadow, `canvas_best`, the archives and the resume state through the existing
  name-driven paths; the only thing to check is that `model::from_averaged` builds the forward-
  only model with the table present, which it will if it constructs from the config's `carry`.
- **The parity oracle's coverage (canvas side):** the V9 `PyTorch` oracle records un-augmented,
  no-carry batches, and with `carry` off it must pass bit for bit as before. It does NOT cover a
  carry batch — the second forward and the argmax between them are outside its recording. Either
  O3 records a carry-batch oracle (the `PyTorch` side is a second forward and an `argmax`, small)
  or the registration states the gap and reads the smoke's P0 as the gate instead. Left to O3;
  named here so it is not discovered on a box.
- **Cost, for the rental estimate:** a carry batch runs two forwards and one backward, so at the
  09-19 profile (forward 12 ms, backward 30 ms, draw overlapped) a carry step costs ~1.3x a
  plain one; with half the batches carried, an 80-epoch run grows by ~15 %, about an hour on the
  slow-core box.
- **What the `NONE` row means at round 0 and in the no-carry training half:** the same zero
  vector, so "no carry" in training and "round 0" at decode are one condition, which is what
  makes the decode's first round in-distribution for the model. If the crate lets the `NONE` row
  train, that stays true (both see the trained row); if it freezes it at zero, also true. Either
  is fine; the crate should say which.


## Crate-side closure (2026-09-21, v0.2.0, commit `9d3502f`)

Written by the crate, not by canvas. The Ask above is left exactly as filed, including the one
citation it invented and then retracted itself.

### The one divergence: it is called `self_conditioning`

The Ask says `carry_ids`, `carry_emb`, `"carry"`. The crate ships `self_cond_ids`,
`self_cond_emb.weight`, and the config key `"self_conditioning"`.

This is not a preference. `docs/adding-a-model.md` gained an auxiliary-input policy the day before
this landed, and rule 4 of it says to name a mechanism after the literature rather than after the
caller. `carry` is canvas's internal vocabulary; the next reader of the crate will know Chen, Zhang
and Hinton (2022) as self-conditioning. canvas is free to keep calling its own local variable
`carry`; only the crate-facing names changed.

**Action for canvas:** the companion `config.json` written by `train::write_companion_config` must
emit `"self_conditioning"`, not `"carry"`. An unrecognised key reads as `false`, so a mismatch does
not error, it silently disables the feature. That is the one place this rename can bite.

### The Ask's own two corrections: both confirmed

1. **The add goes before `HookPoint::Embed`.** Confirmed and implemented. It sits after the token
   embedding and before the positional add, so `Embed` and every later hook see the stream the
   logits came from.
2. **The loader precedent was not established.** Confirmed: `othello-mdlm-plain-gpt-loader.md`
   records no missing-tensor-as-zeros rule. The requirement stood on its own and is met via
   `VarBuilder::contains_tensor`, which does exist in candle-nn 0.11, so canvas does not need its
   fallback plan of writing zero tables into old checkpoints.

### The nine hazards found during validation

| # | hazard | disposition |
|---|---|---|
| 1 | `weight_shapes`' `else` arm would have given the table a `randn` draw | explicit `is_zero_init` branch; the table is zeroed, and the entry is appended **last** so the seeded RNG stream for every other tensor is untouched whatever the init rule becomes |
| 2 | `OthelloGptConfig` is a plain `pub struct`; adding a field breaks literals | resolved by the v0.2.0 API pass: the struct is `#[non_exhaustive]` and `new()` kept its arity, with `with_self_conditioning` added |
| 3 | `HookSpec` is the wrong vehicle for the input | ruled out: `HookSpec`'s own docs guarantee it "holds hook points and intervention descriptions, never activations" |
| 4 | trait method versus inherent method | inherent, per policy rule 1, so the other five backends do not inherit a method they cannot implement |
| 5 | `backward_reaches_every_parameter` hard-codes 29 parameters | unaffected: `tiny_config()` stays `self_conditioning: false`. A `true` fixture would have 30 |
| 6 | the Ask's gradient test is imprecise about candle's embedding backward | candle yields a gradient for the **whole** table with zeros in unused rows, so the shipped test asserts on row *values* and keeps `NONE` out of the batch |
| 7 | "bit-identical" via an all-`NONE` grid rests on `x + 0.0 == x` | the `None` path skips the add entirely rather than adding a zero, so it is exact by construction; the all-`NONE` path is also asserted, and is exact for every value a real checkpoint holds |
| 8 | "freeze the `NONE` row at zero" is not free | one table is one `Var`; a row cannot be frozen without a gradient mask. **The `NONE` row trains like any other.** This answers the Ask's closing question |
| 9 | dtype on the absent-table path | the zero table is built at `vb.dtype()` and `vb.device()`, so a `BF16` model does not silently get an `F32` table |

### One thing the Ask asked for that turned out to matter more than it looked

The id-range check. `candle`'s `index_select` raises `InvalidIndex` on CPU, but **the CUDA kernel
does not bounds-check**, so an out-of-range id would read past the table: a silent wrong answer on
GPU only, which is the failure mode this crate can least afford. The check is implemented as one
`max_all` plus a scalar read rather than a host copy of the grid, because canvas runs two forwards
per training step. Note this is stricter than the crate treats `input_ids`, which are unvalidated;
the asymmetry is deliberate.

### Verification

Five tests, each **mutation-checked**: removing the zero-init fails the bit-identical parity test;
inserting the shape-table entry first instead of last fails the seeded-stream test; accepting the
carry without adding it fails both the logits and gradient tests. Notably the stream test still
passes when only the zero-init is removed, which is the point of appending last: the two properties
are independent and both are load-bearing.

The `PyTorch` fp32 oracle (`tests/validate_othello_forward.rs`) reports the same numbers as before
the change, to every digit, on both devices: CPU 4.18e-5 / 2.59e-4, CUDA 3.37e-5 / 2.44e-4.

### Not done, and why

**Fixed-point forcing** (the registered follow-up) needs no further crate change, as the Ask says:
it is the same channel fed from a multi-step rollout. Nothing was added for it.

The Ask's decode-time question about whether the carry should hold `NONE` at masked positions or
the model's own reading everywhere is a canvas decode choice, as the Ask itself notes. The crate
takes any grid of valid ids and does not constrain it.
