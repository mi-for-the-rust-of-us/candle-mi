# Seeded init promises more than `StdRng` backs

> **Status: IMPLEMENTED**, and left unmarked until now. The report recommended option B, freeze
> the algorithm by having `init` pass `ChaCha8Rng` from `rand_chacha`. What shipped is stronger
> than B and stops short of C: `src/util/rng.rs` is the crate's single construction point, and it
> derives the 32-byte key in-crate with `SplitMix64` before calling `ChaCha8Rng::from_seed`,
> deliberately avoiding `SeedableRng::seed_from_u64` because that convenience carries no stability
> guarantee of its own. So the frozen span is wider than the report asked for, without owning a
> PRNG as option C would have. `util::randn` then takes `ChaCha8Rng` concretely rather than
> `impl Rng`, which makes the frozen generator the only one that can reach the sampler. This
> blockquote is late: the folder's own rule is that a report whose status is missing after its
> asks have shipped reads as open when it is not.
>
> **Completed in v0.2.0 (2026-09-21).** One step of that span had no test behind it, which the
> module doc's claim ("no dependency bump can move it") did not distinguish from the rest. The
> `SplitMix64` derivation was pinned to its published vector; the `ChaCha8` step was not, and the
> only other test asserted `draw(0) == draw(0)` and `draw(0) != draw(1)`, which hold for any
> deterministic generator. `seeded_stream_matches_the_chacha8_specification` now pins the first
> four draws of `seeded(0)` to values derived independently from RFC 8439's block function at 8
> rounds, so it detects a wrong stream and not merely a changed one. Mutation-checked: keeping the
> key and advancing the stream leaves both older tests green and fails only the new one.
>
> The report's timing argument was also borne out. It warned that switching would change the
> weights for a given seed exactly once, and that the window was half closed because
> `canvas_ema_e10` existed on disk. That cost was paid at the switch; nothing since has moved the
> stream, and the new test is what keeps it that way.

**Date:** July 27, 2026
**Source:** askesis `canvas` leg — design review of v0.1.20, after publication
**Affected area:** `src/util/randn.rs`, `OthelloGpt::init` (`src/diffusion/othello.rs`)
**Severity:** Durability — **not a defect**, and nothing is blocked. A documented promise slightly
wider than what backs it, with a closing window on the cheap fix.

---

## The promise

`src/util/randn.rs`'s module doc:

> a model is reproducible from `(config, seed)` alone, independent of the device RNG

and the v0.1.20 CHANGELOG says the same of `OthelloGpt::init`. Both are true **at a fixed `rand`
version**, and not guaranteed across one.

## What backs it

`randn_f32` draws from `rand::rngs::StdRng`. `rand` 0.8.7 documents that type as (verbatim,
`src/rngs/std.rs:25`):

> The algorithm is deterministic but should not be considered reproducible due to dependence on
> configuration and possible replacement in future library versions. For a secure reproducible
> generator, we recommend use of the `rand_chacha` crate directly.

So upstream both declines the guarantee and names the remedy. Historically the type has changed
implementation before (`HC-128` → `ChaCha12`), so this is not hypothetical.

## What is, and is not, at risk

**Not at risk — the canvas cross-backend parity protocol.** It exports the initial weights once via
`VarMap::save` and has *both* backends load that one file; it never regenerates from a seed on two
sides. Immune to any generator change, deliberate or accidental. (An earlier draft of this concern
overstated it here; corrected.)

**Not at risk — anything that would break loudly.** A test pinning numbers derived from init would
fail on a bump, which is the good outcome.

**At risk — reproducing a past run.** "Train v1 with seed 0" a year from now, after a `rand` bump,
could yield different weights under an unchanged seed and an unchanged config. That is exactly the
property `init` was added to provide.

## Options

- **A — document only.** Qualify the claim in `randn.rs` and on `init`: reproducible for a fixed
  `rand`; for durable reproducibility, save the `VarMap`. ~10 lines, no behaviour change. Leaves the
  promise narrowed rather than met.
- **B — freeze the algorithm.** Make `randn_f32` generic (`&mut impl Rng`) instead of
  `StdRng`-typed, and have `init` pass `ChaCha8Rng::seed_from_u64` from `rand_chacha`, which commits
  to precisely the stability `StdRng` disclaims.
- **C — own it, as `pweak::rng` does.** A frozen `SplitMix64` in-crate. Maximum control, but it
  means owning and testing a PRNG when a crate already in the tree guarantees the property. `pweak`
  wrote its own because it wanted zero dependencies and a Zig-port parity target; candle-mi has
  neither constraint.

## Recommendation: B, and the argument is timing

Two facts make B cheap. `rand_chacha` 0.3.1 is **already in the dependency tree** — it *is*
`StdRng`'s implementation under `rand` 0.8 — so declaring it directly adds no compile unit. And
`randn_f32`'s public signature currently names `StdRng`, so generic-ising removes a concrete RNG
type from the API surface, which is the better shape regardless.

The switch changes the weights for a given seed exactly **once**.

**Updated 2026-07-28 — the window this paragraph called open has half closed.** It previously read
"free today: no `canvas` model has been trained". One now has: stage 1 (e0→e10) ran overnight and
is archived as `canvas_ema_e10`. The weights are on disk, so nothing is *lost* by switching — but
`(config, seed 0)` would stop reproducing that model's initialization, which downgrades e10 from
regenerable to merely preserved. The cost is now a line in the run's provenance rather than
nothing, and it grows with each further stage.

**Suggested slot: still v0.1.21, and sooner rather than later.** If the appetite is smaller, A is a
legitimate stopping point — the honest version of the promise costs ten lines.

## The other two v0.1.21 candidates — same root cause

This report is one of three findings with a single origin, worth stating once: **candle-nn has no
notion of training state that outlives a process.** `VarMap::save`/`load` serializes model weights
and nothing else. The item above is about reproducing an **initialization**; the two below are
about reproducing a **continuation**. All three only bite once something is trained from scratch
for long enough that a process boundary matters — which is exactly the regime candle is least used
in, and why none of them has an upstream fix.

### 1. Promote a checkpointable `AdamW` (a default-off `training` feature)

Stock `candle_nn::AdamW` cannot be checkpointed: the per-parameter moments live in a private
`VarAdamW` and `step_t` is a private field, with only `new_lr`/`params`/`set_params` exposed. The
whole 201-line `optim.rs` is byte-identical between 0.9.2 and 0.11.0. It is not an encapsulation
decision — `SGD` in the same file exposes `into_inner()`; `AdamW` just never got an accessor,
because there was no consumer.

canvas needed one (a 40-epoch run is ~21 h and is trained in ~5 h stages; resuming without the
moments resets Adam's bias correction at every boundary, landing an artefact exactly where V11
reads emergence off consecutive checkpoints). It now has one in `canvas/src/optim.rs`:
**candle's update rule transcribed verbatim**, with the state moved from private `Var`s to a named
serializable map, held to candle's own trajectory by `canvas/tests/optim_parity.rs` at `< 1e-6`
over 1/2/17/120 steps, plus a V10 power control showing that silently losing the moments *would*
be visible.

It was deliberately written free of canvas types so promotion is a file move rather than a
rewrite. **Timing (Éric, 2026-07-28): wait until e30.** Three real 5 h resume cycles should shape
the API before it is frozen in a published crate — graduate on demonstrated use, not on
anticipation.

### 2. Upstream the accessors to candle-nn

The narrower fix belongs upstream: give `AdamW` a `step_t()`/`set_step_t()` and a `moments()`
iterator over `(param, first, second)`. `Var` has interior mutability, so reading for a save and
`set`ting for a restore need no `&mut`. ~15 lines, no behaviour change, and `SGD::into_inner()` is
the precedent sitting in the same file.

When it lands, candle-mi drops its vendored copy and calls stock. Until then the copy is the
prototype — which is the right order anyway, since the upstream API is easier to argue for once a
real resume cycle has exercised it.

## Unrelated, noted in passing

`Cargo.lock` resolves both `rand` 0.8.7 and 0.9.5 (and `rand_chacha` 0.3.1 and 0.9.0). Harmless, but
it is the kind of duplicate the v0.1.18 `windows-sys` work tracked deliberately.
