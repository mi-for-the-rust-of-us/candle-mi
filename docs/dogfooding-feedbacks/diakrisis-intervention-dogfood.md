# Steering helpers: re-export `position_delta`, and gate `steering` with the backends

> **Status: IMPLEMENTED**, both findings, and left unmarked until now. Finding 1 shipped as the
> recommended one-line re-export (`src/steering/mod.rs`), and `position_delta` is additionally
> exported at the crate root. Finding 2 shipped as the recommended `any(backend)` gate on
> `pub mod steering`. This blockquote is late: the folder's own rule is that a report whose status
> is missing after its asks have shipped reads as open when it is not.
>
> **Revisited in v0.2.0 (2026-09-21).** Finding 2's symmetry had quietly broken. v0.2.0 added
> `stoicheia` to `hooks::apply_intervention`'s cfg, because both `stoicheia` backends began
> honouring interventions, but `pub mod steering` kept the older three-feature predicate. That
> reopened exactly this report's asymmetry in the opposite direction: under `--features stoicheia`
> the applier existed and the builders did not, so a user wanting a single-position
> `Intervention::Add` would have reinvented `position_delta` again. **There are three gates, not
> two** -- the module, `hooks::apply_intervention`, and the crate-root re-export -- and the first
> attempt at this fix widened only two of them, so the items resolved at
> `candle_mi::steering::position_delta` but not at `candle_mi::position_delta` under that one
> feature. All three now name the same four features, and `src/lib.rs` carries a comment pointing
> here so the next person to widen one widens the others.
>
> Finding 1 has a second postscript worth recording: v0.2.0 found the crate itself had made the
> mistake this report describes. `CrossLayerTranscoder::prepare_hook_injection` and
> `Sae::prepare_hook_injection` each built the single-position payload inline instead of calling
> the helper, and **both copies were missing the bounds check**, so `position == seq_len` produced
> a `[1, seq_len + 1, d]` payload rather than an error. Both now call `position_delta`.

**Date:** July 12, 2026
**Source:** askesis `diakrisis` — the Rust replication of the Othello-MDLM study (M3, causal intervention / `elenchos`)
**Affected area:** `src/steering/mod.rs` (re-exports) + `src/lib.rs` (module gating), relative to `src/hooks.rs`
**Severity:** Papercut — discoverability + feature-gating coherence; **not a blocker** (everything works today)

---

## The use case

`diakrisis` (askesis/rust/) is an independent Rust replication of the Othello-MDLM world-model
study, built on candle-mi as a deliberate dogfood. The engine and the MINE/YOURS linear probe are
done (the probe reproduces the PyTorch `rel_2w` = 0.79 @ L6 to ~1e-3). **M3 is the causal
intervention** (`elenchos`): add the probe's `MINE − YOURS` direction to a block's residual stream
at one position to *flip* a board cell's colour, then read whether the model's next-move prediction
follows the **counterfactual** legal set — the Nanda RQ3 test, reproducing the MDLM floor.

The intervention rides entirely on candle-mi's surface, and **that surface is exactly right**:

- `OthelloGpt` + `MIBackend::forward(&idx, &hooks)`,
- `HookSpec::capture(HookPoint::ResidPost(l))` + `HookCache::require` for the residual norm,
- `HookSpec::intervene(HookPoint::ResidPost(l), Intervention::Add(delta))` for the edit.

Once the pieces are found, a single-position residual edit is ~3 lines. No friction in the hook
model. Two papercuts surfaced around *finding and gating* the steering helpers.

## Finding 1 — `position_delta` is undiscoverable (re-export it)

The natural payload for "add a vector at one sequence position, zero elsewhere" is
`position_delta(direction, position, seq_len) -> [1, seq, hidden]`. It already exists and is
exactly what a positional residual edit needs — but it lives at
`candle_mi::steering::contrastive::position_delta` and is **not re-exported** at `steering::`, nor
referenced anywhere near the `Intervention` / hook docs.

Result: on the first pass I *reinvented it* (a `[seq, 1]` one-hot × `edit` broadcast → reshape),
duplicating a purpose-built helper that even error-checks `position < seq_len` and the 1-D shape.

**Recommendation** (cheap, high-value): re-export the general-purpose helpers at the module root —

```rust
// src/steering/mod.rs
pub use contrastive::{build_contrastive_direction, contrastive_intervention, position_delta};
```

so `candle_mi::steering::position_delta` resolves, and/or mention `position_delta` in the
`Intervention::Add` rustdoc ("to fire at a single position, build the payload with
[`steering::position_delta`]"). `position_delta` and `contrastive_intervention` are not specific to
the contrastive method — they're generic intervention-payload builders that happen to live in the
`contrastive` submodule.

## Finding 2 — the builder is ungated, the applier is backend-gated

The steering builders and the `Intervention` enum are compiled unconditionally:

```rust
// src/lib.rs
pub mod sparse;      // ungated
pub mod steering;    // ungated
```

but the code that *applies* an intervention into a forward pass is gated behind the backends:

```rust
// src/hooks.rs
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
pub(crate) fn apply_intervention(...) { ... }
```

So a backend-less build (`--no-default-features`) compiles a `steering` module whose interventions
can *never* be applied — dead surface. It's an asymmetry, not a bug (steering is dependency-light,
so this is about coherence, not compile time or binary size).

**Recommendation:** gate `steering` (and likely `sparse`) behind the **same** `any(backend)` cfg as
`apply_intervention`, so builder and applier appear/disappear together —

```rust
#[cfg(any(feature = "transformer", feature = "rwkv", feature = "diffusion"))]
pub mod steering;
```

A dedicated `steering` feature is the heavier alternative; the `any(backend)` gate is the
minimal-churn fix and matches the existing `apply_intervention` predicate exactly.

## What "good" looks like

- `candle_mi::steering::position_delta` resolves (re-exported), discoverable from the `Intervention`
  docs — so the next dogfooder doesn't reinvent it.
- `steering` compiles iff some backend that can apply its output is enabled.

Neither blocks `diakrisis`: M3 uses `steering::contrastive::position_delta` today. These are polish
items for the next candle-mi pass, recorded here from the intervention dogfood.
