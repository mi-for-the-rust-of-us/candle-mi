# Hook Point Reference and Intervention Walkthrough

> Hook points are named locations in a model's forward pass where activations
> can be **captured** (read) or **intervened on** (modified).  candle-mi
> follows the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
> naming convention, extended with RWKV-specific hook points.

---

## Table of Contents

- [Overview](#overview)
- [Hook Points](#hook-points)
  - [Naming Convention](#naming-convention)
  - [Ordering and Map Keys](#ordering-and-map-keys)
  - [Transformer Hook Points](#transformer-hook-points)
  - [RWKV Hook Points](#rwkv-hook-points)
- [HookSpec: Declaring Captures and Interventions](#hookspec-declaring-captures-and-interventions)
  - [Captures](#captures)
  - [Interventions](#interventions)
  - [Combining Captures and Interventions](#combining-captures-and-interventions)
  - [Merging Specs](#merging-specs)
- [HookCache: Retrieving Results](#hookcache-retrieving-results)
  - [Enumerating What Was Captured](#enumerating-what-was-captured)
- [Intervention Types](#intervention-types)
  - [Replace](#replace)
  - [PatchAt (Activation Patching)](#patchat-activation-patching)
  - [Add (Steering)](#add-steering)
  - [Knockout](#knockout)
  - [Scale](#scale)
  - [Zero](#zero)
- [RWKV State Interventions](#rwkv-state-interventions)
  - [State Knockout](#state-knockout)
  - [State Steering](#state-steering)
- [Zero-Overhead Guarantee](#zero-overhead-guarantee)
- [Worked Examples](#worked-examples)
  - [1. Capture Attention Patterns](#1-capture-attention-patterns)
  - [2. Logit Lens via Residual Stream](#2-logit-lens-via-residual-stream)
  - [3. Attention Knockout](#3-attention-knockout)
  - [4. Activation Patching](#4-activation-patching)
  - [5. RWKV State Knockout](#5-rwkv-state-knockout)

---

## Overview

The hook system has three components:

| Type | Role |
|------|------|
| [`HookPoint`](#hook-points) | Identifies **where** in the forward pass to act |
| [`HookSpec`](#hookspec-declaring-captures-and-interventions) | Declares **what** to capture and **which** interventions to apply |
| [`HookCache`](#hookcache-retrieving-results) | Stores the **results**: output logits + any captured tensors |

The flow is always:

```rust
// 1. Declare what you want
let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5));

// 2. Run the forward pass
let cache = model.forward(&input, &hooks)?;

// 3. Retrieve results
let logits = cache.output();                              // always present
let attn = cache.require(&HookPoint::AttnPattern(5))?;   // captured tensor
```

When `hooks` is empty, the forward pass has **zero overhead** — no extra
clones, no allocations.  See [Zero-Overhead Guarantee](#zero-overhead-guarantee).

---

## Hook Points

### Naming Convention

Every `HookPoint` variant maps to a TransformerLens-style string via
`Display` and `FromStr`:

```rust
use candle_mi::HookPoint;

let hook = HookPoint::AttnPattern(5);
assert_eq!(hook.to_string(), "blocks.5.attn.hook_pattern");

let parsed: HookPoint = "blocks.5.attn.hook_pattern".parse().unwrap();
assert_eq!(parsed, hook);
```

API methods accept `Into<HookPoint>`, so both styles work interchangeably:

```rust
hooks.capture(HookPoint::AttnPattern(5));      // enum — compile-time checked
hooks.capture("blocks.5.attn.hook_pattern");   // string — TransformerLens style
```

Unknown strings parse as `HookPoint::Custom(s)`, providing an escape hatch
for backend-specific hook points.

### Ordering and Map Keys

`HookPoint` derives `Ord`, so it keys a `BTreeMap` directly:

```rust
use std::collections::BTreeMap;

// Deterministic iteration order, no string keys, no wildcard match arm.
let by_hook: BTreeMap<&HookPoint, &Tensor> = cache.captures().collect();
```

This matters for callers whose results must not depend on `HashMap` iteration
order.  `HookPoint` is `#[non_exhaustive]`, so a downstream crate can never
match it exhaustively; before `Ord`, `to_string()` was the only total operation
left, and string keys are what such a caller is usually trying to avoid.

**The order is total, but its relation to hook semantics is unspecified.**
Derived ordering follows variant declaration order, so inserting a variant can
reorder existing ones in a patch release.  Rely on it for within-run determinism
and for map keys; never persist it, compare it across versions, or read layer
order into it.  To order by hook semantics, sort by the semantic fields
explicitly.

### Transformer Hook Points

The table below lists all hook points in the `GenericTransformer` forward
pass, in execution order.  All hook points support both **capture** and
**intervention**.

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, hidden]` | After token embedding (and optional embedding scale) |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, hidden]` | Residual stream before layer `i` |
| `AttnQ(i)` | `blocks.{i}.attn.hook_q` | `[batch, n_heads, seq, head_dim]` | Query vectors (before RoPE) |
| `AttnK(i)` | `blocks.{i}.attn.hook_k` | `[batch, n_kv_heads, seq, head_dim]` | Key vectors (before RoPE) |
| `AttnV(i)` | `blocks.{i}.attn.hook_v` | `[batch, n_kv_heads, seq, head_dim]` | Value vectors |
| `AttnScores(i)` | `blocks.{i}.attn.hook_scores` | `[batch, n_heads, seq_q, seq_k]` | Pre-softmax attention logits |
| `AttnPattern(i)` | `blocks.{i}.attn.hook_pattern` | `[batch, n_heads, seq_q, seq_k]` | Post-softmax attention probabilities |
| `AttnOut(i)` | `blocks.{i}.hook_attn_out` | `[batch, seq, hidden]` | Attention output (after `o_proj`) |
| `ResidMid(i)` | `blocks.{i}.hook_resid_mid` | `[batch, seq, hidden]` | Residual stream after attention, before MLP |
| `MlpPre(i)` | `blocks.{i}.mlp.hook_pre` | `[batch, seq, hidden]` | After mid-layer norm (MLP input) |
| `MlpPost(i)` | `blocks.{i}.mlp.hook_post` | `[batch, seq, hidden]` | MLP output (before optional post-feedforward norm) |
| `MlpOut(i)` | `blocks.{i}.hook_mlp_out` | `[batch, seq, hidden]` | After optional post-feedforward norm (Gemma 2 only; otherwise same as `MlpPost`) |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, hidden]` | Residual stream after full layer `i` |
| `FinalNorm` | `hook_final_norm` | `[batch, seq, hidden]` | After final layer norm (before logit projection) |

> **The logits are not a hook point.**  They are the forward pass's output, read
> with `cache.output()`, not with `cache.get(&hook)`.  `FinalNorm` is the last
> capturable point before the unembedding projection; to get logits from any
> captured activation, project it with `MIModel::project_to_vocab()`.

**Notes:**

- `n_kv_heads` may differ from `n_heads` in grouped-query attention (GQA).
  LLaMA 3, Qwen2, Gemma, Phi-3, and Mistral all use GQA.
- Q and K are captured **before** rotary position embedding (RoPE).  This
  matches TransformerLens behavior and is the right level for
  interventions: if you replace Q or K, RoPE is applied to the replacement.
- `AttnScores` is the standard point for knockout masks (`Intervention::Knockout`).
  The mask is added pre-softmax, so `-inf` entries become zero probability
  after softmax.
- **`Intervention::PatchAt` is accepted only where the Shape column reads
  `[batch, seq, hidden]`**, because only there is dim 1 the sequence.  At the
  five attention hook points dim 1 is a head, so a positional patch would
  overwrite a head; it is refused with `MIError::Intervention` rather than
  written to the wrong axis.  `HookPoint::accepts_positional_patch()` answers
  the question directly.  See [PatchAt](#patchat-activation-patching).
- `MlpOut` differs from `MlpPost` only for Gemma 2, which has a
  post-feedforward layernorm.  For all other architectures the tensors are
  identical.

### RWKV Hook Points

The `GenericRwkv` backend (RWKV-6 Finch and RWKV-7 Goose) exposes these
hook points.

**Since v0.2.0**, the residual-stream points (`Embed`, `ResidPre`, `ResidPost`,
`FinalNorm`) honour interventions like any other backend's.  Before v0.2.0 only
`Embed` did, and the rest ignored an intervention without error.

The three **diagnostic read-outs** (`RwkvState`, `RwkvDecay`,
`RwkvEffectiveAttn`) are capture-only, and an intervention aimed at one is now
**refused** with `MIError::Intervention` rather than silently dropped.  They are
computed after the recurrence that produced them, so an edit there could not
reach any computation.  To modify the recurrence itself, use the dedicated
[State Intervention](#rwkv-state-interventions) API, which the error message
also points you to.

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, hidden]` | After token embedding |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, hidden]` | Residual stream before layer `i` |
| `RwkvState(i)` | `blocks.{i}.rwkv.hook_state` | `[batch, n_heads, head_dim, head_dim]` | Accumulated WKV recurrent state after layer `i` |
| `RwkvDecay(i)` | `blocks.{i}.rwkv.hook_decay` | `[batch, seq, n_heads, head_dim]` | Per-timestep decay weights |
| `RwkvEffectiveAttn(i)` | `blocks.{i}.rwkv.hook_effective_attn` | `[batch, n_heads, seq_q, seq_src]` | Effective attention derived from WKV recurrence (ReLU + L1 normalized) |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, hidden]` | Residual stream after full layer `i` |
| `FinalNorm` | `hook_final_norm` | `[batch, seq, hidden]` | After final layer norm |

> As above, the logits are `cache.output()`, not a hook point.

**Notes:**

- `RwkvEffectiveAttn` is **computed on demand**: it is only calculated when
  you request capture of that hook point.  This avoids the overhead of
  deriving the attention matrix from the WKV recurrence when not needed.
- `RwkvState` contains the accumulated key-value outer product state from
  the WKV recurrence — the RWKV equivalent of a KV cache, but compressed
  into a fixed-size matrix per head.

### Stoicheia Hook Points

The stoicheia backends (`StoicheiaRnn`, `StoicheiaTransformer`) expose hook
points for ARC's [AlgZoo](https://github.com/alignment-research-center/alg-zoo)
tiny models.

**`StoicheiaRnn`** — uses `HookPoint::Custom(String)` for per-timestep
capture (the RNN cell has no natural correspondence to transformer layers):

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Custom` | `rnn.hook_pre_activation.{t}` | `[batch, H]` | Before ReLU at timestep `t` |
| `Custom` | `rnn.hook_hidden.{t}` | `[batch, H]` | Hidden state after timestep `t` |
| `Custom` | `rnn.hook_final_state` | `[batch, H]` | Final hidden state `h_n` |
| `Custom` | `rnn.hook_output` | `[batch, output_size]` | After output projection |

Per-timestep hooks (`{t}`) are only captured when explicitly requested —
the backend pre-scans which timesteps are in the `HookSpec` to avoid
per-step string allocation (zero overhead when no hooks are active).

**`StoicheiaTransformer`** — reuses standard `HookPoint` variants since
the attention-only architecture maps naturally:

| Hook Point | String | Shape | Description |
|------------|--------|-------|-------------|
| `Embed` | `hook_embed` | `[batch, seq, H]` | After token + positional embedding |
| `ResidPre(i)` | `blocks.{i}.hook_resid_pre` | `[batch, seq, H]` | Before attention layer `i` |
| `AttnScores(i)` | `blocks.{i}.attn.hook_scores` | `[batch, 1, seq, seq]` | Pre-softmax attention |
| `AttnPattern(i)` | `blocks.{i}.attn.hook_pattern` | `[batch, 1, seq, seq]` | Post-softmax attention |
| `AttnOut(i)` | `blocks.{i}.hook_attn_out` | `[batch, seq, H]` | Attention output |
| `ResidPost(i)` | `blocks.{i}.hook_resid_post` | `[batch, seq, H]` | After residual add |

No `MlpPre/Post/Out` (no MLP blocks), no `FinalNorm` (no normalization),
no `ResidMid` (no MLP means `ResidPost` = after attention). Attention is
full bidirectional (no causal mask), single-head.

**Since v0.2.0**, every point listed above honours interventions. Before v0.2.0
both stoicheia backends were capture-only and ignored an intervention without
error; `hooks::apply_intervention` was not even compiled under the `stoicheia`
feature. `AttnScores` and `AttnPattern` now fire *inside* the attention layer,
so an intervention there reaches the weighted sum rather than a value the layer
had already consumed.

`StoicheiaRnn`'s per-timestep `Custom` points honour interventions too. An edit
at `rnn.hook_hidden.{t}` propagates into later timesteps through the recurrence,
which is the point of steering an RNN.

---

## HookSpec: Declaring Captures and Interventions

`HookSpec` is the single configuration object passed to every `forward()`
call.  It declares both what to capture and where to intervene.

### Captures

Request a tensor snapshot at a hook point:

```rust
use candle_mi::{HookPoint, HookSpec};

let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5))
     .capture(HookPoint::ResidPost(5))
     .capture("blocks.5.hook_resid_pre");  // string form works too
```

`capture_all()` is the bulk form, taking anything iterable whose items are
`Into<HookPoint>`:

```rust
let mut hooks = HookSpec::new();
hooks.capture_all((0..model.num_layers()).map(HookPoint::ResidPost))
     .capture_all(["hook_embed", "hook_final_norm"]);
```

To *build* a capture-only spec rather than extend one, collect into it:

```rust
let hooks: HookSpec = (0..model.num_layers()).map(HookPoint::ResidPost).collect();
```

`HookSpec` is `Clone`, and that is a guarantee rather than an accident of the
derive: a spec holds hook points and intervention descriptions, never
activations, so a table of per-step specs handing out clones is cheap.

> There is deliberately no `Extend<HookPoint>` impl.  `HookSpec::extend()` is an
> inherent method that merges another **spec** (see [Merging Specs](#merging-specs)),
> and inherent methods win method resolution, so an `Extend` impl would make
> `spec.extend(some_iterator)` fail to compile.  Use `capture_all()` instead.

### Interventions

Register a modification at a hook point:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention};

let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

Multiple interventions can target the same hook point — they are applied
in registration order:

```rust
hooks.intervene(HookPoint::AttnScores(5), Intervention::Scale(0.5));
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

**An intervention is never silently dropped.** Every backend either applies it,
or returns `MIError::Intervention` explaining why that hook point cannot accept
one. This matters more here than in most crates: a registered intervention that
quietly does nothing makes a causal experiment report an effect of exactly zero,
which is indistinguishable from a genuine null. Three backends behaved that way
before v0.2.0 (see the RWKV and stoicheia sections above), and
`BACKENDS.md`'s conformance checklist now requires a test that would catch it.

### Combining Captures and Interventions

Captures and interventions are independent.  You can capture a hook point
without intervening, intervene without capturing, or both:

```rust
let mut hooks = HookSpec::new();
// Capture attention patterns AND knock out an edge
hooks.capture(HookPoint::AttnPattern(5));
hooks.intervene(HookPoint::AttnScores(5), Intervention::Knockout(mask));
```

### Merging Specs

Use `extend()` to merge two `HookSpec` instances.  This is useful when
combining intervention sources (e.g., CLT suppress + inject):

```rust
let mut combined = HookSpec::new();
combined.extend(&capture_spec);
combined.extend(&intervention_spec);
```

### Query Methods

| Method | Returns |
|--------|---------|
| `is_empty()` | `true` if no captures, interventions, or state specs |
| `num_captures()` | Number of requested captures |
| `num_interventions()` | Number of registered interventions |
| `captures()` | Iterator over the requested hook points (arbitrary order) |
| `is_captured(&hook)` | Whether a specific hook point will be captured |
| `has_intervention_at(&hook)` | Whether any intervention targets a hook point |

---

## HookCache: Retrieving Results

`HookCache` is returned by `forward()`.  It always contains the output
logits and any tensors captured via `HookSpec::capture()`.

```rust
let cache = model.forward(&input, &hooks)?;

// Output logits — always present. This is THE logit tap: there is no
// HookPoint for logits, because they are the forward pass's output.
let logits = cache.output();               // &Tensor, shape [batch, seq, vocab]
let logits = cache.into_output();          // Tensor (consumes cache)

// Captured tensors, by hook point
let attn = cache.get(&HookPoint::AttnPattern(5));          // Option<&Tensor>
let attn = cache.require(&HookPoint::AttnPattern(5))?;     // Result<&Tensor>

// Captured tensors, all of them
for (hook, tensor) in cache.captures() { /* ... */ }       // (&HookPoint, &Tensor)
let owned = cache.into_captures();                         // (HookPoint, Tensor)

// Metadata
let n = cache.num_captures();
```

`get()` returns `None` if the hook point was not captured; `require()`
returns `MIError::Hook` with a descriptive message.

### Enumerating What Was Captured

`captures()` walks everything the forward pass stored.  Without it, a harness
wanting *everything that was captured* has to keep its own copy of the request
and re-derive the keys, discovering absence one `get()` at a time; that is an
error path the caller does not otherwise need.

**Order is arbitrary** for both `captures()` methods, because the backing
stores are a `HashMap` and a `HashSet`.  Collect into a `BTreeMap` or
`BTreeSet` for a deterministic walk (`HookPoint` implements `Ord`, see
[Ordering and Map Keys](#ordering-and-map-keys)); a `BTreeMap`'s iteration
order cannot depend on the order things were inserted into it.

```rust
use std::collections::BTreeMap;

let by_hook: BTreeMap<&HookPoint, &Tensor> = cache.captures().collect();
for (hook, tensor) in &by_hook {
    println!("{hook}: {:?}", tensor.dims());
}
```

`into_captures()` yields owned tensors and is mutually exclusive with
`into_output()`, since both consume the cache.  To keep both, clone the output
first: candle's `Tensor` is reference-counted, so `cache.output().clone()`
costs a refcount, not a copy of the logits.

---

## Intervention Types

The `Intervention` enum provides six modification primitives.  Five of them
work at any transformer hook point that supports interventions; `PatchAt` is
restricted to the sequence-major hook points, for the reason given below.

### Replace

Replace the tensor entirely with a provided value:

```rust
Intervention::Replace(new_tensor)
```

**Use cases:** whole-tensor substitution, such as replacing an attention
pattern outright for a steering experiment.

**Shape requirement:** `new_tensor` must match the original tensor's shape.

To overwrite a single sequence position, reach for `PatchAt` instead of
capturing the activation, splicing a row in and handing the whole tensor back
through `Replace`.

### PatchAt (Activation Patching)

Overwrite one sequence position, leaving every other position untouched:

```rust
Intervention::PatchAt { position: 4, value: donor_row }
```

**Use cases:** activation patching, the standard causal instrument. Run the
recipient's forward pass, but at one hook point and one position substitute a
row taken from a donor pass, then read whether the prediction moves.

**Shape requirement:** the activation at the hook point is
`[batch, seq_len, hidden]`; `value` is `[hidden]`, `[1, 1, hidden]` or
`[batch, 1, hidden]`.  The first two broadcast across the batch, the way `Add`
does.  A row taken from a donor capture with `donor.narrow(1, position, 1)` is
already `[1, 1, hidden]` and can be passed straight in.  Dtype conversion is
automatic, as with `Add`.

**The recipient needs no captures.** `PatchAt` edits the residual stream in
flight, so a causal trace does not have to run the recipient once to store its
activations and then splice them by hand.  Only the donor pass is captured.

**Accepted hook points.** `PatchAt` is accepted exactly where the activation is
`[batch, seq_len, hidden]`, so that dim 1 is the sequence:

| Accepted | Rejected |
|---|---|
| `Embed`, `ResidPre`, `AttnOut`, `ResidMid`, `MlpPre`, `MlpPost`, `MlpOut`, `ResidPost`, `FinalNorm` | `AttnQ`, `AttnK`, `AttnV`, `AttnScores`, `AttnPattern`, `RwkvState`, `RwkvDecay`, `RwkvEffectiveAttn`, `Custom` |

At the five attention hook points dim 1 is a **head**, not a position:
`AttnQ` is `[batch, n_heads, seq_len, head_dim]`, `AttnK` and `AttnV` are
`[batch, n_kv_heads, seq_len, head_dim]` (they are captured before the
grouped-query broadcast), and `AttnScores` and `AttnPattern` are
`[batch, n_heads, seq_len, seq_len]`, which have two sequence axes and so no
unambiguous single position.  A positional write there would overwrite a head
and return a plausible figure rather than an error, so it is refused with
`MIError::Intervention`.  Ask a hook point directly with
`HookPoint::accepts_positional_patch()`.

Patching a key or value is a coherent thing to want, but it is a different
operation rather than a wider version of this one: `AttnK` and `AttnV` are
pre-broadcast, so a write there lands on a KV head and fans out to
`n_heads / n_kv_heads` query heads downstream.

### Add (Steering)

Add a vector to the activation via broadcasting:

```rust
Intervention::Add(direction_vector)
```

**Use cases:** residual stream steering (add a direction vector scaled by
a coefficient), feature injection.

**Shape requirement:** `direction_vector` must be broadcastable to the
tensor's shape.  Dtype conversion is automatic — if you inject an F32
vector into a BF16 forward pass, the vector is cast to BF16 before
addition.

### Knockout

Add a pre-softmax mask to attention scores:

```rust
Intervention::Knockout(mask)
```

The mask contains `0.0` for positions to keep and `-inf` for positions to
knock out.  After softmax, `-inf` entries become zero probability.

**Use cases:** ablating specific attention edges (e.g., "what happens if
head 3 cannot attend from position 7 to position 0?").

**Shape requirement:** `mask` must be broadcastable to
`[batch, n_heads, seq_q, seq_k]`.

Use `create_knockout_mask()` to build masks from a `KnockoutSpec`:

```rust
use candle_mi::{KnockoutSpec, HookPoint, HookSpec, Intervention, create_knockout_mask};

let spec = KnockoutSpec::new()
    .layer(target_layer)
    .edge(query_pos, key_pos);    // knock out one edge

let mask = create_knockout_mask(
    &spec, n_heads, seq_len, device, candle_core::DType::F32,
)?;

let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(target_layer), Intervention::Knockout(mask));
```

### Scale

Multiply all attention weights by a constant factor:

```rust
Intervention::Scale(2.0)
```

**Use cases:** amplifying or dampening attention at a layer, probing
attention sensitivity.

### Zero

Zero the tensor entirely:

```rust
Intervention::Zero
```

**Use cases:** complete ablation of a component (e.g., zero the MLP output
at a layer to measure its contribution).

---

## RWKV State Interventions

RWKV models have a recurrent state that accumulates information across
token positions.  Standard `Intervention` variants operate on tensors at
hook points; state interventions operate on the WKV recurrence loop itself.

### State Knockout

Skip the key-value write at specified token positions, making those tokens
invisible to all future positions:

```rust
use candle_mi::StateKnockoutSpec;

let spec = StateKnockoutSpec::new()
    .position(0)              // knock out position 0
    .position(3)              // and position 3
    .layer(12);               // only at layer 12

let mut hooks = HookSpec::new();
hooks.set_state_knockout(spec);
```

**Layer targeting:**

| Method | Effect |
|--------|--------|
| (default) | All layers |
| `.layer(i)` | Single layer |
| `.layers(&[2, 5, 8])` | Specific layers |
| `.layer_range(5, 10)` | Inclusive range |

### State Steering

Scale the key-value write at specified positions by a factor:

```rust
use candle_mi::StateSteeringSpec;

let spec = StateSteeringSpec::new(2.0)   // amplify 2x
    .position(0)
    .layer_range(0, 11);

let mut hooks = HookSpec::new();
hooks.set_state_steering(spec);
```

**Scale semantics:**

| Scale | Effect |
|-------|--------|
| `0.0` | Knockout (equivalent to `StateKnockoutSpec`) |
| `1.0` | No-op (normal forward pass) |
| `< 1.0` | Dampen the token's state contribution |
| `> 1.0` | Amplify the token's state contribution |

**Priority:** if both `state_knockout` and `state_steering` are set,
knockout takes priority at positions where both apply.

---

## Zero-Overhead Guarantee

When `HookSpec` is empty (no captures, no interventions, no state specs),
the forward pass is identical to a plain forward pass:

- **No tensor clones** — capture checks are `HashSet::contains()` returning
  `false`, which skips the `.clone()`.
- **No extra allocations** — intervention lists are empty; iteration is
  a no-op.
- **Minimal branch cost** — each hook point is a single `if` check.

This guarantee is verified by benchmarks (see `design/hook-overhead.md`):
+11.5% GPU overhead with full capture of all 194 hook points, within noise
on CPU.

---

## Worked Examples

### 1. Capture Attention Patterns

Capture the post-softmax attention pattern at layer 5 and inspect its shape:

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

let model = MIModel::from_pretrained("meta-llama/Llama-3.2-1B")?;
let tokenizer = model.tokenizer().unwrap();

let tokens = tokenizer.encode("The capital of France is")?;
let input = candle_core::Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;

let mut hooks = HookSpec::new();
hooks.capture(HookPoint::AttnPattern(5));

let cache = model.forward(&input, &hooks)?;
let attn = cache.require(&HookPoint::AttnPattern(5))?;
// Shape: [1, n_heads, seq_len, seq_len]
println!("Attention shape: {:?}", attn.dims());
```

### 2. Logit Lens via Residual Stream

Capture residual streams at every layer and project each to vocabulary
logits:

```rust
use candle_mi::{HookPoint, HookSpec, MIModel};

let mut hooks = HookSpec::new();
for layer in 0..model.num_layers() {
    hooks.capture(HookPoint::ResidPost(layer));
}

let cache = model.forward(&input, &hooks)?;

for layer in 0..model.num_layers() {
    let resid = cache.require(&HookPoint::ResidPost(layer))?;
    // Extract last position: [1, seq, hidden] → [1, hidden]
    let last = resid.get(0)?.get(seq_len - 1)?.unsqueeze(0)?;
    let logits = model.project_to_vocab(&last)?;
    let token_id = candle_mi::sample_token(&logits.flatten_all()?, 0.0)?;
    let token_text = tokenizer.decode(&[token_id])?;
    println!("Layer {layer:>2}: {token_text}");
}
```

### 3. Attention Knockout

Knock out the attention edge from the last token to position 0 at a middle
layer:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention, KnockoutSpec, create_knockout_mask};

let target_layer = model.num_layers() / 2;
let spec = KnockoutSpec::new()
    .layer(target_layer)
    .edge(seq_len - 1, 0);   // last token cannot attend to position 0

let mask = create_knockout_mask(
    &spec, model.num_heads(), seq_len, model.device(), candle_core::DType::F32,
)?;

// Baseline (no intervention)
let baseline = model.forward(&input, &HookSpec::new())?;

// Ablated (with knockout)
let mut hooks = HookSpec::new();
hooks.intervene(HookPoint::AttnScores(target_layer), Intervention::Knockout(mask));
let ablated = model.forward(&input, &hooks)?;

// Compare via KL divergence
let result = candle_mi::AblationResult::new(
    baseline.output().get(0)?.get(seq_len - 1)?,
    ablated.output().get(0)?.get(seq_len - 1)?,
    spec,
);
println!("KL divergence: {:.6}", result.kl_divergence()?);
```

### 4. Activation Patching

Patch clean activations into a corrupted forward pass at a specific layer:

```rust
use candle_mi::{HookPoint, HookSpec, Intervention};

// 1. Run clean forward, capturing residuals
let mut capture_hooks = HookSpec::new();
capture_hooks.capture(HookPoint::ResidPost(target_layer));
let clean_cache = model.forward(&clean_input, &capture_hooks)?;
let clean_resid = clean_cache.require(&HookPoint::ResidPost(target_layer))?.clone();

// 2. Run corrupted forward, replacing the residual with clean
let mut patch_hooks = HookSpec::new();
patch_hooks.intervene(
    HookPoint::ResidPost(target_layer),
    Intervention::Replace(clean_resid),
);
let patched_cache = model.forward(&corrupted_input, &patch_hooks)?;

// 3. Compare patched output to clean and corrupted baselines
```

### 5. RWKV State Knockout

Make a token invisible in the RWKV recurrent state at a specific layer:

```rust
use candle_mi::{HookSpec, MIModel, StateAblationResult, StateKnockoutSpec};

let model = MIModel::from_pretrained("RWKV/v6-Finch-1B6-HF")?;

// Knock out position 0 at the middle layer
let spec = StateKnockoutSpec::new()
    .position(0)
    .layer(model.num_layers() / 2);

let baseline = model.forward(&input, &HookSpec::new())?;

let mut hooks = HookSpec::new();
hooks.set_state_knockout(spec.clone());
let ablated = model.forward(&input, &hooks)?;

let result = StateAblationResult::new(
    baseline.output().get(0)?.get(seq_len - 1)?,
    ablated.output().get(0)?.get(seq_len - 1)?,
    spec,
);
println!("KL divergence: {:.6}", result.kl_divergence()?);
```
