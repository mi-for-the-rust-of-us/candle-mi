# Design: `Intervention::AddAtPositions`

**Status:** Proposed **|** **Date:** March 20, 2026 **|** **Relates to:** Roadmap §8 item 10 (`apply_intervention` consolidation), §8 item 2 (hook system) **|** **Origin:** [PR #1](https://github.com/mi-for-the-rust-of-us/candle-mi/pull/1) by hankxu19 (closed) — feature idea retained, implementation reworked

## Question

Should candle-mi support position-specific vector injection as a first-class `Intervention` variant?

## Context

The existing `Intervention` enum provides five primitives — `Replace`, `Add`, `Knockout`, `Scale`, `Zero`. (As of March 2026. `PatchAt` was added in September 2026; see the resolved `ReplaceAtPositions` section below.) All operate uniformly across the sequence dimension: `Add` broadcasts the same vector to every position, `Replace` swaps the entire tensor.

However, MI experiments often need **heterogeneous per-position injection**:

- Inject CLT feature "Paris" at position 3 and feature "Berlin" at position 7 in a single forward pass, then study how the model resolves the conflict through attention patterns.
- Multi-site interaction studies: inject different steering vectors at different positions simultaneously to test whether features are independent or interact. Currently requires exponentially many single-position forward passes.
- [K-BERT](https://arxiv.org/abs/1909.07606) (Liu et al., 2019) pioneered per-position knowledge injection at the embedding layer for encoder-only models. The injection *mechanism* — adding entity embeddings at specific token positions — transfers directly to decoder-only MI experiments.
- The anacrousis recurrent feedback loop (`inject_feedback_at_position` in `src/transformer/mod.rs`) already implements exactly this pattern as a special-case internal helper — building a sparse `[1, seq_len, d_model]` delta tensor and adding it via `broadcast_add`.
- **Update, v0.2.0:** the *single*-position case is now one shared helper, `util::inject::position_delta`, reached publicly as `steering::position_delta`. Contrastive steering, `CLT` injection and `SAE` injection all route through it; two of those three had hand-rolled it and neither bounds-checked the position. That narrows this proposal to what it is actually about: *heterogeneous* per-position injection in a single pass, which the shared helper does not cover.

The pattern is general enough to be a first-class intervention.

## Recommendation

### New variant

Add one variant to `Intervention`:

```rust
/// Add different vectors at specific token positions (sparse injection).
///
/// Each entry is `(position, vector)` where `vector` has shape `[d_model]`.
/// Positions not listed are unmodified. All vectors are scaled by `scale`
/// before addition.
///
/// # Shapes
/// - `positions[].1`: `[d_model]`
/// - target tensor: `[batch, seq_len, d_model]`
AddAtPositions {
    /// `(token_position, injection_vector)` pairs.
    positions: Vec<(usize, Tensor)>,
    /// Multiplicative scale applied to all vectors before addition.
    /// Use `1.0` for unscaled injection.
    scale: f64,
},
```

`scale` uses `f64` to match `Intervention::Scale(f64)`.

`Intervention` is `#[non_exhaustive]`, so adding a variant is non-breaking (downstream `match` arms must already have a wildcard).

### Shared primitive

Extract the sparse-delta construction into a `pub(crate)` helper in `hooks.rs`, called by both `apply_intervention` (for the new variant) and the existing `inject_feedback_at_position` (which becomes a thin wrapper):

```rust
/// Build a sparse delta tensor with vectors placed at specific positions.
///
/// # Shapes
/// - returns: `[1, seq_len, d_model]`
pub(crate) fn build_sparse_delta(
    seq_len: usize,
    d_model: usize,
    positions: &[(usize, &Tensor)],
    scale: f64,
    device: &Device,
    target_dtype: DType,
) -> Result<Tensor> {
    let mut delta_data = vec![0.0_f32; seq_len * d_model];
    // CAST: f64 → f32, scale factor for host-side multiplication
    let scale_f32 = scale as f32;
    for &(pos, vector) in positions {
        // PROMOTE: injection vector may be BF16; F32 for host-side accumulation
        let vec_f32: Vec<f32> = vector
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        if vec_f32.len() != d_model {
            return Err(MIError::Intervention(format!(
                "injection vector has {} elements, expected d_model={d_model}",
                vec_f32.len()
            )));
        }
        let start = pos * d_model;
        let dest = delta_data.get_mut(start..start + d_model).ok_or_else(|| {
            MIError::Intervention(format!(
                "injection position {pos} out of bounds (seq_len={seq_len})"
            ))
        })?;
        for (dst, &src) in dest.iter_mut().zip(vec_f32.iter()) {
            *dst = src * scale_f32;
        }
    }
    let delta = Tensor::from_vec(delta_data, (1, seq_len, d_model), device)?
        .to_dtype(target_dtype)?;
    Ok(delta)
}
```

Key properties:
- **One allocation, one `broadcast_add`** regardless of how many positions — no per-position kernel launches.
- **Accumulates on the host in F32** then casts to target dtype — matches the existing `inject_feedback_at_position` pattern.
- **Bounds-checked** via `.get_mut()` with `MIError::Intervention` on out-of-bounds.

### Refactor feedback path

`inject_feedback_at_position` in `src/transformer/mod.rs` becomes:

```rust
fn inject_feedback_at_position(
    hidden: &Tensor,
    vector: &Tensor,
    position: usize,
) -> Result<Tensor> {
    let delta = build_sparse_delta(
        hidden.dim(1)?,
        hidden.dim(2)?,
        &[(position, vector)],
        1.0,
        hidden.device(),
        hidden.dtype(),
    )?;
    Ok(hidden.broadcast_add(&delta)?)
}
```

This eliminates the code duplication flagged in the PR #1 review (point 1) and satisfies design decision 10: all intervention logic flows through `hooks.rs`.

### New arm in `apply_intervention`

```rust
Intervention::AddAtPositions { positions, scale } => {
    let seq_len = tensor.dim(1)?;
    let d_model = tensor.dim(2)?;
    let pos_refs: Vec<(usize, &Tensor)> =
        positions.iter().map(|(p, t)| (*p, t)).collect();
    let delta = build_sparse_delta(
        seq_len, d_model, &pos_refs, *scale,
        tensor.device(), tensor.dtype(),
    )?;
    Ok(tensor.broadcast_add(&delta)?)
}
```

## What this does NOT include

### `embed_tokens` method

Initially considered: adding `fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor>` to `MIBackend` for raw embedding lookup without a forward pass. **Deferred** — a researcher can already obtain token embeddings by capturing at `HookPoint::Embed`. `embed_tokens` saves one forward pass but adds a trait method (semver-breaking for external implementors). Revisit when probing (Phase 10+) or embedding analysis creates concrete demand.

### Encoder-only (BERT) support

K-BERT is an encoder-only architecture with bidirectional attention, visible matrix masking, and position encoding adjustments. None of that is in scope. candle-mi supports decoder-only causal LMs. The injection *mechanism* is borrowed from K-BERT; the architecture is not. See [Roadmap §3.4](../ROADMAP.md) for what the generic transformer does not cover.

### `ReplaceAtPositions` — RESOLVED, shipped as `Intervention::PatchAt`

Replacing (not adding) embeddings at specific positions would be the other half of K-BERT-style injection. Not needed for current MI use cases (steering is additive). Can be added later as another `Intervention` variant if demand arises.

> **Demand arose.** Reported from askesis `canvas` ([`positional-replace-has-no-intervention.md`](../docs/dogfooding-feedbacks/positional-replace-has-no-intervention.md)) for activation patching, which is the standard causal instrument and wants exactly this. Shipped as `Intervention::PatchAt { position, value }` — see [`patch-at-position.md`](patch-at-position.md).
>
> Three deliberate divergences from the sketch above, worth carrying back if `AddAtPositions` is ever built:
>
> 1. **Singular, not plural.** One position, not a `Vec<(usize, Tensor)>`. The plural form here is motivated by heterogeneous multi-site injection; patching has no such use, and every extra degree of freedom in an intervention API is a degree of freedom in which a probe can be silently wrong.
> 2. **A masked `where_cond`, not `build_sparse_delta`.** The host-side helper proposed above routes through `to_vec1()`, which forces a device synchronisation and detaches from the autograd graph. The on-device alternatives that look obvious, `Tensor::slice_scatter` and a three-way `Tensor::cat`, are both **silently wrong on CUDA when the donor row is a view** (candle's `copy_strided_src` sizes its CUDA copy from the storage rather than the view, so it overruns into the following positions); see [`patch-at-position.md`](patch-at-position.md). `AddAtPositions` would face the same trap the moment its injection vector came from a capture rather than from `Tensor::new`.
> 3. **The hook point is now an argument to `apply_intervention`.** "Target tensor: `[batch, seq_len, d_model]`" is assumed throughout this document, but nothing enforced it: dim 1 is a head at the five attention hook points, so a positional write there would silently overwrite a head. `HookPoint::accepts_positional_patch` makes that a rejection. **`AddAtPositions` would need the same guard** — its `tensor.dim(1)?` / `tensor.dim(2)?` in the sketched match arm reads `n_heads` and `seq_len` at `AttnQ`, not `seq_len` and `d_model`.

## K-BERT experimental paradigm on decoder-only models

Although candle-mi does not support BERT, `AddAtPositions` at `HookPoint::Embed` enables knowledge-injection experiments on any supported model:

```rust
// 1. Obtain a "knowledge" embedding (e.g., from the model's own vocabulary)
let mut capture = HookSpec::new();
capture.capture(HookPoint::Embed);
let emb_cache = model.forward(&knowledge_tokens, &capture)?;
let knowledge_emb = emb_cache.require(&HookPoint::Embed)?
    .get(0)?.get(0)?;  // [d_model]

// 2. Inject it at a specific position during the real forward pass
let mut hooks = HookSpec::new();
hooks.intervene(
    HookPoint::Embed,
    Intervention::AddAtPositions {
        positions: vec![(target_position, knowledge_emb)],
        scale: 1.0,
    },
);

// 3. Capture downstream effects
hooks.capture(HookPoint::AttnPattern(5));
hooks.capture(HookPoint::ResidPost(10));
let cache = model.forward(&input_tokens, &hooks)?;

// 4. Study how the injection propagates through the network
```

This injects external information at specific positions and lets the model process it naturally — no visible matrix, no position encoding adjustment. For MI research, this is often what you want: an unguarded injection whose propagation you can trace through the layers.

## Tests

1. **Single position** — inject at one position, verify the delta is applied correctly.
2. **Multi-position** — inject at 2+ positions in one intervention, verify all are applied.
3. **Scale factor** — verify `scale != 1.0` scales the injection vector.
4. **Out-of-bounds** — verify position ≥ `seq_len` returns `MIError::Intervention`.
5. **Wrong vector size** — inject a vector with length ≠ `d_model`, verify `MIError::Intervention` is returned.
6. **Dtype coercion** — inject an F32 vector into a BF16 forward pass, verify no panic.
7. **Feedback path** — verify `inject_feedback_at_position` still produces identical results after refactoring to use `build_sparse_delta`.

## Implementation order

1. `build_sparse_delta` in `hooks.rs`
2. `AddAtPositions` variant + arm in `apply_intervention`
3. Refactor `inject_feedback_at_position` to call `build_sparse_delta`
4. Tests
5. Update `HOOKS.md` with `AddAtPositions` documentation and K-BERT paradigm note
6. Update `CHANGELOG.md`

## See also

- [HOOKS.md](../HOOKS.md) — user-facing intervention reference
- [intervention-api.md](intervention-api.md) — original intervention API design
- PR #1 review — [7-point review](https://github.com/mi-for-the-rust-of-us/candle-mi/pull/1) that informed this design
- Liu, W. et al. ["K-BERT: Enabling Language Representation with Knowledge Graph."](https://arxiv.org/abs/1909.07606) AAAI 2020.
