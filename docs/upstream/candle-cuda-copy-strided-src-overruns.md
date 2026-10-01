# CUDA `copy_strided_src` sizes its copy from the storage, not the view

**Title for the issue:** CUDA `copy_strided_src` copies past the end of an offset source view, silently corrupting `slice_scatter` (CPU disagrees)

**Status: MERGED, NOT YET RELEASED.** Filed 2026-09-02 as
[candle#3940](https://github.com/huggingface/candle/issues/3940), with the fix proposed the same day as
[candle#3944](https://github.com/huggingface/candle/pull/3944), **merged 2026-09-24 15:06 UTC** by `EricLBuehler` as
[`73e25aa`](https://github.com/huggingface/candle/commit/73e25aace6f001cd06cb792629f4335a761caa6c), which auto-closed the issue two seconds later. Twenty-two days from filing
to merge. It went in as authored (+36/-3, the same two files, no review changes),
approved with "Thanks @PCfVW". CI was never the obstacle: on 2026-09-04 `ivarflakstad`
merged `main` into the branch (`8898eb2`), and a maintainer's push needs no approval,
so that run passed and the thirty-day expiry stopped mattering.

As of 2026-10-01 the newest `candle-core` on crates.io is still 0.11.0 (2026-06-26),
which predates the fix, and candle-mi pins `0.11`. The workaround below therefore
stays until candle-mi moves to a release that contains `73e25aa`.

Searched first: no existing issue or PR covered this path (see "Prior art" below).
Confirmed present on `main` at `d5fee525` as well as on the pinned 0.11.0.

The fix sizes the copy from `src_l.shape().elem_count()`, keeping the two storage
bounds as a defensive clamp. The argument that carried it is that **CUDA was the
outlier**: the CPU path already takes the length from
`StridedBlocks::SingleBlock { len }` and the Metal contiguous path already uses
`src_l.shape().elem_count()`, so the patch makes CUDA agree with the two backends
that agreed with each other. Verified on an RTX 5060 Ti: the added `slice_scatter`
assertions fail on CUDA before the change (row 3 comes back as the donor's next
row) and pass after, with the full `candle-core` suite green on CPU and CUDA.

## Workaround, up front

Until candle-mi depends on a release containing `73e25aa`, do not pass a source
that is a **view over a larger storage** to `Tensor::slice_scatter` on CUDA. Either materialise it into a tensor that owns
its storage exactly (`.contiguous()` is **not** enough, since a `narrow` view is
already contiguous and keeps its parent's storage), or avoid the copy entirely.

candle-mi took the second route for `Intervention::PatchAt`: a masked
`where_cond` over three same-shape contiguous operands, which touches each
element once through its own layout and so has no dependence on storage
provenance. See `design/patch-at-position.md`.

## Summary

`copy_strided_src(dst, dst_offset, src_l)` is expected to copy the elements
described by `src_l`, that is `src_l.shape().elem_count()` of them. The CPU
backend does. The CUDA backend computes the length from **whole-storage sizes**
instead, so when the source is an offset view whose storage continues past the
view, and the destination has room, it copies too much:

```rust
// candle-core/src/cpu_backend/mod.rs:902, copies exactly the view's length
StridedBlocks::SingleBlock { start_offset, len } =>
    dst[dst_offset..dst_offset + len]
        .copy_from_slice(&src[start_offset..start_offset + len])

// candle-core/src/cuda_backend/mod.rs:1217, derives the length from storage
fn slice_src_and_dst<'a, T>(...) -> (CudaView<'a, T>, CudaViewMut<'a, T>) {
    let src_offset = src_l.start_offset();
    let to_copy = dst
        .len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}
```

`to_copy` should be `src_l.shape().elem_count()`. It is only clamped to that by
accident, when the destination happens to be no larger than the source view,
which is the case for `Tensor::copy` / `try_clone`, and is why this has gone
unnoticed. It is not the case for `slice_scatter0`, where `dst` is the whole
output and `dst_offset` points into it.

The result is silent: no error, no panic, plausible numbers, and only on GPU.

## Reproduction

```rust
use candle_core::{Device, Tensor};

fn main() -> candle_core::Result<()> {
    for device in [Device::Cpu, Device::new_cuda(0)?] {
        // A [1, 4, 3] recipient, and a donor row taken as a view: the donor's
        // storage holds all four rows, and the view starts at row 2.
        let base = Tensor::new(
            &[[[0f32, 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]]],
            &device,
        )?;
        let donor = Tensor::new(
            &[[90f32, 91., 92.], [93., 94., 95.], [96., 97., 98.], [99., 100., 101.]],
            &device,
        )?;
        let row = donor.narrow(0, 2, 1)?.unsqueeze(0)?; // [1, 1, 3], offset 6

        let out = base.slice_scatter(&row, 1, 2)?;
        println!("{device:?}: {:?}", out.flatten_all()?.to_vec1::<f32>()?);
    }
    Ok(())
}
```

Expected on both devices, patching only position 2:

```
[0, 1, 2, 3, 4, 5, 96, 97, 98, 9, 10, 11]
```

CPU prints that. CUDA prints position 3 clobbered with the donor's *next* row:

```
[0, 1, 2, 3, 4, 5, 96, 97, 98, 99, 100, 101]
```

`Tensor::cat` over the same narrowed pieces was tested alongside and is **not**
affected on either backend, so the fault is reachable through `slice_scatter`'s
use of the helper rather than through every caller of it:

```
Cpu  slice_scatter: [0, 1, 2, 3, 4, 5, 96, 97, 98,  9,  10,  11]   correct
Cpu  cat          : [0, 1, 2, 3, 4, 5, 96, 97, 98,  9,  10,  11]   correct
Cuda slice_scatter: [0, 1, 2, 3, 4, 5, 96, 97, 98, 99, 100, 101]   WRONG
Cuda cat          : [0, 1, 2, 3, 4, 5, 96, 97, 98,  9,  10,  11]   correct
```

## Suggested fix

In `slice_src_and_dst`, size the copy from the source layout rather than from the
storage:

```rust
let to_copy = src_l.shape().elem_count();
```

keeping the existing clamp against `dst.len() - dst_offset` if a defensive bound
is still wanted. The function is only reached from the `src_l.is_contiguous()`
branch of `copy_strided_src`, where `elem_count` is exactly the number of
elements the caller means to move.

## How it was found, and why it is worth fixing rather than documenting

Downstream in `candle-mi`, an activation-patching intervention overwrites one
sequence position of a `[batch, seq_len, hidden]` residual stream with a row
taken from another forward pass. That donor row is naturally a view into a
captured activation. On CUDA the patch also overwrote **every position after**
the patch site.

The symptom was not a crash. It was a causal-tracing table reporting 100%
recovery at every layer for every token position but the last, including the
final layer, where patching a non-final position cannot affect the logits at all.
A plausible figure. Unit tests passed on both CPU **and** CUDA, because they
built the source with `Tensor::new`, which owns its storage exactly; only a
donor row taken as a view reproduces it.

## Prior art (searched 2026-09-02, none covering this path)

candle has a recurring family of view-start-offset bugs, which is the argument
for fixing the shared helper rather than each call site:

| # | what | backend wrong |
|---|---|---|
| [3874](https://github.com/huggingface/candle/issues/3874) | CPU `sort_last_dim` / `arg_sort_last_dim` ignore the view's start offset | CPU (CUDA and Metal correct) |
| [3735](https://github.com/huggingface/candle/issues/3735) / [3736](https://github.com/huggingface/candle/pull/3736) | CPU conv2d backward, non-contiguous kernels | CPU |
| [3893](https://github.com/huggingface/candle/issues/3893) / [3894](https://github.com/huggingface/candle/pull/3894) | the same conv2d bug on CUDA, fixed on CPU but not GPU | CUDA |
| [3853](https://github.com/huggingface/candle/pull/3853) | CUDA conv im2col reads the original strided kernel | CUDA |
| [3836](https://github.com/huggingface/candle/pull/3836) | non-contiguous conv kernels on Metal, conv1d on CPU | Metal, CPU |

Searched issues and PRs for `slice_scatter`, `copy_strided_src`,
`slice_src_and_dst`, `memcpy_dtod`, `start_offset cuda`, `scatter offset`, and
`cat narrow cuda`. The two `slice_scatter` hits ([988](https://github.com/huggingface/candle/pull/988),
[3062](https://github.com/huggingface/candle/pull/3062)) are a dim-type fix and a
doc-example fix. **Nothing reports this.**

This bug is the mirror image of 3874: there the CPU ignored the offset and the
GPUs were right; here CUDA over-reads from the offset and the CPU is right. Both
are the same underlying gap: a backend reading a view as though it were the
whole storage.
