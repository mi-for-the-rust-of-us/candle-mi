# Store the first gradient directly instead of adding it into fresh zeros

**Title for the PR:** `candle-core`: store the first gradient of each node directly instead of
`zeros_like` + `add` — measured 25% off a training step

**Status: FILED (2026-08-01)** as [candle#3822](https://github.com/huggingface/candle/pull/3822),
from `PCfVW:gradstore-accumulate` (one commit on `origin/main` `6e823a43`; `backprop.rs` is
byte-identical between main and 0.11.0, so the same commit validates on both). Pre-flight at
filing time: fmt clean, clippy zero findings, candle-core 171/171, stock candle-nn 58/58
against the patched core.

**Prior art missed at filing (found 2026-10-01):** [candle#3578](https://github.com/huggingface/candle/pull/3578) by
`TeunVerstraaten`, opened 2026-06-04, makes the same change in the same file. Our
triage comment [comment-5926788462](https://github.com/huggingface/candle/pull/3823#issuecomment-5926788462) proposes landing #3578 and closing this one. Details in
[README.md](README.md#what-happened-checked-2026-10-01).

## What this changes

`Tensor::backward` accumulates gradients through `GradStore::or_insert`, which initializes every
node's slot with `zeros_like` and lets the call site `add` into it:

```rust
let sum_grad = grads.or_insert(arg)?;
*sum_grad = sum_grad.add(&arg_grad)?;
```

Every gradient edge therefore pays a full-size zero fill (one write) plus a full-size `add` (two
reads and a write, into another fresh allocation) — **including the case where the node has a
single consumer and the "sum" has exactly one term, which in a transformer is nearly every
node.** `PyTorch` stores the first gradient and accumulates only from the second onward
(`InputBuffer::add` stores the first gradient into the empty slot); this PR gives candle the same behaviour:

```rust
fn accumulate(&mut self, tensor: &Tensor, grad: &Tensor) -> Result<()> {
    match self.0.entry(tensor.id()) {
        Entry::Occupied(mut entry) => {
            let sum_grad = entry.get_mut();
            *sum_grad = sum_grad.add(grad)?;
        }
        Entry::Vacant(entry) => {
            entry.insert(grad.contiguous()?);
        }
    }
    Ok(())
}
```

plus an `accumulate_neg` twin for the six gradient rules that subtract (`Neg`, `Cos`, `Tanh`,
`Recip`, and the right-hand sides of `Sub` and `Div`), so their fan-in path keeps the single `sub` it has today. 66 of the 71
`or_insert` call sites convert mechanically. Four keep the accumulator by design — `Gather` and
`IndexSelect` scatter/index-add **into** a dense base, and the two `UpsampleNearest` sites
overwrite their slot (a pre-existing behaviour this PR deliberately does not touch). Net diff:
115 insertions, 133 deletions, one file.

Two details that keep the change strictly behaviour-preserving:

- **`contiguous()` on insert.** The old zeros+add always produced a contiguous gradient; a
  directly stored first gradient may be a strided view (a transpose backward, say). For already
  contiguous gradients — the common case — `contiguous()` returns self and costs nothing; for a
  view it costs one copy where the old path cost a zeros-write plus an add.
- **Expressions that read the accumulator's metadata** (`sum_grad.dims()` in the broadcast
  rules, `sum_grad.device()` in `ToDevice`) are rewritten to read the target tensor's, which is
  what the accumulator's shape and device are defined to be.

## Why it matters, measured

On a 10.7M-parameter transformer training step (6 layers, d=384, batch 128, fp32, RTX 5060 Ti),
profiled with Nsight Systems: the zeros+add initialization traffic is **~900 `badd_f32`
launches per step and roughly a third of total GPU time**. The kernels themselves are fine —
they run at ~80% of the card's spec bandwidth — it is purely the number of full-tensor round
trips.

After the change, same machine, same protocol, differenced over 20-vs-120-step runs:

| | before | after |
|---|---|---|
| `backward()` (per-phase sync profile) | 413.7 ms | **251.8 ms** |
| whole step, wall clock | 0.521 s | **0.391 s** (−25%) |
| training throughput | 31.4k tok/s | **41.9k tok/s** |
| `badd_f32` launches per iteration (nsys) | 976, median 196 µs | **401, median 8.4 µs** |
| gemm time (nsys) | — | unchanged |

The launch-count and median collapse are the mechanism check: the surviving adds are the small
bias-gradient reductions and genuine fan-in, the full-tensor swarm is gone. Against an identical
`PyTorch` step (99 ms, same arithmetic — see correctness) this takes candle from 5.1× to 3.9×.

## Correctness evidence

- `cargo test -p candle-core`: **171 passed on main, 170 on the 0.11.0 branch, 0 failed**,
  including the `grad_tests` suite (7 tests) on both.
- A downstream 64-test training suite passes unchanged, including a staged-equals-continuous
  resume property held at 1e-5.
- The strongest gate: a whole-loop parity oracle that replays 20 recorded optimizer steps of the
  real model in `PyTorch` (bit-exact batches, identical init). With this change the worst
  per-step loss disagreement over 20 steps is **1.78e-07** — *below the 2.0e-7 disagreement of
  stock candle with itself across CPU/GPU*, measured on the same rig. The patched backward
  computes the same function to the limit our instruments can resolve.
- The only bitwise-visible semantic difference is signed zeros: `0.0 + (-0.0)` is `+0.0`, so the
  old path could turn a `-0.0` gradient into `+0.0` where the new path preserves it. This
  matches `PyTorch`'s behaviour, since it stores the first gradient too.

## Costs and risks

- `GradStore` gains two private helpers; `or_insert` stays (four call sites still need it).
  Public API unchanged.
- Nodes whose first gradient is inserted hold that tensor rather than a fresh sum; since candle
  tensors are immutable and every fan-in accumulation rebinds the entry to a new tensor, aliasing
  is unobservable. (The `Add` rule inserts the same incoming gradient for both operands on first
  touch; the first accumulation into either replaces that entry, never the storage.)
- Second-order behaviour is unchanged: `backward` detaches each popped gradient exactly as
  before; what is stored differs only by not being routed through an `add` with zeros.

## The measurement instrument, for reproduction

`profile_step.rs` + `nsys profile -t cuda` + `nsys stats --report cuda_gpu_kern_sum` on any
training loop will show the before/after directly: the `badd_f32` row's instance count is the
signature. The workload used here is the askesis `canvas` leg (blocksworld masked-diffusion
planner), but nothing in the change or the measurement is specific to it — every candle training
loop pays the zeros+add tax on every gradient edge today.
