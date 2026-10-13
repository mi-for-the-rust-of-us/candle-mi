> **Status: PARTLY IMPLEMENTED on branch `clt-train`** (2026-10-13, unreleased), from the canvas leg
> of askesis. §2–5 shipped there (`fed8287` through `0200d3e`): `clt::train` (`TrainableClt`,
> `forward_detached`, `export_circuit_tracer`, the sort-based `top_k`),
> `CrossLayerTranscoder::open_local`, `CltConfig::top_k`. §8 items 1–2 shipped as a **variant** in
> `fd2994b` (`clt::replacement`): the forward is the caller's closure and the transcoder sits behind
> a `Reconstruct` trait with a `ScaledClt` adapter — not the `&impl MIBackend` signature §8
> sketches, which could neither run canvas's own forward nor carry the training scales (Consistency
> pass, item 6). **ASKED** for v0.2.3: §8 item 3 (the example). §9's two `candle` fixes are upstream
> candidates, awaiting the maintainer. The new tests are CPU-only but not yet in a CI lane
> (Consistency pass, item 3).

# A CLT trains on candle-mi now, and three `candle` behaviours — not the math — set what a step costs

**Date:** October 13, 2026
**Source:** askesis `canvas` leg — the CLT leg (`reference/canvas/docs/design-canvas-clt.md`), steps 1–3:
the trainable transcoder, its training loop (`canvas clt-train`), the first pilots on the optimal-trained model O
**Affected area:** `src/clt/train.rs` (new), `src/clt/mod.rs` (`open_local`, `CltConfig::top_k`, `encode`); upstream
`candle-core/src/backprop.rs` (`Op::Narrow` backward), `candle-kernels/src/ternary.cu` (`where`), the comparison
kernels
**Severity:** Feature + Throughput — **not a defect**: every number was right from the first run (oracle parity at
`1e-4`); what the crate and `candle` decided was what a step costs and how much memory an evaluation holds
**Platform:** Windows 11, WDDM; spill measured with `hmn spill` (hypomnesis) — load-bearing for §3 only
**Measured against:** candle-mi 0.2.2 + branch `clt-train`, `candle-core`/`candle-nn` 0.11.0 (sibling clone, branch
`gradstore-accumulate-0.11`), candle-fused-attn 0.4.0, `rustc` 1.99.0, NVIDIA GeForce RTX 5060 Ti (sm_120, 16 GB),
driver 610.88, `F32` throughout

---

## 1. The use case

The canvas planner's optimal-trained model O (6 layers × 384, an `OthelloGpt` trained with
candle-mi) picks the right block to sacrifice in blocksworld deadlocks — 0.94 on the optimal move
where several blocks are candidates — yet no linear probe at the decision cell can say which block
it will pick (Me21: at the shuffled-label null), and it does not rank moves by the optimal distance
it plainly carries (Me24). Two readings remain: a universal plan that computes the choice without
writing it anywhere, or a target written as a SPARSE feature a linear probe misses — which is what
"On the Biology of a Large Language Model" found for Haiku 3.5's rhyme word. A cross-layer
transcoder on O, then attribution and causal steering at the decision points, separates them.

The agreed split: candle-mi owns the transcoder and its loss, canvas owns the data and the loop —
the split already used for `OthelloGpt` plus `optim::AdamW`, and the crate's own scope (`training` =
"checkpointable `AdamW` … still no training loop"). The oracle is askesis
`reference/scripts/clt.py`, the `PyTorch` CLT the MDLM-owt transcoders were trained with, run on CPU
torch.

## 2. What the crate needed, and now has on `clt-train`

**MEASURED — oracle parity.** `scripts/clt_train_validation.py` runs `clt.py`'s own class (imported
from its file, not re-implemented) on a fixed case and writes every intermediate;
`tests/validate_clt_train.rs` (CPU, in CI) loads the same weights into `TrainableClt`:

```text
test forward_matches_the_reference ... ok
test gradients_and_one_adam_step_match_the_reference ... ok
test export_folds_the_scales_into_raw_units ... ok
test open_local_encodes_and_decodes_like_the_trained_clt ... ok
```

Bar `1e-4` (the reference's own), on the `TopK` activations, the reconstruction, the loss, the
gradient of every parameter, and the parameters after one Adam step (`optim::AdamW` with zero weight
decay is `torch.optim.Adam`). The fixture's `TopK` binds everywhere (72 active of 288 = exactly k
per row), so the selection is tested, not bypassed.

- **`export_circuit_tracer`** writes the layout `CrossLayerTranscoder` already reads (`W_enc_{l}` /
  `W_dec_{l}`, `F32`), folding the per-layer training scales into the weights; **`open_local(dir)`**
  opens it with nothing downloaded; **`CltConfig::top_k`** makes `encode` keep the k largest, as the
  CLT did in training (`None` — unchanged behaviour — for every published transcoder). The round
  trip above proves a CLT trained here is analysed by the same `encode` / `decoder_vector` code as a
  published one.

## 3. Evaluation through `Var`s holds the whole graph — hence `forward_detached`

**HYPOTHESIS, caught before the run (arithmetic, not a measured spill).** The faithfulness gate runs
the model with every MLP output replaced by the transcoder's reconstruction, closed loop, over every
position of 128 evaluation rows: 128 × 240 = 30,720 positions × 6 layers × 4,096 features ≈ 3 GB per
activation tensor. Through `forward`, every op on a `Var` is recorded, so each such intermediate
stays reachable until the result drops — several alive at once, past the card's 16 GB. Éric saw
shared-memory growth in the performance monitor and asked; the arithmetic answered.

**MEASURED, after the fix.** `TrainableClt::forward_detached` (same numbers, detached parameters — a
doctest asserts the difference is 0) plus chunks of 2,048 positions in canvas: peak **6.9 GiB of
15.7, no spill** at full evaluation size (`hmn spill`: peak shared 158 MiB against an 80 MiB
baseline, 0 episodes).

The general point for the crate: any trainable module will be evaluated, and the tracked forward is
the wrong one to evaluate with. `forward_detached` is the pattern; the crate may want it wherever a
module is both trained and probed.

## 4. `TopK` cost 147 ms a step because `candle`'s `where` and comparison kernels drop to their strided path

The first profile (canvas `examples/clt_step_profile.rs`, a device sync at each phase boundary, 20
steps after 5 of warm-up; 2,048 vectors × 6 layers × 4,096 features):

```text
  draw          17.8 ms    5.4 %
  model         48.6 ms   14.8 %
  scale          1.4 ms    0.4 %
  encode       164.8 ms   50.3 %
  decode        13.6 ms    4.2 %
  loss           3.5 ms    1.1 %
  backward      74.4 ms   22.7 %
  optim          3.4 ms    1.0 %
  TOTAL        327.5 ms per step
  top_k alone on [2048, 6, 4096]: 150.0 ms per call
```

**MEASURED — Nsight Systems** (`nsys profile --trace=cuda --sample=none --cpuctxsw=none`, the
candle-fused-attn recipe; 12 `TopK` calls in the capture):

```text
 32.6 %    949.5 ms    192 x    4945.2 us  where_u8_f32
 16.2 %    473.4 ms    192 x    2465.4 us  eq_f32
 12.5 %    365.4 ms    228 x    1602.5 us  void magma_sgemmEx_kernel<float, float, float, (bool)1, (boo
 11.7 %    341.4 ms    204 x    1673.7 us  fast_max_f32
```

The first `TopK` removed the row maximum k times: `max_keepdim`, `broadcast_eq` against it,
`where_cond` against a broadcast `-inf`. `candle-kernels/src/ternary.cu` has a contiguous fast path
for `where` — taken only when all three layouts are contiguous; one stride-0 operand sends 50M
elements through three `get_strided_index` calls each. Same for the comparison against a broadcast
threshold. 4.95 + 2.47 + 1.67 ms, 16 times.

**MEASURED — the fix.** One descending row sort (`arg_sort_last_dim`), the k-th value gathered as
one column, one compare. On the GPU at a training step's shape, against the replaced algorithm
(canvas `examples/top_k_check.rs`):

```text
dense: max |new - old| = 0 | most active per row 16 (k = 16) | sort 30.58 ms vs maxima 147.42 ms
sparse (shifted by -2.5): max |new - old| = 0 | most active per row 16 (k = 16) | sort 27.38 ms vs maxima 149.52 ms
```

A two-stage variant (sort 1,024-wide chunks, keep each chunk's top k, sort the 64 candidates) is
exact too and no faster (31.7 ms): not kept.

**Mine, recorded as mine.** I wrote in the module docs that `candle`'s sort is "limited to rows of
1,024 columns on `CUDA`" — from memory of an older `candle`, and it steered the first design to the
16-pass loop. The kernel this crate builds against caps the *thread block* at 1,024 and loops ("each
thread handles multiple elements if ncols_pad > blockDim.x", `candle-kernels/src/sort.cu`;
`candle-core/src/sort.rs:87–93`). The docs are corrected in `eb21341`. I had also estimated `TopK`
at ~40 ms on paper; it measured 147. Paper estimates of this step summed to 150–200 ms against 327
measured — which is why the profile came before any optimisation.

## 5. A per-layer slice of a stacked block costs a full-size copy in backward

**MEASURED.** The decoder sliced the `[n, layers, features]` activations once per source layer.
`candle-core`'s `Op::Narrow` backward (`backprop.rs:568`) builds zero padding for both sides and a
`Tensor::cat` to the full argument shape, then accumulates — six full-size `[2048, 6, 4096]` tensors
(200 MB each) per step, visible in the capture as `copy2d_f32` + `badd_f32` (~26 ms a step
together). Keeping one tensor per source layer end to end (`encode_layers` / `decode_layers`,
`0200d3e`; the public `encode` / `forward` still return the stacked block):

```text
  STEADY       161.6 ms per step (no phase syncs)      before
  STEADY       134.5 ms per step (no phase syncs)      after
```

Same math: oracle parity, the 67 `clt` tests and the doctests all pass unchanged.

## 6. The method note: a phase-synced profile measures its own syncs

**MEASURED.** The 17.8–21.9 ms "draw" phase above is not the draw:

```text
  draw on the CPU device: 0.06 ms per call
  draw on the GPU, back to back: 0.07 ms per call
```

and the phase table's total overstates the step by ~48 ms (209.8 phase-synced against 161.6 steady,
same build). A sync at every boundary attributes time; only a run with no phase syncs measures it.
The profiler now prints both, and §4–5's numbers are quoted against the steady figure. The allocator
was not the culprit — canvas `bench_alloc.rs` had already measured it (`free_async` ~1 µs,
steady-state alloc 0.06 ms), and reading that record instead of re-measuring saved the detour.

## 7. Where it stands

| | steady step | 20,000 steps |
|---|---|---|
| first version (phase-synced 327.5 ms) | not measured | ~2.5 h (pilot, evaluations included) |
| sort-based `TopK` | 161.6 ms | |
| per-layer activations | **134.5 ms** | **~52 min** (0.155 s with evaluations) |

Size pilots on O s0, 1,000 steps, same seed / batches / evaluation set (512 decision cells):

| features per layer | loss recovered | agreement | dead | s/step | peak VRAM |
|---|---|---|---|---|---|
| 2,048 | 0.790 | 78.5 % | 64.1 % | 0.10 | 5.2 GiB |
| 4,096 | 0.826 | 78.7 % | 77.2 % | 0.155 | 6.8 GiB |
| 8,192 | 0.825 | 81.4 % | 85.7 % | 0.265 | 10.7 GiB |

## 8. Ask (v0.2.3), in order of value

1. **The closed-loop replacement model, in the crate.** canvas has it (`src/clt.rs`,
   `replacement_logits`), and nothing in it is canvas-specific: pass `t` runs with layers `< t`
   already replaced (`Intervention::Replace` at `MlpOut(l)`), captures `ResidMid(t)`, decodes layer
   `t` from features encoded at the SPLICED residuals; a last pass replaces every layer. Detached
   and chunked (§3). It is the object the Biology paper builds everything on, the foundation of a
   real attribution graph, and it applies as-is to PUBLISHED circuit-tracer CLTs (same `ResidMid` →
   `MlpOut` convention), which would give every candle-mi user a faithfulness check on their own
   prompts. Something like

   ```rust
   pub enum Splice { Clean, Replaced, Ablated }
   pub fn replacement_logits(
       backend: &impl MIBackend,
       clt: &impl Reconstruct,          // TrainableClt today; CrossLayerTranscoder too
       input_ids: &Tensor,
       splice: Splice,
   ) -> Result<Tensor>;                  // [batch, seq, vocab]
   ```

   Whether that is a trait over the two transcoder types, two functions, or a method on each is the
   crate's call.
2. **The faithfulness metrics** as small pure functions with doctests: loss recovered against
   zero-ablated MLPs (`1 - (spliced - clean) / (ablated - clean)`), FVU per layer, the dead-feature
   count. canvas has all three; they are ten lines each and every CLT user needs them.
3. **An example, `clt_train_tiny`**: a complete loop on a small model, as a recipe — the loop stays
   out of the library, but a reader should not have to find canvas to learn how the pieces fit.

## 9. Upstream `candle` candidates (not candle-mi asks)

- **A broadcast-aware fast path in `where` and the comparison kernels.** A stride-0 operand along
  the last dimension is the common case (a per-row threshold, a scalar fill); the strided path costs
  ~5× here.
- **`Op::Narrow` backward as a slice-add into the accumulated gradient** instead of `cat` with zero
  padding to the full shape.

Both are measured above and both would reach every `candle` user; whether to file them, and through
candle-patched or directly, is the maintainer's decision.

## Not asked for

- **A training loop in candle-mi.** The loop, the data and the schedule stay with the consumer (§1).
- **Dead-latent revival (`AuxK`, Gao et al. 2024).** Most features never fire on the evaluation
  positions (§7); whether to add an auxiliary loss is a method decision for the consumer, and the
  oracle `clt.py` has none.
- **TF32 for the transcoder's matmuls.** ~4× matmul throughput, but the model's own forward must
  stay `F32` — the faithfulness gate measures against it.
- **A fused top-k kernel.** `TopK` is now ~20 % of the step; if it becomes worth fusing, it belongs
  in candle-fused-nn, after reading ATen's `TensorTopK.cu`.

## What worked, and is worth keeping

- **The hook surface did the replacement model unaided.** `ResidMid(l)`, `MlpOut(l)` and
  `Intervention::Replace` were all it needed; no new hook point, no callback.
- **`optim::AdamW`'s checkpointable state.** A resumed run picks up exactly (`resumed at step 20`),
  because canvas draws each step's batch as a pure function of `(seed, step)` and the optimiser's
  moments come back with it.
- **The fused attention is untouched by these captures.** No attention-internal hook, so
  `OthelloGpt` keeps candle-fused-attn's path while the CLT reads around it.
- **`hmn spill`** answered "does it spill?" in one wrapped command, every run.

## How to reproduce

From askesis-acsp14 `reference/canvas` (candle-mi on `clt-train`, the sibling clone), release build
with `--features cuda,fused-attn`, O s0 = `data/runs/acspv_sasinv_pxc_opt_s0_d1/canvas_ema_e80`,
data `data/canvas_acsp14_sasinv_opt/arm_d1`:

```text
examples/clt_step_profile <checkpoint> <data> --steps 20 --warmup 5      # phases + steady step   (canvas 5b68ed3, bbbd6b8)
examples/top_k_check                                                       # sort vs maxima, on the GPU (f70a6ce)
bash clt_size_pilots.sh                                                    # §7's sizes             (b61f23e)
canvas clt-train --checkpoint <ck> --data-dir <d> --out-dir <o> --features 4096 --steps 20000 --eval-every 500 --eval-rows 128
```

candle-mi side: `cargo test --no-default-features --features clt,training --test
validate_clt_train`; the fixture is regenerated with `<python with torch>
scripts/clt_train_validation.py [path/to/clt.py]`.

## Consistency pass (2026-10-13, same day), against the files

An independent read-only pass checked every claim above against the repositories, the logs and the
Nsight CSV. The numbers held — commit contents, versions, `backprop.rs:568`, `sort.rs:87–93`, the
verbatim `sort.cu` comment, `ternary.cu`'s three strided lookups, 72 of 288 active, the 67 tests,
the Nsight rows digit for digit against `data/analysis/clt_profile/kern_sum.csv`, the phase and
size-pilot tables, the `training` and `bench_alloc` quotes. What did not hold, corrected here
rather than in the prose above:

1. **§1, Me21 "at the shuffled-label null" — too strong.** The logbook reads *near* the null, with
   a faint, consistent tilt for O: O exceeds its own null by +0.4 to +7.5 points, and on two of five
   seeds it is indistinguishable from it.
2. **§1, "0.94 on the optimal move" — one worked example, not a rate.** O plans optimally on
   99.9–100 % of the problems where the `US` policy does not (Me19); 0.94 is O s0's probability on
   the single optimal sacrifice among four live blocks in one example.
3. **§2, "(CPU, in CI)" — wrong.** No CI lane runs `validate_clt_train` (nor, since,
   `validate_clt_replacement`): `ci.yml`'s clt lane runs `--lib` only, its training lane only
   `validate_optim_parity`. The same claim stood in `CHANGELOG.md` and both test headers; corrected
   there. Adding the two files to a lane is the maintainer's call.
4. **§7, "~52 min (0.155 s with evaluations)" — the pilots' rate, not the run's.** The pilots
   evaluated every 250 steps; the 20,000-step run evaluates every 500 and logs 0.142 s/step
   (`data/clt/O_s0_f4096.log`): ~47 min.
5. **§6, "§4–5's numbers are quoted against the steady figure" — only §5's are.** §4's 327.5 ms is
   the phase-synced total, its `TopK` times are per call, and §4's "150–200 against 327" compares
   paper estimates with that phase-synced total.
6. **§8 item 1, "nothing in it is canvas-specific" — wrong in two ways, and the sketched signature
   inherits both.** canvas runs O through `forward_no_carry` (a `NONE` carry grid), not
   `MIBackend::forward` (no carry), and the reconstruction needs the training scales. `fd2994b`
   therefore took a forward closure and a scaled adapter (status above).
7. **§8 item 2, "ten lines each" — less.** `loss_recovered` is one line, `fvu_per_layer` four, and
   the dead measure was an inline share in `evaluate`, not a count.
8. **§4 and §9, the comparison kernels "drop to their strided path" — half-strided.**
   `binary_op_macros.cuh` has a branch for one contiguous side, and `eq` takes it: only the
   broadcast operand pays the per-element index arithmetic. `where` is fully strided as stated.
9. **§3, "6.9 GiB of 15.7 … 158 MiB against 80 MiB" — from the 300-step pilot**
   (`data/clt/pilot`, printed to the terminal, no log file). The nearest logged run, `size_4096`,
   reads 6.8 GiB and 171 / 91 MiB.
10. **§6, "17.8–21.9 ms" — 21.9 is from the profile after `0200d3e`,** whose table is not shown
    above (draw 21.9 ms, steady 134.5 ms).
11. **Unlabelled estimates.** TF32 "~4×" (Not asked for) is unmeasured and likely wrong for this
    card: NVIDIA's GeForce specifications list dense TF32 tensor throughput about equal to FP32 —
    HYPOTHESIS at best. "~5×" for the strided path (§9) has no contiguous baseline measured. "~20 %
    of the step" for `TopK` divides the stacked-shape call (29 ms) by the per-layer step (134.5 ms);
    the per-layer `top_k` was not timed.
12. **Not verbatim.** The §5 `STEADY` lines carry an added "before / after"; the Nsight block is a
    reformatting of `kern_sum.csv` from a shorter capture (`--steps 5 --warmup 2`, 12 `TopK` calls),
    not of the 20-step profile; "How to reproduce" omits the capture itself:
    `nsys profile --trace=cuda --sample=none --cpuctxsw=none -o <rep> examples/clt_step_profile ...`
    then `nsys stats --report cuda_gpu_kern_sum --format csv <rep>.nsys-rep`.
13. **Style.** The title's "three `candle` behaviours" are §3's graph retention (memory), §4's
    strided kernels and §5's `Narrow` backward (speed) — §9 lists only the two kernel-level ones as
    upstream candidates. The severity's house word is "Throughput ceiling". §3's HYPOTHESIS should
    have named its settling measurement: the evaluation unchunked under `hmn spill`, not run because
    the arithmetic sufficed. §8 item 2 gave no Rust signatures — `fd2994b`'s are
    `loss_recovered(f64, f64, f64) -> f64`, `fvu_per_layer(&Tensor, &Tensor) -> Result<Vec<f32>>`,
    `cross_entropy_at`, `top_tokens_at`, `dead_fraction`.
