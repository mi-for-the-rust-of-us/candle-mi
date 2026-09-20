# Metal preflight: what to run the day a Mac is available

> **Status: DEFERRED by decision**, not by oversight. A macOS CI lane and the
> `metal` verification below are postponed until a local Mac is available to
> preflight them, so that a red CI lane is not the first time anyone looks at
> this. Findings recorded 2026-09-20 during the v0.2.0 audit; the memory-module
> half was first raised 2026-07-01.

This file exists so the eventual Metal pass is a checklist rather than an
investigation. Everything below was established by reading the crate, not by
running anything on Apple hardware, which is the whole point.

## What `metal` actually is in this crate

    metal = ["candle-core/metal", "candle-nn/metal"]      # Cargo.toml:433

**candle-mi contains zero `#[cfg(feature = "metal")]` code.** The feature is a
pure passthrough that switches candle's own backend; there is no Metal shader,
no Metal device management and no Metal branch anywhere in `src/`. Do not go
looking for candle-mi Metal code when the Mac arrives: there is none, and that
is correct. What needs verifying is that candle-mi's *existing* code runs on
candle's Metal backend.

Separately, `hypomnesis`'s own `metal` feature **is** enabled (`Cargo.toml:108`)
for VRAM measurement, and that one does contain real platform code.

## The three gaps, and the check that closes each

### 1. Nothing has ever compiled this feature

No CI lane and no `preflight.ps1` step enables `metal`. The string appears in
neither. So the first build on a Mac will be the first build anywhere.

This matters more than it sounds. Every op in candle-mi's forward passes routes
through candle's Metal kernels, including several the CUDA and CPU paths reach
by different routes: the `nn_ops` fused-versus-composed dispatch
(`Tensor::track_op`), `where_cond` in `hooks::patch_at`, `index_select` backward
in the training path, and `broadcast_matmul` in the rank-preserving projections.
Whether candle 0.11's Metal backend implements all of them is simply unknown.

    cargo build   --no-default-features --features "metal,transformer"
    cargo clippy  --all-targets --no-default-features --features "metal,transformer" -- -W clippy::pedantic
    cargo build   --no-default-features --features "metal,diffusion"
    cargo build   --no-default-features --features "metal,rwkv,rwkv-tokenizer"
    cargo build   --no-default-features --features "metal,stoicheia"

Expect: clean. Any failure here is a genuine finding and belongs in an upstream
candle issue, not a local workaround.

### 2. `from_pretrained` cannot select a Metal device

`MIModel::select_device` (`src/backend.rs:418`) is:

    Device::cuda_if_available(0)     // CUDA GPU 0, or CPU fallback

There is no Metal branch. Its doc comment says "CUDA GPU 0, or CPU fallback", so
it is not lying, but the consequence is easy to miss during a preflight: on a
Mac, with `--features metal`, the documented quick-start path
`MIModel::from_pretrained(model_id)` **silently runs on CPU**. A slow first run
is the only symptom.

`MIModel::new(backend, device)` (`src/backend.rs:428`) is public and takes a
caller-supplied `Device`, so `Device::new_metal(0)?` reaches the Metal path
today. Use that for any timing comparison, or the numbers will be CPU numbers.

The fix, when the time comes, is a `#[cfg(feature = "metal")]` branch in
`select_device` tried before the CUDA/CPU fallback. It is small, and it should
land *with* a machine to test it on rather than before.

### 3. The memory module's Metal path is untested

Already recorded as `docs/audit/VALIDATION_AUDIT_2026-07-01.md` section 1.12,
still open, and the reason it stayed open is exactly this file's trigger:

> the live Metal path is untestable on the available hardware (Windows + CUDA
> GPU; CI is `ubuntu-latest`), so it can't be closed here. A compile-only macOS
> CI lane would catch Metal-path *build* breaks, but the "512 MiB -> exact
> delta" runtime check needs an Apple device.

`src/memory.rs:125` branches on `device.is_cuda() || device.is_metal()`, and no
live test has ever taken the second arm. The CUDA ground-truth tests in
`tests/validate_memory.rs` are the template to mirror: allocate a known size,
assert the VRAM delta.

## Why the lane cannot be `--all-features`

`--all-features` enables `cuda` and `metal` together. `cuda` pulls `cudarc` and
`candle-kernels`; `metal` pulls `objc2-metal` and `objc2-foundation`, neither of
which is target-gated in `candle-core`'s manifest. A Mac cannot build `cuda`
(Apple has not shipped NVIDIA support for years) and nothing else can build
`metal`. **So `--all-features` is unbuildable on any single machine, including
the Mac**, and the Metal lanes must name their features explicitly, as above.

This is the same reason `CLAUDE.md`'s pre-commit clippy command excludes
`metal`, and the same shape of reasoning already recorded in `ci.yml:199` and
`scripts/preflight.ps1:103` about per-feature rustdoc lanes.

## Checklist for the day

1. Section 1's five build/clippy commands. Record any op candle's Metal backend
   does not implement.
2. `cargo test --no-default-features --features "metal,stoicheia"` and the same
   for `diffusion`: these are the download-free suites, so they run immediately.
3. Forward-pass parity against the existing oracles, which is the part only
   Apple hardware can give. `tests/validate_othello_forward.rs` is the cheapest
   meaningful one (~25 M params, fixture-gated). Compare max-abs logit delta
   against the house bars: **~1e-3 CPU, 5e-3 GPU**. Record the number, since a
   Metal null band has never been measured and every later Metal claim will be
   read against it.
4. The section 1.12 memory ground-truth test, mirroring the CUDA pair.
5. Only then add the `macos-latest` CI lane, green on the first push.

`scripts/preflight.ps1` is PowerShell and will not run on macOS. Either invoke
the commands above directly or port the lane list; do not assume the script
works there.

## What to update when this is done

- This file: status from DEFERRED to the date and the measured numbers.
- `docs/audit/VALIDATION_AUDIT_2026-07-01.md` section 1.12: close it.
- `README.md`: the feature table (line ~224) and the "direct CUDA/Metal access"
  claim (line ~84) are currently ahead of what is verified. Either becomes true
  once this passes, or should be softened if it does not.
- `CLAUDE.md`: the pre-commit clippy line, if `metal` becomes runnable locally.
- `.github/workflows/ci.yml`: the new `macos-latest` lane.
