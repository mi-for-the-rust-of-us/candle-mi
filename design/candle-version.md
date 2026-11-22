# Design: Minimum Supported Candle Version

**Status:** Implemented
**Relates to:** Roadmap §8 item 6

## Question

What candle version should candle-mi target, and how should we pin it?

## Context

candle is pre-1.0 (currently 0.11.x). Breaking changes between minor versions are possible.

## Recommendation

Track a minor version in `Cargo.toml` and update incrementally:

```toml
[dependencies]
candle-core = "0.11"
candle-nn = "0.11"
```

> **As implemented:** the shipped `Cargo.toml` uses the caret range `"0.11"`
> (`>=0.11.0, <0.12.0`), not an exact pin. The caret admits compatible `0.11.x`
> patch upgrades automatically; CI's rolling matrix plus the pinned `Cargo.lock`
> provide the compatibility check the exact pin was meant to guarantee. The
> `0.9 → 0.11` jump (2026-07-12) compiled with no library source changes and
> built cleanly on CUDA (candle 0.11's new `cudaforge` kernel-build path).
>
> **Since v0.2.3** the floor is `"0.11.1"` (`>=0.11.1, <0.12.0`): 0.11.1 is the
> first candle release on `zip` 9 (0.11.0's `zip` ^8.6 carried four RustSec
> advisories), and a `"0.11"` floor would let a consumer's lockfile keep 0.11.0
> while candle-mi is tested against 0.11.1 only. 0.11.1 also needs Rust 1.94 on
> x86_64 without declaring a `rust-version` (huggingface/candle#4097), one of
> the two causes of candle-mi's 1.95 MSRV; `Cargo.toml` records both.

Test against the resolved version in CI. Widen the range only after verifying compatibility.

## Open questions

- Should we support multiple candle versions via feature flags, or just track the latest?
- When candle reaches 1.0, switch to standard semver ranges.
