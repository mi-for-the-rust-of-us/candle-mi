// SPDX-License-Identifier: MIT OR Apache-2.0

//! Activation, attention, and KV caching for efficient forward passes.
//!
//! - [`ActivationCache`] — per-layer last-token residual stream activations.
//! - [`AttentionCache`] — per-layer post-softmax attention patterns.
//! - [`FullActivationCache`] — all-position residual stream activations.
//! - `KVCache` — key/value cache, unused and not re-exported, so it is
//!   unreachable outside the crate; see `kv`'s module documentation.

mod activation;
mod attention;
mod kv;

pub use activation::{ActivationCache, FullActivationCache};
pub use attention::AttentionCache;
// `kv::KVCache` is deliberately NOT re-exported. It has no caller, so a
// re-export here would itself be dead code, and its absence is what keeps the
// type out of the public API. A future incremental-decode path adds this back.
