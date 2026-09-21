// SPDX-License-Identifier: MIT OR Apache-2.0

//! A key/value cache that **nothing in this crate currently uses**.
//!
//! Inherited from `plip-rs`, candle-mi's predecessor, where it was live in
//! every `forward_*`. candle-mi's redesign around a single hook-aware
//! [`MIBackend::forward`](crate::backend::MIBackend::forward) orphaned it: the
//! generation helpers re-run a full forward per step, and no backbone threads
//! a cache through. It is retained rather than deleted because the data
//! structure is correct and tested, and a future incremental-decode path would
//! want exactly this shape.
//!
//! Until such a path exists it is unreachable from outside the crate: `kv` is
//! a private module and nothing re-exports the type, so the `pub` below is
//! capped at `cache`'s own subtree. (`pub(crate)` would be identical in effect
//! and `clippy::redundant_pub_crate` rejects it here.) It was public and
//! root-re-exported up to v0.1.24, whose
//! documentation claimed it "enables efficient token-by-token generation with
//! `O(1)` complexity per token instead of `O(n)`". That described what a KV
//! cache does in general, not anything candle-mi implements, and advertising
//! a capability the crate does not have is worse than having no type at all.
//! Demoted in v0.2.0.
//!
//! ## Memory Layout
//!
//! Each layer stores:
//! - keys: `[batch, num_kv_heads, seq_len, head_dim]`
//! - values: `[batch, num_kv_heads, seq_len, head_dim]`
//!
//! ## Memory Estimation
//!
//! For a 7B model (typical hyperparameters):
//! - `num_kv_heads` = 8 (GQA)
//! - `head_dim` = 128
//! - `num_layers` = 32
//! - dtype = BF16 (2 bytes)
//!
//! Per token: 8 * 128 * 2 * 2 * 32 = 128KB
//! For 2048 tokens: ~256MB

use candle_core::Tensor;

use crate::error::{MIError, Result};

/// Per-layer key/value cache, **not wired into any forward path**.
///
/// Stores the key and value tensors from previous positions so they would not
/// need recomputing at each generation step. No backbone consults it today;
/// see the module documentation for why it is kept.
///
/// # Shapes
///
/// - `keys[i]`: `[batch, num_kv_heads, seq_len, head_dim]`
/// - `values[i]`: `[batch, num_kv_heads, seq_len, head_dim]`
// RETAINED: no caller outside this module's own tests, by design rather than
// oversight, so `dead_code` would fire on every item here under
// `#![deny(warnings)]`. The alternative to the allow is deleting a correct,
// tested structure that an incremental-decode path would have to write again.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Cached key tensors per layer: `[batch, num_kv_heads, seq_len, head_dim]`.
    keys: Vec<Option<Tensor>>,
    /// Cached value tensors per layer: `[batch, num_kv_heads, seq_len, head_dim]`.
    values: Vec<Option<Tensor>>,
}

// RETAINED: see the note on the struct; the whole inherent API is unused.
#[allow(dead_code)]
impl KVCache {
    /// Create a new empty cache for the given number of layers.
    #[must_use]
    pub fn new(n_layers: usize) -> Self {
        Self {
            keys: vec![None; n_layers],
            values: vec![None; n_layers],
        }
    }

    /// Current sequence length from the cache (0 if empty).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if a cached tensor has an unexpected shape.
    pub fn seq_len(&self) -> Result<usize> {
        match self.keys.iter().find_map(Option::as_ref) {
            Some(k) => Ok(k.dim(2)?),
            None => Ok(0),
        }
    }

    /// Whether the cache is empty (no layers have been populated).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.iter().all(Option::is_none)
    }

    /// Number of layers in the cache.
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.keys.len()
    }

    /// Clear all cached tensors.
    pub fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
    }

    /// Get mutable references to the cache entry for a specific layer.
    ///
    /// Returns `(&mut Option<Tensor>, &mut Option<Tensor>)` for (key, value).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if `layer` is out of range.
    pub fn layer_mut(
        &mut self,
        layer: usize,
    ) -> Result<(&mut Option<Tensor>, &mut Option<Tensor>)> {
        if layer >= self.keys.len() {
            return Err(MIError::Hook(format!(
                "layer {layer} out of range for KV cache"
            )));
        }
        // Bounds checked above; keys and values are separate fields so the
        // borrow checker allows simultaneous mutable borrows.
        #[allow(clippy::indexing_slicing)]
        Ok((&mut self.keys[layer], &mut self.values[layer]))
    }

    /// Estimate memory usage in bytes.
    ///
    /// Returns the total memory used by all cached tensors.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let key_mem: usize = self
            .keys
            .iter()
            .filter_map(Option::as_ref)
            .map(|k| k.elem_count() * k.dtype().size_in_bytes())
            .sum();
        let value_mem: usize = self
            .values
            .iter()
            .filter_map(Option::as_ref)
            .map(|v| v.elem_count() * v.dtype().size_in_bytes())
            .sum();
        key_mem + value_mem
    }

    /// Trim the cache to keep only the last `max_seq_len` tokens.
    ///
    /// Useful for memory-constrained scenarios with long sequences.
    /// Returns `Ok(true)` if trimming occurred, `Ok(false)` if no
    /// trimming was needed.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn trim_to(&mut self, max_seq_len: usize) -> Result<bool> {
        let current_len = self.seq_len()?;
        if current_len <= max_seq_len {
            return Ok(false);
        }

        let trim_start = current_len - max_seq_len;

        for tensor in self.keys.iter_mut().flatten() {
            *tensor = tensor.narrow(2, trim_start, max_seq_len)?;
        }
        for tensor in self.values.iter_mut().flatten() {
            *tensor = tensor.narrow(2, trim_start, max_seq_len)?;
        }
        Ok(true)
    }

    /// Check if cache exceeds memory limit and trim if needed.
    ///
    /// Trims to ~75% of current length if memory limit is exceeded.
    /// Returns `Ok(true)` if trimming occurred.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn enforce_memory_limit(&mut self, max_bytes: usize) -> Result<bool> {
        let current = self.memory_usage();
        if current > max_bytes {
            let current_len = self.seq_len()?;
            let target_len = (current_len * 3) / 4;
            if target_len > 0 {
                self.trim_to(target_len)?;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new(0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn new_cache() {
        let cache = KVCache::new(32);
        assert_eq!(cache.n_layers(), 32);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len().unwrap(), 0);
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn clear_cache() {
        let mut cache = KVCache::new(4);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn layer_mut_valid() {
        let mut cache = KVCache::new(4);
        let (k, v) = cache.layer_mut(2).unwrap();
        assert!(k.is_none());
        assert!(v.is_none());
    }

    #[test]
    fn layer_mut_out_of_range() {
        let mut cache = KVCache::new(4);
        assert!(cache.layer_mut(10).is_err());
    }

    #[test]
    fn default_cache() {
        let cache = KVCache::default();
        assert_eq!(cache.n_layers(), 0);
        assert!(cache.is_empty());
    }
}
