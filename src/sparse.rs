// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared sparse-feature types used by both CLT and SAE modules.
//!
//! [`FeatureId`] is a marker trait for feature identifiers, and
//! [`SparseActivations`] stores the non-zero activations in descending
//! magnitude order.  These live here so that the public type identity is
//! stable regardless of which feature flags are enabled.

/// Marker trait for feature identifiers in sparse activation vectors.
///
/// Implemented by `CltFeatureId` (CLT features with layer + index,
/// requires `clt` feature) and `SaeFeatureId` (SAE features with index
/// only, requires `sae` feature).
pub trait FeatureId:
    std::fmt::Debug
    + Clone
    + Copy
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + std::hash::Hash
    + std::fmt::Display
{
}

/// Sparse representation of feature activations.
///
/// Only features with non-zero activation are stored,
/// sorted by activation magnitude in descending order.
///
/// Generic over the feature identifier type `F`:
/// - `CltFeatureId` for CLT features (layer + index, requires `clt` feature)
/// - `SaeFeatureId` for SAE features (index only, requires `sae` feature)
// EXHAUSTIVE: a newtype over the activation map, with nothing else to carry.
#[derive(Debug, Clone)]
pub struct SparseActivations<F: FeatureId> {
    /// Active features with their activation magnitudes, sorted descending.
    pub features: Vec<(F, f32)>,
}

impl<F: FeatureId> SparseActivations<F> {
    /// Number of active features.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether no features are active.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Truncate to the top-k most active features.
    pub fn truncate(&mut self, k: usize) {
        self.features.truncate(k);
    }
}
