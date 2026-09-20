// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recurrent feedback types for anacrousis experiments.
//!
//! Anacrousis (ἀνάκρουσις — "the upbeat before the first full bar") re-runs
//! a subset of transformer layers ("commitment layers") with optional feedback
//! injection, giving the model extra depth to sustain planning signals through
//! generation.
//!
//! Two modes are supported:
//!
//! - **Prefill-only** (`sustained: false`): the recurrent multi-pass applies
//!   at every step (since candle-mi recomputes from scratch without KV cache),
//!   but feedback is injected only at the original prompt positions.
//!
//! - **Sustained** (`sustained: true`): in addition to the original feedback
//!   positions, feedback is also injected at the current last token at each
//!   autoregressive step — the transformer analog of the DRC's per-tick
//!   recurrence (Taufeeque et al., 2024).
//!
//! The `depth` parameter (default 2) controls how many times the recurrent
//! layer block is executed. Higher depths give the model more iterations to
//! propagate planning signals, at the cost of proportionally more compute.

use candle_core::Tensor;

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// RecurrentFeedbackEntry
// ---------------------------------------------------------------------------

/// A single feedback injection between recurrent passes.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RecurrentFeedbackEntry {
    /// Token position in the sequence to inject feedback at.
    pub position: usize,
    /// Feedback direction vector.
    ///
    /// # Shapes
    /// - `[d_model]`
    pub vector: Tensor,
    /// Amplification strength.
    pub strength: f32,
}

// ---------------------------------------------------------------------------
// RecurrentPassSpec
// ---------------------------------------------------------------------------

/// Specification for a recurrent multi-pass forward through a layer block.
///
/// The recurrence re-runs layers `loop_start..=loop_end` a total of `depth`
/// times, with optional feedback injected into the hidden state between
/// passes.
///
/// # Without feedback
///
/// Each subsequent pass receives the previous pass's output (true
/// recurrence — extra depth).
///
/// # With feedback
///
/// Each subsequent pass resets to the saved pre-loop hidden state plus
/// feedback vectors: `hidden[position] += strength * vector`.
/// This means every recurrent pass sees the same clean input with the
/// nudge applied — the layers process `H₀ + nudge` rather than
/// iterating on their own output, which would cause degeneration.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct RecurrentPassSpec {
    /// First layer of the recurrent block (inclusive).
    pub loop_start: usize,
    /// Last layer of the recurrent block (inclusive).
    pub loop_end: usize,
    /// Feedback vectors to inject between passes.
    ///
    /// When present, each recurrent pass resets to the saved pre-loop state
    /// and injects these vectors before re-running the loop layers.
    /// If empty, each pass receives the previous pass's output unmodified
    /// (pure depth increase).
    pub feedback: Vec<RecurrentFeedbackEntry>,
    /// If true, also inject feedback at the current last token position
    /// during each autoregressive generation step (sustained recurrence).
    ///
    /// If false, feedback is only injected at the original prompt positions
    /// (prefill-only recurrence).
    pub sustained: bool,
    /// Number of times to run the recurrent layer block.
    ///
    /// Must be at least 1 (a single pass, no recurrence). The default is 2
    /// (one initial pass plus one recurrent pass with feedback injection).
    pub depth: usize,
}

impl RecurrentPassSpec {
    /// Create a spec with no feedback (pure double-pass, depth 2).
    #[must_use]
    pub const fn no_feedback(loop_start: usize, loop_end: usize) -> Self {
        Self {
            loop_start,
            loop_end,
            feedback: Vec::new(),
            sustained: false,
            depth: 2,
        }
    }

    /// Set the sustained flag (builder pattern).
    #[must_use]
    pub const fn with_sustained(mut self, sustained: bool) -> Self {
        self.sustained = sustained;
        self
    }

    /// Set the recurrence depth (builder pattern).
    ///
    /// `depth` is the total number of times the recurrent layer block is
    /// executed. The default is 2 (one initial pass plus one recurrent
    /// pass). A depth of 1 means no recurrence (single pass).
    #[must_use]
    pub const fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// Add a feedback entry.
    pub fn add_feedback(&mut self, position: usize, vector: Tensor, strength: f32) {
        self.feedback.push(RecurrentFeedbackEntry {
            position,
            vector,
            strength,
        });
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if the layer range is invalid,
    /// feedback positions exceed sequence length, or feedback vectors
    /// have the wrong dimension.
    pub fn validate(&self, n_layers: usize, seq_len: usize, d_model: usize) -> Result<()> {
        if self.depth == 0 {
            return Err(MIError::Intervention("depth must be >= 1 (got 0)".into()));
        }
        if self.loop_start > self.loop_end {
            return Err(MIError::Intervention(format!(
                "loop_start ({}) > loop_end ({})",
                self.loop_start, self.loop_end
            )));
        }
        if self.loop_end >= n_layers {
            return Err(MIError::Intervention(format!(
                "loop_end ({}) >= n_layers ({})",
                self.loop_end, n_layers
            )));
        }
        for entry in &self.feedback {
            if entry.position >= seq_len {
                return Err(MIError::Intervention(format!(
                    "feedback position {} >= seq_len {}",
                    entry.position, seq_len
                )));
            }
            let vec_dim = entry.vector.dim(0).map_err(|e| {
                MIError::Intervention(format!("feedback vector dimension error: {e}"))
            })?;
            if vec_dim != d_model {
                return Err(MIError::Intervention(format!(
                    "feedback vector dim {vec_dim} != d_model {d_model}"
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_feedback_builder() {
        let spec = RecurrentPassSpec::no_feedback(14, 15);
        assert_eq!(spec.loop_start, 14);
        assert_eq!(spec.loop_end, 15);
        assert!(spec.feedback.is_empty());
        assert!(!spec.sustained);
    }

    #[test]
    fn with_sustained_builder() {
        let spec = RecurrentPassSpec::no_feedback(14, 15).with_sustained(true);
        assert!(spec.sustained);
    }

    #[test]
    fn add_feedback_entry() {
        let mut spec = RecurrentPassSpec::no_feedback(14, 15);
        let vec = Tensor::zeros(2048, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        spec.add_feedback(5, vec, 2.0);
        assert_eq!(spec.feedback.len(), 1);
        assert_eq!(spec.feedback[0].position, 5);
        assert!((spec.feedback[0].strength - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn validate_good_spec() {
        let spec = RecurrentPassSpec::no_feedback(14, 15);
        assert!(spec.validate(16, 10, 2048).is_ok());
    }

    #[test]
    fn validate_start_gt_end() {
        let spec = RecurrentPassSpec::no_feedback(15, 14);
        let err = spec.validate(16, 10, 2048);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("loop_start"));
    }

    #[test]
    fn validate_end_out_of_range() {
        let spec = RecurrentPassSpec::no_feedback(14, 16);
        let err = spec.validate(16, 10, 2048);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("loop_end"));
    }

    #[test]
    fn validate_feedback_position_out_of_range() {
        let mut spec = RecurrentPassSpec::no_feedback(14, 15);
        let vec = Tensor::zeros(2048, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        spec.add_feedback(20, vec, 1.0);
        let err = spec.validate(16, 10, 2048);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("position"));
    }

    #[test]
    fn default_depth_is_two() {
        let spec = RecurrentPassSpec::no_feedback(14, 15);
        assert_eq!(spec.depth, 2);
    }

    #[test]
    fn with_depth_builder() {
        let spec = RecurrentPassSpec::no_feedback(14, 15).with_depth(4);
        assert_eq!(spec.depth, 4);
    }

    #[test]
    fn validate_depth_zero() {
        let spec = RecurrentPassSpec::no_feedback(14, 15).with_depth(0);
        let err = spec.validate(16, 10, 2048);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("depth"));
    }

    #[test]
    fn validate_depth_one() {
        let spec = RecurrentPassSpec::no_feedback(14, 15).with_depth(1);
        assert!(spec.validate(16, 10, 2048).is_ok());
    }

    #[test]
    fn validate_feedback_wrong_dim() {
        let mut spec = RecurrentPassSpec::no_feedback(14, 15);
        let vec = Tensor::zeros(1024, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        spec.add_feedback(5, vec, 1.0);
        let err = spec.validate(16, 10, 2048);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("d_model"));
    }
}
