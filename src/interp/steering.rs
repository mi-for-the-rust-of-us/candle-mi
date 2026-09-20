// SPDX-License-Identifier: MIT OR Apache-2.0

//! Steering calibration and dose-response curves.
//!
//! Utilities for measuring baseline attention levels and calibrating
//! steering targets for dose-response experiments.
//!
//! This module provides the data structures and analysis methods.
//! The actual forward passes and attention extraction are performed
//! by the caller using [`crate::MIBackend`] and [`crate::HookSpec`].

/// Standard dose levels for dose-response experiments.
///
/// These are multipliers for the baseline attention level:
/// - `0.5`: Reduce attention by half
/// - `1.0`: Baseline (no change)
/// - `2.0`: Double the attention
/// - `3.0`: Triple the attention
/// - `4.0`: Quadruple the attention
/// - `6.0`: Six times baseline
pub const DOSE_LEVELS: &[f32] = &[0.5, 1.0, 2.0, 3.0, 4.0, 6.0];

/// Calibration data for steering experiments.
///
/// Stores measured baseline attention levels for two conditions (source and
/// target) and provides methods for computing scale factors and dose levels.
///
/// # Example
///
/// ```
/// use candle_mi::interp::steering::SteeringCalibration;
///
/// let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);
/// let scale = cal.scale_factor_to_source();
/// assert!((scale - 3.6).abs() < 1e-5);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[must_use]
pub struct SteeringCalibration {
    /// Mean attention for source condition samples.
    pub source_baseline: f32,
    /// Mean attention for target condition samples.
    pub target_baseline: f32,
    /// Recommended target attention level (defaults to source baseline).
    pub recommended_target: f32,
    /// Ratio of source to target attention (`source / target`).
    pub attention_ratio: f32,
    /// Layer index used for calibration.
    pub layer: usize,
    /// Number of source condition samples.
    pub n_source_samples: usize,
    /// Number of target condition samples.
    pub n_target_samples: usize,
}

impl SteeringCalibration {
    /// Create a new calibration result.
    ///
    /// `source_baseline` is the mean attention for the source condition
    /// (e.g., Python doctests). `target_baseline` is the mean attention
    /// for the target condition (e.g., Rust tests). The recommended
    /// target defaults to the source baseline.
    pub fn new(
        source_baseline: f32,
        target_baseline: f32,
        layer: usize,
        n_source_samples: usize,
        n_target_samples: usize,
    ) -> Self {
        let attention_ratio = if target_baseline > 1e-10 {
            source_baseline / target_baseline
        } else {
            0.0
        };

        Self {
            source_baseline,
            target_baseline,
            recommended_target: source_baseline,
            attention_ratio,
            layer,
            n_source_samples,
            n_target_samples,
        }
    }

    /// Set a custom recommended target.
    pub const fn with_target(mut self, target: f32) -> Self {
        self.recommended_target = target;
        self
    }

    /// Calculate the scale factor needed to boost target attention to a value.
    #[must_use]
    pub fn scale_factor_for_target(&self, target: f32) -> f32 {
        if self.target_baseline > 1e-10 {
            target / self.target_baseline
        } else {
            1.0
        }
    }

    /// Calculate the scale factor to boost target to source level.
    #[must_use]
    pub fn scale_factor_to_source(&self) -> f32 {
        self.scale_factor_for_target(self.source_baseline)
    }

    /// Get dose levels as absolute attention values (based on target baseline).
    #[must_use]
    pub fn dose_levels_absolute(&self) -> Vec<(f32, f32)> {
        DOSE_LEVELS
            .iter()
            .map(|&scale| (scale, self.target_baseline * scale))
            .collect()
    }
}

/// A single data point on a dose-response curve.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DoseResponsePoint {
    /// Scale factor applied.
    pub scale_factor: f32,
    /// Resulting attention level (after steering).
    pub attention_level: f32,
    /// KL divergence from baseline.
    pub kl_divergence: f32,
}

/// Full dose-response curve for a single sample.
///
/// Tracks how attention and KL divergence change as the steering
/// scale factor varies from low (dampening) to high (amplifying).
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct DoseResponseCurve {
    /// Sample identifier.
    pub sample_id: String,
    /// Condition label (e.g., "python", "rust").
    pub condition: String,
    /// Layer used for steering.
    pub layer: usize,
    /// Baseline attention (no steering).
    pub baseline_attention: f32,
    /// Data points on the curve.
    pub points: Vec<DoseResponsePoint>,
}

impl DoseResponseCurve {
    /// Create a new dose-response curve.
    #[must_use]
    pub const fn new(
        sample_id: String,
        condition: String,
        layer: usize,
        baseline_attention: f32,
    ) -> Self {
        Self {
            sample_id,
            condition,
            layer,
            baseline_attention,
            points: Vec::new(),
        }
    }

    /// Add a data point to the curve.
    pub fn add_point(&mut self, scale_factor: f32, attention_level: f32, kl_divergence: f32) {
        self.points.push(DoseResponsePoint {
            scale_factor,
            attention_level,
            kl_divergence,
        });
    }

    /// Find the scale factor that achieves target attention level.
    ///
    /// Uses linear interpolation between data points. Returns `None`
    /// if the target is outside the measured range.
    #[allow(clippy::indexing_slicing)] // windows(2) guarantees exactly 2 elements
    #[must_use]
    pub fn scale_for_target(&self, target: f32) -> Option<f32> {
        for window in self.points.windows(2) {
            let (p1, p2) = (&window[0], &window[1]);
            if (p1.attention_level <= target && target <= p2.attention_level)
                || (p2.attention_level <= target && target <= p1.attention_level)
            {
                let denom = p2.attention_level - p1.attention_level;
                if denom.abs() < 1e-10 {
                    continue;
                }
                let t = (target - p1.attention_level) / denom;
                return Some(t.mul_add(p2.scale_factor - p1.scale_factor, p1.scale_factor));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn calibration_new() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);

        assert!((cal.source_baseline - 0.09).abs() < 1e-6);
        assert!((cal.target_baseline - 0.025).abs() < 1e-6);
        assert!((cal.attention_ratio - 3.6).abs() < 1e-6);
        assert_eq!(cal.layer, 16);
    }

    #[test]
    fn scale_factor_calculation() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);

        let scale = cal.scale_factor_to_source();
        assert!((scale - 3.6).abs() < 1e-6);

        let scale = cal.scale_factor_for_target(0.05);
        assert!((scale - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dose_levels_count() {
        assert_eq!(DOSE_LEVELS.len(), 6);
        assert!((DOSE_LEVELS[0] - 0.5).abs() < 1e-6);
        assert!((DOSE_LEVELS[1] - 1.0).abs() < 1e-6);
        assert!((DOSE_LEVELS[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dose_response_curve_interpolation() {
        let mut curve =
            DoseResponseCurve::new("test_sample".to_string(), "rust".to_string(), 16, 0.025);

        curve.add_point(1.0, 0.025, 0.0);
        curve.add_point(2.0, 0.050, 0.01);
        curve.add_point(4.0, 0.100, 0.05);

        let scale = curve.scale_for_target(0.075);
        assert!(scale.is_some());
        let scale = scale.unwrap();
        assert!(scale > 2.0 && scale < 4.0);
    }

    #[test]
    fn dose_response_out_of_range() {
        let mut curve = DoseResponseCurve::new("test".to_string(), "rust".to_string(), 16, 0.025);
        curve.add_point(1.0, 0.025, 0.0);
        curve.add_point(2.0, 0.050, 0.01);

        assert!(curve.scale_for_target(0.200).is_none());
    }

    #[test]
    fn calibration_with_custom_target() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10).with_target(0.05);
        assert!((cal.recommended_target - 0.05).abs() < 1e-6);
    }

    #[test]
    fn dose_levels_absolute() {
        let cal = SteeringCalibration::new(0.09, 0.025, 16, 10, 10);
        let levels = cal.dose_levels_absolute();
        assert_eq!(levels.len(), 6);
        // First: 0.5 * 0.025 = 0.0125
        assert!((levels.first().unwrap().1 - 0.0125).abs() < 1e-6);
    }
}
