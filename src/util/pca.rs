// SPDX-License-Identifier: MIT OR Apache-2.0

//! PCA via power iteration with deflation.
//!
//! Provides [`pca_top_k`] for computing the top principal components of a
//! data matrix using only candle tensor ops (`matmul`, arithmetic,
//! normalization).  This runs transparently on CPU or GPU with zero
//! host↔device transfers.
//!
//! The algorithm works on the **kernel matrix** (`X @ X^T`), which is
//! efficient when the number of samples `n` is much smaller than the
//! number of features `d` (e.g., 150 × 2304 for the character-count
//! helix experiment).

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a PCA decomposition via power iteration.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal component directions, shape `[k, n_features]`.
    ///
    /// Each row is a unit-length direction in feature space, ordered by
    /// decreasing eigenvalue.
    pub components: Tensor,

    /// Eigenvalues of the kernel matrix, one per component.
    pub eigenvalues: Vec<f32>,

    /// Fraction of total variance explained by each component.
    ///
    /// Values sum to at most 1.0.  The total variance is the trace of the
    /// (centered) kernel matrix.
    pub explained_variance_ratio: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the top `k` principal components of `matrix` via power iteration
/// with deflation on the kernel matrix.
///
/// # Shapes
///
/// - `matrix`: `[n_samples, n_features]` — each row is one observation
/// - returns [`PcaResult`] with `components` of shape `[k, n_features]`
///
/// # Algorithm
///
/// 1. Center the matrix (subtract column means).
/// 2. Build the kernel `K = X @ X^T` — shape `[n, n]`.
/// 3. For each of `k` components: power-iterate on `K`, extract the
///    eigenvalue, recover the PC direction in feature space via
///    `w = X^T @ v / ‖X^T @ v‖`, then deflate `K`.
/// 4. Explained variance ratios are `λ_i / trace(K_original)`.
///
/// # Errors
///
/// Returns [`MIError::Model`](crate::MIError::Model) if any tensor operation fails (shape
/// mismatch, device error, etc.).
pub fn pca_top_k(matrix: &Tensor, k: usize, n_iter: usize) -> Result<PcaResult> {
    let device = matrix.device();
    let (n, _d) = matrix.dims2()?;

    // 1. Center: subtract column means
    // CAST: usize → f64, n is small (≤ 150 for helix); exact in f64 mantissa
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    let mean = (matrix.sum(0)? / (n as f64))?; // [d]
    let centered = matrix.broadcast_sub(&mean)?; // [n, d]

    // 2. Kernel matrix K = X @ X^T  →  [n, n]
    // CONTIGUOUS: transpose produces non-unit strides; matmul requires contiguous layout
    let centered_t = centered.t()?.contiguous()?; // [d, n]
    let k_original = centered.matmul(&centered_t)?; // [n, n]
    let mut k_mat = k_original.copy()?;

    // Total variance = trace(K) — sum of diagonal elements
    let trace = trace_2d(&k_original, n)?;

    let mut eigenvalues = Vec::with_capacity(k);
    let mut components = Vec::with_capacity(k); // in feature space [d]

    // 3. Power iteration + deflation
    for _ in 0..k {
        let v = power_iterate(&k_mat, n, n_iter, device)?; // [n]

        // Eigenvalue: λ = v^T K v
        let kv = k_mat.matmul(&v.unsqueeze(1)?)?; // [n, 1]
        let lambda_t = v.unsqueeze(0)?.matmul(&kv)?; // [1, 1]
        // PROMOTE: extract eigenvalue as F32 for precision
        let lambda: f32 = lambda_t
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_scalar()?;

        // Recover PC direction in feature space: w = X^T @ v, normalise
        let w = centered_t.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // [d]
        let w_norm = w.sqr()?.sum_all()?.sqrt()?;
        let w_unit = w.broadcast_div(&w_norm)?; // [d]

        // Deflate: K ← K − λ v v^T
        let vvt = v.unsqueeze(1)?.matmul(&v.unsqueeze(0)?)?; // [n, n]
        let lambda_f64 = f64::from(lambda);
        k_mat = (k_mat - (vvt * lambda_f64)?)?;

        eigenvalues.push(lambda);
        components.push(w_unit);
    }

    // 4. Explained variance ratios
    let explained_variance_ratio: Vec<f32> = eigenvalues
        .iter()
        .map(|&lam| if trace > 0.0 { lam / trace } else { 0.0 })
        .collect();

    // Stack components: [k, d]
    let comp_refs: Vec<&Tensor> = components.iter().collect();
    let stacked = Tensor::stack(&comp_refs, 0)?;

    Ok(PcaResult {
        components: stacked,
        eigenvalues,
        explained_variance_ratio,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run `n_iter` rounds of power iteration on `mat` to find the dominant
/// eigenvector.  Returns a unit-length vector of shape `[n]`.
fn power_iterate(mat: &Tensor, n: usize, n_iter: usize, device: &Device) -> Result<Tensor> {
    // Initialise with a random unit vector
    let mut v = Tensor::randn(0.0_f32, 1.0, (n,), device)?;
    let v_norm = v.sqr()?.sum_all()?.sqrt()?;
    v = v.broadcast_div(&v_norm)?;

    for _ in 0..n_iter {
        // v ← K v / ‖K v‖
        let kv = mat.matmul(&v.unsqueeze(1)?)?.squeeze(1)?; // [n]
        let norm = kv.sqr()?.sum_all()?.sqrt()?;
        v = kv.broadcast_div(&norm)?;
    }

    Ok(v)
}

/// Compute the trace of a 2-D square tensor as an `f32` by extracting
/// diagonal elements one by one.
///
/// Candle has no `diagonal()` method (still true as of 0.11), so we narrow
/// each row to a single element and sum.
fn trace_2d(mat: &Tensor, n: usize) -> Result<f32> {
    let mut sum = 0.0_f32;
    for i in 0..n {
        // Extract element [i, i]: narrow row i, then narrow col i
        // PROMOTE: extract as F32 for accumulation precision
        let val: f32 = mat
            .narrow(0, i, 1)?
            .narrow(1, i, 1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_scalar()?;
        sum += val;
    }
    Ok(sum)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: PCA on a tiny matrix recovers the dominant direction.
    #[test]
    fn pca_smoke() -> Result<()> {
        // 5 points along (1, 0) + small noise in dim 2
        let data = Tensor::new(
            &[
                [1.0_f32, 0.0],
                [2.0, 0.1],
                [3.0, -0.1],
                [4.0, 0.05],
                [5.0, -0.05],
            ],
            &Device::Cpu,
        )?;

        let result = pca_top_k(&data, 2, 50)?;

        // PC1 should capture most variance (> 99%)
        assert!(
            result.explained_variance_ratio[0] > 0.99,
            "PC1 variance ratio {:.4} should be > 0.99",
            result.explained_variance_ratio[0],
        );

        // PC1 direction should be close to (1, 0) or (-1, 0)
        let pc1: Vec<f32> = result.components.get(0)?.to_vec1()?;
        assert!(
            pc1[0].abs() > 0.99,
            "PC1[0] = {:.4}, expected close to ±1.0",
            pc1[0],
        );

        // Two components should sum to ~1.0
        let total: f32 = result.explained_variance_ratio.iter().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "Total variance {total:.4} should be ~1.0",
        );

        Ok(())
    }
}
