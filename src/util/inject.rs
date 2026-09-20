// SPDX-License-Identifier: MIT OR Apache-2.0

//! Building intervention payloads that act at a single sequence position.
//!
//! [`Intervention::Add`](crate::hooks::Intervention::Add) broadcasts over every
//! position, so targeting one position means handing it a payload that is zero
//! everywhere else. Three call sites needed that shape: contrastive steering,
//! `CLT` feature injection and `SAE` feature injection.
//!
//! This module is **ungated** on purpose. `steering` is compiled only under
//! `any(transformer, rwkv, diffusion)`, while `clt` and `sae` each compile
//! standalone, so the shared implementation cannot live in `steering` without
//! breaking those lanes. The public entry point remains
//! `steering::contrastive::position_delta`, which delegates here.

use candle_core::Tensor;

use crate::error::{MIError, Result};

/// Build a `[1, seq_len, hidden]` payload carrying `direction` at `position`
/// and zeros elsewhere.
///
/// # Shapes
/// - `direction`: `[hidden]`
/// - returns: `[1, seq_len, hidden]`, same dtype as `direction`
///
/// # Errors
///
/// Returns [`MIError::Config`] if `position >= seq_len`, or if `direction` is
/// not 1-D `[hidden]`.
/// Returns [`MIError::Model`] on tensor construction failure.
pub fn position_delta(direction: &Tensor, position: usize, seq_len: usize) -> Result<Tensor> {
    if position >= seq_len {
        return Err(MIError::Config(format!(
            "position_delta: position {position} >= seq_len {seq_len}"
        )));
    }
    let dims = direction.dims();
    if dims.len() != 1 {
        return Err(MIError::Config(format!(
            "position_delta: direction must be 1-D [hidden]; got shape {dims:?}"
        )));
    }

    // Build a [seq_len, hidden] tensor by stacking per-position rows: the
    // chosen position holds `direction`, all others hold zeros_like(direction).
    let zero_row = direction.zeros_like()?;
    // BORROW: rows is a Vec<&Tensor> for Tensor::stack; entries borrow either
    // `direction` (for the chosen position) or `zero_row` (for all others).
    let rows: Vec<&Tensor> = (0..seq_len)
        .map(|i| if i == position { direction } else { &zero_row })
        .collect();
    let stacked = Tensor::stack(&rows, 0)?;
    // Add batch dim -> [1, seq_len, hidden].
    Ok(stacked.unsqueeze(0)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn places_the_direction_at_the_requested_position() {
        let dev = Device::Cpu;
        let dir = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let out = position_delta(&dir, 1, 3).unwrap();

        assert_eq!(out.dims(), &[1, 3, 3]);
        let v: Vec<Vec<f32>> = out.squeeze(0).unwrap().to_vec2().unwrap();
        assert_eq!(v[0], vec![0.0, 0.0, 0.0]);
        assert_eq!(v[1], vec![1.0, 2.0, 3.0]);
        assert_eq!(v[2], vec![0.0, 0.0, 0.0]);
    }

    /// The bound the three former copies did not check. At `position == seq_len`
    /// the old narrow/cat form produced a `[1, seq_len + 1, hidden]` payload
    /// instead of failing, which then broadcast wrongly at the hook point.
    #[test]
    fn rejects_a_position_past_the_sequence() {
        let dev = Device::Cpu;
        let dir = Tensor::new(&[1.0f32, 2.0], &dev).unwrap();
        assert!(position_delta(&dir, 3, 3).is_err());
        assert!(position_delta(&dir, 4, 3).is_err());
    }

    #[test]
    fn rejects_a_non_1d_direction() {
        let dev = Device::Cpu;
        let dir = Tensor::zeros((2, 2), DType::F32, &dev).unwrap();
        assert!(position_delta(&dir, 0, 3).is_err());
    }

    #[test]
    fn preserves_the_direction_dtype() {
        let dev = Device::Cpu;
        let dir = Tensor::zeros(4, DType::F64, &dev).unwrap();
        let out = position_delta(&dir, 0, 2).unwrap();
        assert_eq!(out.dtype(), DType::F64);
    }
}
