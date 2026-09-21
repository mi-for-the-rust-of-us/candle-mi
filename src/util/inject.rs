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

    // Assemble from at most three blocks rather than stacking `seq_len` rows:
    // the zeros before the position, the row itself, and the zeros after. The
    // stacking form built one tensor reference per position, which is fine at
    // `block_size = 60` and wasteful at a transformer's sequence length, where
    // this is reached through `CLT`/`SAE` feature injection.
    let hidden = *dims.first().unwrap_or(&0);
    let row = direction.reshape((1, 1, hidden))?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if position > 0 {
        parts.push(Tensor::zeros(
            (1, position, hidden),
            direction.dtype(),
            direction.device(),
        )?);
    }
    parts.push(row);
    let after = seq_len - position - 1;
    if after > 0 {
        parts.push(Tensor::zeros(
            (1, after, hidden),
            direction.dtype(),
            direction.device(),
        )?);
    }
    // Single-block case: no concatenation needed.
    if parts.len() == 1 {
        // INDEX: len checked to be 1 on the line above.
        #[allow(clippy::indexing_slicing)]
        return Ok(parts.swap_remove(0));
    }
    Ok(Tensor::cat(&parts, 1)?)
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

    /// The three-block assembly has a branch per edge, so both ends and the
    /// degenerate single-position case need exercising, not just the middle.
    #[test]
    fn places_the_direction_at_either_end_and_in_a_length_one_sequence() {
        let dev = Device::Cpu;
        let dir = Tensor::new(&[1.0f32, 2.0], &dev).unwrap();

        // First position: no leading zero block.
        let v: Vec<Vec<f32>> = position_delta(&dir, 0, 3)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2()
            .unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0], vec![0.0, 0.0], vec![0.0, 0.0]]);

        // Last position: no trailing zero block.
        let v: Vec<Vec<f32>> = position_delta(&dir, 2, 3)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2()
            .unwrap();
        assert_eq!(v, vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 2.0]]);

        // seq_len == 1: the row alone, no concatenation at all.
        let out = position_delta(&dir, 0, 1).unwrap();
        assert_eq!(out.dims(), &[1, 1, 2]);
        let v: Vec<Vec<f32>> = out.squeeze(0).unwrap().to_vec2().unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0]]);
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
