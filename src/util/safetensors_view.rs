// SPDX-License-Identifier: MIT OR Apache-2.0

//! Building a candle [`Tensor`] from a `safetensors` tensor view.
//!
//! Shared by the `clt` and `sae` loaders, which are independent features, so
//! this lives in the ungated `util` module rather than in either of them.

use candle_core::{DType, Device, Tensor};

use crate::error::{MIError, Result};

/// Convert a `safetensors` view into a candle [`Tensor`] on `device`.
///
/// `kind` names the caller (`"CLT"`, `"SAE"`) so the unsupported-dtype message
/// stays as specific as it was when each loader carried its own copy.
///
/// # Shapes
/// - returns: the view's own shape, unchanged.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the view's dtype is not `BF16`, `F16` or
/// `F32`.
/// Returns [`MIError::Model`] if the raw buffer cannot be interpreted as a
/// tensor of that shape and dtype.
pub fn tensor_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    device: &Device,
    kind: &str,
) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    #[allow(clippy::wildcard_enum_match_arm)]
    // EXHAUSTIVE: safetensors exposes many dtypes; these loaders only use float types
    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::F32 => DType::F32,
        other => {
            return Err(MIError::Config(format!(
                "unsupported {kind} tensor dtype: {other:?}"
            )));
        }
    };
    Ok(Tensor::from_raw_buffer(view.data(), dtype, &shape, device)?)
}
