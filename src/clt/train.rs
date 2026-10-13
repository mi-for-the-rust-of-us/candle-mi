// SPDX-License-Identifier: MIT OR Apache-2.0

//! A **trainable** cross-layer transcoder: the model and its loss, not a training loop.
//!
//! [`CrossLayerTranscoder`](super::CrossLayerTranscoder) loads a CLT someone else trained; this module builds one
//! whose parameters are `candle` [`Var`]s, so a caller's loop can fit it with `optim::AdamW` (`training` feature; the loop, the data
//! and the schedule stay with the caller, as everywhere in candle-mi). Once trained,
//! [`export_circuit_tracer`](TrainableClt::export_circuit_tracer) writes the circuit-tracer layout that
//! [`CrossLayerTranscoder`](super::CrossLayerTranscoder) reads, so a CLT trained here is analysed with the same
//! `encode` / `inject` / attribution code as a published one.
//!
//! # Architecture
//!
//! `L` layers, model width `d`, `f` features per layer, `k` active per (position, layer):
//!
//! ```text
//! encode:  a[l]   = TopK_k( ReLU( x[l] · W_enc[l] + b_enc[l] ) )                 x[l]: the layer-l input
//! decode:  y[t]   = b_dec[t] + Σ_{l ≤ t} a[l] · W_dec[l → t]                        y[t]: the layer-t output
//! loss:    mean over (positions, layers, d) of (y - target)²
//! ```
//!
//! The decoder is strictly triangular: a source layer writes only to itself and the layers above it. This is the
//! reference of askesis (`reference/scripts/clt.py`), reproduced to its stated bar (`1e-4`) by
//! `tests/validate_clt_train.rs` on a fixture that script's own class produced. Which activations feed it is the
//! caller's choice (the circuit-tracer convention, and candle-mi's `encode`, read the residual stream before the MLP and
//! reconstruct the MLP output).
//!
//! **`TopK` and ties.** The k-th largest activation is found by k successive row maxima (`candle`'s sort kernel is limited
//! to rows of 1,024 columns on `CUDA`, and a CLT layer has thousands of features). Values equal to a maximum are removed
//! together, so exact ties among positive activations keep more than k features where `torch.topk` keeps exactly k;
//! with real-valued activations this has probability zero. Rows with fewer than k positive activations keep them all,
//! as `torch.topk` does (the rest are zeros either way).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{D, DType, Device, Tensor, Var};

use crate::error::{MIError, Result};

/// The shape of a [`TrainableClt`]. Built with [`new`](Self::new); fields stay readable and assignable.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainableCltConfig {
    /// Number of layers the CLT reads and writes (the base model's layer count).
    pub n_layers: usize,
    /// Width of the activations read and written (the base model's `d_model`).
    pub d_model: usize,
    /// Features per layer.
    pub n_features: usize,
    /// Features kept active per (position, layer) by the `TopK` sparsifier.
    pub top_k: usize,
}

impl TrainableCltConfig {
    /// A configuration; validated when a [`TrainableClt`] is built from it.
    #[must_use]
    pub const fn new(n_layers: usize, d_model: usize, n_features: usize, top_k: usize) -> Self {
        Self {
            n_layers,
            d_model,
            n_features,
            top_k,
        }
    }

    /// Reject degenerate shapes before any tensor is allocated.
    fn validate(&self) -> Result<Self> {
        if self.n_layers == 0 || self.d_model == 0 || self.n_features == 0 {
            return Err(MIError::Config(format!(
                "trainable CLT dimensions must be positive (n_layers {}, d_model {}, n_features {})",
                self.n_layers, self.d_model, self.n_features
            )));
        }
        if self.top_k == 0 || self.top_k > self.n_features {
            return Err(MIError::Config(format!(
                "top_k {} out of range (1..={})",
                self.top_k, self.n_features
            )));
        }
        Ok(*self)
    }
}

/// A cross-layer transcoder whose parameters are trainable [`Var`]s.
///
/// See the [module documentation](self) for the architecture.
pub struct TrainableClt {
    /// Shape of the transcoder.
    config: TrainableCltConfig,
    /// Encoder weights, one `[d_model, n_features]` matrix per layer.
    ///
    /// # Shapes
    /// - `w_enc`: `[n_layers, d_model, n_features]`
    w_enc: Var,
    /// Encoder biases.
    ///
    /// # Shapes
    /// - `b_enc`: `[n_layers, n_features]`
    b_enc: Var,
    /// Decoder weights, one per SOURCE layer `l`, writing layers `l..n_layers`: output column `(t - l) * d_model + j`
    /// is component `j` of target layer `t`.
    ///
    /// # Shapes
    /// - `w_dec[l]`: `[n_features, (n_layers - l) * d_model]`
    w_dec: Vec<Var>,
    /// Decoder biases, one per target layer.
    ///
    /// # Shapes
    /// - `b_dec`: `[n_layers, d_model]`
    b_dec: Var,
}

/// `n` uniform samples in `[-bound, bound)` from a seeded xorshift: device-independent, so a seed means one
/// initialisation on every backend.
fn uniform(n: usize, bound: f32, state: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            // CAST: u64 → f32, the top 24 bits are exactly representable; the value is a sample in [0, 1)
            #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
            let unit = (*state >> 40) as f32 / 16_777_216.0;
            (2.0f32.mul_add(unit, -1.0)) * bound
        })
        .collect()
}

impl TrainableClt {
    /// A freshly initialised transcoder: weights uniform in `±1/sqrt(n_features)` (`PyTorch`'s default for these
    /// shapes in the reference), biases zero.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a dimension is zero or `top_k` exceeds `n_features`.
    /// Returns [`MIError::Model`] on tensor allocation failure.
    pub fn new(config: &TrainableCltConfig, seed: u64, device: &Device) -> Result<Self> {
        let c = config.validate()?;
        // CAST: usize → f32, a feature count is far below 2^24; the bound is an initialisation scale
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        let bound = 1.0 / (c.n_features as f32).sqrt();
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let w_enc = Tensor::from_vec(
            uniform(c.n_layers * c.d_model * c.n_features, bound, &mut state),
            (c.n_layers, c.d_model, c.n_features),
            device,
        )?;
        let b_enc = Tensor::zeros((c.n_layers, c.n_features), DType::F32, device)?;
        let w_dec = (0..c.n_layers)
            .map(|l| {
                let cols = (c.n_layers - l) * c.d_model;
                Tensor::from_vec(
                    uniform(c.n_features * cols, bound, &mut state),
                    (c.n_features, cols),
                    device,
                )
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let b_dec = Tensor::zeros((c.n_layers, c.d_model), DType::F32, device)?;
        Self::from_tensors(&c, &w_enc, &b_enc, &w_dec, &b_dec)
    }

    /// A transcoder from given parameter values (copied into fresh [`Var`]s).
    ///
    /// # Shapes
    /// - `w_enc`: `[n_layers, d_model, n_features]`
    /// - `b_enc`: `[n_layers, n_features]`
    /// - `w_dec[l]`: `[n_features, (n_layers - l) * d_model]`, one per layer
    /// - `b_dec`: `[n_layers, d_model]`
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the configuration is degenerate or a tensor's shape does not match it.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn from_tensors(
        config: &TrainableCltConfig,
        w_enc: &Tensor,
        b_enc: &Tensor,
        w_dec: &[Tensor],
        b_dec: &Tensor,
    ) -> Result<Self> {
        let c = config.validate()?;
        let check = |name: &str, t: &Tensor, want: &[usize]| -> Result<()> {
            if t.dims() == want {
                Ok(())
            } else {
                Err(MIError::Config(format!(
                    "{name} has shape {:?} (expected {want:?})",
                    t.dims()
                )))
            }
        };
        check("w_enc", w_enc, &[c.n_layers, c.d_model, c.n_features])?;
        check("b_enc", b_enc, &[c.n_layers, c.n_features])?;
        check("b_dec", b_dec, &[c.n_layers, c.d_model])?;
        if w_dec.len() != c.n_layers {
            return Err(MIError::Config(format!(
                "w_dec has {} source layers (expected {})",
                w_dec.len(),
                c.n_layers
            )));
        }
        for (l, w) in w_dec.iter().enumerate() {
            check(
                &format!("w_dec[{l}]"),
                w,
                &[c.n_features, (c.n_layers - l) * c.d_model],
            )?;
        }
        // PROMOTE: the parameters are trained in F32 whatever dtype the caller's tensors carry (BF16 checkpoints included)
        Ok(Self {
            config: c,
            w_enc: Var::from_tensor(&w_enc.to_dtype(DType::F32)?)?,
            b_enc: Var::from_tensor(&b_enc.to_dtype(DType::F32)?)?,
            w_dec: w_dec
                .iter()
                .map(|w| Var::from_tensor(&w.to_dtype(DType::F32)?))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            b_dec: Var::from_tensor(&b_dec.to_dtype(DType::F32)?)?,
        })
    }

    /// The transcoder's shape.
    #[must_use]
    pub const fn config(&self) -> TrainableCltConfig {
        self.config
    }

    /// Every parameter with a stable name (`w_enc`, `b_enc`, `w_dec_{l}`, `b_dec`), as `optim::AdamW::new` takes them
    /// (`training` feature).
    #[must_use]
    pub fn vars(&self) -> Vec<(String, Var)> {
        // BORROW: explicit .to_owned() — the optimiser keys its state by owned names
        let mut vars = vec![
            ("w_enc".to_owned(), self.w_enc.clone()),
            ("b_enc".to_owned(), self.b_enc.clone()),
            ("b_dec".to_owned(), self.b_dec.clone()),
        ];
        vars.extend(
            self.w_dec
                .iter()
                .enumerate()
                .map(|(l, w)| (format!("w_dec_{l}"), w.clone())),
        );
        vars
    }

    /// Check an activation tensor is `[n_positions, n_layers, width]`.
    fn check_activations(&self, name: &str, t: &Tensor, width: usize) -> Result<usize> {
        match t.dims() {
            &[n, l, w] if l == self.config.n_layers && w == width => Ok(n),
            dims => Err(MIError::Config(format!(
                "{name} has shape {dims:?} (expected [n_positions, {}, {width}])",
                self.config.n_layers
            ))),
        }
    }

    /// Sparse feature activations: `TopK_k(ReLU(x · W_enc + b_enc))` per (position, layer).
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]` -- the layer inputs
    /// - returns: `[n_positions, n_layers, n_features]`, at most `top_k` non-zero per (position, layer)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` does not have the expected shape.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        self.check_activations("x", x, self.config.d_model)?;
        // CONTIGUOUS: the layer-major transpose has non-unit strides; the batched matmul needs a contiguous layout
        let by_layer = x.transpose(0, 1)?.contiguous()?;
        let pre = by_layer
            .matmul(self.w_enc.as_tensor())?
            .transpose(0, 1)?
            .broadcast_add(&self.b_enc.as_tensor().unsqueeze(0)?)?;
        top_k(&pre.relu()?, self.config.top_k)
    }

    /// Reconstruction from feature activations, through the triangular decoder.
    ///
    /// # Shapes
    /// - `acts`: `[n_positions, n_layers, n_features]`
    /// - returns: `[n_positions, n_layers, d_model]` -- the reconstructed layer outputs
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `acts` does not have the expected shape.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn decode(&self, acts: &Tensor) -> Result<Tensor> {
        let n = self.check_activations("acts", acts, self.config.n_features)?;
        let (layers, d) = (self.config.n_layers, self.config.d_model);
        let mut recon = self
            .b_dec
            .as_tensor()
            .unsqueeze(0)?
            .broadcast_as((n, layers, d))?
            .contiguous()?;
        // EXPLICIT: the reconstruction accumulates source layer by source layer; a fold would hide the running sum
        for (l, w) in self.w_dec.iter().enumerate() {
            // CONTIGUOUS: one layer's slice of the activations is strided; the matmul needs a contiguous layout
            let source = acts.narrow(1, l, 1)?.squeeze(1)?.contiguous()?;
            let written = source.matmul(w.as_tensor())?.reshape((n, layers - l, d))?;
            recon = (recon + written.pad_with_zeros(1, l, 0)?)?;
        }
        Ok(recon)
    }

    /// Encode then decode.
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]`
    /// - returns: (`[n_positions, n_layers, d_model]` reconstruction, `[n_positions, n_layers, n_features]` activations)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` does not have the expected shape.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let acts = self.encode(x)?;
        Ok((self.decode(&acts)?, acts))
    }

    /// The training loss: the mean over (positions, layers, `d_model`) of the squared reconstruction error --
    /// the reference's reduction (the per-layer MSE averaged over layers, not summed).
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]` -- the layer inputs
    /// - `target`: `[n_positions, n_layers, d_model]` -- the layer outputs to reconstruct
    /// - returns: `[]` (a scalar)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a tensor does not have the expected shape.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn loss(&self, x: &Tensor, target: &Tensor) -> Result<Tensor> {
        self.check_activations("target", target, self.config.d_model)?;
        let (recon, _) = self.forward(x)?;
        Ok((recon - target)?.sqr()?.mean_all()?)
    }

    /// Write the transcoder in the circuit-tracer layout [`CrossLayerTranscoder`](super::CrossLayerTranscoder) reads,
    /// in raw activation units.
    ///
    /// Training usually scales each layer's inputs and outputs by a scalar (`x / scale_in[l]`, `target /
    /// scale_out[l]`); the export folds those scales into the weights, so the written CLT takes the model's raw
    /// activations: `W_enc_{l} = (W_enc[l] / scale_in[l])ᵀ`, `W_dec_{l}[:, t - l, :] = W_dec[l → t] · scale_out[t]`,
    /// `b_dec_{l} = b_dec[l] · scale_out[l]` (the `TopK` selection is unchanged). Pass ones for an unscaled CLT.
    ///
    /// Files, per layer `l`, in `F32`: `W_enc_{l}.safetensors` (`W_enc_{l}` `[n_features, d_model]`, `b_enc_{l}`
    /// `[n_features]`, `b_dec_{l}` `[d_model]`) and `W_dec_{l}.safetensors` (`W_dec_{l}` `[n_features, n_layers - l,
    /// d_model]`); plus `candle_mi_clt.json`, which records `top_k` (the circuit-tracer layout has no field for it)
    /// and the model it was trained on.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if a scale slice does not have one entry per layer, or a scale is not positive.
    /// Returns [`MIError::Io`] if the directory or a file cannot be written.
    /// Returns [`MIError::Model`] on tensor failure.
    pub fn export_circuit_tracer(
        &self,
        dir: &Path,
        scale_in: &[f32],
        scale_out: &[f32],
        model_name: &str,
    ) -> Result<()> {
        let (layers, d) = (self.config.n_layers, self.config.d_model);
        for (name, scales) in [("scale_in", scale_in), ("scale_out", scale_out)] {
            if scales.len() != layers || scales.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
                return Err(MIError::Config(format!(
                    "{name} must hold {layers} positive finite values (got {scales:?})"
                )));
            }
        }
        std::fs::create_dir_all(dir)?;
        let device = Device::Cpu;
        let w_enc = self.w_enc.as_tensor().to_device(&device)?;
        let b_enc = self.b_enc.as_tensor().to_device(&device)?;
        let b_dec = self.b_dec.as_tensor().to_device(&device)?;
        let out_scales = Tensor::from_slice(scale_out, layers, &device)?;
        for (l, (w, (s_in, s_out))) in self
            .w_dec
            .iter()
            .zip(scale_in.iter().zip(scale_out))
            .enumerate()
        {
            let encoder: HashMap<String, Tensor> = HashMap::from([
                (
                    format!("W_enc_{l}"),
                    // CONTIGUOUS: the transpose is strided; safetensors serialises a contiguous buffer
                    w_enc
                        .get(l)?
                        .affine(1.0 / f64::from(*s_in), 0.0)?
                        .t()?
                        .contiguous()?,
                ),
                (format!("b_enc_{l}"), b_enc.get(l)?),
                (
                    format!("b_dec_{l}"),
                    b_dec.get(l)?.affine(f64::from(*s_out), 0.0)?,
                ),
            ]);
            candle_core::safetensors::save(&encoder, dir.join(format!("W_enc_{l}.safetensors")))?;
            let targets = out_scales
                .narrow(0, l, layers - l)?
                .reshape((1, layers - l, 1))?;
            let w_dec = w
                .as_tensor()
                .to_device(&device)?
                .reshape((self.config.n_features, layers - l, d))?
                .broadcast_mul(&targets)?;
            let decoder: HashMap<String, Tensor> = HashMap::from([(format!("W_dec_{l}"), w_dec)]);
            candle_core::safetensors::save(&decoder, dir.join(format!("W_dec_{l}.safetensors")))?;
        }
        let meta = serde_json::json!({
            "format": "circuit-tracer cross-layer transcoder (per-layer W_enc_<l> / W_dec_<l> safetensors)",
            "activation": "topk_relu",
            "top_k": self.config.top_k,
            "n_layers": layers,
            "d_model": d,
            "n_features_per_layer": self.config.n_features,
            "model_name": model_name,
            "written_by": concat!("candle-mi ", env!("CARGO_PKG_VERSION"), " clt::train"),
        });
        std::fs::write(dir.join("candle_mi_clt.json"), format!("{meta:#}\n"))?;
        Ok(())
    }
}

/// Keep the `k` largest entries of each row (last dimension), zero the rest; the selection is a constant mask, so the
/// gradient flows to the kept entries only (as `torch.topk` + `scatter` in the reference).
///
/// # Shapes
/// - `acts`: `[..., n_features]`, non-negative (post-`ReLU`)
/// - returns: same shape
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
pub fn top_k(acts: &Tensor, k: usize) -> Result<Tensor> {
    let frozen = acts.detach();
    let removed = Tensor::new(f32::NEG_INFINITY, acts.device())?
        .to_dtype(acts.dtype())?
        .broadcast_as(acts.shape())?;
    let mut work = frozen.clone();
    let mut kth = frozen.max_keepdim(D::Minus1)?;
    // EXPLICIT: each pass removes the current row maxima from `work`; the state carried between passes is the point
    for _ in 0..k {
        kth = work.max_keepdim(D::Minus1)?;
        let hit = work.broadcast_eq(&kth)?;
        work = hit.where_cond(&removed, &work)?;
    }
    let keep = frozen.broadcast_ge(&kth)?.to_dtype(acts.dtype())?;
    Ok((acts * keep)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
// EXPLICIT: unit-test code — assertions panic by design.
mod tests {
    use super::*;

    const fn tiny() -> TrainableCltConfig {
        TrainableCltConfig::new(3, 4, 6, 2)
    }

    #[test]
    fn top_k_keeps_k_and_all_when_fewer_are_positive() {
        let acts = Tensor::new(
            &[[0.5f32, 3.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.7, 0.0, 0.0]],
            &Device::Cpu,
        )
        .unwrap();
        let kept = top_k(&acts, 2).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(kept[0], vec![0.0, 3.0, 0.0, 2.0, 0.0]);
        assert_eq!(kept[1], vec![0.0, 0.0, 0.7, 0.0, 0.0]);
    }

    #[test]
    fn decoder_is_strictly_triangular() {
        // A feature at source layer 2 must leave layers 0 and 1 at their bias.
        let clt = TrainableClt::new(&tiny(), 7, &Device::Cpu).unwrap();
        let mut acts = vec![0f32; 3 * 6];
        acts[2 * 6 + 1] = 1.5;
        let acts = Tensor::from_vec(acts, (1, 3, 6), &Device::Cpu).unwrap();
        let recon = clt.decode(&acts).unwrap().to_vec3::<f32>().unwrap();
        assert!(recon[0][0].iter().chain(&recon[0][1]).all(|v| *v == 0.0));
        assert!(recon[0][2].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn shapes_are_checked() {
        let clt = TrainableClt::new(&tiny(), 7, &Device::Cpu).unwrap();
        let wrong = Tensor::zeros((5, 2, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(clt.encode(&wrong).is_err());
        assert!(
            TrainableClt::new(&TrainableCltConfig { top_k: 7, ..tiny() }, 0, &Device::Cpu).is_err()
        );
    }

    #[test]
    fn a_seed_is_one_initialisation() {
        let a = TrainableClt::new(&tiny(), 3, &Device::Cpu).unwrap();
        let b = TrainableClt::new(&tiny(), 3, &Device::Cpu).unwrap();
        let diff = (a.w_enc.as_tensor() - b.w_enc.as_tensor())
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap();
        assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
    }
}
