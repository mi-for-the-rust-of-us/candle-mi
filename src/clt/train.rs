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
//! **`TopK` and ties.** The k-th largest activation of each row is read off one descending row sort (`candle`'s
//! bitonic sort, which handles rows wider than its 1,024-thread block), then every entry at least that large is kept.
//! Exact ties at the k-th value therefore keep more than k features where `torch.topk` keeps exactly k; with
//! real-valued activations this has probability zero. Rows with fewer than k positive activations keep them all,
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
    ///
    /// ```
    /// use candle_mi::clt::train::TrainableCltConfig;
    ///
    /// let config = TrainableCltConfig::new(6, 384, 4096, 16);
    /// assert_eq!((config.n_layers, config.top_k), (6, 16));
    /// ```
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
///
/// ```
/// use candle_core::{DType, Device, Tensor};
/// use candle_mi::clt::train::{TrainableClt, TrainableCltConfig};
///
/// // 2 layers, width 4, 8 features per layer, 2 active per (position, layer).
/// let clt = TrainableClt::new(&TrainableCltConfig::new(2, 4, 8, 2), 0, &Device::Cpu)?;
/// let x = Tensor::ones((3, 2, 4), DType::F32, &Device::Cpu)?;
/// let (recon, acts) = clt.forward(&x)?;
/// assert_eq!((recon.dims(), acts.dims()), (&[3, 2, 4][..], &[3, 2, 8][..]));
/// let most_active = acts.gt(0.0)?.to_dtype(DType::F32)?.sum(2)?.flatten_all()?.max(0)?;
/// assert!(most_active.to_scalar::<f32>()? <= 2.0);
/// assert_eq!(clt.loss(&x, &x)?.dims(), &[] as &[usize]);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
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
        let layers = self.encode_layers(self.w_enc.as_tensor(), self.b_enc.as_tensor(), x)?;
        Ok(Tensor::stack(&layers, 1)?)
    }

    /// The activations of each source layer as its OWN tensor, `[n_positions, n_features]`, with
    /// the encoder parameters given (tracked or detached).
    ///
    /// One tensor per layer, rather than a `[n_positions, n_layers, n_features]` block sliced per
    /// layer, is a measured choice: the backward of each per-layer slice of the block allocates zero
    /// padding and a full-size concatenation, six times per step (the `Narrow` backward of
    /// `candle-core`).
    fn encode_layers(&self, w_enc: &Tensor, b_enc: &Tensor, x: &Tensor) -> Result<Vec<Tensor>> {
        self.check_activations("x", x, self.config.d_model)?;
        (0..self.config.n_layers)
            .map(|l| {
                // CONTIGUOUS: one layer's slice of the inputs is strided; the matmul needs a contiguous layout
                let input = x.narrow(1, l, 1)?.squeeze(1)?.contiguous()?;
                let pre = input
                    .matmul(&w_enc.get(l)?)?
                    .broadcast_add(&b_enc.get(l)?)?;
                top_k(&pre.relu()?, self.config.top_k)
            })
            .collect()
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
        self.check_activations("acts", acts, self.config.n_features)?;
        let layers = (0..self.config.n_layers)
            .map(|l| acts.narrow(1, l, 1)?.squeeze(1))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let w_dec: Vec<Tensor> = self.w_dec.iter().map(|w| w.as_tensor().clone()).collect();
        self.decode_layers(&w_dec, self.b_dec.as_tensor(), &layers)
    }

    /// [`decode`](Self::decode) from one activation tensor per source layer (`[n_positions,
    /// n_features]` each), with the decoder parameters given (tracked or detached).
    fn decode_layers(&self, w_dec: &[Tensor], b_dec: &Tensor, acts: &[Tensor]) -> Result<Tensor> {
        let (layers, d) = (self.config.n_layers, self.config.d_model);
        let n = match acts.first().map(Tensor::dims) {
            Some(&[n, f]) if acts.len() == layers && f == self.config.n_features => n,
            _ => {
                return Err(MIError::Config(format!(
                    "decoder input must be {layers} tensors [n_positions, {}]",
                    self.config.n_features
                )));
            }
        };
        let mut recon = b_dec
            .unsqueeze(0)?
            .broadcast_as((n, layers, d))?
            .contiguous()?;
        // EXPLICIT: the reconstruction accumulates source layer by source layer; a fold would hide the running sum
        for (l, (w, source)) in w_dec.iter().zip(acts).enumerate() {
            // CONTIGUOUS: a layer's activations may come from a slice; the matmul needs a contiguous layout
            let written = source
                .contiguous()?
                .matmul(w)?
                .reshape((n, layers - l, d))?;
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
        let layers = self.encode_layers(self.w_enc.as_tensor(), self.b_enc.as_tensor(), x)?;
        let w_dec: Vec<Tensor> = self.w_dec.iter().map(|w| w.as_tensor().clone()).collect();
        let recon = self.decode_layers(&w_dec, self.b_dec.as_tensor(), &layers)?;
        Ok((recon, Tensor::stack(&layers, 1)?))
    }

    /// [`forward`](Self::forward) on DETACHED parameters: the same numbers, and no backward graph.
    ///
    /// Every op on a [`Var`] is recorded for `backward`, so an evaluation through
    /// [`forward`](Self::forward) keeps each intermediate reachable from the parameters until the
    /// result is dropped -- over thousands of positions, gigabytes. Use this wherever no gradient
    /// is wanted (evaluation, a replacement model, analysis).
    ///
    /// # Shapes
    /// - `x`: `[n_positions, n_layers, d_model]`
    /// - returns: (`[n_positions, n_layers, d_model]` reconstruction, `[n_positions, n_layers, n_features]` activations)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if `x` does not have the expected shape.
    /// Returns [`MIError::Model`] on tensor failure.
    ///
    /// ```
    /// use candle_core::{DType, Device, Tensor};
    /// use candle_mi::clt::train::{TrainableClt, TrainableCltConfig};
    ///
    /// let clt = TrainableClt::new(&TrainableCltConfig::new(2, 4, 8, 2), 1, &Device::Cpu)?;
    /// let x = Tensor::ones((3, 2, 4), DType::F32, &Device::Cpu)?;
    /// let (tracked, _) = clt.forward(&x)?;
    /// let (detached, _) = clt.forward_detached(&x)?;
    /// let diff = (tracked - detached)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    /// assert_eq!(diff, 0.0);
    /// # Ok::<(), candle_mi::MIError>(())
    /// ```
    pub fn forward_detached(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let layers = self.encode_layers(
            &self.w_enc.as_tensor().detach(),
            &self.b_enc.as_tensor().detach(),
            &x.detach(),
        )?;
        let w_dec: Vec<Tensor> = self.w_dec.iter().map(|w| w.as_tensor().detach()).collect();
        let recon = self.decode_layers(&w_dec, &self.b_dec.as_tensor().detach(), &layers)?;
        Ok((recon, Tensor::stack(&layers, 1)?))
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
        // The per-layer path end to end: the stacked activations of `forward` are not needed here.
        let layers = self.encode_layers(self.w_enc.as_tensor(), self.b_enc.as_tensor(), x)?;
        let w_dec: Vec<Tensor> = self.w_dec.iter().map(|w| w.as_tensor().clone()).collect();
        let recon = self.decode_layers(&w_dec, self.b_dec.as_tensor(), &layers)?;
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

/// Keep the `k` largest entries of each row (last dimension), zero the rest.
///
/// The selection is a constant mask, so the gradient flows to the kept entries only (as `torch.topk` + `scatter` in
/// the reference).
///
/// ```
/// use candle_core::{Device, Tensor};
/// use candle_mi::clt::train::top_k;
///
/// let acts = Tensor::new(&[[0.5f32, 3.0, 0.0, 2.0], [0.0, 0.7, 0.0, 0.0]], &Device::Cpu)?;
/// // Row 0 keeps its two largest; row 1 has a single positive entry and keeps it.
/// assert_eq!(top_k(&acts, 2)?.to_vec2::<f32>()?, [[0.0, 3.0, 0.0, 2.0], [0.0, 0.7, 0.0, 0.0]]);
/// # Ok::<(), candle_mi::MIError>(())
/// ```
///
/// # Shapes
/// - `acts`: `[..., n_features]`, non-negative (post-`ReLU`)
/// - returns: same shape
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
pub fn top_k(acts: &Tensor, k: usize) -> Result<Tensor> {
    // The selection is a constant: computed on a detached copy, so no gradient flows through the sort.
    // CONTIGUOUS: the row sort requires a contiguous layout
    let frozen = acts.detach().contiguous()?;
    let order = frozen.arg_sort_last_dim(false)?;
    let kth_index = order.narrow(D::Minus1, k - 1, 1)?.contiguous()?;
    let kth = frozen.gather(&kth_index, D::Minus1)?;
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
        // Same seed, same weights: the summed absolute difference is exactly zero.
        assert!(diff.to_scalar::<f32>().unwrap().abs() < f32::MIN_POSITIVE);
    }
}
