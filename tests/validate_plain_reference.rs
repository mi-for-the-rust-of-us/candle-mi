// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]
// EXPLICIT: integration-test code — assertions panic by design.
// EXPLICIT: `suboptimal_flops` — the reference transcribes the equations as
// written (`acc += x * w`); fusing them into `mul_add` would change nothing that
// matters in `f64` against a `1e-5` bound, and the plain form is what reviews.
// EXPLICIT: `many_single_char_names` / `similar_names` — the reference forward
// uses the standard names of the equations it transcribes (`q`, `k`, `v`, `o`,
// `w`, `b`, `d`, `h`, `s`, `t`), which is what makes it reviewable against them.

//! **`PlainTransformer` against an independent reference forward**
//! (`design/plain-transformer.md` §7.1).
//!
//! The reference below is written with plain loops over `f64`, from the
//! equations in the design, and never touches candle: it shares no code with
//! the backbone, only the weights. It runs in `f64` so its own rounding is
//! negligible next to the bound. It is compared against `PlainTransformer` on
//! seeded random weights across **every** option combination: norm none /
//! `LayerNorm` / `RmsNorm`, each activation, causal or not, biases or not,
//! positions or not; always with `n_head * head_dim != d_model` and
//! `output_size != vocab_size`; plus one non-default `attn_scale` (`2.0`,
//! moderate on purpose: softmax multiplies a score difference by the scale).
//!
//! Tolerance, fixed before any run, per compared tensor `x` (the reference's):
//! `max |Δ| <= 1e-5 * max(1, max |x|)`. This is the only in-Rust check with a
//! tolerance, because it is not candle on both sides.
//!
//! Exact `GELU` needs `erf`, which `std` lacks; it uses Abramowitz and Stegun
//! 7.1.26 (max absolute error `1.5e-7`), well inside the bound.
//!
//! CPU-only, a fraction of a second, no download: runs in CI (step
//! "Tests (plain)").

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_mi::{
    Activation, HookPoint, HookSpec, MIBackend, NormType, PlainTransformer, PlainTransformerConfig,
};

const RTOL: f64 = 1e-5;

// --- weights ----------------------------------------------------------------

/// Deterministic values in `[-0.5, 0.5)` from a 64-bit LCG.
fn lcg_values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // CAST: u64 → f32 via the top 24 bits, exactly representable
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let unit = (state >> 40) as f32 / 16_777_216.0;
            unit - 0.5
        })
        .collect()
}

/// The weights twice: as candle tensors for the model, as `f64` vectors for
/// the reference. Norm weights are centred on 1 so the norms are not near-zero.
fn weights(
    config: &PlainTransformerConfig,
    seed: u64,
) -> (HashMap<String, Tensor>, HashMap<String, Vec<f64>>) {
    let mut tensors = HashMap::new();
    let mut plain = HashMap::new();
    for (i, (name, dims)) in config.weight_shapes().into_iter().enumerate() {
        let n: usize = dims.iter().product();
        // CAST: usize → u64, a tensor index
        #[allow(clippy::as_conversions)]
        let mut values = lcg_values(seed * 1000 + i as u64, n);
        let is_norm_weight = (name.contains(".ln") || name.starts_with("ln_f"))
            && std::path::Path::new(&name)
                .extension()
                .is_some_and(|e| e == "weight");
        if is_norm_weight {
            for v in &mut values {
                *v += 1.0;
            }
        }
        plain.insert(name.clone(), values.iter().map(|&v| f64::from(v)).collect());
        tensors.insert(name, Tensor::from_vec(values, dims, &Device::Cpu).unwrap());
    }
    (tensors, plain)
}

// --- the reference forward ------------------------------------------------------

/// `[rows, cols]` matrix as a flat row-major `Vec<f64>`.
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Mat {
    fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }
}

/// `y = x · Wᵀ + b` with `W` in the `nn.Linear` layout `[out, in]`.
fn linear(x: &Mat, w: &[f64], b: Option<&[f64]>, out: usize) -> Mat {
    let mut data = vec![0.0; x.rows * out];
    for r in 0..x.rows {
        for o in 0..out {
            let mut acc = b.map_or(0.0, |b| b[o]);
            for i in 0..x.cols {
                acc += x.at(r, i) * w[o * x.cols + i];
            }
            data[r * out + o] = acc;
        }
    }
    Mat {
        rows: x.rows,
        cols: out,
        data,
    }
}

fn norm(x: &Mat, kind: Option<NormType>, eps: f64, w: Option<&[f64]>, b: Option<&[f64]>) -> Mat {
    let Some(kind) = kind else {
        return Mat {
            rows: x.rows,
            cols: x.cols,
            data: x.data.clone(),
        };
    };
    let w = w.unwrap();
    // CAST: usize → f64, a small dimension
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let n = x.cols as f64;
    let mut data = vec![0.0; x.rows * x.cols];
    for r in 0..x.rows {
        let row: Vec<f64> = (0..x.cols).map(|c| x.at(r, c)).collect();
        match kind {
            NormType::LayerNorm => {
                let mean = row.iter().sum::<f64>() / n;
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                let inv = 1.0 / (var + eps).sqrt();
                for c in 0..x.cols {
                    data[r * x.cols + c] = (row[c] - mean) * inv * w[c] + b.unwrap()[c];
                }
            }
            NormType::RmsNorm => {
                let ms = row.iter().map(|v| v * v).sum::<f64>() / n;
                let inv = 1.0 / (ms + eps).sqrt();
                for c in 0..x.cols {
                    data[r * x.cols + c] = row[c] * inv * w[c];
                }
            }
            NormType::GemmaRmsNorm => panic!("PlainTransformer does not support GemmaRmsNorm"),
            // `NormType` is `#[non_exhaustive]`: a future variant must be added here.
            _ => panic!("reference does not implement {kind:?}"),
        }
    }
    Mat {
        rows: x.rows,
        cols: x.cols,
        data,
    }
}

/// Abramowitz and Stegun 7.1.26, max absolute error 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn act(x: f64, a: Activation) -> f64 {
    match a {
        Activation::Relu => x.max(0.0),
        Activation::Gelu => 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)),
        Activation::GeluApprox => {
            let c = (2.0 / std::f64::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * x.mul_add(0.044_715 * x * x, x)).tanh())
        }
        Activation::Silu => x / (1.0 + (-x).exp()),
        other => panic!("reference does not implement {other:?}"),
    }
}

/// What the reference produces, for one sequence.
struct RefOut {
    embed: Mat,
    resid_mid: Vec<Mat>,
    resid_post: Vec<Mat>,
    /// Per layer, `[head][query][key]` flattened.
    pattern: Vec<Vec<f64>>,
    mlp_act: Vec<Mat>,
    logits: Mat,
}

fn reference(cfg: &PlainTransformerConfig, w: &HashMap<String, Vec<f64>>, ids: &[u32]) -> RefOut {
    let (d, h, hd, m) = (cfg.d_model, cfg.n_head, cfg.head_dim, cfg.mlp_hidden);
    let a = h * hd;
    let s = ids.len();
    let get = |name: &str| w.get(name).map(Vec::as_slice);
    let bias = |name: &str| if cfg.bias { get(name) } else { None };

    // Embedding (+ positions).
    let tok = get("tok_emb.weight").unwrap();
    let mut data = vec![0.0; s * d];
    for (p, &id) in ids.iter().enumerate() {
        // CAST: u32 → usize, a token id
        #[allow(clippy::as_conversions)]
        let id = id as usize;
        for c in 0..d {
            data[p * d + c] = tok[id * d + c];
            if let Some(pos) = get("pos_emb.weight") {
                data[p * d + c] += pos[p * d + c];
            }
        }
    }
    let mut x = Mat {
        rows: s,
        cols: d,
        data,
    };
    let embed = Mat {
        rows: s,
        cols: d,
        data: x.data.clone(),
    };

    let (mut resid_mid, mut resid_post, mut pattern, mut mlp_act) =
        (vec![], vec![], vec![], vec![]);
    for l in 0..cfg.n_layer {
        let pre = |site: &str| format!("blocks.{l}.{site}");
        // Attention.
        let n1 = norm(
            &x,
            cfg.norm,
            cfg.norm_eps,
            get(&pre("ln1.weight")),
            get(&pre("ln1.bias")),
        );
        let q = linear(
            &n1,
            get(&pre("attn.q.weight")).unwrap(),
            bias(&pre("attn.q.bias")),
            a,
        );
        let k = linear(
            &n1,
            get(&pre("attn.k.weight")).unwrap(),
            bias(&pre("attn.k.bias")),
            a,
        );
        let v = linear(
            &n1,
            get(&pre("attn.v.weight")).unwrap(),
            bias(&pre("attn.v.bias")),
            a,
        );
        let mut merged = Mat {
            rows: s,
            cols: a,
            data: vec![0.0; s * a],
        };
        let mut pat = vec![0.0; h * s * s];
        for head in 0..h {
            for qi in 0..s {
                let mut scores = vec![f64::NEG_INFINITY; s];
                for (ki, score) in scores.iter_mut().enumerate() {
                    if cfg.causal && ki > qi {
                        continue;
                    }
                    let mut dot = 0.0;
                    for e in 0..hd {
                        dot += q.at(qi, head * hd + e) * k.at(ki, head * hd + e);
                    }
                    *score = cfg.attn_scale * dot;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|&sc| (sc - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for ki in 0..s {
                    let p = exps[ki] / sum;
                    pat[(head * s + qi) * s + ki] = p;
                    for e in 0..hd {
                        merged.data[qi * a + head * hd + e] += p * v.at(ki, head * hd + e);
                    }
                }
            }
        }
        let attn_out = linear(
            &merged,
            get(&pre("attn.o.weight")).unwrap(),
            bias(&pre("attn.o.bias")),
            d,
        );
        for (xv, av) in x.data.iter_mut().zip(&attn_out.data) {
            *xv += av;
        }
        resid_mid.push(Mat {
            rows: s,
            cols: d,
            data: x.data.clone(),
        });
        pattern.push(pat);

        // MLP.
        let n2 = norm(
            &x,
            cfg.norm,
            cfg.norm_eps,
            get(&pre("ln2.weight")),
            get(&pre("ln2.bias")),
        );
        let mut up = linear(
            &n2,
            get(&pre("mlp.fc.weight")).unwrap(),
            bias(&pre("mlp.fc.bias")),
            m,
        );
        for val in &mut up.data {
            *val = act(*val, cfg.activation);
        }
        let mlp_out = linear(
            &up,
            get(&pre("mlp.proj.weight")).unwrap(),
            bias(&pre("mlp.proj.bias")),
            d,
        );
        for (xv, mv) in x.data.iter_mut().zip(&mlp_out.data) {
            *xv += mv;
        }
        mlp_act.push(up);
        resid_post.push(Mat {
            rows: s,
            cols: d,
            data: x.data.clone(),
        });
    }

    let nf = norm(
        &x,
        cfg.norm,
        cfg.norm_eps,
        get("ln_f.weight"),
        get("ln_f.bias"),
    );
    let logits = linear(&nf, get("unembed.weight").unwrap(), None, cfg.output_size);
    RefOut {
        embed,
        resid_mid,
        resid_post,
        pattern,
        mlp_act,
        logits,
    }
}

// --- comparison -----------------------------------------------------------------

/// `max |Δ|` and the bound `RTOL * max(1, max |x|)` for one compared tensor.
fn check(what: &str, label: &str, candle: &Tensor, reference: &[f64]) -> f64 {
    let got: Vec<f32> = candle.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got.len(), reference.len(), "{label} {what}: length");
    let scale = reference.iter().fold(1.0_f64, |acc, v| acc.max(v.abs()));
    let max_diff = got
        .iter()
        .zip(reference)
        .map(|(&g, &r)| (f64::from(g) - r).abs())
        .fold(0.0_f64, f64::max);
    let bound = RTOL * scale;
    assert!(
        max_diff <= bound,
        "{label} {what}: max |Δ| {max_diff:.3e} > bound {bound:.3e} (max |x| {scale:.3e})"
    );
    max_diff / scale
}

fn run_case(cfg: &PlainTransformerConfig, label: &str, seed: u64, device: &Device) -> f64 {
    let (tensors, plain) = weights(cfg, seed);
    let model = PlainTransformer::from_tensors(cfg.clone(), tensors, device).unwrap();
    let ids_vec: Vec<u32> = vec![0, 3, 6, 1, 4];
    let ids = Tensor::new(ids_vec.as_slice(), device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let mut spec = HookSpec::new();
    spec.capture(HookPoint::Embed);
    for l in 0..cfg.n_layer {
        spec.capture(HookPoint::AttnPattern(l))
            .capture(HookPoint::ResidMid(l))
            .capture(HookPoint::MlpAct(l))
            .capture(HookPoint::ResidPost(l));
    }
    let cache = model.forward(&ids, &spec).unwrap();
    let want = reference(cfg, &plain, &ids_vec);

    let mut worst = 0.0_f64;
    let mut note = |r: f64| worst = worst.max(r);
    note(check(
        "Embed",
        label,
        cache.require(&HookPoint::Embed).unwrap(),
        &want.embed.data,
    ));
    for l in 0..cfg.n_layer {
        note(check(
            &format!("AttnPattern({l})"),
            label,
            cache.require(&HookPoint::AttnPattern(l)).unwrap(),
            &want.pattern[l],
        ));
        note(check(
            &format!("ResidMid({l})"),
            label,
            cache.require(&HookPoint::ResidMid(l)).unwrap(),
            &want.resid_mid[l].data,
        ));
        note(check(
            &format!("MlpAct({l})"),
            label,
            cache.require(&HookPoint::MlpAct(l)).unwrap(),
            &want.mlp_act[l].data,
        ));
        note(check(
            &format!("ResidPost({l})"),
            label,
            cache.require(&HookPoint::ResidPost(l)).unwrap(),
            &want.resid_post[l].data,
        ));
    }
    note(check("logits", label, cache.output(), &want.logits.data));
    worst
}

/// 2 layers, 3 heads of 4 dims in a 10-dim residual (`n_head * head_dim` =
/// 12), MLP width 6, 7 input tokens, 5 outputs.
fn base() -> PlainTransformerConfig {
    PlainTransformerConfig::new(7, 2, 10, 3, 4, 6)
        .unwrap()
        .with_output_size(5)
}

/// The CUDA device, if this build has `cuda` and a device is present.
fn cuda_device() -> Option<Device> {
    let device = Device::cuda_if_available(0).unwrap();
    device.is_cuda().then_some(device)
}

/// The 96-configuration sweep on `device`.
fn sweep(device: &Device) {
    let norms = [None, Some(NormType::LayerNorm), Some(NormType::RmsNorm)];
    let activations = [
        Activation::Relu,
        Activation::Gelu,
        Activation::GeluApprox,
        Activation::Silu,
    ];
    let mut cases = 0_u32;
    let mut worst = (0.0_f64, String::new());
    for norm in norms {
        for activation in activations {
            for causal in [false, true] {
                for bias in [false, true] {
                    for positions in [false, true] {
                        let mut cfg = base()
                            .with_activation(activation)
                            .with_causal(causal)
                            .with_bias(bias);
                        cfg.norm = norm;
                        if positions {
                            cfg = cfg.with_positions(5);
                        }
                        let label = format!(
                            "[norm {norm:?}, {activation:?}, causal {causal}, bias {bias}, positions {positions}]"
                        );
                        let r = run_case(&cfg, &label, u64::from(cases) + 1, device);
                        if r > worst.0 {
                            worst = (r, label);
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 96, "the sweep must cover every combination");
    eprintln!(
        "plain reference sweep on {device:?}: {cases} configurations within \
         max |Δ| <= 1e-5 * max(1, max |x|); worst relative {:.3e} at {}",
        worst.0, worst.1
    );
}

/// The non-default `attn_scale` case on `device`.
fn non_default_scale(device: &Device) {
    let cfg = base()
        .with_norm(NormType::LayerNorm)
        .with_activation(Activation::Gelu)
        .with_bias(true)
        .with_positions(5)
        .with_attn_scale(2.0);
    let r = run_case(&cfg, "[attn_scale 2.0]", 777, device);
    eprintln!("plain reference on {device:?}, attn_scale 2.0: worst relative {r:.3e}");
}

#[test]
fn matches_the_reference_on_every_option_combination() {
    sweep(&Device::Cpu);
}

#[test]
fn matches_the_reference_with_a_non_default_attn_scale() {
    non_default_scale(&Device::Cpu);
}

/// The same sweep and the same bound on CUDA. Not part of CI (no GPU there):
/// run with `--include-ignored` on a CUDA machine.
#[test]
#[ignore = "CUDA: run by hand on a GPU"]
fn matches_the_reference_on_cuda() {
    let Some(device) = cuda_device() else {
        eprintln!("no CUDA device: nothing to report");
        return;
    };
    sweep(&device);
    non_default_scale(&device);
}
