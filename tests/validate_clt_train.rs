// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)]
// EXPLICIT: integration-test code — assertions panic by design.

//! **The trainable CLT is askesis's reference CLT, and this proves it.**
//!
//! `scripts/clt_train_validation.py` runs the `CrossLayerTranscoder` of askesis (`reference/scripts/clt.py`, the class
//! its MDLM-owt CLTs were trained with) on one tiny fixed case and records every intermediate in
//! `scripts/clt_train_reference.json`. This test loads the same weights into [`TrainableClt`] and checks, at the
//! reference's stated bar (`1e-4`): the `TopK` activations, the reconstruction, the loss, the gradient of every
//! parameter, and the parameters after one Adam step (candle-mi's `AdamW` with zero weight decay is `torch.optim.Adam`).
//! Then the export: the circuit-tracer files, read back, reproduce the encoder and decoder in raw units, and
//! `CrossLayerTranscoder::open_local` on them encodes (with `TopK`) and decodes exactly as the trained CLT.
//!
//! CPU-only and tiny, so it is not `#[ignore]`d. Not yet in a CI lane: `ci.yml`'s clt lane
//! runs `--lib` only.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_mi::clt::train::{TrainableClt, TrainableCltConfig};
use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::optim::{AdamW, ParamsAdamW};
use serde_json::Value;

const BAR: f32 = 1e-4;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/scripts/clt_train_reference.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture present"))
        .expect("fixture parses")
}

/// A nested JSON array as a flat `Vec<f32>` and its shape.
fn flat(v: &Value) -> (Vec<f32>, Vec<usize>) {
    match v {
        Value::Array(items) => {
            let mut data = Vec::new();
            let mut inner = Vec::new();
            for item in items {
                let (d, s) = flat(item);
                data.extend(d);
                inner = s;
            }
            let mut shape = vec![items.len()];
            shape.extend(inner);
            (data, shape)
        }
        Value::Number(n) => (vec![n.as_f64().unwrap() as f32], vec![]),
        other @ (Value::Null | Value::Bool(_) | Value::String(_) | Value::Object(_)) => {
            panic!("unexpected JSON value {other}")
        }
    }
}

fn tensor(v: &Value) -> Tensor {
    let (data, shape) = flat(v);
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "shapes differ");
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn assert_close(name: &str, ours: &Tensor, reference: &Tensor) {
    let diff = max_abs_diff(ours, reference);
    assert!(diff <= BAR, "{name}: max |diff| {diff:e} > {BAR:e}");
}

struct Case {
    clt: TrainableClt,
    x: Tensor,
    y: Tensor,
    fx: Value,
}

/// The reference's weights in [`TrainableClt`]'s layout: `torch.nn.Linear` stores the decoder as
/// `[(L - l) * d, f]`, ours is its transpose.
fn case() -> Case {
    let fx = fixture();
    let c = &fx["config"];
    let n = |k: &str| c[k].as_u64().unwrap() as usize;
    let config = TrainableCltConfig::new(n("n_layers"), n("d_model"), n("n_features"), n("top_k"));
    let p = &fx["params"];
    let w_dec: Vec<Tensor> = (0..config.n_layers)
        .map(|l| {
            tensor(&p[format!("dec_{l}")])
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
        })
        .collect();
    let clt = TrainableClt::from_tensors(
        &config,
        &tensor(&p["W_enc"]),
        &tensor(&p["b_enc"]),
        &w_dec,
        &tensor(&p["b_dec"]),
    )
    .unwrap();
    Case {
        clt,
        x: tensor(&fx["x"]),
        y: tensor(&fx["y"]),
        fx,
    }
}

/// Our parameter `name` in the reference's layout (decoders transposed back).
fn as_reference(name: &str, t: &Tensor) -> (String, Tensor) {
    // BORROW: explicit .to_owned() — the reference's parameter names are owned keys
    name.strip_prefix("w_dec_").map_or_else(
        || {
            (
                if name == "w_enc" { "W_enc" } else { name }.to_owned(),
                t.clone(),
            )
        },
        |l| (format!("dec_{l}"), t.t().unwrap().contiguous().unwrap()),
    )
}

#[test]
fn forward_matches_the_reference() {
    let Case { clt, x, y, fx } = case();
    let (recon, acts) = clt.forward(&x).unwrap();
    assert_close("acts", &acts, &tensor(&fx["acts"]));
    assert_close("recon", &recon, &tensor(&fx["recon"]));
    let loss = clt.loss(&x, &y).unwrap().to_scalar::<f32>().unwrap();
    let reference = fx["loss"].as_f64().unwrap() as f32;
    assert!(
        (loss - reference).abs() <= BAR,
        "loss {loss} vs {reference}"
    );
    // TopK is binding in this fixture: exactly k active per (token, layer).
    let active = acts
        .gt(0.0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    // CAST: f32 → usize, a count of 0/1 entries summed exactly in f32 (far below 2^24)
    #[allow(clippy::cast_sign_loss)]
    let active = active as usize;
    let c = clt.config();
    let x_dims = x.dims();
    assert_eq!(active, x_dims[0] * c.n_layers * c.top_k);
}

#[test]
fn gradients_and_one_adam_step_match_the_reference() {
    let Case { clt, x, y, fx } = case();
    let loss = clt.loss(&x, &y).unwrap();
    let grads = loss.backward().unwrap();
    for (name, var) in clt.vars() {
        let g = grads
            .get(var.as_tensor())
            .expect("every parameter gets a gradient");
        let (ref_name, ours) = as_reference(&name, g);
        assert_close(
            &format!("grad {name}"),
            &ours,
            &tensor(&fx["grads"][&ref_name]),
        );
    }
    let a = &fx["adam"];
    let mut opt = AdamW::new(
        clt.vars(),
        ParamsAdamW {
            lr: a["lr"].as_f64().unwrap(),
            beta1: a["beta1"].as_f64().unwrap(),
            beta2: a["beta2"].as_f64().unwrap(),
            eps: a["eps"].as_f64().unwrap(),
            weight_decay: 0.0,
        },
    )
    .unwrap();
    opt.step(&grads).unwrap();
    for (name, var) in clt.vars() {
        let (ref_name, ours) = as_reference(&name, var.as_tensor());
        assert_close(
            &format!("after step {name}"),
            &ours,
            &tensor(&fx["after_step"][&ref_name]),
        );
    }
}

#[test]
fn export_folds_the_scales_into_raw_units() {
    let Case {
        clt,
        x,
        fx: _,
        y: _,
    } = case();
    let c = clt.config();
    let scale_in = [0.5f32, 2.0, 1.25];
    let scale_out = [3.0f32, 0.25, 1.5];
    let dir = std::env::temp_dir().join(format!("candle_mi_clt_export_{}", std::process::id()));
    clt.export_circuit_tracer(&dir, &scale_in, &scale_out, "fixture")
        .unwrap();
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("candle_mi_clt.json")).unwrap())
            .unwrap();
    assert_eq!(meta["top_k"].as_u64().unwrap() as usize, c.top_k);

    // Raw activations: x_raw[l] = x[l] * scale_in[l]. Reading the exported files, the raw CLT must give the same
    // activations as the scaled one, and a reconstruction equal to the scaled one times scale_out[t].
    let s_in = Tensor::from_slice(&scale_in, (1, c.n_layers, 1), &Device::Cpu).unwrap();
    let s_out = Tensor::from_slice(&scale_out, (1, c.n_layers, 1), &Device::Cpu).unwrap();
    let x_raw = x.broadcast_mul(&s_in).unwrap();
    let load = |name: &str| -> HashMap<String, Tensor> {
        candle_core::safetensors::load(dir.join(name), &Device::Cpu).unwrap()
    };
    let mut w_enc = Vec::new();
    let mut b_enc = Vec::new();
    let mut b_dec = Vec::new();
    let mut w_dec = Vec::new();
    for l in 0..c.n_layers {
        let e = load(&format!("W_enc_{l}.safetensors"));
        assert_eq!(e[&format!("W_enc_{l}")].dims(), &[c.n_features, c.d_model]);
        w_enc.push(e[&format!("W_enc_{l}")].t().unwrap());
        b_enc.push(e[&format!("b_enc_{l}")].clone());
        b_dec.push(e[&format!("b_dec_{l}")].clone());
        let dmat = load(&format!("W_dec_{l}.safetensors"))[&format!("W_dec_{l}")].clone();
        assert_eq!(dmat.dims(), &[c.n_features, c.n_layers - l, c.d_model]);
        w_dec.push(
            dmat.reshape((c.n_features, (c.n_layers - l) * c.d_model))
                .unwrap(),
        );
    }
    let raw = TrainableClt::from_tensors(
        &c,
        &Tensor::stack(&w_enc, 0).unwrap(),
        &Tensor::stack(&b_enc, 0).unwrap(),
        &w_dec,
        &Tensor::stack(&b_dec, 0).unwrap(),
    )
    .unwrap();
    let (recon, acts) = clt.forward(&x).unwrap();
    let (recon_raw, acts_raw) = raw.forward(&x_raw).unwrap();
    assert_close("raw acts", &acts_raw, &acts);
    assert_close(
        "raw recon",
        &recon_raw,
        &recon.broadcast_mul(&s_out).unwrap(),
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn open_local_encodes_and_decodes_like_the_trained_clt() {
    let Case { clt, x, .. } = case();
    let c = clt.config();
    let ones = vec![1.0f32; c.n_layers];
    let dir = std::env::temp_dir().join(format!("candle_mi_clt_open_local_{}", std::process::id()));
    clt.export_circuit_tracer(&dir, &ones, &ones, "fixture")
        .unwrap();
    let mut local = CrossLayerTranscoder::open_local(&dir).unwrap();
    let cfg = local.config();
    assert_eq!(
        (
            cfg.n_layers,
            cfg.d_model,
            cfg.n_features_per_layer,
            cfg.top_k,
            cfg.model_name.as_str()
        ),
        (
            c.n_layers,
            c.d_model,
            c.n_features,
            Some(c.top_k),
            "fixture"
        )
    );

    // encode, one position at a time, equals the trained CLT's TopK activations.
    let acts = clt.encode(&x).unwrap();
    for l in 0..c.n_layers {
        local.load_encoder(l, &Device::Cpu).unwrap();
        for n in 0..x.dims()[0] {
            let residual = x.get(n).unwrap().get(l).unwrap();
            let sparse = local.encode(&residual, l).unwrap();
            assert!(sparse.features.len() <= c.top_k, "more than top_k active");
            let mut dense = vec![0f32; c.n_features];
            for (id, v) in &sparse.features {
                assert_eq!(id.layer, l);
                dense[id.index] = *v;
            }
            let dense = Tensor::from_vec(dense, c.n_features, &Device::Cpu).unwrap();
            assert_close(
                &format!("encode n{n} l{l}"),
                &dense,
                &acts.get(n).unwrap().get(l).unwrap(),
            );
        }
    }

    // decoder_vector(feature, target) is the trained decoder's row, at the target layer's block.
    let (_, w_dec_1) = clt
        .vars()
        .into_iter()
        .find(|(name, _)| name == "w_dec_1")
        .unwrap();
    let feature = CltFeatureId { layer: 1, index: 3 };
    for target in 1..c.n_layers {
        let ours = local
            .decoder_vector(&feature, target, &Device::Cpu)
            .unwrap();
        let trained = w_dec_1
            .as_tensor()
            .get(3)
            .unwrap()
            .narrow(0, (target - 1) * c.d_model, c.d_model)
            .unwrap();
        assert_close(&format!("decoder_vector target {target}"), &ours, &trained);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
