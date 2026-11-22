// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for `PlainTransformer`: config, the weight table, the strict
//! loader, input validation and the hook surface. The numerical checks against
//! independent references (a plain-loop forward, `OthelloGpt`, a hand
//! construction, `Tracr`) live in `tests/`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor};

use super::{PlainTransformer, PlainTransformerConfig};
use crate::backend::MIBackend;
use crate::config::{Activation, NormType};
use crate::error::MIError;
use crate::hooks::{HookPoint, HookSpec, Intervention};

/// Deterministic values in `[-0.5, 0.5)` from a 64-bit LCG, so the tests need
/// no RNG crate and are identical on every run and platform.
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

/// A full, correctly shaped tensor map for `config`, with seeded values.
fn tensors_for(config: &PlainTransformerConfig, seed: u64) -> HashMap<String, Tensor> {
    let dev = Device::Cpu;
    config
        .weight_shapes()
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let n = dims.iter().product();
            // CAST: usize → u64, a tensor index, far below u64::MAX
            #[allow(clippy::as_conversions)]
            let values = lcg_values(seed + i as u64, n);
            (name, Tensor::from_vec(values, dims, &dev).unwrap())
        })
        .collect()
}

/// Two layers, 3 heads of 4 dims in a 10-dim residual (so `n_head * head_dim`
/// is 12, not `d_model`), MLP width 6, 7 input tokens, 5 outputs.
fn odd_config() -> PlainTransformerConfig {
    PlainTransformerConfig::new(7, 2, 10, 3, 4, 6)
        .unwrap()
        .with_output_size(5)
        .with_positions(8)
        .with_bias(true)
}

fn model(config: PlainTransformerConfig, seed: u64) -> PlainTransformer {
    let tensors = tensors_for(&config, seed);
    PlainTransformer::from_tensors(config, tensors, &Device::Cpu).unwrap()
}

fn ids(rows: &[&[u32]]) -> Tensor {
    let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (rows.len(), rows[0].len()), &Device::Cpu).unwrap()
}

fn config_err(result: crate::error::Result<impl Sized>) -> String {
    match result {
        Err(MIError::Config(msg)) => msg,
        Err(other) => panic!("expected MIError::Config, got {other:?}"),
        Ok(_) => panic!("expected MIError::Config, got Ok"),
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
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

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

// --- config -----------------------------------------------------------------

#[test]
fn new_gives_the_constructed_model_defaults() {
    let cfg = PlainTransformerConfig::new(7, 2, 10, 3, 4, 6).unwrap();
    assert_eq!(cfg.output_size, 7, "output_size defaults to vocab_size");
    assert_eq!(cfg.max_positions, None);
    assert_eq!(cfg.activation, Activation::Relu);
    assert_eq!(cfg.norm, None);
    assert!(!cfg.causal);
    assert!(!cfg.bias);
    assert_eq!(cfg.attn_width(), 12);
}

// The default scale is the f64 `OthelloGpt` computes, to the bit, including at
// a head_dim whose square root is irrational (48), where an f32 scale or a
// different formula would differ in the last place.
#[test]
fn default_attn_scale_is_othellogpt_f64_formula_to_the_bit() {
    for head_dim in [1_usize, 4, 48, 64, 96] {
        let cfg = PlainTransformerConfig::new(3, 1, 8, 1, head_dim, 4).unwrap();
        // CAST: usize → f64, test dimension
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let expected = 1.0 / (head_dim as f64).sqrt();
        assert_eq!(
            cfg.attn_scale.to_bits(),
            expected.to_bits(),
            "head_dim {head_dim}"
        );
    }
}

#[test]
fn validate_rejects_each_invalid_config() {
    for (name, cfg) in [
        ("vocab_size", PlainTransformerConfig::new(0, 1, 4, 1, 4, 4)),
        ("n_layer", PlainTransformerConfig::new(3, 0, 4, 1, 4, 4)),
        ("d_model", PlainTransformerConfig::new(3, 1, 0, 1, 4, 4)),
        ("n_head", PlainTransformerConfig::new(3, 1, 4, 0, 4, 4)),
        ("head_dim", PlainTransformerConfig::new(3, 1, 4, 1, 0, 4)),
        ("mlp_hidden", PlainTransformerConfig::new(3, 1, 4, 1, 4, 0)),
    ] {
        assert!(config_err(cfg).contains(name), "{name}");
    }

    let base = PlainTransformerConfig::new(3, 1, 4, 1, 4, 4).unwrap();
    assert!(config_err(base.clone().with_output_size(0).validate()).contains("output_size"));
    let mut zero_pos = base.clone();
    zero_pos.max_positions = Some(0);
    assert!(config_err(zero_pos.validate()).contains("max_positions"));
    assert!(config_err(base.clone().with_attn_scale(f64::NAN).validate()).contains("attn_scale"));
    assert!(
        config_err(base.clone().with_attn_scale(f64::INFINITY).validate()).contains("attn_scale")
    );
    assert!(
        config_err(base.clone().with_norm(NormType::GemmaRmsNorm).validate())
            .contains("GemmaRmsNorm")
    );
    assert!(
        config_err(
            base.clone()
                .with_norm(NormType::LayerNorm)
                .with_norm_eps(0.0)
                .validate()
        )
        .contains("norm_eps")
    );
    // An epsilon is irrelevant without a norm, so it is not checked then.
    assert!(base.with_norm_eps(0.0).validate().is_ok());
}

// --- weight table -----------------------------------------------------------

#[test]
fn weight_shapes_lists_exactly_the_configured_tensors() {
    let bare = PlainTransformerConfig::new(7, 2, 10, 3, 4, 6).unwrap();
    // tok_emb + per layer (q, k, v, o, fc, proj) + unembed.
    assert_eq!(bare.weight_shapes().len(), 1 + 2 * 6 + 1);

    let full = odd_config().with_norm(NormType::LayerNorm);
    // + pos_emb; per layer + 6 biases + ln1/ln2 weight and bias; + ln_f weight and bias.
    assert_eq!(full.weight_shapes().len(), 2 + 2 * (6 + 6 + 4) + 2 + 1);

    let rms = odd_config().with_norm(NormType::RmsNorm);
    // RmsNorm has a weight and no bias at each of the 5 norm sites.
    assert_eq!(rms.weight_shapes().len(), 2 + 2 * (6 + 6 + 2) + 1 + 1);

    let shapes: HashMap<String, Vec<usize>> = full.weight_shapes().into_iter().collect();
    assert_eq!(shapes["tok_emb.weight"], vec![7, 10]);
    assert_eq!(shapes["pos_emb.weight"], vec![8, 10]);
    assert_eq!(shapes["blocks.1.attn.q.weight"], vec![12, 10]);
    assert_eq!(shapes["blocks.1.attn.v.bias"], vec![12]);
    assert_eq!(shapes["blocks.1.attn.o.weight"], vec![10, 12]);
    assert_eq!(shapes["blocks.0.mlp.fc.weight"], vec![6, 10]);
    assert_eq!(shapes["blocks.0.mlp.proj.weight"], vec![10, 6]);
    assert_eq!(shapes["ln_f.bias"], vec![10]);
    assert_eq!(shapes["unembed.weight"], vec![5, 10]);
    assert!(
        !shapes.contains_key("unembed.bias"),
        "the unembed has no bias"
    );
}

// --- strict loader ------------------------------------------------------------

#[test]
fn from_tensors_accepts_exactly_the_table() {
    let cfg = odd_config().with_norm(NormType::LayerNorm);
    let tensors = tensors_for(&cfg, 1);
    assert!(PlainTransformer::from_tensors(cfg, tensors, &Device::Cpu).is_ok());
}

#[test]
fn from_tensors_reports_every_problem_by_name() {
    let cfg = odd_config();
    let mut tensors = tensors_for(&cfg, 2);
    let dev = Device::Cpu;

    // One of each kind of mistake.
    tensors.remove("blocks.0.attn.k.weight"); // missing
    tensors.insert(
        "blocks.0.attn.k.wieght".to_string(), // the typo the check exists for
        Tensor::zeros((12, 10), DType::F32, &dev).unwrap(),
    );
    tensors.insert(
        "blocks.1.mlp.fc.weight".to_string(), // wrong shape (transposed)
        Tensor::zeros((10, 6), DType::F32, &dev).unwrap(),
    );
    tensors.insert(
        "unembed.weight".to_string(), // wrong dtype
        Tensor::zeros((5, 10), DType::F64, &dev).unwrap(),
    );

    let msg = config_err(PlainTransformer::from_tensors(cfg, tensors, &dev));
    assert!(msg.contains("4 problem(s)"), "{msg}");
    assert!(
        msg.contains("missing tensor `blocks.0.attn.k.weight`"),
        "{msg}"
    );
    assert!(
        msg.contains("unexpected tensor `blocks.0.attn.k.wieght`"),
        "{msg}"
    );
    assert!(
        msg.contains("`blocks.1.mlp.fc.weight` has shape [10, 6] (expected [6, 10])"),
        "{msg}"
    );
    assert!(
        msg.contains("`unembed.weight` has dtype F64 (expected F32)"),
        "{msg}"
    );
}

// With `bias = false`, biases are unexpected tensors, not silently ignored: a
// Tracr export (which always carries zero biases) must be loaded with
// `bias = true`, as the design requires.
#[test]
fn from_tensors_rejects_biases_the_config_does_not_have() {
    let with_bias = odd_config();
    let tensors = tensors_for(&with_bias, 3);
    let without_bias = with_bias.with_bias(false);
    let msg = config_err(PlainTransformer::from_tensors(
        without_bias,
        tensors,
        &Device::Cpu,
    ));
    assert!(
        msg.contains("unexpected tensor `blocks.0.attn.q.bias`"),
        "{msg}"
    );
}

// --- input validation ---------------------------------------------------------

#[test]
fn forward_rejects_inputs_it_cannot_take() {
    let m = model(odd_config(), 4);
    let model_err = |t: Tensor| match MIBackend::forward(&m, &t, &HookSpec::new()) {
        Err(MIError::Model(e)) => e.to_string(),
        Err(other) => panic!("expected MIError::Model, got {other:?}"),
        Ok(_) => panic!("expected an error"),
    };
    // id == vocab_size is out of range (ids are 0..7).
    assert!(model_err(ids(&[&[0, 7]])).contains("input id 7 out of range"));
    // 9 positions with max_positions = 8.
    assert!(model_err(ids(&[&[0; 9]])).contains("exceeds max_positions 8"));
    // Not U32.
    let i64_ids = Tensor::new(&[[0_i64, 1]], &Device::Cpu).unwrap();
    assert!(model_err(i64_ids).contains("not U32"));
    // Empty.
    let empty = Tensor::zeros((1, 0), DType::U32, &Device::Cpu).unwrap();
    assert!(model_err(empty).contains("empty"));
}

#[test]
fn no_positions_means_no_length_limit() {
    let cfg = PlainTransformerConfig::new(7, 1, 10, 3, 4, 6).unwrap();
    let m = model(cfg, 5);
    let out = MIBackend::forward(&m, &ids(&[&[1; 40]]), &HookSpec::new()).unwrap();
    assert_eq!(out.output().dims(), &[1, 40, 7]);
}

// --- hook surface -------------------------------------------------------------

#[test]
fn every_hook_fires_with_its_documented_shape() {
    let cfg = odd_config().with_norm(NormType::LayerNorm);
    let m = model(cfg, 6);
    let input = ids(&[&[1, 2, 3, 4], &[6, 5, 0, 1]]);
    let (batch, seq, d_model, heads, head_dim, mlp_hidden, out) = (2, 4, 10, 3, 4, 6, 5);

    let mut spec = HookSpec::new();
    spec.capture(HookPoint::Embed).capture(HookPoint::FinalNorm);
    for l in 0..2 {
        for p in [
            HookPoint::ResidPre(l),
            HookPoint::AttnQ(l),
            HookPoint::AttnK(l),
            HookPoint::AttnV(l),
            HookPoint::AttnScores(l),
            HookPoint::AttnPattern(l),
            HookPoint::AttnOut(l),
            HookPoint::ResidMid(l),
            HookPoint::MlpPre(l),
            HookPoint::MlpAct(l),
            HookPoint::MlpPost(l),
            HookPoint::MlpOut(l),
            HookPoint::ResidPost(l),
        ] {
            spec.capture(p);
        }
    }
    let cache = MIBackend::forward(&m, &input, &spec).unwrap();
    let dims = |p: HookPoint| cache.require(&p).unwrap().dims().to_vec();

    assert_eq!(cache.output().dims(), &[batch, seq, out]);
    assert_eq!(dims(HookPoint::Embed), vec![batch, seq, d_model]);
    assert_eq!(dims(HookPoint::FinalNorm), vec![batch, seq, d_model]);
    for l in 0..2 {
        assert_eq!(dims(HookPoint::ResidPre(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::AttnQ(l)), vec![batch, heads, seq, head_dim]);
        assert_eq!(dims(HookPoint::AttnK(l)), vec![batch, heads, seq, head_dim]);
        assert_eq!(dims(HookPoint::AttnV(l)), vec![batch, heads, seq, head_dim]);
        assert_eq!(dims(HookPoint::AttnScores(l)), vec![batch, heads, seq, seq]);
        assert_eq!(
            dims(HookPoint::AttnPattern(l)),
            vec![batch, heads, seq, seq]
        );
        assert_eq!(dims(HookPoint::AttnOut(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::ResidMid(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::MlpPre(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::MlpAct(l)), vec![batch, seq, mlp_hidden]);
        assert_eq!(dims(HookPoint::MlpPost(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::MlpOut(l)), vec![batch, seq, d_model]);
        assert_eq!(dims(HookPoint::ResidPost(l)), vec![batch, seq, d_model]);

        // MlpPost and MlpOut are the same tensor here (no post-MLP norm).
        let post = cache.require(&HookPoint::MlpPost(l)).unwrap();
        let mlp_out = cache.require(&HookPoint::MlpOut(l)).unwrap();
        assert_eq!(bits(post), bits(mlp_out));
        // ReLU neurons are never negative.
        let act = cache.require(&HookPoint::MlpAct(l)).unwrap();
        let min: f32 = act
            .flatten_all()
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(min >= 0.0, "ReLU neuron below zero: {min}");
        // Each attention row is a distribution.
        let pattern = cache.require(&HookPoint::AttnPattern(l)).unwrap();
        let row_sums = pattern.sum(3).unwrap();
        let ones = Tensor::ones((batch, heads, seq), DType::F32, &Device::Cpu).unwrap();
        assert!(max_abs_diff(&row_sums, &ones) < 1e-6);
    }
}

// Capturing reads a tensor and puts it back untouched: the logits with every
// hook captured are the logits with none, to the bit.
#[test]
fn capturing_does_not_change_the_output() {
    let m = model(odd_config().with_norm(NormType::RmsNorm), 7);
    let input = ids(&[&[1, 2, 3, 4, 5]]);
    let plain = MIBackend::forward(&m, &input, &HookSpec::new()).unwrap();
    let mut spec = HookSpec::new();
    for l in 0..2 {
        spec.capture(HookPoint::AttnScores(l))
            .capture(HookPoint::MlpAct(l))
            .capture(HookPoint::ResidPost(l));
    }
    spec.capture(HookPoint::FinalNorm);
    let captured = MIBackend::forward(&m, &input, &spec).unwrap();
    assert_eq!(bits(plain.output()), bits(captured.output()));
}

#[test]
fn causal_mask_zeroes_every_future_position_exactly() {
    let m = model(odd_config().with_causal(true), 8);
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::AttnPattern(0));
    let cache = MIBackend::forward(&m, &ids(&[&[1, 2, 3, 4, 5, 6]]), &spec).unwrap();
    let pattern = cache.require(&HookPoint::AttnPattern(0)).unwrap();
    let values = pattern
        .i((0, .., .., ..))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    for head in &values {
        for (q, row) in head.iter().enumerate() {
            for (k, &p) in row.iter().enumerate() {
                if k > q {
                    assert_eq!(p.to_bits(), 0_f32.to_bits(), "future key {k} at query {q}");
                }
            }
        }
    }
}

// Zeroing the neurons at `MlpAct` zeroes the MLP's contribution exactly (no
// MLP biases in this config), so the residual leaves the layer unchanged.
#[test]
fn zero_intervention_at_mlp_act_removes_the_mlp() {
    let cfg = PlainTransformerConfig::new(7, 1, 10, 3, 4, 6)
        .unwrap()
        .with_positions(8);
    let m = model(cfg, 9);
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::ResidMid(0))
        .capture(HookPoint::MlpOut(0))
        .capture(HookPoint::ResidPost(0))
        .intervene(HookPoint::MlpAct(0), Intervention::Zero);
    let cache = MIBackend::forward(&m, &ids(&[&[1, 2, 3]]), &spec).unwrap();
    let mlp_out = cache.require(&HookPoint::MlpOut(0)).unwrap();
    assert!(bits(mlp_out).iter().all(|&b| b == 0), "MLP output not zero");
    assert_eq!(
        bits(cache.require(&HookPoint::ResidMid(0)).unwrap()),
        bits(cache.require(&HookPoint::ResidPost(0)).unwrap())
    );
}

// A positional patch at `MlpAct` (width mlp_hidden, not d_model) writes one
// position's neurons and leaves the others. A capture at the intervened point
// holds the value *before* the intervention (`hook_point` captures, then
// intervenes), so the effect is read downstream, at `MlpOut`.
#[test]
fn patch_at_mlp_act_writes_one_position() {
    let cfg = PlainTransformerConfig::new(7, 1, 10, 3, 4, 6)
        .unwrap()
        .with_positions(8);
    let tensors = tensors_for(&cfg, 10);
    let w_proj = tensors["blocks.0.mlp.proj.weight"].clone(); // [d_model, mlp_hidden]
    let m = PlainTransformer::from_tensors(cfg, tensors, &Device::Cpu).unwrap();
    let input = ids(&[&[1, 2, 3]]);
    let value = Tensor::full(2.5_f32, 6, &Device::Cpu).unwrap();

    let mut spec = HookSpec::new();
    spec.capture(HookPoint::MlpOut(0)).intervene(
        HookPoint::MlpAct(0),
        Intervention::PatchAt {
            position: 1,
            value: value.clone(),
        },
    );
    let patched = MIBackend::forward(&m, &input, &spec).unwrap();
    let mut clean_spec = HookSpec::new();
    clean_spec.capture(HookPoint::MlpOut(0));
    let clean = MIBackend::forward(&m, &input, &clean_spec).unwrap();

    let p = patched.require(&HookPoint::MlpOut(0)).unwrap();
    let c = clean.require(&HookPoint::MlpOut(0)).unwrap();
    // Position 1 now carries the patched neurons through the down projection
    // (no bias in this config). A tolerance, not bits: the expectation is
    // computed by a differently shaped matmul.
    let expected = value
        .reshape((1, 6))
        .unwrap()
        .matmul(&w_proj.t().unwrap())
        .unwrap()
        .squeeze(0)
        .unwrap();
    assert!(max_abs_diff(&p.i((0, 1)).unwrap(), &expected) < 1e-6);
    assert!(max_abs_diff(&p.i((0, 1)).unwrap(), &c.i((0, 1)).unwrap()) > 1e-3);
    // The other positions are untouched (the MLP is position-wise).
    assert_eq!(bits(&p.i((0, 0)).unwrap()), bits(&c.i((0, 0)).unwrap()));
    assert_eq!(bits(&p.i((0, 2)).unwrap()), bits(&c.i((0, 2)).unwrap()));
}

// `BACKENDS.md`'s conformance case: `Intervention::Zero` at `ResidPost(0)` must
// change the output. The weights are non-zero (seeded), since zeroing an
// already-zero residual is a no-op and would certify nothing.
#[test]
fn honours_intervention_at_resid_post() {
    let m = model(odd_config(), 13);
    let input = ids(&[&[1, 2, 3, 4]]);
    let clean = MIBackend::forward(&m, &input, &HookSpec::new()).unwrap();
    let mut spec = HookSpec::new();
    spec.intervene(HookPoint::ResidPost(0), Intervention::Zero);
    let zeroed = MIBackend::forward(&m, &input, &spec).unwrap();
    assert!(
        max_abs_diff(clean.output(), zeroed.output()) > 1e-3,
        "Zero at ResidPost(0) did not change the output: interventions are not wired"
    );
}

// The recommended knockout: an edge knocked out at `AttnScores` receives
// exactly zero attention, and the other rows are untouched.
#[test]
fn knockout_at_attn_scores_removes_exactly_that_edge() {
    use crate::interp::intervention::{KnockoutSpec, create_knockout_mask};

    let m = model(odd_config(), 14);
    let input = ids(&[&[1, 2, 3, 4]]);
    let spec_ko = KnockoutSpec::new().layer(0).edge(3, 0);
    let mask = create_knockout_mask(&spec_ko, 3, 4, &Device::Cpu, DType::F32).unwrap();

    let mut spec = HookSpec::new();
    spec.capture(HookPoint::AttnPattern(0))
        .intervene(HookPoint::AttnScores(0), Intervention::Knockout(mask));
    let ko = MIBackend::forward(&m, &input, &spec).unwrap();
    let mut clean_spec = HookSpec::new();
    clean_spec.capture(HookPoint::AttnPattern(0));
    let clean = MIBackend::forward(&m, &input, &clean_spec).unwrap();

    let p = ko.require(&HookPoint::AttnPattern(0)).unwrap();
    let c = clean.require(&HookPoint::AttnPattern(0)).unwrap();
    for head in 0..3 {
        let edge: f32 = p.i((0, head, 3, 0)).unwrap().to_scalar().unwrap();
        assert_eq!(
            edge.to_bits(),
            0_f32.to_bits(),
            "head {head}: knocked-out edge"
        );
        let before: f32 = c.i((0, head, 3, 0)).unwrap().to_scalar().unwrap();
        assert!(
            before > 0.0,
            "head {head}: the edge carried attention before"
        );
        for query in 0..3 {
            assert_eq!(
                bits(&p.i((0, head, query)).unwrap()),
                bits(&c.i((0, head, query)).unwrap()),
                "head {head}, query {query}: a row without a knocked-out edge changed"
            );
        }
    }
}

// `BACKENDS.md`: `project_to_vocab` preserves rank, and each position of the
// rank-3 result equals the rank-2 projection of that position.
#[test]
fn project_to_vocab_preserves_rank() {
    let m = model(odd_config().with_norm(NormType::LayerNorm), 15);
    let hidden = Tensor::from_vec(lcg_values(99, 2 * 3 * 10), (2, 3, 10), &Device::Cpu).unwrap();
    let seq_logits = m.project_to_vocab(&hidden).unwrap();
    assert_eq!(seq_logits.dims(), &[2, 3, 5]);
    for b in 0..2 {
        for s in 0..3 {
            let position = hidden.i((b, s)).unwrap().unsqueeze(0).unwrap();
            let expected = m.project_to_vocab(&position).unwrap();
            assert_eq!(expected.dims(), &[1, 5]);
            let actual = seq_logits.i((b, s)).unwrap().unsqueeze(0).unwrap();
            assert!(
                max_abs_diff(&actual, &expected) < 1e-6,
                "position ({b}, {s})"
            );
        }
    }
}

// `project_to_vocab` on the last residual is the logits, to the bit: the same
// final norm and unembed, so a logit lens at the last layer reads the output.
#[test]
fn project_to_vocab_of_last_residual_is_the_logits() {
    for norm in [None, Some(NormType::LayerNorm), Some(NormType::RmsNorm)] {
        let mut cfg = odd_config();
        cfg.norm = norm;
        let m = model(cfg, 11);
        let mut spec = HookSpec::new();
        spec.capture(HookPoint::ResidPost(1));
        let cache = MIBackend::forward(&m, &ids(&[&[1, 2, 3, 4]]), &spec).unwrap();
        let lens = m
            .project_to_vocab(cache.require(&HookPoint::ResidPost(1)).unwrap())
            .unwrap();
        assert_eq!(bits(&lens), bits(cache.output()), "norm {norm:?}");
    }
}

#[test]
fn vocab_size_is_the_logit_width_and_embedding_vector_indexes_the_input_table() {
    let cfg = odd_config();
    let tensors = tensors_for(&cfg, 12);
    let tok_emb = tensors["tok_emb.weight"].clone();
    let m = PlainTransformer::from_tensors(cfg, tensors, &Device::Cpu).unwrap();
    assert_eq!(m.vocab_size(), 5, "logit width (output_size)");
    assert_eq!(m.config().vocab_size, 7, "input vocabulary");
    assert_eq!(m.hidden_size(), 10);
    assert_eq!(m.num_heads(), 3);
    assert_eq!(m.num_layers(), 2);
    assert_eq!(
        bits(&m.embedding_vector(6).unwrap()),
        bits(&tok_emb.i(6).unwrap())
    );
    assert!(
        m.embedding_vector(7).is_err(),
        "id 7 is past the input table"
    );
}
