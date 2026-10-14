// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc
)]
// EXPLICIT: integration-test code — assertions panic by design.

//! **The replacement model splices what it says, closed loop.**
//!
//! A tiny seeded `OthelloGpt` and reconstructors whose output is known: the model's own clean MLP
//! outputs (so the replaced model must BE the clean model), zeros (so it must be the ablated
//! model), and the clean outputs shifted by a constant while recording what they are fed (so pass
//! `t` must see the residual of a run with layers `< t` already replaced — the closed loop, which
//! an open-loop splice would fail). Then a real `TrainableClt` through `ScaledClt`.
//!
//! CPU-only and tiny, so it is not `#[ignore]`d and runs in CI (`ci.yml`, step "Tests (CLT train +
//! replacement)"; mirrored in `publish.yml` and `scripts/preflight.ps1`).

use std::cell::RefCell;

use candle_core::{DType, Device, Tensor};
use candle_mi::clt::replacement::{
    Reconstruct, ScaledClt, Splice, cross_entropy_at, loss_recovered, replacement_logits_for,
    replacement_logits_where,
};
use candle_mi::clt::train::{TrainableClt, TrainableCltConfig};
use candle_mi::diffusion::{OthelloGpt, OthelloGptConfig};
use candle_mi::{HookPoint, HookSpec, Intervention, MIBackend};
use candle_nn::VarMap;

const LAYERS: usize = 3;
const D: usize = 8;

fn model() -> (OthelloGpt, Tensor) {
    let device = Device::Cpu;
    let config = OthelloGptConfig::new(12, 6, LAYERS, 2, D, false).unwrap();
    let varmap = VarMap::new();
    let model = OthelloGpt::init(config, &varmap, &device, 20_261_013).unwrap();
    let ids = Tensor::new(&[[1u32, 4, 7, 2, 9, 3], [5, 5, 0, 11, 6, 8]], &device).unwrap();
    (model, ids)
}

/// Every layer's clean MLP output, `[batch * seq, LAYERS, D]`.
fn clean_mlp(model: &OthelloGpt, ids: &Tensor) -> Tensor {
    let mut spec = HookSpec::new();
    for l in 0..LAYERS {
        spec.capture(HookPoint::MlpOut(l));
    }
    let cache = model.forward(ids, &spec).unwrap();
    let per_layer: Vec<Tensor> = (0..LAYERS)
        .map(|l| {
            cache
                .get(&HookPoint::MlpOut(l))
                .unwrap()
                .reshape(((), D))
                .unwrap()
        })
        .collect();
    Tensor::stack(&per_layer, 1).unwrap()
}

/// A reconstructor that ignores its input and returns `fixed`, recording every input it is fed.
struct Fixed {
    fixed: Tensor,
    fed: RefCell<Vec<Tensor>>,
}

impl Reconstruct for Fixed {
    fn n_layers(&self) -> usize {
        LAYERS
    }
    fn reconstruct(&self, x: &Tensor) -> candle_mi::Result<Tensor> {
        self.fed.borrow_mut().push(x.clone());
        Ok(self.fixed.clone())
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

#[test]
fn clean_is_the_model_and_replacing_with_its_own_mlps_changes_nothing() {
    let (model, ids) = model();
    let plain = model
        .forward(&ids, &HookSpec::new())
        .unwrap()
        .output()
        .clone();
    let own = Fixed {
        fixed: clean_mlp(&model, &ids),
        fed: RefCell::new(Vec::new()),
    };
    let clean = replacement_logits_for(&model, &ids, &own, Splice::Clean).unwrap();
    let replaced = replacement_logits_for(&model, &ids, &own, Splice::Replaced).unwrap();
    assert_eq!(max_abs_diff(&clean, &plain), 0.0);
    assert!(
        max_abs_diff(&replaced, &clean) < 1e-6,
        "{}",
        max_abs_diff(&replaced, &clean)
    );
    assert_eq!(
        own.fed.borrow().len(),
        LAYERS,
        "one reconstruction per pass"
    );
}

#[test]
fn replacing_with_zeros_is_ablating() {
    let (model, ids) = model();
    let zeros = Fixed {
        fixed: Tensor::zeros((12, LAYERS, D), DType::F32, &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let replaced = replacement_logits_for(&model, &ids, &zeros, Splice::Replaced).unwrap();
    let ablated = replacement_logits_for(&model, &ids, &zeros, Splice::Ablated).unwrap();
    assert_eq!(max_abs_diff(&replaced, &ablated), 0.0);
}

#[test]
fn pass_t_reads_the_spliced_residual_closed_loop() {
    let (model, ids) = model();
    let clean = clean_mlp(&model, &ids);
    let shifted = Fixed {
        fixed: clean.affine(1.0, 0.5).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let _ = replacement_logits_for(&model, &ids, &shifted, Splice::Replaced).unwrap();
    let last = LAYERS - 1;
    // What pass `last` must have read: ResidMid(last) of a run with layers < last replaced.
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::ResidMid(last));
    for l in 0..last {
        let r = shifted
            .fixed
            .narrow(1, l, 1)
            .unwrap()
            .reshape((2, 6, D))
            .unwrap();
        spec.intervene(HookPoint::MlpOut(l), Intervention::Replace(r));
    }
    let spliced = model.forward(&ids, &spec).unwrap();
    let want = spliced
        .get(&HookPoint::ResidMid(last))
        .unwrap()
        .reshape(((), D))
        .unwrap();
    let fed = shifted.fed.borrow()[last]
        .narrow(1, last, 1)
        .unwrap()
        .squeeze(1)
        .unwrap();
    assert!(
        max_abs_diff(&fed, &want) < 1e-6,
        "pass {last} was not fed the spliced residual"
    );
    // And it is not the clean residual: the loop is closed, not open.
    let mut plain = HookSpec::new();
    plain.capture(HookPoint::ResidMid(last));
    let open = model.forward(&ids, &plain).unwrap();
    let open = open
        .get(&HookPoint::ResidMid(last))
        .unwrap()
        .reshape(((), D))
        .unwrap();
    assert!(
        max_abs_diff(&fed, &open) > 1e-3,
        "the shift should reach layer {last}"
    );
}

#[test]
fn a_trained_transcoder_runs_through_scaled_clt() {
    let (model, ids) = model();
    let clt =
        TrainableClt::new(&TrainableCltConfig::new(LAYERS, D, 16, 4), 3, &Device::Cpu).unwrap();
    let scaled = ScaledClt::new(&clt, &[1.0, 2.0, 0.5], &[1.5, 1.0, 2.0]).unwrap();
    assert!(ScaledClt::new(&clt, &[1.0], &[1.0, 1.0, 1.0]).is_err());
    let positions: Vec<u32> = (0..12).collect();
    let targets: Vec<u32> = ids.flatten_all().unwrap().to_vec1().unwrap();
    let ce = |s| {
        let logits = replacement_logits_for(&model, &ids, &scaled, s).unwrap();
        assert_eq!(logits.dims(), &[2, 6, 12]);
        cross_entropy_at(&logits, &positions, &targets).unwrap()
    };
    let recovered = loss_recovered(ce(Splice::Clean), ce(Splice::Replaced), ce(Splice::Ablated));
    assert!(recovered.is_finite());
}

/// The model's forward, as `replacement_logits_where` takes it.
fn forward_of<'a>(
    model: &'a OthelloGpt,
    ids: &'a Tensor,
) -> impl Fn(&HookSpec) -> candle_mi::Result<candle_mi::HookCache> + 'a {
    move |hooks| model.forward(ids, hooks)
}

#[test]
fn splicing_every_position_is_the_plain_replacement_model() {
    let (model, ids) = model();
    let clt =
        TrainableClt::new(&TrainableCltConfig::new(LAYERS, D, 16, 4), 5, &Device::Cpu).unwrap();
    let scaled = ScaledClt::new(&clt, &[1.0, 2.0, 0.5], &[1.5, 1.0, 2.0]).unwrap();
    let everywhere = Tensor::ones((2, 6), DType::F32, &Device::Cpu).unwrap();
    for splice in [Splice::Replaced, Splice::Ablated] {
        let plain = replacement_logits_for(&model, &ids, &scaled, splice).unwrap();
        let masked =
            replacement_logits_where(forward_of(&model, &ids), &scaled, splice, &everywhere)
                .unwrap();
        assert!(
            max_abs_diff(&plain, &masked) < 1e-5,
            "{splice:?}: {}",
            max_abs_diff(&plain, &masked)
        );
    }
}

#[test]
fn splicing_no_position_is_the_clean_model() {
    let (model, ids) = model();
    let zeros = Fixed {
        fixed: Tensor::zeros((12, LAYERS, D), DType::F32, &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let clean = replacement_logits_for(&model, &ids, &zeros, Splice::Clean).unwrap();
    let nowhere = Tensor::zeros((2, 6), DType::U8, &Device::Cpu).unwrap();
    for splice in [Splice::Replaced, Splice::Ablated] {
        let masked =
            replacement_logits_where(forward_of(&model, &ids), &zeros, splice, &nowhere).unwrap();
        assert!(max_abs_diff(&clean, &masked) < 1e-6, "{splice:?}");
    }
}

#[test]
fn a_row_left_unmarked_keeps_the_model_while_a_marked_row_is_ablated() {
    // Attention mixes positions within a row, never across rows: so with row 0 marked and row 1
    // not, row 0 must be the ablated model's and row 1 the clean model's.
    let (model, ids) = model();
    let zeros = Fixed {
        fixed: Tensor::zeros((12, LAYERS, D), DType::F32, &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let row_zero = Tensor::new(&[[1u32; 6], [0u32; 6]], &Device::Cpu).unwrap();
    let mixed = replacement_logits_where(
        forward_of(&model, &ids),
        &zeros,
        Splice::Replaced,
        &row_zero,
    )
    .unwrap();
    let clean = replacement_logits_for(&model, &ids, &zeros, Splice::Clean).unwrap();
    let ablated = replacement_logits_for(&model, &ids, &zeros, Splice::Ablated).unwrap();
    let row = |t: &Tensor, r: usize| t.narrow(0, r, 1).unwrap();
    assert!(max_abs_diff(&row(&mixed, 0), &row(&ablated, 0)) < 1e-6);
    assert!(max_abs_diff(&row(&mixed, 1), &row(&clean, 1)) < 1e-6);
}

#[test]
fn a_mask_of_the_wrong_shape_is_refused() {
    let (model, ids) = model();
    let zeros = Fixed {
        fixed: Tensor::zeros((12, LAYERS, D), DType::F32, &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let wrong = Tensor::ones((2, 5), DType::F32, &Device::Cpu).unwrap();
    assert!(
        replacement_logits_where(forward_of(&model, &ids), &zeros, Splice::Replaced, &wrong)
            .is_err()
    );
}

#[test]
fn a_mask_within_a_row_ablates_exactly_the_marked_positions_closed_loop() {
    // An independent oracle: zero each layer's MLP output at the marked positions only, layer by
    // layer, each pass running on the residual the previous zeroings produced.
    let (model, ids) = model();
    let zeros = Fixed {
        fixed: Tensor::zeros((12, LAYERS, D), DType::F32, &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let marks = [[1u32, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1]];
    let at = Tensor::new(&marks, &Device::Cpu).unwrap();
    let got =
        replacement_logits_where(forward_of(&model, &ids), &zeros, Splice::Replaced, &at).unwrap();
    let keep = Tensor::new(
        &[
            [[0f32], [1.0], [0.0], [0.0], [1.0], [1.0]],
            [[1.0], [0.0], [1.0], [1.0], [0.0], [0.0]],
        ],
        &Device::Cpu,
    )
    .unwrap();
    let mut done: Vec<Tensor> = Vec::new();
    for t in 0..LAYERS {
        let mut spec = HookSpec::new();
        spec.capture(HookPoint::MlpOut(t));
        for (l, r) in done.iter().enumerate() {
            spec.intervene(HookPoint::MlpOut(l), Intervention::Replace(r.clone()));
        }
        let own = model
            .forward(&ids, &spec)
            .unwrap()
            .get(&HookPoint::MlpOut(t))
            .unwrap()
            .clone();
        done.push(own.broadcast_mul(&keep).unwrap());
    }
    let mut spec = HookSpec::new();
    for (l, r) in done.into_iter().enumerate() {
        spec.intervene(HookPoint::MlpOut(l), Intervention::Replace(r));
    }
    let want = model.forward(&ids, &spec).unwrap().output().clone();
    assert!(
        max_abs_diff(&got, &want) < 1e-6,
        "{}",
        max_abs_diff(&got, &want)
    );
}

#[test]
fn a_non_finite_reconstruction_never_reaches_an_unmarked_position() {
    let (model, ids) = model();
    let nan = Fixed {
        fixed: Tensor::full(f32::NAN, (12, LAYERS, D), &Device::Cpu).unwrap(),
        fed: RefCell::new(Vec::new()),
    };
    let clean = replacement_logits_for(&model, &ids, &nan, Splice::Clean).unwrap();
    let nowhere = Tensor::zeros((2, 6), DType::U8, &Device::Cpu).unwrap();
    let masked =
        replacement_logits_where(forward_of(&model, &ids), &nan, Splice::Replaced, &nowhere)
            .unwrap();
    assert!(
        max_abs_diff(&clean, &masked) < 1e-6,
        "an unmarked position took the NaN reconstruction"
    );
}
