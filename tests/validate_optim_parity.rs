// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::as_conversions,
    clippy::cast_precision_loss
)]
// EXPLICIT: integration-test code — assertions panic by design.

//! **The optimizer is candle's, and this proves it.**
//!
//! `candle_mi::optim::AdamW` exists only to make the moments and the step
//! counter reachable; the update rule is transcribed from `candle-nn` 0.11.0.
//! So the acceptance test is not "does it descend" — it is "does it reproduce
//! candle's trajectory, step for step". Anything less and the honest
//! description would be "a hand-rolled optimizer", with all the doubt that
//! carries.
//!
//! The third test is the one a staged run actually rests on: a checkpointed and
//! resumed optimizer must continue exactly as one that never stopped. The
//! fourth is its power control — without it, the third could pass vacuously.
//!
//! CPU-only and tiny, so unlike the oracle suite these are **not** `#[ignore]`d
//! and run in CI.

use candle_core::{Device, Tensor, Var};
use candle_mi::optim::{AdamW, ParamsAdamW};

/// A fixed, non-symmetric starting point — asymmetric so a coordinate mix-up
/// cannot hide behind a symmetric fixture.
const START: [f32; 6] = [0.7, -1.3, 2.5, 0.02, -0.4, 1.1];
/// A fixed target, so the loss (and hence every gradient) is deterministic.
const TARGET: [f32; 6] = [-0.2, 0.9, 1.0, -1.7, 0.35, 0.0];

const fn params() -> ParamsAdamW {
    // Non-zero weight decay so the decoupled-decay path is exercised too, not
    // just the moments.
    ParamsAdamW {
        lr: 0.05,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.1,
    }
}

/// `sum((w - target)^2)` — convex, deterministic, gradient touches every coord.
fn loss_of(w: &Var, target: &Tensor) -> Tensor {
    (w.as_tensor() - target)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
}

/// The learning rate at `step`, so both optimizers see the same varying
/// schedule — a constant rate would not exercise `set_learning_rate`.
fn lr_at(step: usize) -> f64 {
    // CAST: usize → f64, an optimizer step count, far inside f64's exactly-representable
    // integer range
    0.05 * (step as f64).mul_add(0.01, 1.0).recip()
}

fn with_candle(steps: usize) -> Vec<f32> {
    use candle_nn::Optimizer;

    let device = Device::Cpu;
    let w = Var::from_slice(&START, (START.len(),), &device).unwrap();
    let target = Tensor::from_slice(&TARGET, (TARGET.len(),), &device).unwrap();
    let mut opt = candle_nn::AdamW::new(vec![w.clone()], params()).unwrap();
    for step in 0..steps {
        opt.set_learning_rate(lr_at(step));
        let loss = loss_of(&w, &target);
        opt.step(&loss.backward().unwrap()).unwrap();
    }
    w.as_tensor().to_vec1::<f32>().unwrap()
}

fn with_ours(steps: usize) -> Vec<f32> {
    let device = Device::Cpu;
    let w = Var::from_slice(&START, (START.len(),), &device).unwrap();
    let target = Tensor::from_slice(&TARGET, (TARGET.len(),), &device).unwrap();
    let mut opt = AdamW::new(vec![("w".to_owned(), w.clone())], params()).unwrap();
    for step in 0..steps {
        opt.set_learning_rate(lr_at(step));
        let loss = loss_of(&w, &target);
        opt.step(&loss.backward().unwrap()).unwrap();
    }
    w.as_tensor().to_vec1::<f32>().unwrap()
}

#[test]
fn our_adamw_matches_candles_step_for_step() {
    for steps in [1_usize, 2, 17, 120] {
        let (theirs, ours) = (with_candle(steps), with_ours(steps));
        for (index, (a, b)) in theirs.iter().zip(ours.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "after {steps} steps, coordinate {index} differs: candle {a} vs ours {b}"
            );
        }
    }
}

#[test]
fn the_parameters_actually_moved() {
    // Guards the test above against passing vacuously: two optimizers that both
    // did nothing would also "agree".
    let settled = with_ours(120);
    let moved: f32 = settled
        .iter()
        .zip(START.iter())
        .map(|(end, start)| (end - start).abs())
        .fold(0.0, f32::max);
    assert!(
        moved > 0.5,
        "the fixture must exercise a real descent (max move {moved})"
    );
}

#[test]
fn a_resumed_optimizer_continues_as_if_it_never_stopped() {
    // The property a staged run rests on. Run 60 steps straight; then run 25,
    // checkpoint the moments AND the step counter, restore both into a fresh
    // optimizer, and run the remaining 35. The two must land on the same
    // parameters.
    let device = Device::Cpu;
    let target = Tensor::from_slice(&TARGET, (TARGET.len(),), &device).unwrap();

    let straight = with_ours(60);

    let w = Var::from_slice(&START, (START.len(),), &device).unwrap();
    let mut opt = AdamW::new(vec![("w".to_owned(), w.clone())], params()).unwrap();
    for step in 0..25 {
        opt.set_learning_rate(lr_at(step));
        let loss = loss_of(&w, &target);
        opt.step(&loss.backward().unwrap()).unwrap();
    }
    let saved_state = opt.state();
    let saved_step = opt.steps_taken();
    let saved_weights = w.as_tensor().to_vec1::<f32>().unwrap();
    drop(opt);

    // A fresh process would rebuild both the parameter and the optimizer from
    // disk; this mimics that without touching the filesystem.
    let resumed_w = Var::from_slice(&saved_weights, (saved_weights.len(),), &device).unwrap();
    let mut resumed = AdamW::new(vec![("w".to_owned(), resumed_w.clone())], params()).unwrap();
    let restored = resumed.restore(&saved_state, saved_step).unwrap();
    assert_eq!(
        restored,
        resumed.len(),
        "the checkpoint must carry every parameter's moments"
    );
    assert_eq!(
        resumed.steps_taken(),
        25,
        "bias correction must resume at the right step"
    );

    for step in 25..60 {
        resumed.set_learning_rate(lr_at(step));
        let loss = loss_of(&resumed_w, &target);
        resumed.step(&loss.backward().unwrap()).unwrap();
    }
    let staged = resumed_w.as_tensor().to_vec1::<f32>().unwrap();

    for (index, (a, b)) in straight.iter().zip(staged.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "resumed run diverged at coordinate {index}: continuous {a} vs staged {b}"
        );
    }
}

#[test]
fn dropping_the_moments_would_be_detectable() {
    // The power control for the test above: if a resume silently lost the
    // moments, would we notice? Restore ONLY the step counter and check the
    // trajectory diverges — otherwise the resume test proves nothing.
    let device = Device::Cpu;
    let target = Tensor::from_slice(&TARGET, (TARGET.len(),), &device).unwrap();
    let straight = with_ours(60);

    let w = Var::from_slice(&START, (START.len(),), &device).unwrap();
    let mut opt = AdamW::new(vec![("w".to_owned(), w.clone())], params()).unwrap();
    for step in 0..25 {
        opt.set_learning_rate(lr_at(step));
        let loss = loss_of(&w, &target);
        opt.step(&loss.backward().unwrap()).unwrap();
    }
    let weights = w.as_tensor().to_vec1::<f32>().unwrap();

    let cold_w = Var::from_slice(&weights, (weights.len(),), &device).unwrap();
    let mut cold = AdamW::new(vec![("w".to_owned(), cold_w.clone())], params()).unwrap();
    // Deliberately NOT restoring the moments — only the step counter.
    cold.restore(&std::collections::BTreeMap::new(), 25)
        .unwrap();
    for step in 25..60 {
        cold.set_learning_rate(lr_at(step));
        let loss = loss_of(&cold_w, &target);
        cold.step(&loss.backward().unwrap()).unwrap();
    }
    let lost = cold_w.as_tensor().to_vec1::<f32>().unwrap();

    let divergence: f32 = straight
        .iter()
        .zip(lost.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        divergence > 1e-4,
        "losing the moments must be visible, else the resume test is vacuous (max diff {divergence})"
    );
}
