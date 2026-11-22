// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::float_cmp
)]
// EXPLICIT: integration-test code — assertions panic by design.
// EXPLICIT: `float_cmp` — every value asserted here is exact by construction
// (see the module docs), so the assertions compare bits, not tolerances.

//! **A hand construction with exactly known outputs**
//! (`design/plain-transformer.md` §7.3).
//!
//! One head in the spirit of the SNLP "establish" step: the query is an open
//! condition `p`, the keys are step effects, and the head attends to the steps
//! that add `p`. An MLP on top turns "which step was attended" into an
//! evidence value. The layout follows `Tracr`'s discipline: the residual stream
//! is a sum of named one-hot subspaces, there is no norm, and a constant `one`
//! direction plays the part of a bias.
//!
//! ```text
//! residual (14):  e_p0 e_p1 e_p2 | c_p0 c_p1 c_p2 | s_A s_B s_D | o_A o_B o_D | evidence | one
//! tokens:         A adds p0      B adds p1        D adds p0 and p2
//!                 C0, C1, C2: the open condition p0, p1, p2
//! head:           q = c, k = e, v = s, o-projection writes v into o
//!                 score(C_p -> t) = scale * [t adds p]
//! MLP:            n_j = ReLU(o_j - 0.25 * one),  evidence = sum_j n_j
//! unembed:        logits = (o_A, o_B, o_D, evidence)
//! ```
//!
//! **Why the asserted values are exact.** Softmax computes `exp(s - max) / sum`.
//! The attended (or tied) entries are `exp(0) = 1` exactly. Every other key
//! scores `scale` below them, and with `scale = 200` that gap is past `F32`
//! `exp` underflow (about 104), so the others are exactly `0` and the sum is
//! exactly the number of winners. All weights are `0`, `1` or `-0.25`, so every
//! product and sum downstream is exact too.
//!
//! **The tie is built exact, and checked exact first.** `A` and `D` both add
//! `p0`. Their key rows differ only in `e_p2`, where `C0`'s query is exactly
//! zero, so the two scores are the same terms in the same order: equal to the
//! bit. Positions are off (a positional embedding could break that), and
//! `seq <= 4` keeps both key rows in one matmul tile. The test asserts the
//! bitwise equality at `AttnScores` before it asserts the `0.5 / 0.5` pattern;
//! if that equality ever fails, suspect the kernel before the construction.
//! Under a large scale any difference between the two scores would pick a
//! winner: a tie is a knife-edge, and a construction that relies on one has to
//! make it exact by design.
//!
//! A third case uses `scale = 30`: a gap above the exactness bound
//! `24 ln 2 + ln(seq)` (about 17.8 here) but below underflow, so the winner is
//! exactly `1.0` while the losers are `exp(-30)`, tiny but **not zero**. That is
//! what "one-hot to `F32` precision" means when the gap is not huge.
//!
//! CPU-only, milliseconds: runs in CI (step "Tests (plain)").

use std::collections::HashMap;

use candle_core::{Device, IndexOp, Tensor};
use candle_mi::{HookPoint, HookSpec, MIBackend, PlainTransformer, PlainTransformerConfig};

// Residual-stream directions.
const E: usize = 0; // e_p0, e_p1, e_p2
const C: usize = 3; // c_p0, c_p1, c_p2
const S: usize = 6; // s_A, s_B, s_D
const O: usize = 9; // o_A, o_B, o_D
const EVIDENCE: usize = 12;
const ONE: usize = 13;
const D_MODEL: usize = 14;

// Token ids.
const TOK_A: u32 = 0;
const TOK_B: u32 = 1;
const TOK_D: u32 = 2;
const TOK_C0: u32 = 3;
const TOK_C1: u32 = 4;

/// Steps, as (token row, propositions added, step index).
const STEPS: [(usize, &[usize], usize); 3] = [(0, &[0], 0), (1, &[1], 1), (2, &[0, 2], 2)];

/// A `[rows, cols]` matrix of zeros with the given ones set.
fn matrix(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> Tensor {
    let mut data = vec![0.0_f32; rows * cols];
    for &(r, c, v) in entries {
        data[r * cols + c] = v;
    }
    Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
}

fn construction(attn_scale: f64, device: &Device) -> PlainTransformer {
    // 6 input tokens (A, B, D, C0, C1, C2), 1 layer, 1 head of 3 dims, 3 neurons,
    // 4 logits. No norm, no positions, no biases: the defaults.
    let config = PlainTransformerConfig::new(6, 1, D_MODEL, 1, 3, 3)
        .unwrap()
        .with_output_size(4)
        .with_attn_scale(attn_scale);

    // Token embedding: steps set their effects and their identity; a condition
    // sets its proposition; every token sets `one`.
    let mut emb = Vec::new();
    for &(row, adds, step) in &STEPS {
        for &p in adds {
            emb.push((row, E + p, 1.0));
        }
        emb.push((row, S + step, 1.0));
        emb.push((row, ONE, 1.0));
    }
    for p in 0..3 {
        emb.push((3 + p, C + p, 1.0));
        emb.push((3 + p, ONE, 1.0));
    }

    let ident =
        |from: usize| -> Vec<(usize, usize, f32)> { (0..3).map(|j| (j, from + j, 1.0)).collect() };
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("tok_emb.weight".into(), matrix(6, D_MODEL, &emb));
    tensors.insert(
        "blocks.0.attn.q.weight".into(),
        matrix(3, D_MODEL, &ident(C)),
    );
    tensors.insert(
        "blocks.0.attn.k.weight".into(),
        matrix(3, D_MODEL, &ident(E)),
    );
    tensors.insert(
        "blocks.0.attn.v.weight".into(),
        matrix(3, D_MODEL, &ident(S)),
    );
    let o_proj: Vec<(usize, usize, f32)> = (0..3).map(|j| (O + j, j, 1.0)).collect();
    tensors.insert("blocks.0.attn.o.weight".into(), matrix(D_MODEL, 3, &o_proj));
    let mut fc: Vec<(usize, usize, f32)> = (0..3).map(|j| (j, O + j, 1.0)).collect();
    fc.extend((0..3).map(|j| (j, ONE, -0.25)));
    tensors.insert("blocks.0.mlp.fc.weight".into(), matrix(3, D_MODEL, &fc));
    let proj: Vec<(usize, usize, f32)> = (0..3).map(|j| (EVIDENCE, j, 1.0)).collect();
    tensors.insert("blocks.0.mlp.proj.weight".into(), matrix(D_MODEL, 3, &proj));
    let mut unembed: Vec<(usize, usize, f32)> = (0..3).map(|j| (j, O + j, 1.0)).collect();
    unembed.push((3, EVIDENCE, 1.0));
    tensors.insert("unembed.weight".into(), matrix(4, D_MODEL, &unembed));

    PlainTransformer::from_tensors(config, tensors, device).unwrap()
}

struct Reading {
    scores: Vec<f32>,
    pattern: Vec<f32>,
    neurons: Vec<f32>,
    logits: Vec<f32>,
}

/// Run `ids` and read everything at the last position (the condition).
fn read(model: &PlainTransformer, ids: &[u32], device: &Device) -> Reading {
    let q = ids.len() - 1;
    let input = Tensor::new(ids, device).unwrap().unsqueeze(0).unwrap();
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::AttnScores(0))
        .capture(HookPoint::AttnPattern(0))
        .capture(HookPoint::MlpAct(0));
    let cache = model.forward(&input, &spec).unwrap();
    let row = |p: HookPoint| -> Vec<f32> {
        cache
            .require(&p)
            .unwrap()
            .i((0, 0, q))
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    Reading {
        scores: row(HookPoint::AttnScores(0)),
        pattern: row(HookPoint::AttnPattern(0)),
        neurons: cache
            .require(&HookPoint::MlpAct(0))
            .unwrap()
            .i((0, q))
            .unwrap()
            .to_vec1()
            .unwrap(),
        logits: cache.output().i((0, q)).unwrap().to_vec1().unwrap(),
    }
}

/// The CUDA device, if this build has `cuda` and a device is present.
fn cuda_device() -> Option<Device> {
    let device = Device::cuda_if_available(0).unwrap();
    device.is_cuda().then_some(device)
}

fn one_establisher(device: &Device) {
    // [A, B, C1]: only B adds p1.
    let r = read(
        &construction(200.0, device),
        &[TOK_A, TOK_B, TOK_C1],
        device,
    );
    assert_eq!(r.scores, vec![0.0, 200.0, 0.0], "score = scale * [adds p1]");
    // Gap 200 > ~104 (F32 exp underflow): the losers are exact zeros.
    assert_eq!(r.pattern, vec![0.0, 1.0, 0.0]);
    // n_B = ReLU(1 - 0.25); the others ReLU(0 - 0.25) = 0.
    assert_eq!(r.neurons, vec![0.0, 0.75, 0.0]);
    // logits = (o_A, o_B, o_D, evidence).
    assert_eq!(r.logits, vec![0.0, 1.0, 0.0, 0.75]);
}

fn tie(device: &Device) {
    // [A, B, D, C0]: A and D both add p0. seq = 4, so both key rows sit in one tile.
    let r = read(
        &construction(200.0, device),
        &[TOK_A, TOK_B, TOK_D, TOK_C0],
        device,
    );
    // The tie is checked exact BEFORE the pattern is read. If this fails, suspect
    // the kernel first: the two products are the same terms in the same order.
    assert_eq!(
        r.scores[0].to_bits(),
        r.scores[2].to_bits(),
        "tied scores must be equal to the bit: {} vs {}",
        r.scores[0],
        r.scores[2]
    );
    assert_eq!(r.scores, vec![200.0, 0.0, 200.0, 0.0]);
    // Two winners at exp(0) = 1, the rest underflow to 0: the sum is exactly 2.
    assert_eq!(r.pattern, vec![0.5, 0.0, 0.5, 0.0]);
    // n_A = n_D = ReLU(0.5 - 0.25) = 0.25: half the evidence of one establisher.
    assert_eq!(r.neurons, vec![0.25, 0.0, 0.25]);
    assert_eq!(r.logits, vec![0.5, 0.0, 0.5, 0.5]);
}

fn below_underflow(device: &Device) {
    // Gap 30: above the exactness bound 24 ln 2 + ln 3 ~ 17.7, below ~104.
    let r = read(&construction(30.0, device), &[TOK_A, TOK_B, TOK_C1], device);
    assert_eq!(r.pattern[1], 1.0, "winner exactly 1.0");
    let half_ulp_of_one = f32::EPSILON / 2.0; // 2^-24
    for loser in [r.pattern[0], r.pattern[2]] {
        assert!(loser > 0.0, "a loser is exp(-30), not zero");
        assert!(loser < half_ulp_of_one, "a loser is below 2^-24: {loser:e}");
    }
    // The attention output carries the losers' values: o_A = p(A) * v_A = exp(-30)
    // (v is one-hot), so logit o_A is that tiny number, not zero...
    assert_eq!(r.logits[0].to_bits(), r.pattern[0].to_bits(), "o_A = p(A)");
    assert!(r.logits[0] > 0.0);
    // ...and the ReLU threshold erases it from the neurons: n_A = ReLU(exp(-30)
    // - 0.25) = 0 exactly, while n_B = ReLU(1 - 0.25) = 0.75 exactly. A
    // threshold in the MLP is what turns a not-quite-one-hot pattern back into
    // exact readings.
    assert_eq!(r.neurons, vec![0.0, 0.75, 0.0]);
}

#[test]
fn one_establisher_gives_an_exact_one_hot_and_known_neurons() {
    one_establisher(&Device::Cpu);
}

#[test]
fn a_tie_gives_an_exact_half_half_and_known_neurons() {
    tie(&Device::Cpu);
}

#[test]
fn below_underflow_the_winner_is_exactly_one_and_the_losers_are_tiny_but_not_zero() {
    below_underflow(&Device::Cpu);
}

/// The same three cases, the same exact values, on CUDA: whether the exactness
/// argument (exact products, `exp` underflowing to `0`, a bit-equal tie) holds
/// on the GPU too. Not part of CI (no GPU there): run with `--include-ignored`
/// on a CUDA machine.
#[test]
#[ignore = "CUDA: run by hand on a GPU"]
fn exact_readings_on_cuda() {
    let Some(device) = cuda_device() else {
        eprintln!("no CUDA device: nothing to report");
        return;
    };
    one_establisher(&device);
    tie(&device);
    below_underflow(&device);
    eprintln!("CUDA: one establisher, tie and below-underflow readings all exact");
}
