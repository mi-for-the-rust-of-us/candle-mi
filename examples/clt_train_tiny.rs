// SPDX-License-Identifier: MIT OR Apache-2.0

//! A complete cross-layer transcoder (CLT) training recipe, offline, on CPU, in seconds.
//!
//! ```bash
//! cargo run --no-default-features --features clt,diffusion,training --example clt_train_tiny
//! ```
//!
//! This is the command CI runs. There is no `--release`: the `dev` profile (`opt-level = 1` in
//! this crate) finishes in seconds, and `--release` would rebuild candle with fat LTO for no
//! gain here (it works too). `--no-default-features` drops `cuda`, which needs `nvcc` to build;
//! the example uses the CPU on purpose, since a GPU gains nothing on tensors this small.
//!
//! **What it does:**
//!
//! 1. Builds a tiny, seeded, random-weight `OthelloGpt` (`OthelloGpt::init`): the only model
//!    that can be constructed with no download.
//! 2. Generates seeded pseudo-random token sequences (no `rand` dependency). They are **not**
//!    legal Othello games; with random weights there is nothing to be legal for.
//! 3. Captures, for every layer, the residual stream entering the MLP sublayer (`ResidMid`, the
//!    CLT's input) and the MLP's output (`MlpOut`, its target) in one forward, then detaches them: the model is built over a `VarMap`, so its activations would otherwise drag
//!    the backbone's graph into every `backward()`.
//! 4. Fits one scalar scale per layer and side (the RMS of the activations), so the CLT trains on
//!    `x / scale_in` and `target / scale_out`.
//! 5. Trains a [`TrainableClt`](candle_mi::clt::train::TrainableClt) with
//!    [`AdamW`](candle_mi::optim::AdamW). The loop is plain code in this file: the crate ships
//!    the model and its loss, not a training loop.
//! 6. Reads the result back in the model's raw units on a held-out batch: fraction of variance
//!    unexplained per layer, dead features, and the closed-loop replacement model
//!    (`replacement_logits_for` with `Splice::Clean`, `Replaced` and `Ablated`), then the
//!    splice confined to some positions (`replacement_logits_where`).
//! 7. Exports the CLT in the circuit-tracer layout and reloads it with
//!    `CrossLayerTranscoder::open_local`: the code that reads a published CLT reads this one.
//!
//! The example asserts its own invariants and exits non-zero if one fails, which is what lets CI
//! run it rather than only build it. **The weights are random, so what the CLT learns about the
//! model means nothing**: the numbers demonstrate the mechanics, they are not findings.

use std::fmt::Display;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_mi::clt::replacement::{
    ScaledClt, Splice, cross_entropy_at, dead_fraction, fvu_per_layer, loss_recovered,
    replacement_logits_for, replacement_logits_where,
};
use candle_mi::clt::train::{TrainableClt, TrainableCltConfig};
use candle_mi::clt::{CltFeatureId, CrossLayerTranscoder};
use candle_mi::diffusion::{OthelloGpt, OthelloGptConfig};
use candle_mi::optim::{AdamW, ParamsAdamW};
use candle_mi::{HookPoint, HookSpec, MIBackend, MIError};
use candle_nn::VarMap;

/// Layers of the backbone, and so of the CLT.
const LAYERS: usize = 3;
/// Width of the residual stream (`n_embd`).
const D: usize = 16;
/// Attention heads of the backbone; must divide [`D`].
const HEADS: usize = 2;
/// Vocabulary size of the backbone; token ids are drawn from `0..VOCAB`.
const VOCAB: usize = 12;
/// Sequence length: positions per row.
const SEQ: usize = 6;
/// Features per layer in the CLT.
const N_FEATURES: usize = 64;
/// Features kept active per (position, layer) by the `TopK` sparsifier.
const TOP_K: usize = 4;
/// Seed of the backbone's initialisation, the CLT's initialisation and the token generator.
const SEED: u64 = 20_261_031;
/// Mixed into [`SEED`] (by exclusive or) so the held-out batch is a different token stream.
const HELD_OUT_SALT: u64 = 0x5eed;
/// Sequences in the training batch.
const TRAIN_SEQS: usize = 300;
/// Sequences in the held-out batch (a different seed).
const EVAL_SEQS: usize = 100;
/// Optimiser steps.
const STEPS: usize = 300;
/// Steps between loss lines.
const LOG_EVERY: usize = 100;
/// `AdamW` learning rate (the betas and epsilon in [`fit`] are Adam's usual defaults).
const LEARNING_RATE: f64 = 3e-3;
/// The training loss must fall to at most this fraction of its first value.
const LOSS_FALL: f32 = 0.5;
/// Largest held-out fraction of variance unexplained, per layer, for the reconstruction to count
/// as working. A CLT fed the wrong way round (`x * scale_in`) sits far above it.
const FVU_CEILING: f32 = 0.5;
/// Smallest `ablated - clean` cross-entropy gap (nats) at which `loss_recovered` is a ratio of
/// something rather than of noise.
const GAP_FLOOR: f64 = 1e-4;
/// Largest disagreement tolerated between a reloaded encoder and the trained CLT's activations.
const ENCODE_TOLERANCE: f32 = 1e-3;
/// The feature index whose decoder vectors are compared after the reload (in every layer).
const DECODER_FEATURE: usize = 3;
/// Largest relative disagreement tolerated between a reloaded decoder vector and the trained
/// one times its `scale_out`. Relative, because the scaled vectors are tiny (the output scales
/// are about 0.007), so an absolute tolerance would pass a decoder missing its scale.
const DECODER_TOLERANCE: f32 = 1e-4;
/// Stride between the held-out *positions* (flattened `batch * SEQ` rows) whose encodings are
/// compared after the reload. 25 is coprime to [`SEQ`], so the sample visits every position of
/// a sequence, and it keeps the comparison to a few dozen encodings.
const RELOAD_STRIDE: usize = 25;

/// Counts the self-checks that held; one that does not holds the example back (non-zero exit).
struct Checks {
    /// Self-checks passed so far.
    passed: usize,
}

impl Checks {
    /// Record one invariant: print it and fail the run if it does not hold.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if `holds` is `false`.
    fn expect(&mut self, holds: bool, what: &str) -> candle_mi::Result<()> {
        if !holds {
            return Err(MIError::Model(candle_core::Error::Msg(format!(
                "self-check failed ({what})"
            ))));
        }
        self.passed += 1;
        println!("  [ok] {what}");
        Ok(())
    }
}

/// A checked integer conversion; the example performs no `as` cast.
///
/// # Errors
///
/// Returns [`MIError::Config`] if `value` does not fit in `U`.
fn convert<T: Copy + Display, U: TryFrom<T>>(value: T) -> candle_mi::Result<U> {
    U::try_from(value).map_err(|_| {
        MIError::Config(format!(
            "value {value} out of range for {}",
            std::any::type_name::<U>()
        ))
    })
}

/// Seeded pseudo-random token ids: a linear congruential generator, so there is no `rand`
/// dependency and the stream cannot move under a future release.
///
/// # Shapes
/// - returns: (`n_seqs * SEQ` flat ids, the same ids as `[n_seqs, SEQ]` `u32`)
///
/// # Errors
///
/// Returns [`MIError::Config`] if an id does not fit in `u32`.
/// Returns [`MIError::Model`] on tensor failure.
fn random_ids(n_seqs: usize, seed: u64, device: &Device) -> candle_mi::Result<(Vec<u32>, Tensor)> {
    let vocab: u64 = convert(VOCAB)?;
    let mut state = seed;
    let mut flat = Vec::with_capacity(n_seqs * SEQ);
    // Knuth's MMIX constants. The high bits of an LCG are the well-mixed ones, hence `>> 33`.
    // EXPLICIT: the generator carries `state` from one draw to the next; `.map()` would hide it
    for _ in 0..n_seqs * SEQ {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        flat.push(convert::<u64, u32>((state >> 33) % vocab)?);
    }
    let ids = Tensor::from_vec(flat.clone(), (n_seqs, SEQ), device)?;
    Ok((flat, ids))
}

/// Every layer's MLP input and output on `ids`, detached, stacked over layers.
///
/// # Shapes
/// - `ids`: `[batch, SEQ]` -- token ids
/// - returns: (`x`, `target`), each `[batch * SEQ, LAYERS, D]` -- the `ResidMid` (MLP input) and
///   `MlpOut` (MLP output) of every layer, in the model's raw units
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the backbone does not provide a hook point.
/// Returns [`MIError::Model`] on tensor failure.
fn capture_mlp_io(model: &OthelloGpt, ids: &Tensor) -> candle_mi::Result<(Tensor, Tensor)> {
    let mut spec = HookSpec::new();
    for layer in 0..LAYERS {
        spec.capture(HookPoint::ResidMid(layer));
        spec.capture(HookPoint::MlpOut(layer));
    }
    let cache = model.forward(ids, &spec)?;
    let stack = |point: fn(usize) -> HookPoint| -> candle_mi::Result<Tensor> {
        let per_layer = (0..LAYERS)
            .map(|layer| Ok(cache.require(&point(layer))?.reshape(((), D))?))
            .collect::<candle_mi::Result<Vec<Tensor>>>()?;
        // The backbone is built over a `VarMap`, so these tensors carry its graph; the CLT
        // trains on data, not through the backbone.
        Ok(Tensor::stack(&per_layer, 1)?.detach())
    };
    Ok((stack(HookPoint::ResidMid)?, stack(HookPoint::MlpOut)?))
}

/// The RMS of each layer's column: the per-layer scale the CLT is trained under.
///
/// # Shapes
/// - `activations`: `[rows, LAYERS, D]`
/// - returns: `LAYERS` positive scalars
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
fn layer_rms(activations: &Tensor) -> candle_mi::Result<Vec<f32>> {
    (0..LAYERS)
        .map(|layer| {
            Ok(activations
                .narrow(1, layer, 1)?
                .sqr()?
                .mean_all()?
                .sqrt()?
                .to_scalar::<f32>()?)
        })
        .collect()
}

/// `activations / scales`, one scale per layer.
///
/// # Shapes
/// - `activations`: `[rows, LAYERS, D]`
/// - returns: `[rows, LAYERS, D]`
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
fn scaled(activations: &Tensor, scales: &[f32]) -> candle_mi::Result<Tensor> {
    let per_layer = Tensor::from_slice(scales, (1, LAYERS, 1), activations.device())?;
    Ok(activations.broadcast_div(&per_layer)?)
}

/// Fit `clt` to `(xs, ts)` with a plain `AdamW` loop; returns the first and the last loss.
///
/// # Shapes
/// - `xs`: `[rows, LAYERS, D]` -- scaled MLP inputs
/// - `ts`: `[rows, LAYERS, D]` -- scaled MLP outputs
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure.
fn fit(clt: &TrainableClt, xs: &Tensor, ts: &Tensor) -> candle_mi::Result<(f32, f32)> {
    let mut optimizer = AdamW::new(
        clt.vars(),
        ParamsAdamW {
            lr: LEARNING_RATE,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        },
    )?;
    let first = clt.loss(xs, ts)?.to_scalar::<f32>()?;
    // EXPLICIT: each step updates the optimiser's moments and the CLT's `Var`s in place
    for step in 0..STEPS {
        let loss = clt.loss(xs, ts)?;
        if step % LOG_EVERY == 0 {
            println!("  step {step:>4}  loss {:.5}", loss.to_scalar::<f32>()?);
        }
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }
    let last = clt.loss(xs, ts)?.to_scalar::<f32>()?;
    println!("  step {STEPS:>4}  loss {last:.5}");
    Ok((first, last))
}

/// The largest absolute elementwise difference between two tensors of one shape.
///
/// # Errors
///
/// Returns [`MIError::Model`] on tensor failure (including a shape mismatch).
fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_mi::Result<f32> {
    Ok((a - b)?.abs()?.max_all()?.to_scalar::<f32>()?)
}

/// The largest relative disagreement between the reloaded decoder vectors of feature
/// [`DECODER_FEATURE`] (every source layer, every target layer it writes to) and the trained
/// decoder rows times `scale_out`, which is what the export folded into them.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the trained CLT lacks a decoder parameter or `scale_out` a layer.
/// Returns [`MIError::Model`] on tensor failure.
fn decoder_disagreement(
    clt: &TrainableClt,
    local: &mut CrossLayerTranscoder,
    scale_out: &[f32],
    device: &Device,
) -> candle_mi::Result<f32> {
    let trained = clt.vars();
    let mut worst = 0.0_f32;
    for source in 0..LAYERS {
        let wanted = format!("w_dec_{source}");
        let (_, w_dec) = trained
            .iter()
            .find(|(name, _)| *name == wanted)
            .ok_or_else(|| MIError::Config(format!("trained CLT has no parameter {wanted}")))?;
        let feature = CltFeatureId {
            layer: source,
            index: DECODER_FEATURE,
        };
        for target in source..LAYERS {
            let scale = scale_out.get(target).copied().ok_or_else(|| {
                MIError::Config(format!("scale_out has no layer {target} ({LAYERS} layers)"))
            })?;
            let reloaded = local.decoder_vector(&feature, target, device)?;
            let expected = w_dec
                .as_tensor()
                .get(DECODER_FEATURE)?
                .narrow(0, (target - source) * D, D)?
                .affine(f64::from(scale), 0.0)?;
            let size = expected.abs()?.max_all()?.to_scalar::<f32>()?;
            worst = worst.max(max_abs_diff(&reloaded, &expected)? / size);
        }
    }
    Ok(worst)
}

/// Export `clt` as circuit-tracer files, reload them with the loader that reads published CLTs,
/// and check that it encodes like the trained CLT on every `RELOAD_STRIDE`-th position.
///
/// Both folds of the export are checked through the real loader, with non-unit scales: the
/// encoder on a sample of positions, the decoder on one feature per layer. This complements
/// `tests/validate_clt_train.rs`, whose `export_folds_the_scales_into_raw_units` reads every
/// exported file with non-unit scales but not through the loader, and whose
/// `open_local_encodes_and_decodes_like_the_trained_clt` uses the loader but unit scales.
///
/// # Shapes
/// - `x_eval`: `[rows, LAYERS, D]` -- raw MLP inputs
///
/// # Errors
///
/// Returns [`MIError::Io`] if the directory cannot be written or removed.
/// Returns [`MIError::Config`] if the reloaded configuration cannot be read.
/// Returns [`MIError::Model`] on tensor failure or a failed self-check.
fn export_and_reload(
    clt: &TrainableClt,
    scale_in: &[f32],
    scale_out: &[f32],
    x_eval: &Tensor,
    checks: &mut Checks,
) -> candle_mi::Result<()> {
    let dir = std::env::temp_dir().join(format!("clt_train_tiny_{}", std::process::id()));
    let outcome = (|| -> candle_mi::Result<()> {
        clt.export_circuit_tracer(&dir, scale_in, scale_out, "clt_train_tiny")?;
        let mut local = CrossLayerTranscoder::open_local(&dir)?;
        let config = local.config();
        checks.expect(
            (
                config.n_layers,
                config.d_model,
                config.n_features_per_layer,
                config.top_k,
            ) == (LAYERS, D, N_FEATURES, Some(TOP_K)),
            "the reloaded CLT has the trained (layers, d_model, features, top_k)",
        )?;
        // The trained CLT's activations on the *scaled* inputs (detached: nothing here trains);
        // the reloaded one takes the raw residual, because the export folded the scales in.
        let (_, trained) = clt.forward_detached(&scaled(x_eval, scale_in)?)?;
        let mut compared = 0_usize;
        let mut worst = 0.0_f32;
        let mut missing = 0_usize;
        let mut within_top_k = true;
        // EXPLICIT: `load_encoder` mutates the loader, and each layer's encoder is loaded once
        for layer in 0..LAYERS {
            local.load_encoder(layer, x_eval.device())?;
            for row in (0..x_eval.dim(0)?).step_by(RELOAD_STRIDE) {
                let residual = x_eval.get(row)?.get(layer)?;
                let sparse = local.encode(&residual, layer)?;
                let dense = trained.get(row)?.get(layer)?.to_vec1::<f32>()?;
                within_top_k &= sparse.features.len() <= TOP_K;
                // Every feature the reloaded encoder keeps has the trained activation...
                for (id, value) in &sparse.features {
                    let want = dense.get(id.index).copied().unwrap_or(f32::INFINITY);
                    worst = worst.max((value - want).abs());
                }
                // ...and every feature the trained CLT clearly fires is kept.
                for (index, want) in dense.iter().enumerate() {
                    let kept = sparse.features.iter().any(|(id, _)| id.index == index);
                    missing += usize::from(*want > ENCODE_TOLERANCE && !kept);
                }
                compared += 1;
            }
        }
        checks.expect(
            within_top_k,
            "no reloaded encoding keeps more than top_k features",
        )?;
        checks.expect(
            worst <= ENCODE_TOLERANCE,
            "every feature the reloaded encoder keeps has the trained activation",
        )?;
        checks.expect(
            missing == 0,
            "every feature the trained CLT clearly fires is kept by the reloaded encoder",
        )?;
        let decoder = decoder_disagreement(clt, &mut local, scale_out, x_eval.device())?;
        checks.expect(
            decoder <= DECODER_TOLERANCE,
            "the reloaded decoder vectors are the trained ones times scale_out",
        )?;
        println!("  largest encoder disagreement over {compared} encodings: {worst:.2e}");
        println!("  largest relative decoder disagreement: {decoder:.2e}");
        Ok(())
    })();
    // Remove the directory whether or not the checks held, then report the first failure.
    let cleanup = std::fs::remove_dir_all(&dir);
    outcome?;
    cleanup?;
    Ok(())
}

/// The replacement model with the CLT confined to the first `marked` positions of each row; the
/// model's own MLPs run on the rest, on the residual the splice has produced. `marked = SEQ` is
/// the plain replacement model and `marked = 0` the clean one.
///
/// # Shapes
/// - `ids`: `[batch, SEQ]` -- token ids
/// - returns: `[batch, SEQ, VOCAB]` -- logits
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the backbone does not provide a hook point.
/// Returns [`MIError::Model`] on tensor failure.
fn confined_splice(
    model: &OthelloGpt,
    ids: &Tensor,
    clt: &ScaledClt<'_>,
    marked: usize,
) -> candle_mi::Result<Tensor> {
    // The mask is `[batch, seq]` and nonzero where the CLT splices.
    let batch = ids.dim(0)?;
    let mask: Vec<f32> = (0..batch * SEQ)
        .map(|i| if i % SEQ < marked { 1.0 } else { 0.0 })
        .collect();
    let at = Tensor::from_vec(mask, (batch, SEQ), ids.device())?;
    replacement_logits_where(
        |hooks| model.forward(ids, hooks),
        clt,
        Splice::Replaced,
        &at,
    )
}

/// The replacement model on the held-out batch: cross-entropy under each splice, `loss_recovered`
/// with its denominator, and the confined splice.
///
/// # Shapes
/// - `eval_ids`: `[batch, SEQ]` -- token ids
/// - `eval_flat`: `batch * SEQ` -- the same ids, flat; the cross-entropy target at each position
///
/// # Errors
///
/// Returns [`MIError::Hook`] if the backbone does not provide a hook point.
/// Returns [`MIError::Config`] if a position index does not fit in `u32`.
/// Returns [`MIError::Model`] on tensor failure or a failed self-check.
fn replacement_readings(
    model: &OthelloGpt,
    eval_ids: &Tensor,
    eval_flat: &[u32],
    scaled_clt: &ScaledClt<'_>,
    checks: &mut Checks,
) -> candle_mi::Result<()> {
    let positions: Vec<u32> = (0..eval_flat.len())
        .map(convert::<usize, u32>)
        .collect::<candle_mi::Result<_>>()?;
    // Bidirectional model, so the target at a position is the token that is there.
    let clean = replacement_logits_for(model, eval_ids, scaled_clt, Splice::Clean)?;
    let replaced = replacement_logits_for(model, eval_ids, scaled_clt, Splice::Replaced)?;
    let ablated = replacement_logits_for(model, eval_ids, scaled_clt, Splice::Ablated)?;
    let ce_clean = cross_entropy_at(&clean, &positions, eval_flat)?;
    let ce_replaced = cross_entropy_at(&replaced, &positions, eval_flat)?;
    let ce_ablated = cross_entropy_at(&ablated, &positions, eval_flat)?;
    let gap = ce_ablated - ce_clean;
    let recovered = loss_recovered(ce_clean, ce_replaced, ce_ablated);
    println!(
        "   cross-entropy: clean {ce_clean:.4}, replaced {ce_replaced:.4}, ablated {ce_ablated:.4}"
    );
    println!(
        "   ablated - clean = {gap:.4} nats; loss_recovered = {recovered:.3}, a ratio of that gap"
    );
    println!("   (with random weights the MLPs barely matter, so it says nothing about the CLT)");
    checks.expect(
        [ce_clean, ce_replaced, ce_ablated, recovered]
            .iter()
            .all(|v| v.is_finite()),
        "the replacement-model readings are finite",
    )?;
    checks.expect(
        gap.abs() > GAP_FLOOR,
        "ablating the MLPs moves the cross-entropy, so loss_recovered is not a ratio of noise",
    )?;
    let moved_replaced = max_abs_diff(&replaced, &clean)?;
    let moved_ablated = max_abs_diff(&ablated, &clean)?;
    checks.expect(
        0.0 < moved_replaced && moved_replaced < moved_ablated,
        "the replaced model moves the logits, but less than zeroing every MLP does",
    )?;

    // The confined splice: marking every position is the plain replacement model, marking none is
    // the clean model, and marking some moves the logits (the model is bidirectional, so a splice
    // at some positions still reaches the others through attention).
    let everywhere = confined_splice(model, eval_ids, scaled_clt, SEQ)?;
    let nowhere = confined_splice(model, eval_ids, scaled_clt, 0)?;
    checks.expect(
        max_abs_diff(&everywhere, &replaced)? < 1e-5 && max_abs_diff(&nowhere, &clean)? < 1e-6,
        "confining the splice to every position is the replacement model, to none the clean one",
    )?;
    let some = confined_splice(model, eval_ids, scaled_clt, SEQ / 2)?;
    let moved_some = max_abs_diff(&some, &clean)?;
    println!(
        "   largest logit change vs clean: replaced {moved_replaced:.4}, ablated {moved_ablated:.4}, replaced at the first {} positions only {moved_some:.4}",
        SEQ / 2
    );
    checks.expect(
        moved_some > 0.0,
        "a splice confined to some positions still moves the logits",
    )?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Run the recipe end to end; the first failed self-check or tensor error ends it.
///
/// # Errors
///
/// Returns the first [`MIError`] raised by a step, including a failed self-check.
fn run() -> candle_mi::Result<()> {
    let started = Instant::now();
    let device = Device::Cpu;
    let mut checks = Checks { passed: 0 };

    // 1. The backbone: seeded, random weights, F32, over a `VarMap` like a model being trained.
    println!("1. Backbone: OthelloGpt, {LAYERS} layers, d_model {D}, random weights (seed {SEED})");
    let config = OthelloGptConfig::new(VOCAB, SEQ, LAYERS, HEADS, D, false)?;
    let varmap = VarMap::new();
    let model = OthelloGpt::init(config, &varmap, &device, SEED)?;

    // 2 and 3. Data, and the activations the CLT will be trained to reconstruct.
    println!(
        "2. Data: {TRAIN_SEQS} training and {EVAL_SEQS} held-out sequences of {SEQ} ids in 0..{VOCAB}"
    );
    println!("   (seeded pseudo-random ids, not legal games)");
    let (_, train_ids) = random_ids(TRAIN_SEQS, SEED, &device)?;
    let (eval_flat, eval_ids) = random_ids(EVAL_SEQS, SEED ^ HELD_OUT_SALT, &device)?;
    let (x_train, target_train) = capture_mlp_io(&model, &train_ids)?;
    let (x_eval, target_eval) = capture_mlp_io(&model, &eval_ids)?;
    println!(
        "3. Captured ResidMid / MlpOut of every layer: {:?} per batch (detached)",
        x_train.dims()
    );

    // 4. One scale per layer and side, from the training batch only.
    let scale_in = layer_rms(&x_train)?;
    let scale_out = layer_rms(&target_train)?;
    println!("4. Scales (RMS per layer) in {scale_in:.4?}, out {scale_out:.4?}");
    let xs = scaled(&x_train, &scale_in)?;
    let ts = scaled(&target_train, &scale_out)?;

    // 5. Train.
    println!(
        "5. Training a CLT: {N_FEATURES} features per layer, top_k {TOP_K}, {STEPS} AdamW steps"
    );
    let clt = TrainableClt::new(
        &TrainableCltConfig::new(LAYERS, D, N_FEATURES, TOP_K),
        SEED,
        &device,
    )?;
    let fit_started = Instant::now();
    let (first, last) = fit(&clt, &xs, &ts)?;
    println!("   fitted in {:.2?}", fit_started.elapsed());
    checks.expect(
        first.is_finite() && last.is_finite() && last < LOSS_FALL * first,
        "the training loss fell by more than half",
    )?;

    // 6. Read it back in the model's raw units, on data the CLT never saw.
    println!("6. Held-out readings (raw units)");
    let scaled_clt = ScaledClt::new(&clt, &scale_in, &scale_out)?;
    let (recon, fired) = scaled_clt.reconstruct_and_fired(&x_eval)?;
    let fvu = fvu_per_layer(&recon, &target_eval)?;
    let dead = dead_fraction(&fired)?;
    println!("   fraction of variance unexplained, per layer: {fvu:.3?}");
    println!(
        "   dead features: {:.1} % (of {} held-out positions, so rare features are overcounted)",
        100.0 * dead,
        eval_flat.len()
    );
    checks.expect(
        fvu.iter().all(|v| *v < FVU_CEILING) && dead < 1.0,
        "the held-out reconstruction explains most of the variance in every layer and some features fire",
    )?;

    replacement_readings(&model, &eval_ids, &eval_flat, &scaled_clt, &mut checks)?;

    // 7. Export, reload through the published-CLT loader, compare.
    println!("7. Export (circuit-tracer layout) and reload with CrossLayerTranscoder::open_local");
    export_and_reload(&clt, &scale_in, &scale_out, &x_eval, &mut checks)?;

    println!(
        "\nAll {} self-checks passed in {:.2?}.",
        checks.passed,
        started.elapsed()
    );
    println!("The weights are random: these readings demonstrate the mechanics, not a finding.");
    Ok(())
}
