// SPDX-License-Identifier: MIT OR Apache-2.0

//! A hand-weighted transformer: one SNLP "establish" step, written as attention.
//!
//! ```bash
//! cargo run --no-default-features --features plain --example plain_construction
//! ```
//!
//! This is the command CI runs: CPU, no download, milliseconds.
//!
//! **What it builds.** A `PlainTransformer` whose every weight is set by hand,
//! in `Tracr`'s discipline: the residual stream is a sum of named one-hot
//! subspaces, there is no norm, and a constant `one` direction plays the part
//! of a bias.
//!
//! ```text
//! residual (14):  e_p0 e_p1 e_p2 | c_p0 c_p1 c_p2 | s_A s_B s_D | o_A o_B o_D | evidence | one
//! tokens:         A adds p0      B adds p1        D adds p0 and p2
//!                 C0, C1, C2: "the open condition is p0 (p1, p2)"
//! head:           query = c, key = e, value = s, the output projection writes v into o
//!                 score(C_p -> t) = scale * [step t adds p]
//! MLP:            neuron n_j = ReLU(o_j - 0.25 * one);  evidence = sum_j n_j
//! unembed:        logits = (o_A, o_B, o_D, evidence)
//! ```
//!
//! **What it shows**, read through the hook surface every backbone shares
//! (`AttnScores`, `AttnPattern`, and the MLP neurons at `MlpAct`):
//!
//! 1. one establisher (`[A, B, C1]`): the head attends to `B` alone, exactly;
//! 2. a tie (`[A, B, D, C0]`): `A` and `D` both add `p0`, the two scores are
//!    equal to the bit, and the pattern is exactly `0.5 / 0.5`: half the
//!    evidence of one establisher, which is what a construction that must
//!    handle two candidate establishers has to account for;
//! 3. a smaller scale (`30`): the winner is still exactly `1.0`, but the losers
//!    are `exp(-30)`, tiny and not zero; the `ReLU` threshold erases them again.
//!
//! The example checks each of these and exits non-zero if one fails. The
//! integration test `tests/validate_plain_construction.rs` asserts the same
//! values; `design/plain-transformer.md` §7.3 explains why they are exact.

use std::collections::HashMap;

use candle_core::{Device, IndexOp, Tensor};
use candle_mi::{
    HookPoint, HookSpec, MIBackend, MIError, PlainTransformer, PlainTransformerConfig,
};

/// First of the three effect directions `e_p0, e_p1, e_p2` (what a step adds).
const E: usize = 0;
/// First of the three condition directions `c_p0, c_p1, c_p2` (the open condition).
const C: usize = 3;
/// First of the three step-identity directions `s_A, s_B, s_D`.
const S: usize = 6;
/// First of the three attended-step directions `o_A, o_B, o_D` (written by the head).
const O: usize = 9;
/// The evidence direction (written by the MLP).
const EVIDENCE: usize = 12;
/// The constant `one` direction, set on every token: `Tracr`'s stand-in for a bias.
const ONE: usize = 13;
/// Width of the residual stream.
const D_MODEL: usize = 14;

/// Steps, as (token row, propositions added, step index, name).
const STEPS: [(usize, &[usize], usize, &str); 3] =
    [(0, &[0], 0, "A"), (1, &[1], 1, "B"), (2, &[0, 2], 2, "D")];
/// Token names, by id.
const TOKENS: [&str; 6] = ["A", "B", "D", "C0", "C1", "C2"];

/// Counts passed self-checks and turns a failed one into an error.
struct Checks {
    /// Number of checks passed so far.
    passed: usize,
}

impl Checks {
    /// Record a check: `Ok` and counted if `ok`, else an error naming `what`.
    fn expect(&mut self, ok: bool, what: &str) -> candle_mi::Result<()> {
        if ok {
            self.passed += 1;
            Ok(())
        } else {
            Err(MIError::Config(format!("self-check failed: {what}")))
        }
    }
}

/// A `[rows, cols]` matrix of zeros with the given entries set.
fn matrix(rows: usize, cols: usize, entries: &[(usize, usize, f32)]) -> candle_mi::Result<Tensor> {
    let mut data = vec![0.0_f32; rows * cols];
    for &(r, c, v) in entries {
        let slot = data
            .get_mut(r * cols + c)
            .filter(|_| c < cols)
            .ok_or_else(|| {
                MIError::Config(format!(
                    "entry ({r}, {c}) outside a [{rows}, {cols}] matrix"
                ))
            })?;
        *slot = v;
    }
    Ok(Tensor::from_vec(data, (rows, cols), &Device::Cpu)?)
}

/// The construction, with the given attention scale.
fn construction(attn_scale: f64) -> candle_mi::Result<PlainTransformer> {
    let config = PlainTransformerConfig::new(6, 1, D_MODEL, 1, 3, 3)?
        .with_output_size(4)
        .with_attn_scale(attn_scale);

    let mut emb = Vec::new();
    for &(row, adds, step, _) in &STEPS {
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
    let o_proj: Vec<(usize, usize, f32)> = (0..3).map(|j| (O + j, j, 1.0)).collect();
    let mut fc: Vec<(usize, usize, f32)> = (0..3).map(|j| (j, O + j, 1.0)).collect();
    fc.extend((0..3).map(|j| (j, ONE, -0.25)));
    let proj: Vec<(usize, usize, f32)> = (0..3).map(|j| (EVIDENCE, j, 1.0)).collect();
    let mut unembed: Vec<(usize, usize, f32)> = (0..3).map(|j| (j, O + j, 1.0)).collect();
    unembed.push((3, EVIDENCE, 1.0));

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("tok_emb.weight".into(), matrix(6, D_MODEL, &emb)?);
    tensors.insert(
        "blocks.0.attn.q.weight".into(),
        matrix(3, D_MODEL, &ident(C))?,
    );
    tensors.insert(
        "blocks.0.attn.k.weight".into(),
        matrix(3, D_MODEL, &ident(E))?,
    );
    tensors.insert(
        "blocks.0.attn.v.weight".into(),
        matrix(3, D_MODEL, &ident(S))?,
    );
    tensors.insert(
        "blocks.0.attn.o.weight".into(),
        matrix(D_MODEL, 3, &o_proj)?,
    );
    tensors.insert("blocks.0.mlp.fc.weight".into(), matrix(3, D_MODEL, &fc)?);
    tensors.insert(
        "blocks.0.mlp.proj.weight".into(),
        matrix(D_MODEL, 3, &proj)?,
    );
    tensors.insert("unembed.weight".into(), matrix(4, D_MODEL, &unembed)?);

    // Strict: a misnamed key here would be an error, not a silent zero.
    PlainTransformer::from_tensors(config, tensors, &Device::Cpu)
}

/// What the head and the MLP do at the last position (the condition).
struct Reading {
    /// Pre-softmax scores, one per key.
    scores: Vec<f32>,
    /// Attention pattern, one per key.
    pattern: Vec<f32>,
    /// The three MLP neurons (`MlpAct`).
    neurons: Vec<f32>,
    /// `(o_A, o_B, o_D, evidence)`.
    logits: Vec<f32>,
}

/// Run `ids` and read the condition's row at every hook of interest.
fn read(model: &PlainTransformer, ids: &[u32]) -> candle_mi::Result<Reading> {
    let q = ids.len() - 1;
    let input = Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?;
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::AttnScores(0))
        .capture(HookPoint::AttnPattern(0))
        .capture(HookPoint::MlpAct(0));
    let cache = model.forward(&input, &spec)?;
    let head_row = |p: HookPoint| -> candle_mi::Result<Vec<f32>> {
        Ok(cache.require(&p)?.i((0, 0, q))?.to_vec1()?)
    };
    Ok(Reading {
        scores: head_row(HookPoint::AttnScores(0))?,
        pattern: head_row(HookPoint::AttnPattern(0))?,
        neurons: cache.require(&HookPoint::MlpAct(0))?.i((0, q))?.to_vec1()?,
        logits: cache.output().i((0, q))?.to_vec1()?,
    })
}

/// Print a reading, one line per hook.
fn show(ids: &[u32], r: &Reading) {
    let names: Vec<&str> = ids
        .iter()
        // CAST: u32 → usize, a token id
        .map(|&id| {
            usize::try_from(id)
                .ok()
                .and_then(|i| TOKENS.get(i))
                .copied()
                .unwrap_or("?")
        })
        .collect();
    println!("   keys:            {names:?}");
    println!("   AttnScores:      {:?}", r.scores);
    println!("   AttnPattern:     {:?}", r.pattern);
    let step_names: Vec<&str> = STEPS.iter().map(|s| s.3).collect();
    println!("   MlpAct {step_names:?}: {:?}", r.neurons);
    println!("   logits (o_A, o_B, o_D, evidence): {:?}", r.logits);
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Build the construction, read the three cases, check each.
///
/// # Errors
///
/// Returns the first [`MIError`] raised, including a failed self-check.
#[allow(clippy::float_cmp)] // every value checked here is exact by construction
fn run() -> candle_mi::Result<()> {
    let mut checks = Checks { passed: 0 };
    let sharp = construction(200.0)?;

    println!("1. One establisher: [A, B, C1], only B adds p1 (attn_scale 200)");
    let ids = [0, 1, 4];
    let r = read(&sharp, &ids)?;
    show(&ids, &r);
    checks.expect(
        r.pattern == [0.0, 1.0, 0.0],
        "the head attends to B alone, exactly",
    )?;
    checks.expect(
        r.neurons == [0.0, 0.75, 0.0],
        "neuron n_B = ReLU(1 - 0.25) = 0.75",
    )?;
    checks.expect(r.logits.get(3) == Some(&0.75), "evidence 0.75")?;

    println!("\n2. A tie: [A, B, D, C0], A and D both add p0 (attn_scale 200)");
    let ids = [0, 1, 2, 3];
    let r = read(&sharp, &ids)?;
    show(&ids, &r);
    checks.expect(
        r.scores.first().map(|s| s.to_bits()) == r.scores.get(2).map(|s| s.to_bits()),
        "the two tied scores are equal to the bit (if not, suspect the kernel first)",
    )?;
    checks.expect(
        r.pattern == [0.5, 0.0, 0.5, 0.0],
        "an exact 0.5 / 0.5 pattern",
    )?;
    checks.expect(
        r.neurons == [0.25, 0.0, 0.25],
        "n_A = n_D = ReLU(0.5 - 0.25) = 0.25",
    )?;
    checks.expect(
        r.logits.get(3) == Some(&0.5),
        "evidence 0.5: half of one establisher's",
    )?;

    println!(
        "\n3. One establisher at attn_scale 30: gap above the exactness bound, below underflow"
    );
    let ids = [0, 1, 4];
    let r = read(&construction(30.0)?, &ids)?;
    show(&ids, &r);
    checks.expect(
        r.pattern.get(1) == Some(&1.0),
        "the winner is still exactly 1.0",
    )?;
    checks.expect(
        r.pattern
            .first()
            .is_some_and(|&loser| loser > 0.0 && loser < f32::EPSILON / 2.0),
        "a loser is exp(-30): not zero, below 2^-24",
    )?;
    checks.expect(
        r.neurons == [0.0, 0.75, 0.0],
        "the ReLU threshold erases the losers from the neurons",
    )?;

    println!("\nAll {} self-checks passed.", checks.passed);
    Ok(())
}
