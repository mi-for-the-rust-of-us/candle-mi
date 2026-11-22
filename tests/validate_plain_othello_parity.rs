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

//! **`PlainTransformer` configured as `OthelloGpt` is `OthelloGpt`, bit for bit**
//! (`design/plain-transformer.md` §7.2).
//!
//! Both backbones are candle, running the same operations in the same order, so
//! there is no tolerance here: every common hook point and the logits must
//! agree to the last bit. On a mismatch the test names **the first hook point,
//! in forward order, where a bit differs** (and the largest difference there),
//! which is the operation to look at.
//!
//! Both models load the same seeded random weights, `LayerNorm` weights and
//! biases included (an `init`-ed `OthelloGpt` would have unit norms, which
//! would exercise less). `OthelloGpt`'s fused `attn.qkv` is split into the
//! `q`, `k`, `v` tensors `PlainTransformer` reads; its loader concatenates them
//! back, so the projection weight is the same bits.
//!
//! Four configurations, so neither a scale nor a mask difference can hide:
//! `n_embd = 512, n_head = 8` (`OthelloGpt`'s own width, `head_dim = 64`, scale
//! `0.125`, exact in every precision) and `n_embd = 768, n_head = 8`
//! (`head_dim = 96`), each causal and not.
//!
//! Why 96 and not 48: candle multiplies an `F32` tensor by a scalar after
//! casting it to `f32` (`T::from_f64` in the CPU `affine` kernel), so a scale
//! computed in `f32` instead of `f64` is visible only where the two round to
//! different `f32` values. They do for 54 of the head sizes 1 to 256, `96`
//! among them, but not for `48` (nor `64`), where such a mutation passes
//! unseen. With `96` it fails at `AttnScores`, checked by planting it.
//! `self_conditioning` is off (an operation `PlainTransformer` does not have);
//! the attention-internal hooks are captured, so `OthelloGpt` takes its
//! composed attention even when `fused-attn` is compiled in.
//!
//! The guarantee is on CPU, where CI runs it. The CUDA variant is `#[ignore]`d
//! and reported separately: the same shapes should select the same cuBLAS
//! algorithms, but that is not promised.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_mi::{
    Activation, HookPoint, HookSpec, MIBackend, NormType, OthelloGpt, OthelloGptConfig,
    PlainTransformer, PlainTransformerConfig,
};
use candle_nn::VarBuilder;

const VOCAB: usize = 62;
const BLOCK: usize = 60;
const LAYERS: usize = 2;

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

/// `OthelloGpt`'s checkpoint layout (`OthelloGpt::load`'s documented keys),
/// with seeded values: weights scaled to keep activations O(1), norm weights
/// centred on 1, biases small.
fn othello_weights(n_embd: usize, device: &Device) -> HashMap<String, Tensor> {
    let h = n_embd;
    let mut names: Vec<(String, Vec<usize>)> = vec![
        ("tok_emb.weight".into(), vec![VOCAB, h]),
        ("pos_emb.weight".into(), vec![BLOCK, h]),
    ];
    for i in 0..LAYERS {
        for (suffix, dims) in [
            ("ln1.weight", vec![h]),
            ("ln1.bias", vec![h]),
            ("ln2.weight", vec![h]),
            ("ln2.bias", vec![h]),
            ("attn.qkv.weight", vec![3 * h, h]),
            ("attn.qkv.bias", vec![3 * h]),
            ("attn.proj.weight", vec![h, h]),
            ("attn.proj.bias", vec![h]),
            ("mlp.0.weight", vec![4 * h, h]),
            ("mlp.0.bias", vec![4 * h]),
            ("mlp.2.weight", vec![h, 4 * h]),
            ("mlp.2.bias", vec![h]),
        ] {
            names.push((format!("blocks.{i}.{suffix}"), dims));
        }
    }
    names.push(("ln_f.weight".into(), vec![h]));
    names.push(("ln_f.bias".into(), vec![h]));
    names.push(("head.weight".into(), vec![VOCAB, h]));

    names
        .into_iter()
        .enumerate()
        .map(|(i, (name, dims))| {
            let n: usize = dims.iter().product();
            // CAST: usize → u64, a tensor index
            #[allow(clippy::as_conversions)]
            let raw = lcg_values(1_000 + i as u64, n);
            let is_norm_weight = name.ends_with("ln1.weight")
                || name.ends_with("ln2.weight")
                || name == "ln_f.weight";
            let values: Vec<f32> = if is_norm_weight {
                raw.iter().map(|v| 1.0 + v).collect()
            } else if name.contains("emb") {
                raw
            } else {
                raw.iter().map(|v| v * 0.1).collect()
            };
            (name, Tensor::from_vec(values, dims, device).unwrap())
        })
        .collect()
}

/// The same weights under `PlainTransformer`'s names: `attn.qkv` split into
/// `q`, `k`, `v`; `attn.proj` -> `attn.o`; `mlp.0` / `mlp.2` -> `mlp.fc` /
/// `mlp.proj`; `head` -> `unembed`. Norms and embeddings keep their names.
fn plain_weights(othello: &HashMap<String, Tensor>, n_embd: usize) -> HashMap<String, Tensor> {
    let h = n_embd;
    let mut out = HashMap::new();
    for (name, tensor) in othello {
        if let Some(prefix) = name.strip_suffix("attn.qkv.weight") {
            for (j, proj) in ["q", "k", "v"].into_iter().enumerate() {
                out.insert(
                    format!("{prefix}attn.{proj}.weight"),
                    tensor.narrow(0, j * h, h).unwrap(),
                );
            }
        } else if let Some(prefix) = name.strip_suffix("attn.qkv.bias") {
            for (j, proj) in ["q", "k", "v"].into_iter().enumerate() {
                out.insert(
                    format!("{prefix}attn.{proj}.bias"),
                    tensor.narrow(0, j * h, h).unwrap(),
                );
            }
        } else {
            let renamed = name
                .replace("attn.proj.", "attn.o.")
                .replace("mlp.0.", "mlp.fc.")
                .replace("mlp.2.", "mlp.proj.")
                .replace("head.", "unembed.");
            out.insert(renamed, tensor.clone());
        }
    }
    out
}

/// Every hook point both backbones fire, in forward order.
fn common_hooks() -> Vec<HookPoint> {
    let mut hooks = vec![HookPoint::Embed];
    for l in 0..LAYERS {
        hooks.extend([
            HookPoint::ResidPre(l),
            HookPoint::AttnQ(l),
            HookPoint::AttnK(l),
            HookPoint::AttnV(l),
            HookPoint::AttnScores(l),
            HookPoint::AttnPattern(l),
            HookPoint::AttnOut(l),
            HookPoint::ResidMid(l),
            HookPoint::MlpPre(l),
            HookPoint::MlpPost(l),
            HookPoint::MlpOut(l),
            HookPoint::ResidPost(l),
        ]);
    }
    hooks.push(HookPoint::FinalNorm);
    hooks
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

/// Run both backbones and return `Err(first difference)` or `Ok(())`.
fn compare(n_embd: usize, causal: bool, device: &Device) -> Result<(), String> {
    let n_head = 8;
    let head_dim = n_embd / n_head;

    let othello_tensors = othello_weights(n_embd, device);
    let plain_tensors = plain_weights(&othello_tensors, n_embd);

    let othello_cfg = OthelloGptConfig::new(VOCAB, BLOCK, LAYERS, n_head, n_embd, causal).unwrap();
    assert!(!othello_cfg.self_conditioning);
    let othello = OthelloGpt::load(
        othello_cfg,
        VarBuilder::from_tensors(othello_tensors, DType::F32, device),
    )
    .unwrap();

    let plain_cfg =
        PlainTransformerConfig::new(VOCAB, LAYERS, n_embd, n_head, head_dim, 4 * n_embd)
            .unwrap()
            .with_positions(BLOCK)
            .with_norm(NormType::LayerNorm)
            .with_activation(Activation::Gelu)
            .with_bias(true)
            .with_causal(causal);
    let plain = PlainTransformer::from_tensors(plain_cfg, plain_tensors, device).unwrap();

    // Two sequences of 20 tokens covering the vocabulary unevenly.
    let ids: Vec<u32> = (0..40_u32).map(|i| (i * 7 + 3) % 62).collect();
    let input = Tensor::from_vec(ids, (2, 20), device).unwrap();

    let mut spec = HookSpec::new();
    for hook in common_hooks() {
        spec.capture(hook);
    }
    let a = othello.forward(&input, &spec).unwrap();
    let b = plain.forward(&input, &spec).unwrap();

    for hook in common_hooks() {
        let ta = a.require(&hook).unwrap();
        let tb = b.require(&hook).unwrap();
        if ta.dims() != tb.dims() {
            return Err(format!(
                "first difference at {hook}: shapes {:?} vs {:?}",
                ta.dims(),
                tb.dims()
            ));
        }
        if bits(ta) != bits(tb) {
            return Err(format!(
                "first difference at {hook}: max |Δ| {:.3e}",
                max_abs_diff(ta, tb)
            ));
        }
    }
    if bits(a.output()) != bits(b.output()) {
        return Err(format!(
            "first difference at the logits: max |Δ| {:.3e}",
            max_abs_diff(a.output(), b.output())
        ));
    }
    Ok(())
}

#[test]
fn bit_identical_to_othellogpt_on_cpu() {
    let mut failures = Vec::new();
    for n_embd in [512, 768] {
        for causal in [false, true] {
            let label = format!(
                "n_embd {n_embd}, n_head 8 (head_dim {}), causal {causal}",
                n_embd / 8
            );
            match compare(n_embd, causal, &Device::Cpu) {
                Ok(()) => eprintln!("CPU, {label}: bit-identical at every hook and the logits"),
                Err(e) => failures.push(format!("CPU, {label}: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Not promised (see the module docs), so not part of CI: run with
/// `--include-ignored` on a CUDA machine to learn whether the guarantee also
/// holds there.
#[test]
#[ignore = "CUDA: bit-identity is not promised there; reported, run by hand on a GPU"]
fn bit_identical_to_othellogpt_on_cuda() {
    let device = Device::cuda_if_available(0).unwrap();
    if !device.is_cuda() {
        eprintln!("no CUDA device: nothing to report");
        return;
    }
    let mut failures = Vec::new();
    for n_embd in [512, 768] {
        for causal in [false, true] {
            let label = format!("n_embd {n_embd}, causal {causal}");
            match compare(n_embd, causal, &device) {
                Ok(()) => eprintln!("CUDA, {label}: bit-identical at every hook and the logits"),
                Err(e) => failures.push(format!("CUDA, {label}: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
