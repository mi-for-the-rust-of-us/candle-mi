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

//! **`PlainTransformer` reproduces `Tracr`-compiled models** (the `Tracr` oracle,
//! `design/plain-transformer.md` §7.4).
//!
//! `scripts/plain_tracr_validation.py` compiled seven RASP programs from
//! `tracr.compiler.lib` (categorical and numerical outputs, bidirectional and
//! causal, up to six layers, two heads in a layer, a numerical discretising
//! MLP) and exported, for each, the weights in `PlainTransformer`'s layout, the
//! inputs, and what `Tracr` computed: the input embeddings, the residual stream
//! after every attention and every MLP sublayer, and the output. The fixture is
//! `tests/fixtures/plain_tracr/` (committed; Python is needed only to
//! regenerate it).
//!
//! Each model is loaded through the strict `PlainTransformer::from_tensors`, so
//! the exporter's names and shapes are checked too, with `bias = true` (`Tracr`
//! exports zero biases) and the `causal` flag the fixture was compiled with.
//!
//! The pass criterion, fixed before the first run (`JAX` and candle are different
//! `F32` implementations, so it is a tolerance, not bit-identity):
//!
//! - residual stream at every layer (`Embed`, then `ResidMid` and `ResidPost`
//!   against `Tracr`'s residuals): `max |Δ| <= 1e-5 * max(1, max |x_layer|)`,
//!   `x_layer` being `Tracr`'s tensor; relative because a discretising MLP's
//!   residuals are legitimately far above 1;
//! - categorical outputs: the same argmax at every position but the first (the
//!   BOS position, which `Tracr`'s decoder special-cases);
//! - numerical outputs: the output value, with the same relative tolerance.
//!
//! The measured worst case per program is printed. CPU, milliseconds, no
//! download: runs in CI (step "Tests (plain)").

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{Device, IndexOp, Tensor};
use candle_mi::{HookPoint, HookSpec, MIBackend, PlainTransformer, PlainTransformerConfig};
use serde_json::Value;

const RTOL: f64 = 1e-5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plain_tracr")
}

fn usize_of(v: &Value, key: &str) -> usize {
    usize::try_from(
        v[key]
            .as_u64()
            .unwrap_or_else(|| panic!("manifest: no `{key}`")),
    )
    .unwrap()
}

/// `max |Δ| / max(1, max |x|)` for one compared tensor, asserting the bound.
fn check(program: &str, case: usize, what: &str, got: &Tensor, want: &Tensor) -> f64 {
    assert_eq!(
        got.dims(),
        want.dims(),
        "{program} case {case} {what}: shape"
    );
    let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
    let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
    let scale = w
        .iter()
        .fold(1.0_f64, |acc, &v| acc.max(f64::from(v).abs()));
    let max_diff = g
        .iter()
        .zip(&w)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
        .fold(0.0_f64, f64::max);
    let bound = RTOL * scale;
    assert!(
        max_diff <= bound,
        "{program} case {case} {what}: max |Δ| {max_diff:.3e} > bound {bound:.3e} \
         (max |x| {scale:.3e})"
    );
    max_diff / scale
}

/// One input of one program: what to run and what `Tracr` computed for it.
struct Fixture<'a> {
    program: &'a str,
    case_index: usize,
    case: &'a Value,
    n_layer: usize,
    categorical: bool,
    all: &'a HashMap<String, Tensor>,
    device: &'a Device,
}

/// Run one input through `model` and compare it with `Tracr`: every layer's
/// residual, then the output. Returns the worst relative difference.
fn check_case(model: &PlainTransformer, fx: &Fixture<'_>) -> f64 {
    let (name, case_index, case) = (fx.program, fx.case_index, fx.case);
    let ids: Vec<u32> = case["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(ids.len(), case["tokens"].as_array().unwrap().len());
    let input = Tensor::new(ids.as_slice(), fx.device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut spec = HookSpec::new();
    spec.capture(HookPoint::Embed);
    for l in 0..fx.n_layer {
        spec.capture(HookPoint::ResidMid(l))
            .capture(HookPoint::ResidPost(l));
    }
    let cache = model.forward(&input, &spec).unwrap();
    let want = |key: &str| fx.all[&format!("{name}/case{case_index}/{key}")].clone();
    let got = |p: HookPoint| cache.require(&p).unwrap().squeeze(0).unwrap();

    let mut worst = check(
        name,
        case_index,
        "Embed",
        &got(HookPoint::Embed),
        &want("embed"),
    );
    for l in 0..fx.n_layer {
        worst = worst.max(check(
            name,
            case_index,
            &format!("ResidMid({l})"),
            &got(HookPoint::ResidMid(l)),
            &want(&format!("resid_mid_{l}")),
        ));
        worst = worst.max(check(
            name,
            case_index,
            &format!("ResidPost({l})"),
            &got(HookPoint::ResidPost(l)),
            &want(&format!("resid_post_{l}")),
        ));
    }

    let logits = cache.output().squeeze(0).unwrap(); // [seq, output_size]
    if fx.categorical {
        let expected: Vec<u32> = case["argmax"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let got_argmax: Vec<u32> = logits.argmax(1).unwrap().to_vec1().unwrap();
        // Position 0 is BOS, which Tracr's decoder special-cases.
        assert_eq!(
            got_argmax[1..],
            expected[1..],
            "{name} case {case_index}: decoded tokens differ (Tracr decoded {:?})",
            case["decoded"]
        );
    } else {
        let value = logits.i((.., 0)).unwrap();
        worst = worst.max(check(name, case_index, "output", &value, &want("output")));
    }
    worst
}

/// The CUDA device, if this build has `cuda` and a device is present.
fn cuda_device() -> Option<Device> {
    let device = Device::cuda_if_available(0).unwrap();
    device.is_cuda().then_some(device)
}

/// Every program and case of the fixture, on `device`.
fn reproduce(device: &Device) {
    let dir = fixture_dir();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let all = candle_core::safetensors::load(dir.join("tracr.safetensors"), &Device::Cpu).unwrap();
    let programs = manifest["programs"].as_object().unwrap();
    assert_eq!(programs.len(), 7, "the fixture holds seven programs");

    let mut report = Vec::new();
    for (name, entry) in programs {
        let causal = entry["causal"].as_bool().unwrap();
        let categorical = entry["output"].as_str().unwrap() == "categorical";
        let n_layer = usize_of(entry, "n_layer");
        let config = PlainTransformerConfig::new(
            usize_of(entry, "vocab_size"),
            n_layer,
            usize_of(entry, "d_model"),
            usize_of(entry, "n_head"),
            usize_of(entry, "head_dim"),
            usize_of(entry, "mlp_hidden"),
        )
        .unwrap()
        .with_output_size(usize_of(entry, "output_size"))
        .with_positions(usize_of(entry, "max_positions"))
        .with_bias(true)
        .with_causal(causal);

        // The model's weights: this program's tensors, minus the per-case expectations.
        let prefix = format!("{name}/");
        let weights: HashMap<String, Tensor> = all
            .iter()
            .filter_map(|(key, t)| {
                let rest = key.strip_prefix(&prefix)?;
                (!rest.starts_with("case")).then(|| (rest.to_string(), t.clone()))
            })
            .collect();
        let model = PlainTransformer::from_tensors(config, weights, device)
            .unwrap_or_else(|e| panic!("{name}: the strict loader rejected the export: {e}"));

        let mut worst = 0.0_f64;
        for (case_index, case) in entry["cases"].as_array().unwrap().iter().enumerate() {
            let fixture = Fixture {
                program: name,
                case_index,
                case,
                n_layer,
                categorical,
                all: &all,
                device,
            };
            worst = worst.max(check_case(&model, &fixture));
        }
        report.push(format!("{name}: worst relative {worst:.3e}"));
    }
    eprintln!(
        "Tracr oracle on {device:?}, max |Δ| <= 1e-5 * max(1, max |x_layer|) per layer:\n  {}",
        report.join("\n  ")
    );
}

#[test]
fn reproduces_every_tracr_fixture() {
    reproduce(&Device::Cpu);
}

/// The same fixture and the same bound on CUDA. Not part of CI (no GPU there):
/// run with `--include-ignored` on a CUDA machine.
#[test]
#[ignore = "CUDA: run by hand on a GPU"]
fn reproduces_every_tracr_fixture_on_cuda() {
    let Some(device) = cuda_device() else {
        eprintln!("no CUDA device: nothing to report");
        return;
    };
    reproduce(&device);
}
