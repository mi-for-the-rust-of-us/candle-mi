// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test: `BlueLightAI` Qwen3 CLT encoder parity against the
//! from-first-principles Python oracle in `scripts/clt_qwen3_validation.py`.
//!
//! Consumes the frozen reference JSON (`scripts/clt_qwen3_reference.json`)
//! and verifies that candle-mi's [`CrossLayerTranscoder`] encoder produces
//! matching output when fed the same residual vectors. Acceptance bar:
//!
//! - Detected schema is [`TranscoderSchema::CltSplitJumpReLU`].
//! - `(d_model, n_features_per_layer)` match the Python run.
//! - Per test case: active-feature count matches, top-10 feature indices
//!   match exactly, activation magnitudes within `abs diff < 1e-4` at `F32`.
//! - Both `is_cross_layer()` and `is_jump_relu()` schema accessors return
//!   `true` for `CltSplitJumpReLU`.
//!
//! Runs on CPU: the Python oracle is CPU-only, so same-device comparison
//! gives the closest possible numerical match (no CUDA-vs-CPU rounding noise).
//!
//! Requires the 3 encoder safetensors files for layers `{0, 13, 27}` from
//! `bluelightai/clt-qwen3-1.7b-base-20k` cached in
//! `~/.cache/huggingface/hub/` (~240 MiB total — decoders are NOT needed
//! for encoder parity and stay un-downloaded). `#[ignore]`-gated so it
//! does not run by default.
//!
//! Run:
//!   `cargo test --test validate_clt_qwen3 --features clt,transformer -- --ignored`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    missing_docs
)]

mod common;

use common::{json_f32, json_usize, reference_path};

use candle_core::{Device, Tensor};
use candle_mi::clt::{CrossLayerTranscoder, TranscoderSchema};
use std::collections::HashMap;

#[test]
#[ignore = "requires `BlueLightAI` encoder safetensors cached (~240 MiB for 3 layers); run with --ignored"]
// Flat sequence (load reference → open transcoder → assert schema → group cases
// by layer → iterate → compare). Mirrors validate_plt_gemma.rs intentionally.
#[allow(clippy::too_many_lines)]
fn validate_clt_qwen3_encoder_against_python_oracle() {
    // --- Load the frozen reference JSON ---
    let reference_str = std::fs::read_to_string(reference_path("clt_qwen3_reference.json")).expect(
        "failed to read clt_qwen3_reference.json — run scripts/clt_qwen3_validation.py first",
    );
    let reference: serde_json::Value = serde_json::from_str(&reference_str).unwrap();

    let clt_repo = reference["clt_repo"].as_str().unwrap();
    let ref_schema = reference["schema"].as_str().unwrap();
    let d_model = json_usize(&reference["d_model"]);
    let n_features_per_layer = json_usize(&reference["n_features_per_layer"]);
    let test_cases = reference["test_cases"].as_array().unwrap();

    assert_eq!(
        ref_schema, "CltSplitJumpReLU",
        "oracle JSON schema field must be CltSplitJumpReLU"
    );

    println!("Validating `BlueLightAI` CLT encoder parity:");
    println!("  clt_repo: {clt_repo}");
    println!("  d_model = {d_model}, n_features_per_layer = {n_features_per_layer}");
    println!("  {} test cases to check", test_cases.len());

    // --- Open the transcoder; verify schema and dimensions match ---
    // Single-repo flow — `BlueLightAI` ships weights directly at clt_repo
    // (no two-repo curation indirection like GemmaScope).
    let mut clt = CrossLayerTranscoder::open(clt_repo)
        .expect("failed to open `BlueLightAI` CLT — ensure encoder safetensors are in HF cache");

    assert_eq!(
        clt.config().schema,
        TranscoderSchema::CltSplitJumpReLU,
        "open() must detect CltSplitJumpReLU for {clt_repo}"
    );
    assert!(
        clt.config().schema.is_jump_relu(),
        "CltSplitJumpReLU schema must report is_jump_relu() == true"
    );
    assert!(
        clt.config().schema.is_cross_layer(),
        "CltSplitJumpReLU schema must report is_cross_layer() == true \
         (rank-3 cross-layer W_dec, same as plain CltSplit)"
    );
    assert_eq!(
        clt.config().d_model,
        d_model,
        "d_model mismatch with oracle"
    );
    assert_eq!(
        clt.config().n_features_per_layer,
        n_features_per_layer,
        "n_features_per_layer mismatch with oracle"
    );

    let device = Device::Cpu;

    // --- Group test cases by layer so each encoder is loaded exactly once ---
    let mut by_layer: HashMap<usize, Vec<&serde_json::Value>> = HashMap::new();
    for tc in test_cases {
        let layer = json_usize(&tc["layer"]);
        by_layer.entry(layer).or_default().push(tc);
    }

    let mut total_cases = 0_usize;
    let mut max_abs_diff: f32 = 0.0;

    // Iterate layers in sorted order for reproducible output.
    let mut layers: Vec<usize> = by_layer.keys().copied().collect();
    layers.sort_unstable();

    for layer in layers {
        clt.load_encoder(layer, &device).unwrap();
        println!("Layer {layer}:");

        // INDEX: by_layer was populated from the same `layers` keys we are
        // iterating — every `layer` is guaranteed to be a key.
        for tc in &by_layer[&layer] {
            let seed = tc["seed"].as_u64().unwrap();
            let residual_vec: Vec<f32> = tc["residual"]
                .as_array()
                .unwrap()
                .iter()
                .map(json_f32)
                .collect();
            let ref_n_active = json_usize(&tc["n_active"]);
            let ref_top10 = tc["top_10"].as_array().unwrap();

            assert_eq!(
                residual_vec.len(),
                d_model,
                "layer {layer} seed {seed}: residual length {} != d_model {d_model}",
                residual_vec.len()
            );

            // Build the residual tensor on CPU and run the Rust encoder.
            let residual = Tensor::from_vec(residual_vec, (d_model,), &device).unwrap();
            let sparse = clt.encode(&residual, layer).unwrap();

            // --- Active-feature count ---
            // For layer 27 the oracle expects exactly 1 active feature
            // (out-of-distribution random Gaussian residual hits only the
            // lowest-threshold feature). The parity assertion must hold
            // exactly for the same seeded residual.
            assert_eq!(
                sparse.features.len(),
                ref_n_active,
                "layer {layer} seed {seed}: n_active mismatch (Rust {}, Python {})",
                sparse.features.len(),
                ref_n_active
            );

            // --- Top-K indices + activations (K = oracle's top_10 length,
            // which is min(10, n_active); falls to 1 on layer-27 degenerate cases) ---
            for (rank, ref_item) in ref_top10.iter().enumerate() {
                let ref_idx = json_usize(&ref_item["index"]);
                let ref_act = json_f32(&ref_item["activation"]);

                let (rust_fid, rust_act) = sparse.features.get(rank).unwrap_or_else(|| {
                    panic!(
                        "layer {layer} seed {seed}: Rust top-{} shorter than Python's",
                        rank + 1
                    )
                });

                assert_eq!(
                    rust_fid.index, ref_idx,
                    "layer {layer} seed {seed} rank {rank}: index mismatch \
                     (Rust {}, Python {ref_idx})",
                    rust_fid.index
                );
                assert_eq!(
                    rust_fid.layer, layer,
                    "layer {layer} seed {seed} rank {rank}: feature.layer {} != test layer",
                    rust_fid.layer
                );

                let diff = (*rust_act - ref_act).abs();
                assert!(
                    diff < 1e-4,
                    "layer {layer} seed {seed} rank {rank}: activation abs-diff {diff:.2e} >= 1e-4 \
                     (Rust {rust_act}, Python {ref_act})"
                );
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }

            // INDEX: sparse.features[0] — safe because we just validated that
            // top-K is populated (ref_top10.len() ≥ 1 when we enter this block,
            // and sparse.features.len() == ref_n_active ≥ 1).
            let top_feature = sparse.features[0].0;
            println!(
                "  seed={seed:4}: {} active / {} features, top={top_feature}, top-K matches \
                 (max abs-diff so far = {max_abs_diff:.2e})",
                sparse.features.len(),
                n_features_per_layer,
            );
            total_cases += 1;
        }
    }

    println!(
        "\n{total_cases} test cases passed; max abs-diff across all top-K activations = \
         {max_abs_diff:.2e} (bar: 1e-4)"
    );
}
