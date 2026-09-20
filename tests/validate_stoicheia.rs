// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation tests for stoicheia (`AlgZoo`) backends.
//!
//! Loads pre-trained weights from `safetensors` fixtures, runs the same
//! inputs as the `Python` reference, and compares outputs to 6 decimal places.

// Test/example target: these lints are denied crate-wide for library code,
// where a panic is a bug. Here a failed unwrap IS the failure signal, and
// indexing a fixture whose shape the test itself fixes cannot go out of
// bounds. Same allowance as the other 30+ files under tests/ and examples/.
#![allow(clippy::expect_used)]
#![allow(clippy::indexing_slicing)]

use candle_core::{Device, IndexOp, Tensor};
use candle_mi::MIBackend;
use candle_mi::hooks::HookSpec;
use candle_mi::stoicheia::config::{StoicheiaConfig, StoicheiaTask};
use candle_mi::stoicheia::{StoicheiaRnn, StoicheiaTransformer};

/// Parsed reference data from a `Python`-generated JSON file.
struct Reference {
    input: Vec<Vec<f64>>,
    output: Vec<Vec<f64>>,
}

impl Reference {
    fn load(path: &str) -> Self {
        let content = std::fs::read_to_string(path).expect("fixture file missing");
        let json: serde_json::Value = serde_json::from_str(&content).expect("invalid JSON fixture");

        let input: Vec<Vec<f64>> = json["input"]
            .as_array()
            .expect("missing input")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("input row not array")
                    .iter()
                    .map(|v| v.as_f64().expect("input value not float"))
                    .collect()
            })
            .collect();

        let output_raw = &json["output"];
        let output: Vec<Vec<f64>> = output_raw
            .as_array()
            .expect("missing output")
            .iter()
            .map(|row| {
                // Scalar tasks return [[v], [v], ...]; distribution tasks return [[v, v, ...], ...]
                if row.is_array() {
                    row.as_array()
                        .expect("output row not array")
                        .iter()
                        .map(|v| v.as_f64().expect("output value not float"))
                        .collect()
                } else {
                    // Scalar output stored as flat value
                    vec![row.as_f64().expect("output value not float")]
                }
            })
            .collect();

        Self { input, output }
    }
}

/// Compare two `f32` values with tolerance (6 decimal places).
fn assert_close(actual: f32, expected: f64, name: &str, tolerance: f64) {
    // CAST: f32 → f64, widening for comparison
    #[allow(clippy::as_conversions)]
    let actual_f64 = f64::from(actual);
    let diff = (actual_f64 - expected).abs();
    assert!(
        diff < tolerance,
        "{name}: actual={actual}, expected={expected}, diff={diff}"
    );
}

/// Assert that `project_to_vocab` preserves rank.
///
/// A `[batch, seq, hidden]` hidden state must project to `[batch, seq, vocab]`,
/// and every position must equal the rank-2 projection of that same position, so
/// the widened contract cannot quietly compute something different.
///
/// The trait documented rank 2 only, while four of six backends already accepted
/// rank 3 (`Linear` and `LayerNorm` are rank-agnostic) and the two `stoicheia`
/// backends rejected it, because candle's `matmul` requires operands of equal
/// rank. See `docs/dogfooding-feedbacks/interp-api-forces-stringly-typed-hook-handling.md`.
// TRAIT_OBJECT: one helper serves both `stoicheia` backends
fn assert_project_to_vocab_preserves_rank(model: &dyn MIBackend, hidden_size: usize) {
    let (batch, seq) = (2_usize, 3_usize);
    let hidden = Tensor::randn(0.0_f32, 1.0, (batch, seq, hidden_size), &Device::Cpu)
        .expect("failed to create hidden state");

    let logits = model
        .project_to_vocab(&hidden)
        .expect("rank-3 project_to_vocab failed");
    let (out_batch, out_seq, _) = logits.dims3().expect("expected a rank-3 projection");
    assert_eq!(
        (out_batch, out_seq),
        (batch, seq),
        "leading dimensions not preserved"
    );

    let actual: Vec<Vec<Vec<f32>>> = logits.to_vec3().expect("failed to extract logits");
    for (b, actual_b) in actual.iter().enumerate().take(batch) {
        for (s, actual_bs) in actual_b.iter().enumerate().take(seq) {
            let position = hidden
                .i((b, s, ..))
                .expect("failed to slice position")
                .unsqueeze(0)
                .expect("failed to unsqueeze position");
            let expected: Vec<Vec<f32>> = model
                .project_to_vocab(&position)
                .expect("rank-2 project_to_vocab failed")
                .to_vec2()
                .expect("failed to extract rank-2 logits");
            for (col, (&a, &e)) in actual_bs.iter().zip(&expected[0]).enumerate() {
                // Tolerance, not equality: the rank-3 path runs a batched gemm
                // whose accumulation order need not match the rank-2 one.
                assert_close(a, f64::from(e), &format!("[{b}][{s}][{col}]"), 1e-5);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RNN cross-validation
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
fn rnn_2nd_argmax_h2_n2_matches_python() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(
        config,
        "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load RNN fixture");

    let reference = Reference::load("tests/fixtures/stoicheia/ref_2nd_argmax_h2_n2.json");

    // Build input tensor from reference data: [batch=4, seq_len=2]
    let input_data: Vec<f32> = reference
        .input
        .iter()
        // CAST: f64 → f32, reference fixtures are stored at f32 precision anyway
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    let input = Tensor::from_slice(
        &input_data,
        (reference.input.len(), reference.input[0].len()),
        &Device::Cpu,
    )
    .expect("failed to create input tensor");

    // Forward pass
    let cache = model
        .forward(&input, &HookSpec::new())
        .expect("forward pass failed");

    // Output is [batch, 1, output_size] — squeeze the seq dim
    let output = cache.output().squeeze(1).expect("failed to squeeze output");
    let output_vec: Vec<Vec<f32>> = output.to_vec2().expect("failed to extract output");

    // Compare against Python reference
    for (batch_idx, (actual_row, expected_row)) in
        output_vec.iter().zip(&reference.output).enumerate()
    {
        for (col_idx, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
            assert_close(
                actual,
                expected,
                &format!("batch[{batch_idx}][{col_idx}]"),
                1e-4,
            );
        }
    }
}

#[test]
fn rnn_hook_capture() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let model = StoicheiaRnn::load(
        config,
        "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load RNN fixture");

    let input = Tensor::randn(0.0_f32, 1.0, (1, 2), &Device::Cpu).expect("failed to create input");

    let mut hooks = HookSpec::new();
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_hidden.0".into()));
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_hidden.1".into()));
    hooks.capture(candle_mi::HookPoint::Custom("rnn.hook_final_state".into()));

    let cache = model.forward(&input, &hooks).expect("forward pass failed");

    // Hidden states should be [1, 2] (batch=1, hidden=2)
    let h0 = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_hidden.0".into()))
        .expect("hook_hidden.0 not captured");
    assert_eq!(h0.dims(), &[1, 2]);

    let h1 = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_hidden.1".into()))
        .expect("hook_hidden.1 not captured");
    assert_eq!(h1.dims(), &[1, 2]);

    // Final state should equal h1 (last timestep)
    let final_state = cache
        .get(&candle_mi::HookPoint::Custom("rnn.hook_final_state".into()))
        .expect("hook_final_state not captured");
    assert_eq!(final_state.dims(), &[1, 2]);
}

// ---------------------------------------------------------------------------
// Transformer cross-validation
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn transformer_longest_cycle_h4_n4_matches_python() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
    let model = StoicheiaTransformer::load(
        config,
        "tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load transformer fixture");

    let reference = Reference::load("tests/fixtures/stoicheia/ref_longest_cycle_h4_n4.json");

    // Build input tensor from reference data: [batch=4, seq_len=4] (integers)
    // CAST: f64 → u32, reference stores integers as floats in JSON
    #[allow(clippy::as_conversions)]
    let input_data: Vec<u32> = reference
        .input
        .iter()
        // CAST: f64 → u32, fixture values are small non-negative token indices
        .flat_map(|row| row.iter().map(|&v| v as u32))
        .collect();
    let input = Tensor::from_slice(
        &input_data,
        (reference.input.len(), reference.input[0].len()),
        &Device::Cpu,
    )
    .expect("failed to create input tensor");

    // Forward pass
    let cache = model
        .forward(&input, &HookSpec::new())
        .expect("forward pass failed");

    // Output is [batch, 1, output_size] — squeeze the seq dim
    let output = cache.output().squeeze(1).expect("failed to squeeze output");
    let output_vec: Vec<Vec<f32>> = output.to_vec2().expect("failed to extract output");

    // Compare against Python reference
    for (batch_idx, (actual_row, expected_row)) in
        output_vec.iter().zip(&reference.output).enumerate()
    {
        for (col_idx, (&actual, &expected)) in actual_row.iter().zip(expected_row).enumerate() {
            assert_close(
                actual,
                expected,
                &format!("batch[{batch_idx}][{col_idx}]"),
                1e-2, // Transformer logits are large (thousands); use relative-scale tolerance
            );
        }
    }
}

#[test]
fn transformer_hook_capture() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
    let model = StoicheiaTransformer::load(
        config,
        "tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load transformer fixture");

    let input = Tensor::new(&[0_u32, 1, 2, 3], &Device::Cpu)
        .expect("failed to create input")
        .unsqueeze(0)
        .expect("failed to unsqueeze");

    let mut hooks = HookSpec::new();
    hooks.capture(candle_mi::HookPoint::Embed);
    hooks.capture(candle_mi::HookPoint::AttnPattern(0));
    hooks.capture(candle_mi::HookPoint::AttnPattern(1));
    hooks.capture(candle_mi::HookPoint::ResidPost(0));
    hooks.capture(candle_mi::HookPoint::ResidPost(1));

    let cache = model.forward(&input, &hooks).expect("forward pass failed");

    // Embed: [1, 4, 4] (batch=1, seq=4, hidden=4)
    let embed = cache
        .get(&candle_mi::HookPoint::Embed)
        .expect("Embed not captured");
    assert_eq!(embed.dims(), &[1, 4, 4]);

    // AttnPattern: [1, 1, 4, 4] (batch=1, heads=1, seq=4, seq=4)
    let attn0 = cache
        .get(&candle_mi::HookPoint::AttnPattern(0))
        .expect("AttnPattern(0) not captured");
    assert_eq!(attn0.dims(), &[1, 1, 4, 4]);

    // ResidPost: [1, 4, 4] (batch=1, seq=4, hidden=4)
    let resid1 = cache
        .get(&candle_mi::HookPoint::ResidPost(1))
        .expect("ResidPost(1) not captured");
    assert_eq!(resid1.dims(), &[1, 4, 4]);
}

#[test]
fn rnn_project_to_vocab_preserves_rank() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
    let hidden_size = config.hidden_size;
    let model = StoicheiaRnn::load(
        config,
        "tests/fixtures/stoicheia/rnn_2nd_argmax_h2_n2.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load RNN fixture");

    assert_project_to_vocab_preserves_rank(&model, hidden_size);
}

#[test]
fn transformer_project_to_vocab_preserves_rank() {
    let config = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
    let hidden_size = config.hidden_size;
    let model = StoicheiaTransformer::load(
        config,
        "tests/fixtures/stoicheia/transformer_longest_cycle_h4_n4.safetensors",
        &Device::Cpu,
    )
    .expect("failed to load transformer fixture");

    assert_project_to_vocab_preserves_rank(&model, hidden_size);
}
