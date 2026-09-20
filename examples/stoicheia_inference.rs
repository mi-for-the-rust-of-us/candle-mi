// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stoicheia inference example — run an `AlgZoo` model and compare against
//! ground truth.
//!
//! Loads a pre-trained `AlgZoo` RNN or transformer from a `safetensors` file,
//! runs inference on random inputs, and checks predictions against the
//! ground-truth task function.
//!
//! # Usage
//!
//! ```bash
//! cargo run --features stoicheia --release --example stoicheia_inference -- \
//!     --task 2nd-argmax --hidden-size 2 --seq-len 2 \
//!     --weights path/to/rnn_2nd_argmax_h2_n2.safetensors
//! ```

// Test/example target: these lints are denied crate-wide for library code,
// where a panic is a bug. Here a failed unwrap IS the failure signal, and
// indexing a fixture whose shape the test itself fixes cannot go out of
// bounds. Same allowance as the other 30+ files under tests/ and examples/.
#![allow(clippy::panic)]

use candle_core::{Device, Tensor};
use candle_mi::MIBackend;
use candle_mi::hooks::{HookPoint, HookSpec};
use candle_mi::stoicheia::config::{StoicheiaArch, StoicheiaConfig, StoicheiaTask};
use candle_mi::stoicheia::tasks;
use candle_mi::stoicheia::{StoicheiaRnn, StoicheiaTransformer};
use clap::Parser;

/// Run an `AlgZoo` model and compare against ground truth.
#[derive(Parser)]
struct Args {
    /// Task name: 2nd-argmax, argmedian, median, longest-cycle.
    #[arg(long)]
    task: String,
    /// Hidden dimension.
    #[arg(long)]
    hidden_size: usize,
    /// Sequence length.
    #[arg(long)]
    seq_len: usize,
    /// Path to the `.safetensors` weight file.
    #[arg(long)]
    weights: String,
    /// Number of random samples to test.
    #[arg(long, default_value = "100")]
    samples: usize,
}

/// Parse the `--task` argument into a [`StoicheiaTask`].
fn parse_task(s: &str) -> StoicheiaTask {
    match s {
        "2nd-argmax" | "2nd_argmax" => StoicheiaTask::SecondArgmax,
        "argmedian" => StoicheiaTask::Argmedian,
        "median" => StoicheiaTask::Median,
        "longest-cycle" | "longest_cycle" => StoicheiaTask::LongestCycle,
        other => panic!("unknown task: {other}"),
    }
}

fn main() -> candle_mi::Result<()> {
    let args = Args::parse();
    let task = parse_task(&args.task);
    let config = StoicheiaConfig::from_task(task, args.hidden_size, args.seq_len);

    let arch = config.arch;
    let seq_len = config.seq_len;

    println!("Model: {config}");
    println!("Weights: {}", args.weights);
    println!("Samples: {}", args.samples);

    let device = Device::Cpu;

    // Load model based on architecture
    // TRAIT_OBJECT: heterogeneous dispatch between RNN and transformer backends
    let model: Box<dyn MIBackend> = match arch {
        StoicheiaArch::Rnn => Box::new(StoicheiaRnn::load(config, &args.weights, &device)?),
        StoicheiaArch::Transformer => {
            Box::new(StoicheiaTransformer::load(config, &args.weights, &device)?)
        }
        _ => panic!("unsupported architecture: {arch}"),
    };

    // Generate random input
    let input = match arch {
        StoicheiaArch::Rnn => Tensor::randn(0.0_f32, 1.0, (args.samples, seq_len), &device)?,
        StoicheiaArch::Transformer => {
            // CAST: usize → u32, seq_len is small (max 10 in `AlgZoo`)
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            let range = seq_len as u32;
            // CAST: usize → u32, small indices bounded by `range` above
            #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
            let data: Vec<u32> = (0..args.samples * seq_len)
                .map(|i| (i as u32) % range)
                .collect();
            Tensor::from_slice(&data, (args.samples, seq_len), &device)?
        }
        _ => panic!("unsupported architecture: {arch}"),
    };

    // Forward pass with hook capture
    let mut hooks = HookSpec::new();
    if arch == StoicheiaArch::Transformer {
        hooks.capture(HookPoint::AttnPattern(0));
    }

    let cache = model.forward(&input, &hooks)?;
    let output = cache.output().squeeze(1)?;
    let predictions = output.argmax(1)?;
    let pred_vec: Vec<u32> = predictions.to_vec1()?;

    // Compute ground truth
    let targets = match task {
        StoicheiaTask::SecondArgmax => tasks::second_argmax(&input)?,
        StoicheiaTask::Argmedian => tasks::argmedian(&input)?,
        StoicheiaTask::LongestCycle => tasks::longest_cycle(&input)?,
        StoicheiaTask::Median => {
            println!("Median task outputs scalar values, not positions — skipping accuracy.");
            return Ok(());
        }
        _ => panic!("unsupported task: {task}"),
    };
    let target_vec: Vec<u32> = targets.to_vec1()?;

    // Compute accuracy
    let correct = pred_vec
        .iter()
        .zip(&target_vec)
        .filter(|(p, t)| p == t)
        .count();

    // CAST: usize → f64, sample count for percentage
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let accuracy = 100.0 * correct as f64 / args.samples as f64;

    println!("\nAccuracy: {correct}/{} ({accuracy:.1}%)", args.samples);

    // Show first 5 predictions
    println!("\nFirst 5 predictions:");
    for i in 0..5.min(args.samples) {
        // INDEX: i bounded by min(5, samples)
        #[allow(clippy::indexing_slicing)]
        let (p, t) = (pred_vec[i], target_vec[i]);
        let mark = if p == t { "OK" } else { "MISS" };
        println!("  [{i}] predicted={p}, target={t} {mark}");
    }

    // Show attention pattern if captured
    if let Some(attn) = cache.get(&HookPoint::AttnPattern(0)) {
        println!("\nAttention pattern (layer 0), sample 0:");
        println!("  shape: {:?}", attn.dims());
    }

    Ok(())
}
