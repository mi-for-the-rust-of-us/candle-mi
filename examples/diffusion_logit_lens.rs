// SPDX-License-Identifier: MIT OR Apache-2.0

//! Diffusion-time logit lens for `MDLM`.
//!
//! ```bash
//! cargo run --features diffusion,mmap --release --example diffusion_logit_lens
//! ```
//!
//! The autoregressive logit lens reads a `(layer × position)` heatmap.  A
//! masked-diffusion model adds a third axis — the **denoising step** `k` — so
//! the lens becomes a `(k, layer, position)` object.  This example fixes one
//! masked target position and prints the `(layer × step)` slice: for each
//! denoising step `k`, it runs the model on that step's partially-masked state,
//! captures every layer's residual stream, and applies the model's unembedding
//! (`project_to_vocab`, i.e. final-norm + output head) to read each layer's
//! top-1 prediction at the target position.
//!
//! You can watch the prediction **crystallize** — late layers commit earlier
//! than early ones, and the committed token sharpens as `k` advances and the
//! surrounding context gets revealed.

use candle_core::{IndexOp, Tensor};
use candle_mi::{DiffusionSamplingConfig, HookPoint, HookSpec, MIModel, MITokenizer};

/// `HuggingFace` repo id of the MDLM masked-diffusion checkpoint.
const MODEL_ID: &str = "TheQweaker/mdlm-owt-noflash";
/// Number of denoising steps in the sampled trajectory.
const NUM_STEPS: usize = 8;
/// Number of positions to fill after the prompt.
const GEN_LEN: usize = 4;

fn main() -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained(MODEL_ID)?;
    let Ok(tokenizer) = MITokenizer::from_hf_cache("openai-community/gpt2")
        .or_else(|_| MITokenizer::from_hf_cache("gpt2"))
    else {
        println!("GPT-2 tokenizer not found in the HuggingFace cache.");
        println!("Get tokenizer.json from https://huggingface.co/openai-community/gpt2");
        println!(
            "e.g. (dogfooding hf-fm):  hf-fm download-file openai-community/gpt2 tokenizer.json"
        );
        return Ok(());
    };

    // MDLM convention: [MASK] is the final vocab id. Decoder-style diffusion LMs
    // (Dream, a2d-qwen2) use a distinct `<|mask|>` token id instead — supply it
    // when reusing this on them.
    let mask_id = u32::try_from(model.vocab_size() - 1).map_err(|e| {
        candle_mi::MIError::Model(candle_core::Error::Msg(format!("mask id overflow: {e}")))
    })?;

    // Prompt is carried over; GEN_LEN positions are denoised. The target is the
    // first generated position (right after the prompt).
    let prompt = tokenizer.encode_raw("The capital of France is")?;
    let seq_len = prompt.len() + GEN_LEN;
    let target_pos = prompt.len();
    let n_layers = model.num_layers();

    let config = DiffusionSamplingConfig::default()
        .with_seq_len(seq_len)
        .with_num_steps(NUM_STEPS)
        .with_top_k(Some(50));
    let trajectory = candle_mi::diffusion::generate_trajectory(
        model.backend(),
        model.device(),
        mask_id,
        &prompt,
        &config,
    )?;

    println!(
        "MDLM diffusion-time logit lens — prompt {:?}, target position {target_pos} (first generated)",
        "The capital of France is"
    );
    println!("Each cell: top-1 prediction of layer L at denoising step k, via the unembedding.\n");

    // Build one column per denoising step: column[k][layer] = top-1 token string.
    let mut columns: Vec<Vec<String>> = Vec::with_capacity(trajectory.len());
    let mut revealed: Vec<String> = Vec::with_capacity(trajectory.len());
    for state in &trajectory {
        let input = Tensor::new(state.as_slice(), model.device())?.unsqueeze(0)?;
        let mut hooks = HookSpec::new();
        for layer in 0..n_layers {
            hooks.capture(HookPoint::ResidPost(layer));
        }
        let cache = model.forward(&input, &hooks)?;

        let mut column = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let resid = cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
            let at = resid.i((0, target_pos))?.unsqueeze(0)?; // [1, hidden]
            let logits = model.project_to_vocab(&at)?; // [1, vocab]
            let token = candle_mi::sample_token(&logits.flatten_all()?, 0.0)?;
            column.push(decode_cell(&tokenizer, token)?);
        }
        columns.push(column);

        // What the sampler actually has at the target position in this state.
        let actual = state.get(target_pos).copied().unwrap_or(mask_id);
        revealed.push(if actual == mask_id {
            "·".to_owned() // still masked
        } else {
            decode_cell(&tokenizer, actual)?
        });
    }

    // Header: denoising steps.
    print!("layer \\ k |");
    for k in 0..columns.len() {
        print!(" {k:>8}");
    }
    println!();
    print!("  (state) |");
    for cell in &revealed {
        print!(" {cell:>8.8}");
    }
    println!();
    println!("{}", "-".repeat(11 + 9 * columns.len()));

    // Rows: layers, top-1 prediction at the target position.
    for layer in 0..n_layers {
        print!("L{layer:>2}      |");
        for column in &columns {
            let cell = column.get(layer).map_or("?", String::as_str);
            print!(" {cell:>8.8}");
        }
        println!();
    }

    let final_token = trajectory
        .last()
        .and_then(|s| s.get(target_pos).copied())
        .unwrap_or(mask_id);
    println!(
        "\nFinal token at target position: {:?}",
        decode_cell(&tokenizer, final_token)?
    );
    Ok(())
}

/// Decode a single token id to a trimmed display string.
fn decode_cell(tokenizer: &MITokenizer, token: u32) -> candle_mi::Result<String> {
    Ok(tokenizer.decode(&[token])?.trim().to_owned())
}

// (GPT-2 tokenizer discovery now lives in `MITokenizer::from_hf_cache`.)
