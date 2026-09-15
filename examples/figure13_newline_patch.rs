// SPDX-License-Identifier: MIT OR Apache-2.0
//! Newline activation patching: does the residual at a poem's newline carry the
//! rhyme of the line the model has not written yet?
//!
//! ```bash
//! # Layer trace plus the all-layer patch, one minimal pair, one direction
//! cargo run --release --features transformer,mmap --example figure13_newline_patch -- \
//!     --pair p1-a-to-b --model google/gemma-2-2b --k-samples 60 --seed 1 \
//!     --output docs/experiments/figure13-patching/patch_gemma_p1_a2b_s1.json
//!
//! # The identity control on its own: patching a prompt from itself must be a no-op
//! cargo run --release --features transformer,mmap --example figure13_newline_patch -- \
//!     --pair p1-a-to-b --identity-only
//! ```
//!
//! **The question.** Every other causal probe in this line of work writes a
//! cross-layer-transcoder decoder direction into the residual stream, so a null
//! result can always be blamed on the feature-discovery method rather than on
//! the model. Activation patching replaces a residual row with one the model
//! itself produced on another prompt: no transcoder, no feature search, no
//! steering strength. It is therefore the one instrument whose null cannot be
//! explained away that same way, which is why it is worth a separate example.
//!
//! **The design.** Donor and recipient are a *minimal pair*: two prompts
//! identical through line 3 except for that line's final word, which sets a
//! different rhyme. Patching the newline transfers that position's whole state,
//! so with arbitrary prompts a rhyme change would only say that the recipient
//! inherited the donor's context; with a minimal pair, everything the patch can
//! carry is shared except the rhyme.
//!
//! The steering site is the line-3 newline, which is also the final prompt
//! token, so the model must compose the whole of line 4 downstream of it. As in
//! `figure13_newline_steering`, candle-mi keeps no KV cache, so the patch is
//! re-applied at that fixed position at every generation step
//! (`route: "recompute-per-step"` in the output).
//!
//! **What it runs.** A donor forward pass capturing [`HookPoint::ResidPost`] at
//! every layer; then, on the recipient:
//!
//! 1. `baseline` — no patch, for the unpatched rhyme distribution;
//! 2. a **layer trace** — single-layer patch at each layer in turn, reporting
//!    the teacher-forced probability of the donor's and the recipient's rhyme
//!    words at the composed line's final-word slot. One forward per layer, so
//!    this is cheap and runs first;
//! 3. `all-layer` — the donor's newline row patched at *every* layer at once,
//!    which replaces that position's entire state. A null under single-layer
//!    patching alone would admit the escape "the plan is spread across
//!    layers"; this condition closes it;
//! 4. `single-layer` — the layer the trace ranked best, sampled properly;
//! 5. `identity` — donor and recipient are the same prompt. This must be an
//!    exact no-op. The CUDA path of [`Intervention::PatchAt`] was silently
//!    wrong before v0.1.25, so the check runs by default rather than on
//!    request.
//!
//! Rhyme classification is deliberately **not** done here: the composed lines
//! are written out verbatim and classified by CMU rime in the Python layer, the
//! same way `figure13_newline_steering` output is, so both experiments share
//! one phonology.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use candle_mi::{
    FullActivationCache, HookPoint, HookSpec, Intervention, MIModel, MITokenizer,
    extract_token_prob,
};

// ── Minimal-pair presets ──────────────────────────────────────────────────────

/// One member of a minimal pair: a four-line prompt truncated after line 3.
struct Member {
    /// The first three lines plus the line-3 newline. The model composes line 4.
    prompt: &'static str,
    /// Final word of line 3, whose rhyme line 4 is expected to pick up.
    rhyme_word: &'static str,
}

/// A minimal pair plus the direction in which it is patched.
struct Pair {
    /// Preset name, as passed to `--pair`.
    name: &'static str,
    /// The prompt whose newline residual is captured.
    donor: Member,
    /// The prompt that is generated from, with the donor's row patched in.
    recipient: Member,
}

/// Lines 1 and 2 are shared inside a pair; only line 3's last word differs.
/// All four prompts are `AABB`: lines 1 and 2 rhyme together and line 3 opens a
/// new couplet, so swapping line 3's ending does not leave the earlier lines
/// pulling toward the original rhyme.
const PAIRS: &[Pair] = &[
    Pair {
        name: "p1-a-to-b",
        donor: Member {
            prompt: "The stars were twinkling in the night,\n\
                     The lanterns cast a golden light.\n\
                     She wandered in the dark about,\n",
            rhyme_word: "about",
        },
        recipient: Member {
            prompt: "The stars were twinkling in the night,\n\
                     The lanterns cast a golden light.\n\
                     She wandered in the dark alone,\n",
            rhyme_word: "alone",
        },
    },
    Pair {
        name: "p1-b-to-a",
        donor: Member {
            prompt: "The stars were twinkling in the night,\n\
                     The lanterns cast a golden light.\n\
                     She wandered in the dark alone,\n",
            rhyme_word: "alone",
        },
        recipient: Member {
            prompt: "The stars were twinkling in the night,\n\
                     The lanterns cast a golden light.\n\
                     She wandered in the dark about,\n",
            rhyme_word: "about",
        },
    },
    Pair {
        name: "p2-a-to-b",
        donor: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     The world keeps spinning even so,\n",
            rhyme_word: "so",
        },
        recipient: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     The world keeps spinning even now,\n",
            rhyme_word: "now",
        },
    },
    Pair {
        name: "p2-b-to-a",
        donor: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     The world keeps spinning even now,\n",
            rhyme_word: "now",
        },
        recipient: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     The world keeps spinning even so,\n",
            rhyme_word: "so",
        },
    },
    Pair {
        name: "p3-a-to-b",
        donor: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     He raised his voice and gave a shout,\n",
            rhyme_word: "shout",
        },
        recipient: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     He raised his voice and gave a sigh,\n",
            rhyme_word: "sigh",
        },
    },
    Pair {
        name: "p3-b-to-a",
        donor: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     He raised his voice and gave a sigh,\n",
            rhyme_word: "sigh",
        },
        recipient: Member {
            prompt: "A sailor sailed across the bay,\n\
                     And dreamed of home throughout the day.\n\
                     He raised his voice and gave a shout,\n",
            rhyme_word: "shout",
        },
    },
    Pair {
        name: "p4-a-to-b",
        donor: Member {
            prompt: "The sun goes up, the sun goes down,\n\
                     The moon shines bright above the town.\n\
                     Nobody knows or remembers who,\n",
            rhyme_word: "who",
        },
        recipient: Member {
            prompt: "The sun goes up, the sun goes down,\n\
                     The moon shines bright above the town.\n\
                     Nobody knows or remembers when,\n",
            rhyme_word: "when",
        },
    },
    Pair {
        name: "p4-b-to-a",
        donor: Member {
            prompt: "The sun goes up, the sun goes down,\n\
                     The moon shines bright above the town.\n\
                     Nobody knows or remembers when,\n",
            rhyme_word: "when",
        },
        recipient: Member {
            prompt: "The sun goes up, the sun goes down,\n\
                     The moon shines bright above the town.\n\
                     Nobody knows or remembers who,\n",
            rhyme_word: "who",
        },
    },
];

/// Look up a minimal pair by preset name.
///
/// # Errors
/// Returns [`MIError::Config`](candle_mi::MIError::Config) if `name` is not one
/// of the eight preset pair names.
fn select_pair(name: &str) -> candle_mi::Result<&'static Pair> {
    PAIRS.iter().find(|p| p.name == name).ok_or_else(|| {
        let known: Vec<&str> = PAIRS.iter().map(|p| p.name).collect();
        candle_mi::MIError::Config(format!("unknown pair {name} (known: {})", known.join(", ")))
    })
}

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "figure13_newline_patch",
    about = "Newline activation patching between minimal-pair poems (CLT-free)"
)]
struct Args {
    /// Minimal-pair preset and direction, e.g. `p1-a-to-b`.
    #[arg(long, default_value = "p1-a-to-b")]
    pair: String,

    /// `HuggingFace` model ID.
    #[arg(long, default_value = "google/gemma-2-2b")]
    model: String,

    /// Donor prompt override; must end at the line-3 newline.
    #[arg(long)]
    donor_prompt: Option<String>,

    /// Recipient prompt override; must end at the line-3 newline.
    #[arg(long)]
    recipient_prompt: Option<String>,

    /// Word whose probability tracks the donor's rhyme (overrides the preset).
    #[arg(long)]
    donor_word: Option<String>,

    /// Word whose probability tracks the recipient's rhyme (overrides the preset).
    #[arg(long)]
    recipient_word: Option<String>,

    /// Max new tokens per composed line.
    #[arg(long, default_value_t = 15)]
    generate: usize,

    /// Sampled lines per condition.
    #[arg(long, default_value_t = 20)]
    k_samples: usize,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// RNG seed, logged in the output.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Run only the identity control and exit. Useful as a fast device check.
    #[arg(long)]
    identity_only: bool,

    /// Patch this many tokens *before* the line-3 newline instead of at it.
    /// `0` is the newline itself; the registered `mid-line` control uses a
    /// positive offset to land inside line 3, which localizes any effect found.
    #[arg(long, default_value_t = 0)]
    patch_offset: usize,

    /// Run only the unpatched baseline: the prompt-validation phase, which asks
    /// whether the recipient rhymes at all before any patch result is worth
    /// interpreting. Skips the layer trace and both patch conditions.
    #[arg(long)]
    baseline_only: bool,

    /// Output JSON path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

// ── Output schema ─────────────────────────────────────────────────────────────

/// Teacher-forced probabilities under a single-layer patch at one layer.
#[derive(Serialize)]
struct LayerProb {
    /// Layer whose `ResidPost` row was replaced.
    layer: usize,
    /// Probability of the donor's rhyme word at the final-word slot.
    p_donor: f32,
    /// Probability of the recipient's rhyme word at the same slot.
    p_recipient: f32,
}

/// How far apart the donor's and recipient's newline rows are at one layer.
///
/// A null result is only interpretable against this: if the two rows are
/// near-identical, the newline does not encode the upcoming rhyme at all, and
/// the patch had nothing to carry. If they differ and the rhyme still does not
/// move, the newline encodes something, but not a rhyme the model acts on.
#[derive(Serialize)]
struct LayerDivergence {
    /// Layer whose `ResidPost` rows are compared.
    layer: usize,
    /// Cosine similarity between the donor's and the recipient's newline rows.
    cosine: f32,
    /// Euclidean distance between the rows over the recipient row's norm.
    relative_l2: f32,
}

/// One patching condition, with the same readouts as the steering experiment.
#[derive(Serialize)]
struct ConditionResult {
    /// Condition name: `baseline`, `all-layer`, `single-layer`, or `identity`.
    condition: String,
    /// Layers patched; empty for the baseline.
    patch_layers: Vec<usize>,
    /// The greedy composed line, verbatim.
    greedy_line: String,
    /// Teacher-forced probability of the donor's rhyme word.
    m3_p_donor: f32,
    /// Teacher-forced probability of the recipient's rhyme word.
    m3_p_recipient: f32,
    /// Sampled composed lines, classified downstream by CMU rime.
    sampled_lines: Vec<String>,
}

/// The identity control: patching a prompt with its own newline row.
#[derive(Serialize)]
struct IdentityCheck {
    /// Whether the greedy line is byte-identical to the unpatched baseline.
    greedy_matches_baseline: bool,
    /// Largest absolute difference across the two tracked probabilities.
    max_abs_prob_delta: f32,
    /// Whether the check passed; a false here invalidates the run.
    passed: bool,
}

/// Everything one (model, pair, direction, seed) run produces.
#[derive(Serialize)]
struct PatchOutput {
    /// `HuggingFace` model ID.
    model: String,
    /// Minimal-pair preset name and direction.
    pair: String,
    /// Donor prompt, ending at its line-3 newline.
    donor_prompt: String,
    /// Recipient prompt, ending at its line-3 newline.
    recipient_prompt: String,
    /// Donor's line-3 final word.
    donor_word: String,
    /// Recipient's line-3 final word.
    recipient_word: String,
    /// Recipient prompt tokens, for position bookkeeping.
    tokens: Vec<String>,
    /// Index of the donor's newline, where its row is taken.
    donor_newline_index: usize,
    /// Index of the recipient's newline, where the row is written.
    recipient_newline_index: usize,
    /// Number of layers patched by the `all-layer` condition.
    n_layers: usize,
    /// Tokens before the newline at which the patch was applied (`0` = at it).
    patch_offset: usize,
    /// Recipient sequence index actually patched.
    patch_position: usize,
    /// KV-cache route; always `"recompute-per-step"` for candle-mi.
    route: &'static str,
    /// Sampling temperature.
    temperature: f32,
    /// RNG seed.
    seed: u64,
    /// The unpatched greedy line, which defines the teacher-forced slot.
    baseline_greedy_line: String,
    /// Sequence index of the first token of that line's final word.
    final_word_slot: usize,
    /// One entry per condition.
    conditions: Vec<ConditionResult>,
    /// Single-layer patch trace over every layer.
    layer_trace: Vec<LayerProb>,
    /// Per-layer distance between the donor's and the recipient's newline rows.
    newline_divergence: Vec<LayerDivergence>,
    /// Result of the identity control.
    identity_check: IdentityCheck,
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    let pair = select_pair(&args.pair)?;
    let donor_prompt = args
        .donor_prompt
        .clone()
        .unwrap_or_else(|| pair.donor.prompt.to_owned());
    let recipient_prompt = args
        .recipient_prompt
        .clone()
        .unwrap_or_else(|| pair.recipient.prompt.to_owned());
    let donor_word = args
        .donor_word
        .clone()
        .unwrap_or_else(|| pair.donor.rhyme_word.to_owned());
    let recipient_word = args
        .recipient_word
        .clone()
        .unwrap_or_else(|| pair.recipient.rhyme_word.to_owned());

    eprintln!("Model: {}", args.model);
    eprintln!("Pair:  {}", args.pair);
    eprintln!("Donor     (\"{donor_word}\"):\n{donor_prompt}");
    eprintln!("Recipient (\"{recipient_word}\"):\n{recipient_prompt}");

    let model = MIModel::from_pretrained(&args.model)?;
    let tokenizer = model
        .tokenizer()
        .ok_or_else(|| candle_mi::MIError::Tokenizer("model has no bundled tokenizer".into()))?;
    let device = model.device().clone();
    let n_layers = model.num_layers();

    let donor_tokens = tokenizer.encode(&donor_prompt)?;
    let recipient_tokens = tokenizer.encode(&recipient_prompt)?;
    // The steering site is the final prompt token, which is the line-3 newline
    // by construction of the prompts above.
    let donor_newline = donor_tokens.len() - 1;
    let recipient_newline = recipient_tokens.len() - 1;
    assert_newline(tokenizer, &donor_tokens, donor_newline, "donor")?;
    assert_newline(tokenizer, &recipient_tokens, recipient_newline, "recipient")?;
    eprintln!(
        "Donor newline @ {donor_newline} ({} tokens); recipient newline @ {recipient_newline} ({} tokens); {n_layers} layers",
        donor_tokens.len(),
        recipient_tokens.len()
    );

    let patch_pos = recipient_newline
        .checked_sub(args.patch_offset)
        .ok_or_else(|| {
            candle_mi::MIError::Config(format!(
                "patch offset {} exceeds the recipient newline index {recipient_newline}",
                args.patch_offset
            ))
        })?;
    let donor_pos = donor_newline
        .checked_sub(args.patch_offset)
        .ok_or_else(|| {
            candle_mi::MIError::Config(format!(
                "patch offset {} exceeds the donor newline index {donor_newline}",
                args.patch_offset
            ))
        })?;
    if args.patch_offset > 0 {
        eprintln!(
            "Mid-line control: donor row from position {donor_pos} into recipient position {patch_pos} ({} tokens before each newline)",
            args.patch_offset
        );
    }

    // Donor rows are taken at the SAME offset the recipient is patched at, so a
    // mid-line control writes a row the donor actually had there. Under a minimal
    // pair those rows are identical to the recipient's, which makes the control a
    // genuine no-op test rather than a perturbation of unrelated state.
    let donor_rows = capture_donor_rows(&model, &donor_tokens, donor_pos, n_layers, &device)?;
    let self_rows = capture_donor_rows(
        &model,
        &recipient_tokens,
        recipient_newline,
        n_layers,
        &device,
    )?;

    let divergence = newline_divergence(&donor_rows, &self_rows)?;
    if let Some(min) = divergence
        .iter()
        .min_by(|a, b| a.cosine.total_cmp(&b.cosine))
    {
        eprintln!(
            "Newline rows differ least at cosine {:.4} (layer {}); most at relative L2 {:.4}",
            min.cosine,
            min.layer,
            divergence
                .iter()
                .map(|d| d.relative_l2)
                .fold(0.0_f32, f32::max)
        );
    }

    let donor_id = tokenizer.find_token_id(&format!(" {donor_word}"))?;
    let recipient_id = tokenizer.find_token_id(&format!(" {recipient_word}"))?;

    // --- Baseline: the unpatched recipient defines the teacher-forced slot ---
    let baseline_gen = generate(
        &model,
        &recipient_tokens,
        recipient_newline,
        &donor_rows,
        &[],
        args.generate,
        tokenizer,
        None,
        &device,
    )?;
    let baseline_line = tokenizer.decode(&baseline_gen).unwrap_or_default();
    let final_word_slot = final_word_start_index(recipient_tokens.len(), &baseline_gen, tokenizer);
    // Teacher-forced context: the prompt plus the baseline line up to, but not
    // including, its final word. Every condition reads the same slot.
    let mut context = recipient_tokens.clone();
    let n_lead = final_word_slot - recipient_tokens.len();
    let lead = baseline_gen.get(..n_lead).ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "final-word slot {final_word_slot} lies outside the composed line ({} tokens)",
            baseline_gen.len()
        ))
    })?;
    context.extend_from_slice(lead);
    eprintln!("\nBaseline greedy line: {baseline_line:?} (final-word slot {final_word_slot})");

    // --- Identity control, before any result is trusted ---
    let identity = identity_control(
        &model,
        &recipient_tokens,
        recipient_newline,
        &self_rows,
        &context,
        (donor_id, recipient_id),
        &baseline_line,
        args.generate,
        tokenizer,
        &device,
    )?;
    eprintln!(
        "Identity control: greedy {}, max |dP| {:.3e} -> {}",
        if identity.greedy_matches_baseline {
            "identical"
        } else {
            "DIFFERENT"
        },
        identity.max_abs_prob_delta,
        if identity.passed { "pass" } else { "FAIL" }
    );
    if args.identity_only {
        eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
        return if identity.passed {
            Ok(())
        } else {
            Err(candle_mi::MIError::Intervention(
                "identity patch changed the output (patch path is unsound on this device)".into(),
            ))
        };
    }

    // --- Layer trace: one forward per layer, teacher-forced ---
    // Skipped entirely under --baseline-only: the validation phase only needs
    // the unpatched rhyme rate, and the trace is n_layers forward passes.
    let mut layer_trace: Vec<LayerProb> = Vec::with_capacity(n_layers);
    if args.baseline_only {
        eprintln!("\nBaseline only: skipping the layer trace and both patch conditions.");
    } else {
        eprintln!("\nLayer trace (single-layer patch at the newline):");
        for layer in 0..n_layers {
            let hooks = patch_hooks(&donor_rows, &[layer], patch_pos)?;
            let logits = model.forward(&input_tensor(&context, &device)?, &hooks)?;
            layer_trace.push(LayerProb {
                layer,
                p_donor: extract_token_prob(logits.output(), donor_id)?,
                p_recipient: extract_token_prob(logits.output(), recipient_id)?,
            });
        }
    }
    let best_layer = layer_trace
        .iter()
        .max_by(|a, b| a.p_donor.total_cmp(&b.p_donor))
        .map_or(0, |b| b.layer);
    if !args.baseline_only {
        eprintln!("  best P(donor) at layer {best_layer}");
    }

    // --- Sampled conditions ---
    let all_layers: Vec<usize> = (0..n_layers).collect();
    let conditions: Vec<(&str, Vec<usize>)> = if args.baseline_only {
        vec![("baseline", Vec::new())]
    } else {
        vec![
            ("baseline", Vec::new()),
            ("all-layer", all_layers),
            ("single-layer", vec![best_layer]),
        ]
    };
    let mut condition_results = Vec::with_capacity(conditions.len());
    for (name, layers) in conditions {
        let hooks = patch_hooks(&donor_rows, &layers, patch_pos)?;
        let logits = model.forward(&input_tensor(&context, &device)?, &hooks)?;
        let m3_p_donor = extract_token_prob(logits.output(), donor_id)?;
        let m3_p_recipient = extract_token_prob(logits.output(), recipient_id)?;

        let greedy = generate(
            &model,
            &recipient_tokens,
            patch_pos,
            &donor_rows,
            &layers,
            args.generate,
            tokenizer,
            None,
            &device,
        )?;
        let greedy_line = tokenizer.decode(&greedy).unwrap_or_default();

        let mut rng = StdRng::seed_from_u64(args.seed);
        let mut sampled_lines = Vec::with_capacity(args.k_samples);
        for _ in 0..args.k_samples {
            let s = generate(
                &model,
                &recipient_tokens,
                patch_pos,
                &donor_rows,
                &layers,
                args.generate,
                tokenizer,
                Some((&mut rng, args.temperature)),
                &device,
            )?;
            sampled_lines.push(tokenizer.decode(&s).unwrap_or_default());
        }

        eprintln!(
            "  [{name:<12}] greedy={greedy_line:?}  P(donor)={m3_p_donor:.4e}  P(recipient)={m3_p_recipient:.4e}"
        );
        condition_results.push(ConditionResult {
            condition: name.to_owned(),
            patch_layers: layers,
            greedy_line,
            m3_p_donor,
            m3_p_recipient,
            sampled_lines,
        });
    }

    let token_strs = recipient_tokens
        .iter()
        .map(|&id| tokenizer.decode_token(id).unwrap_or_default())
        .collect();

    let output = PatchOutput {
        model: args.model,
        pair: args.pair,
        donor_prompt,
        recipient_prompt,
        donor_word,
        recipient_word,
        tokens: token_strs,
        donor_newline_index: donor_newline,
        recipient_newline_index: recipient_newline,
        n_layers,
        patch_offset: args.patch_offset,
        patch_position: patch_pos,
        route: "recompute-per-step",
        temperature: args.temperature,
        seed: args.seed,
        baseline_greedy_line: baseline_line,
        final_word_slot,
        conditions: condition_results,
        layer_trace,
        newline_divergence: divergence,
        identity_check: identity,
    };
    write_output(&output, args.output.as_deref())?;

    eprintln!("\nTotal elapsed: {:.2?}", t_start.elapsed());
    Ok(())
}

// ── Patching ──────────────────────────────────────────────────────────────────

/// Capture one prompt's residual row at `position`, one per layer.
///
/// # Errors
/// Returns [`MIError::Hook`](candle_mi::MIError::Hook) if a layer's `ResidPost`
/// was not captured.
///
/// # Shapes
/// - returns: `n_layers` tensors of `[hidden]`
fn capture_donor_rows(
    model: &MIModel,
    tokens: &[u32],
    position: usize,
    n_layers: usize,
    device: &Device,
) -> candle_mi::Result<Vec<Tensor>> {
    let mut hooks = HookSpec::new();
    hooks.capture_all((0..n_layers).map(HookPoint::ResidPost));
    let cache = model.forward(&input_tensor(tokens, device)?, &hooks)?;

    let mut acts = FullActivationCache::with_capacity(n_layers);
    for layer in 0..n_layers {
        let resid = cache.require(&HookPoint::ResidPost(layer))?; // [1, seq, hidden]
        acts.push(resid.get(0)?); // [seq, hidden]
    }
    (0..n_layers)
        .map(|layer| acts.get_position(layer, position))
        .collect()
}

/// Build a [`HookSpec`] patching `rows` into `position` at each listed layer.
/// An empty `layers` yields an empty spec, which is the baseline.
///
/// # Errors
/// Returns [`MIError::Intervention`](candle_mi::MIError::Intervention) if a
/// hook point does not accept a positional patch, which cannot happen for
/// `ResidPost` but is checked rather than assumed.
fn patch_hooks(rows: &[Tensor], layers: &[usize], position: usize) -> candle_mi::Result<HookSpec> {
    let mut hooks = HookSpec::new();
    for &layer in layers {
        let point = HookPoint::ResidPost(layer);
        if !point.accepts_positional_patch() {
            return Err(candle_mi::MIError::Intervention(format!(
                "hook point {point:?} does not accept a positional patch"
            )));
        }
        let value = rows
            .get(layer)
            .ok_or_else(|| {
                candle_mi::MIError::Intervention(format!(
                    "no donor row for layer {layer} ({} captured)",
                    rows.len()
                ))
            })?
            .clone();
        hooks.intervene(point, Intervention::PatchAt { position, value });
    }
    Ok(hooks)
}

/// Patch the recipient with its **own** newline row and confirm nothing moves.
/// This is the device-level soundness check for the patch path.
///
/// # Errors
/// Returns any error raised by the underlying forward passes.
fn identity_control(
    model: &MIModel,
    prompt_tokens: &[u32],
    newline: usize,
    self_rows: &[Tensor],
    context: &[u32],
    targets: (u32, u32),
    baseline_line: &str,
    max_new: usize,
    tokenizer: &MITokenizer,
    device: &Device,
) -> candle_mi::Result<IdentityCheck> {
    let all: Vec<usize> = (0..self_rows.len()).collect();
    let hooks = patch_hooks(self_rows, &all, newline)?;

    let unpatched = model.forward(&input_tensor(context, device)?, &HookSpec::new())?;
    let patched = model.forward(&input_tensor(context, device)?, &hooks)?;
    let (donor_id, recipient_id) = targets;
    let d0 = extract_token_prob(unpatched.output(), donor_id)?;
    let d1 = extract_token_prob(patched.output(), donor_id)?;
    let r0 = extract_token_prob(unpatched.output(), recipient_id)?;
    let r1 = extract_token_prob(patched.output(), recipient_id)?;
    let max_abs_prob_delta = (d1 - d0).abs().max((r1 - r0).abs());

    let greedy = generate(
        model,
        prompt_tokens,
        newline,
        self_rows,
        &all,
        max_new,
        tokenizer,
        None,
        device,
    )?;
    let greedy_matches_baseline = tokenizer.decode(&greedy).unwrap_or_default() == baseline_line;

    // A patch that writes a row back over itself is arithmetically a no-op, so
    // the tolerance covers only non-determinism in the kernels, not a real shift.
    let passed = greedy_matches_baseline && max_abs_prob_delta < 1e-6;
    Ok(IdentityCheck {
        greedy_matches_baseline,
        max_abs_prob_delta,
        passed,
    })
}

// ── Generation ────────────────────────────────────────────────────────────────

/// Compose a line, re-applying the newline patch at every step because
/// candle-mi keeps no KV cache. Stops at the first generated newline or after
/// `max_new` tokens; the newline is not included.
///
/// # Errors
/// Returns any error raised by the forward pass or the sampler.
fn generate(
    model: &MIModel,
    prompt_tokens: &[u32],
    patch_pos: usize,
    rows: &[Tensor],
    layers: &[usize],
    max_new: usize,
    tokenizer: &MITokenizer,
    mut sampler: Option<(&mut StdRng, f32)>,
    device: &Device,
) -> candle_mi::Result<Vec<u32>> {
    // The donor rows are captured once by the caller; the spec is cheap to
    // rebuild each step and keeps the patched position fixed as the sequence
    // grows past it.
    let mut current: Vec<u32> = prompt_tokens.to_vec();
    let start = current.len();
    for _ in 0..max_new {
        let hooks = patch_hooks(rows, layers, patch_pos)?;
        let cache = model.forward(&input_tensor(&current, device)?, &hooks)?;
        let logits = last_position_logits(cache.output())?;
        let next = match sampler.as_mut() {
            Some((rng, temp)) => sample_token(&logits, *temp, rng)?,
            None => argmax_token(&logits)?,
        };
        if tokenizer.decode_token(next).is_ok_and(|s| s.contains('\n')) {
            break;
        }
        current.push(next);
    }
    Ok(current.split_off(start))
}

fn input_tensor(tokens: &[u32], device: &Device) -> candle_mi::Result<Tensor> {
    Ok(Tensor::new(tokens, device)?.unsqueeze(0)?)
}

/// Last-position logits `[vocab]` from a `[1, seq, vocab]` tensor.
fn last_position_logits(logits_3d: &Tensor) -> candle_mi::Result<Tensor> {
    let seq = logits_3d.dim(1)?;
    Ok(logits_3d.get(0)?.get(seq - 1)?)
}

fn argmax_token(logits_1d: &Tensor) -> candle_mi::Result<u32> {
    // PROMOTE: force F32 so the argmax is dtype-stable across backends
    let values: Vec<f32> = logits_1d.to_dtype(DType::F32)?.to_vec1()?;
    let best = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i);
    vocab_index(best)
}

fn sample_token(logits_1d: &Tensor, temperature: f32, rng: &mut StdRng) -> candle_mi::Result<u32> {
    // PROMOTE: softmax over BF16 logits loses resolution in the tail we sample from
    let values: Vec<f32> = logits_1d.to_dtype(DType::F32)?.to_vec1()?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scaled: Vec<f32> = values
        .iter()
        .map(|v| ((v - max) / temperature).exp())
        .collect();
    let total: f32 = scaled.iter().sum();
    let mut draw = rng.r#gen::<f32>() * total;
    for (i, w) in scaled.iter().enumerate() {
        draw -= *w;
        if draw <= 0.0 {
            return vocab_index(i);
        }
    }
    // EXPLICIT: floating-point drift can exhaust the draw before the last index
    vocab_index(scaled.len() - 1)
}

/// Per-layer cosine and relative L2 between two sets of newline rows.
///
/// # Errors
/// Returns any error raised by the underlying tensor arithmetic.
///
/// # Shapes
/// - `donor`, `recipient`: `n_layers` tensors of `[hidden]`
fn newline_divergence(
    donor: &[Tensor],
    recipient: &[Tensor],
) -> candle_mi::Result<Vec<LayerDivergence>> {
    donor
        .iter()
        .zip(recipient.iter())
        .enumerate()
        .map(|(layer, (d, r))| {
            // PROMOTE: BF16 residuals lose too much resolution for a cosine
            let d = d.to_dtype(DType::F32)?;
            let r = r.to_dtype(DType::F32)?;
            let dot = (&d * &r)?.sum_all()?.to_scalar::<f32>()?;
            let nd = (&d * &d)?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nr = (&r * &r)?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let diff = (&d - &r)?;
            let nl2 = (&diff * &diff)?.sum_all()?.to_scalar::<f32>()?.sqrt();
            Ok(LayerDivergence {
                layer,
                cosine: if nd > 0.0 && nr > 0.0 {
                    dot / (nd * nr)
                } else {
                    0.0
                },
                relative_l2: if nr > 0.0 { nl2 / nr } else { 0.0 },
            })
        })
        .collect()
}

/// Narrow a vocabulary index to the `u32` the tokenizer API expects.
///
/// # Errors
/// Returns [`MIError::Config`](candle_mi::MIError::Config) if the index does
/// not fit in `u32`, which no supported vocabulary approaches.
fn vocab_index(index: usize) -> candle_mi::Result<u32> {
    u32::try_from(index).map_err(|_| {
        candle_mi::MIError::Config(format!("vocabulary index {index} does not fit in u32"))
    })
}

/// Whether a decoded token begins a new word.
fn is_word_start(token: &str) -> bool {
    token.starts_with(' ') || token.starts_with('\u{2581}') || token.starts_with('\u{0120}')
}

/// Sequence index of the first token of the composed line's final word.
fn final_word_start_index(prompt_len: usize, gen_tokens: &[u32], tokenizer: &MITokenizer) -> usize {
    let mut last_rel = 0_usize;
    for (i, &id) in gen_tokens.iter().enumerate() {
        let s = tokenizer.decode_token(id).unwrap_or_default();
        if i == 0 || is_word_start(&s) {
            last_rel = i;
        }
    }
    prompt_len + last_rel
}

/// Confirm the chosen steering site really is a newline token.
///
/// # Errors
/// Returns [`MIError::Config`](candle_mi::MIError::Config) if the token at
/// `index` contains no newline, which means the prompt was not truncated after
/// line 3.
fn assert_newline(
    tokenizer: &MITokenizer,
    tokens: &[u32],
    index: usize,
    role: &str,
) -> candle_mi::Result<()> {
    let id = *tokens.get(index).ok_or_else(|| {
        candle_mi::MIError::Config(format!(
            "{role} newline index {index} out of range ({} tokens)",
            tokens.len()
        ))
    })?;
    let decoded = tokenizer.decode_token(id).unwrap_or_default();
    if decoded.contains('\n') {
        Ok(())
    } else {
        Err(candle_mi::MIError::Config(format!(
            "{role} prompt does not end at a newline (token {index} is {decoded:?})"
        )))
    }
}

fn write_output(output: &PatchOutput, path: Option<&Path>) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("failed to serialize output: {e}")))?;
    match path {
        Some(p) => {
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).map_err(|e| {
                    candle_mi::MIError::Config(format!("failed to create output directory: {e}"))
                })?;
            }
            std::fs::write(p, json)
                .map_err(|e| candle_mi::MIError::Config(format!("failed to write output: {e}")))?;
            eprintln!("\nWrote {}", p.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
