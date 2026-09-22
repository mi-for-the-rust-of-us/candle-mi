// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test for recurrent feedback (anacrousis) on Llama 3.2 1B.
//!
//! Replicates the plip-rs recurrent block rhyme experiment:
//! 28 conditions × 15 couplets = 420 measurements.
//!
//! Key result (candle-mi, no KV cache): unembed_8-15_s=2.0 converts 11/15
//! (baseline: 9/15). Differs from plip-rs (with KV cache): sustained_14-15_s=1.0
//! converts 11/15 (baseline: 10/15).
//!
//! Requires `meta-llama/Llama-3.2-1B` in HF cache.
//! Run: `cargo test --test validate_anacrousis --features transformer -- --ignored --test-threads=1`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    unsafe_code,
    missing_docs
)]

mod common;

use common::{cuda_device, find_snapshot, safetensors_paths};

use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{
    GenericTransformer, HookSpec, MIBackend, MITokenizer, RecurrentPassSpec, TransformerConfig,
    sample_token,
};

// ---------------------------------------------------------------------------
// Helpers (shared pattern with validate_clt.rs / validate_models.rs)
// ---------------------------------------------------------------------------

fn load_llama(device: &Device) -> (GenericTransformer, MITokenizer, TransformerConfig) {
    let snapshot = find_snapshot("meta-llama/Llama-3.2-1B")
        .expect("meta-llama/Llama-3.2-1B not found in HF cache");
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();
    let dtype = DType::F32;
    let paths = safetensors_paths(&snapshot);
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };
    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();
    (model, tokenizer, config)
}

// ---------------------------------------------------------------------------
// Couplet definitions (all 15 from plip-rs recurrent_block_rhyme.rs)
// ---------------------------------------------------------------------------

struct CoupletDef {
    id: u32,
    _target_word: &'static str,
    line1: &'static str,
    rhyme_family: &'static [&'static str],
}

fn couplet_defs() -> Vec<CoupletDef> {
    vec![
        CoupletDef {
            id: 1,
            _target_word: "light",
            line1: "The moon casts silver light,",
            rhyme_family: &[
                "light", "night", "bright", "sight", "might", "flight", "right", "tight", "white",
                "bite", "kite", "quite", "knight", "delight", "blight", "plight", "slight",
                "fright", "height",
            ],
        },
        CoupletDef {
            id: 2,
            _target_word: "play",
            line1: "The children laugh and play,",
            rhyme_family: &[
                "play", "day", "way", "say", "stay", "sway", "ray", "bay", "may", "lay", "pay",
                "gray", "away", "display", "pray", "stray", "clay", "hay", "decay", "delay",
            ],
        },
        CoupletDef {
            id: 3,
            _target_word: "sound",
            line1: "The thunder makes a sound,",
            rhyme_family: &[
                "sound", "ground", "found", "round", "bound", "around", "mound", "pound", "hound",
                "wound", "profound", "abound", "astound",
            ],
        },
        CoupletDef {
            id: 4,
            _target_word: "rain",
            line1: "The clouds bring heavy rain,",
            rhyme_family: &[
                "rain", "pain", "gain", "main", "vain", "plain", "chain", "train", "brain",
                "strain", "remain", "again", "drain", "lane", "crane", "bane", "wane", "reign",
                "feign",
            ],
        },
        CoupletDef {
            id: 5,
            _target_word: "time",
            line1: "The old clock measures time,",
            rhyme_family: &[
                "time", "rhyme", "climb", "crime", "dime", "lime", "mime", "prime", "chime",
                "sublime", "paradigm", "thyme",
            ],
        },
        CoupletDef {
            id: 6,
            _target_word: "air",
            line1: "The geese fly through the air,",
            rhyme_family: &[
                "air", "there", "fair", "care", "bare", "dare", "rare", "share", "stare", "where",
                "pair", "aware", "compare", "despair", "prayer", "hair", "chair", "bear", "wear",
                "spare", "snare", "glare",
            ],
        },
        CoupletDef {
            id: 7,
            _target_word: "gold",
            line1: "The sunset gleams like gold,",
            rhyme_family: &[
                "gold",
                "old",
                "bold",
                "cold",
                "fold",
                "hold",
                "told",
                "sold",
                "mold",
                "behold",
                "unfold",
                "rolled",
                "controlled",
            ],
        },
        CoupletDef {
            id: 8,
            _target_word: "fire",
            line1: "The embers feed the fire,",
            rhyme_family: &[
                "fire", "hire", "wire", "desire", "tire", "inspire", "acquire", "higher", "entire",
                "admire", "liar", "dire", "sire", "pyre", "mire", "conspire", "expire",
            ],
        },
        CoupletDef {
            id: 9,
            _target_word: "stone",
            line1: "The castle walls of stone,",
            rhyme_family: &[
                "stone", "bone", "tone", "lone", "zone", "throne", "phone", "own", "known",
                "blown", "grown", "shown", "moan", "groan", "clone", "drone",
            ],
        },
        CoupletDef {
            id: 10,
            _target_word: "dream",
            line1: "I wandered through a dream,",
            rhyme_family: &[
                "dream", "stream", "seem", "team", "beam", "cream", "gleam", "scheme", "theme",
                "extreme", "esteem", "scream", "steam",
            ],
        },
        CoupletDef {
            id: 11,
            _target_word: "strange",
            line1: "The silence felt so strange,",
            rhyme_family: &[
                "strange", "change", "range", "arrange", "exchange", "grange",
            ],
        },
        CoupletDef {
            id: 12,
            _target_word: "love",
            line1: "I never knew such love,",
            rhyme_family: &["love", "above", "dove", "of", "shove", "glove", "thereof"],
        },
        CoupletDef {
            id: 13,
            _target_word: "truth",
            line1: "She spoke the honest truth,",
            rhyme_family: &[
                "truth", "youth", "tooth", "booth", "smooth", "sleuth", "ruth", "uncouth",
            ],
        },
        CoupletDef {
            id: 14,
            _target_word: "world",
            line1: "He traveled all the world,",
            rhyme_family: &[
                "world", "curled", "unfurled", "whirled", "hurled", "swirled", "pearled", "furled",
                "twirled",
            ],
        },
        CoupletDef {
            id: 15,
            _target_word: "earth",
            line1: "The seeds lay in the earth,",
            rhyme_family: &[
                "earth", "birth", "worth", "mirth", "berth", "girth", "dearth", "rebirth", "hearth",
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Experiment conditions (28 total, matching plip-rs)
// ---------------------------------------------------------------------------

struct Condition {
    name: String,
    loop_range: Option<(usize, usize)>,
    feedback_strength: Option<f32>,
    sustained: bool,
}

fn experiment_conditions() -> Vec<Condition> {
    let mut conds = vec![Condition {
        name: "baseline".to_string(),
        loop_range: None,
        feedback_strength: None,
        sustained: false,
    }];

    let loop_ranges: &[(usize, usize, &str)] = &[
        (12, 15, "12-15"),
        (10, 15, "10-15"),
        (8, 15, "8-15"),
        (14, 15, "14-15"),
    ];

    for &(start, end, label) in loop_ranges {
        // Double pass (no feedback)
        conds.push(Condition {
            name: format!("double_pass_{label}"),
            loop_range: Some((start, end)),
            feedback_strength: None,
            sustained: false,
        });

        // Unembed feedback at various strengths (prefill only)
        for strength in &[1.0_f32, 2.0, 5.0, 10.0, 20.0] {
            conds.push(Condition {
                name: format!("unembed_{label}_s={strength:.1}"),
                loop_range: Some((start, end)),
                feedback_strength: Some(*strength),
                sustained: false,
            });
        }
    }

    // Sustained feedback during generation (L14-15 only)
    for strength in &[0.5_f32, 1.0, 2.0] {
        conds.push(Condition {
            name: format!("sustained_14-15_s={strength:.1}"),
            loop_range: Some((14, 15)),
            feedback_strength: Some(*strength),
            sustained: true,
        });
    }

    conds
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the last word-like token from generated text.
fn extract_last_word(text: &str) -> String {
    text.split_whitespace()
        .next_back()
        .unwrap_or("")
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Check if a word matches any member of the rhyme family.
fn word_rhymes(word: &str, rhyme_family: &[String]) -> bool {
    let clean = word
        .trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase();
    rhyme_family.contains(&clean)
}

/// Compute L2-normalised average of a set of embedding vectors.
fn averaged_rhyme_direction(
    model: &GenericTransformer,
    tokenizer: &MITokenizer,
    rhyme_words: &[&str],
) -> Tensor {
    let mut embeddings: Vec<Tensor> = Vec::new();

    for word in rhyme_words {
        let with_space = format!(" {word}");
        let ids = tokenizer.encode_raw(&with_space).unwrap();
        let token_id = *ids.last().expect("word produced no tokens");
        let emb = model.embedding_vector(token_id).unwrap();
        embeddings.push(emb);
    }

    let stacked = Tensor::stack(&embeddings, 0).unwrap();
    let avg = stacked.mean(0).unwrap();
    let avg_f32 = avg.to_dtype(DType::F32).unwrap();
    let norm: f32 = avg_f32
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .sqrt()
        .unwrap()
        .to_scalar()
        .unwrap();
    if norm > 1e-8 {
        avg_f32.affine(1.0 / f64::from(norm), 0.0).unwrap()
    } else {
        avg_f32
    }
}

/// Run baseline generation (standard forward, no recurrence).
fn generate_baseline(
    model: &GenericTransformer,
    prompt_tokens: &[u32],
    max_tokens: usize,
    stop_tokens: &[u32],
) -> Vec<u32> {
    let embed_device = model.embedding_vector(0).unwrap().device().clone();
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..max_tokens {
        let input = Tensor::new(tokens.as_slice(), &embed_device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let cache = model.forward(&input, &HookSpec::new()).unwrap();
        let logits = cache.output();
        let seq_len = logits.dim(1).unwrap();
        let last_logits = logits
            .i((.., seq_len - 1, ..))
            .unwrap()
            .squeeze(1)
            .unwrap()
            .flatten_all()
            .unwrap();
        let next_token = sample_token(&last_logits, 0.0).unwrap();
        if stop_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
    }
    tokens
}

// ===========================================================================
// Main test: 28 conditions × 15 couplets
// ===========================================================================

#[test]
#[ignore = "full 28x15 condition-couplet matrix; long-running, run locally"]
fn anacrousis_28x15_full_matrix() {
    let t0 = Instant::now();
    let device = cuda_device().expect("anacrousis test requires a CUDA GPU");
    let (model, tokenizer, _config) = load_llama(&device);
    eprintln!("  model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let couplets = couplet_defs();
    let conditions = experiment_conditions();
    assert_eq!(conditions.len(), 28, "expected 28 conditions");

    let max_tokens: usize = 30;
    let stop_tokens: Vec<u32> = vec![
        128_001, // <|end_of_text|>
        128_009, // <|eot_id|>
    ];

    // Track rhyme success per condition
    let mut rhyme_counts: Vec<(String, usize)> = Vec::new();
    // Track per-couplet results for best condition and baseline
    let mut baseline_rhymes: Vec<bool> = Vec::new();
    let mut best_rhymes: Vec<bool> = Vec::new();

    for cond in &conditions {
        let cond_start = Instant::now();
        let mut rhyme_count = 0_usize;

        for couplet in &couplets {
            let prompt = format!("{}\n", couplet.line1);
            let rhyme_set: Vec<String> = couplet
                .rhyme_family
                .iter()
                .map(|w| (*w).to_lowercase())
                .collect();

            // encode() adds BOS — required for Llama 3.2 correct predictions
            let prompt_ids = tokenizer.encode(&prompt).unwrap();
            let planning_pos = prompt_ids.len() - 1;

            // Generate
            let all_tokens = match cond.loop_range {
                None => generate_baseline(&model, &prompt_ids, max_tokens, &stop_tokens),
                Some((start, end)) => {
                    let mut spec =
                        RecurrentPassSpec::no_feedback(start, end).with_sustained(cond.sustained);
                    if let Some(strength) = cond.feedback_strength {
                        let rhyme_dir =
                            averaged_rhyme_direction(&model, &tokenizer, couplet.rhyme_family);
                        spec.add_feedback(planning_pos, rhyme_dir, strength);
                    }
                    model
                        .generate_recurrent(&prompt_ids, max_tokens, 0.0, &stop_tokens, &spec)
                        .unwrap()
                }
            };

            // Decode only the generated tokens (skip prompt to avoid BOS prefix mismatch)
            let generated_tokens = &all_tokens[prompt_ids.len()..];
            let generated_text = tokenizer.decode(generated_tokens).unwrap();
            let line2 = generated_text.lines().next().unwrap_or("").to_string();

            let last_word = extract_last_word(&line2);
            let rhymes = word_rhymes(&last_word, &rhyme_set);

            if rhymes {
                rhyme_count += 1;
            }

            // Track baseline and best condition per couplet
            // candle-mi (no KV cache): best improvement at unembed_8-15_s=2.0
            if cond.name == "baseline" {
                baseline_rhymes.push(rhymes);
            } else if cond.name == "unembed_8-15_s=2.0" {
                best_rhymes.push(rhymes);
            }

            let marker = if rhymes { "RHYME" } else { "-" };
            eprintln!(
                "  {:<35} [{:<5}] couplet={:>2} last='{}'",
                cond.name, marker, couplet.id, last_word
            );
        }

        eprintln!(
            "  >>> {}: {}/{}  ({:.1}s)\n",
            cond.name,
            rhyme_count,
            couplets.len(),
            cond_start.elapsed().as_secs_f64()
        );
        rhyme_counts.push((cond.name.clone(), rhyme_count));
    }

    // --- Summary ---
    let total_elapsed = t0.elapsed().as_secs_f64();
    eprintln!("\n=== SUMMARY ({total_elapsed:.1}s total) ===");
    for (name, count) in &rhyme_counts {
        eprintln!("  {name:<35} {count}/15");
    }

    // --- Key assertions ---
    //
    // candle-mi results differ from plip-rs because there is no KV cache:
    // every generation step recomputes the full sequence, so the recurrent
    // double-pass applies at every step. This changes the optimal conditions:
    //
    // - Baseline: 9/15 (plip-rs: 10/15)
    // - Best improvement: unembed_8-15_s=2.0 → 11/15 (plip-rs best: sustained_14-15_s=1.0 → 11/15)
    // - Double-pass without feedback: 0/15 (degenerate — out-of-distribution)
    // - High strength (s≥10): degrades below baseline

    // 1. Baseline: 9/15
    let baseline_count = rhyme_counts
        .iter()
        .find(|(n, _)| n == "baseline")
        .map(|(_, c)| *c)
        .unwrap();
    assert_eq!(
        baseline_count, 9,
        "baseline should rhyme 9/15, got {baseline_count}/15"
    );

    // 2. Best condition (unembed_8-15_s=2.0): 11/15, improves over baseline
    let best_count = rhyme_counts
        .iter()
        .find(|(n, _)| n == "unembed_8-15_s=2.0")
        .map(|(_, c)| *c)
        .unwrap();
    assert_eq!(
        best_count, 11,
        "unembed_8-15_s=2.0 should rhyme 11/15, got {best_count}/15"
    );
    assert!(
        best_count > baseline_count,
        "recurrent feedback should improve over baseline"
    );

    // 3. Best condition is a superset of baseline successes
    assert_eq!(baseline_rhymes.len(), 15);
    assert_eq!(best_rhymes.len(), 15);
    for (i, (&baseline, &best)) in baseline_rhymes.iter().zip(best_rhymes.iter()).enumerate() {
        if baseline {
            assert!(
                best,
                "couplet {} rhymes at baseline but NOT at unembed_8-15_s=2.0 (regression)",
                i + 1
            );
        }
    }

    // 4. Double-pass without feedback degrades (out-of-distribution activations)
    for (name, count) in &rhyme_counts {
        if name.starts_with("double_pass") {
            assert_eq!(
                *count, 0,
                "double_pass without feedback should degrade to 0/15, got {count}/15 for {name}"
            );
        }
    }
}
