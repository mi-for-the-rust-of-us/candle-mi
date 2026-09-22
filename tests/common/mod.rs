// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for candle-mi's integration tests.
//!
//! Before this module existed, every oracle test carried its own copy of the
//! same four functions, and the copies had **drifted**: 21 definitions of
//! `hf_cache_dir` across 4 bodies, 20 of [`find_snapshot`] across **6**, 19 of
//! [`cuda_device`], 13 of [`safetensors_paths`].  Most of that was cosmetic
//! (`PathBuf` vs `std::path::PathBuf`), but `find_snapshot` had genuinely
//! divergent semantics, and twelve files were running the unsafe one.  See
//! [`find_snapshot`] for what that cost.
//!
//! It also holds the only three `as` casts the oracle tests need.  They used to
//! appear 149 times across 27 files, each nominally requiring its own
//! `// CAST:` annotation under `CONVENTIONS.md`.  One annotated conversion that
//! everything calls is worth more than 149 restatements of the same reason, and
//! `CONVENTIONS.md` asks for exactly this ("prefer `From`/`Into` ... use `as`
//! only when truncation or wrapping is the deliberate intent").
//!
//! This file lives in `tests/common/` rather than `tests/` so Cargo does not
//! build it as a test binary of its own.  Declare it with `mod common;` in a
//! test and call through `common::`.

// Each test binary pulls in the whole module but uses only the part it needs,
// so unused-item lints fire per binary rather than per definition.
#![allow(dead_code)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::path::{Path, PathBuf};

use candle_core::Device;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Hugging Face cache
// ---------------------------------------------------------------------------

/// Root of the local Hugging Face hub cache.
///
/// `HF_HOME` wins when set, then the per-user default.  `USERPROFILE` is
/// checked before `HOME` because Git Bash on Windows sets both, and `HOME`
/// there points at the MSYS root rather than the Windows profile that
/// `huggingface_hub` actually writes to.
///
/// # Panics
///
/// Panics when none of `HF_HOME`, `USERPROFILE` or `HOME` is set, since every
/// caller needs a cache to read and there is no useful fallback.
#[must_use]
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory (set HF_HOME, USERPROFILE or HOME)");
}

/// Locate a cached snapshot of `model_id` that actually carries weights.
///
/// Returns `None` when the repo is not cached, which is what lets every oracle
/// test skip cleanly instead of failing on a machine without the model.
///
/// **This scans every snapshot directory and validates each one.**  Twelve test
/// files used to do `read_dir(..).next()` instead, taking the first entry
/// whatever it was: with more than one snapshot cached (ordinary after a repo
/// update) that silently selects an arbitrary revision, and a partial download
/// or a stray directory selects something with no weights at all.  Two other
/// files hardcoded a single weight filename, so they saw a sharded repo as
/// absent and skipped rather than running.  Accepting any of the four weight
/// layouts is what lets one implementation serve all of them.
#[must_use]
pub fn find_snapshot(model_id: &str) -> Option<PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    for entry in std::fs::read_dir(snapshots_dir).ok()?.flatten() {
        let path = entry.path();
        if !path.join("config.json").exists() {
            continue;
        }
        let has_weights = path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists()
            || path.join("pytorch_model.bin").exists()
            || path.join("pytorch_model.bin.index.json").exists();
        if has_weights {
            return Some(path);
        }
    }
    None
}

/// Collect the safetensors files for a snapshot, single-file or sharded.
///
/// # Panics
///
/// Panics when the snapshot has neither `model.safetensors` nor a readable
/// `model.safetensors.index.json`; a caller that got here has already passed
/// [`find_snapshot`], so a missing index means the cache is corrupt.
#[must_use]
pub fn safetensors_paths(snapshot: &Path) -> Vec<PathBuf> {
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }
    let index_path = snapshot.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path).unwrap_or_else(|_| {
        panic!(
            "no model.safetensors or index.json in {}",
            snapshot.display()
        )
    });
    let index: Value = serde_json::from_str(&index_str).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();
    let mut shard_names: Vec<String> = weight_map
        .values()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    shard_names.sort();
    shard_names.dedup();
    shard_names.iter().map(|name| snapshot.join(name)).collect()
}

/// A CUDA device, or `None` so a GPU test can skip on a CPU-only machine.
#[must_use]
pub fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0).ok().filter(Device::is_cuda)
}

/// Path to a frozen oracle reference under `scripts/`.
#[must_use]
pub fn reference_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(file_name)
}

// ---------------------------------------------------------------------------
// Oracle JSON readers
// ---------------------------------------------------------------------------
//
// The three conversions every oracle test performs on its reference JSON.
// `serde_json` stores all integers as `u64` and all floats as `f64`, while the
// values are model dimensions, token ids and logits produced by *our own*
// Python oracle scripts in `scripts/` -- not untrusted input. Each cast is
// justified once here instead of at all 149 former call sites.

/// Read a `u64` from the oracle JSON as a `usize`.
///
/// # Panics
///
/// Panics when the value is absent or not an integer, which means the
/// reference JSON and the test have gone out of step and no comparison the
/// test could make would be meaningful.
#[must_use]
pub fn json_usize(v: &Value) -> usize {
    // CAST: u64 → usize, a dimension, count or index emitted by this repo's own
    // oracle scripts; model dimensions are orders of magnitude below usize::MAX
    // on every supported target, and a 32-bit target would still hold them.
    v.as_u64().unwrap() as usize
}

/// Read a `u64` from the oracle JSON as a `u32` token id.
///
/// # Panics
///
/// Panics when the value is absent or not an integer.
#[must_use]
pub fn json_u32(v: &Value) -> u32 {
    // CAST: u64 → u32, a token id or vocabulary index from our own oracle.
    // candle's tokenizers and `Tensor::new` both take u32 ids, and the largest
    // vocabulary candle-mi loads is 256000 (Gemma), well inside u32.
    v.as_u64().unwrap() as u32
}

/// Read an `f64` from the oracle JSON as an `f32`.
///
/// # Panics
///
/// Panics when the value is absent or not a number.
#[must_use]
pub fn json_f32(v: &Value) -> f32 {
    // CAST: f64 → f32, a logit, activation or scale that Python wrote as a
    // double. The comparison it feeds runs in F32 because that is candle-mi's
    // research dtype, so narrowing here matches the oracle's own input dtype
    // rather than losing precision that the assertion would have used.
    v.as_f64().unwrap() as f32
}

/// Read a JSON array of integers as token ids.
///
/// # Panics
///
/// Panics when the value is not an array of integers.
#[must_use]
pub fn json_u32_vec(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(json_u32).collect()
}

/// Read a JSON array of numbers as `f32`.
///
/// # Panics
///
/// Panics when the value is not an array of numbers.
#[must_use]
pub fn json_f32_vec(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(json_f32).collect()
}
