# Example Style Guide

Every example in candle-mi must follow these conventions. This guide captures lessons learned from building 23 examples and ensures consistency across the codebase.

## Table of Contents

- [File Structure](#file-structure)
- [Token Positions](#token-positions)
- [CLI Pattern](#cli-pattern)
- [Runtime Reporting](#runtime-reporting)
- [Memory Reporting](#memory-reporting)
- [JSON Output](#json-output)
- [Model Loading](#model-loading)
- [Cargo.toml Entry](#cargotoml-entry)
- [Run Commands](#run-commands)
- [Annotations (CONVENTIONS.md)](#annotations-conventionsmd)
- [Checklist for New Examples](#checklist-for-new-examples)

## File Structure

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-line description of the example.
//!
//! ```bash
//! cargo run --release --features transformer --example <name>
//! ```
//!
//! **What it does:**
//!
//! 1. Step one...
//! 2. Step two...
//!
//! Paper reference (if replicating):
//! > Author et al. "Title." Venue, Year. <https://...>

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]
```

The `#![allow]` block is standard for all examples. **Since v0.2.0 clippy also lints
example and test targets** (`--all-targets` on every CI lane), so an example that
indexes a slice, unwraps, expects or panics needs the matching allow as well:
`clippy::indexing_slicing`, `clippy::unwrap_used`, `clippy::expect_used`,
`clippy::panic`. Add only the ones the file actually needs; 19 of 42 examples
currently carry `indexing_slicing`. Library code uses deny; examples relax these for readability.

## Token Positions

**When you need token positions or token strings, always use `forward_text` + `label_spans`.** Non-position-aware examples (e.g., `generate`, `logit_lens`) may use bare `encode()` since they only need token IDs. But if you need to know *which token corresponds to which part of the input text*, use the offset-aware API — never manually track character offsets.

```rust
// CORRECT — positions come from the API
let result = model.forward_text(prompt, &hooks)?;
let labels = result.encoding().label_spans(&[
    ("subject", 0..subject.len()),
    ("relation", rel_start..rel_end),
]);
let tokens = result.tokens();  // token strings, no decode loop

// WRONG — ad-hoc offset tracking, fragile BOS handling
let ids = tokenizer.encode(prompt)?;
let mut offset = 0;
for &tid in &ids {
    let decoded = tokenizer.decode(&[tid])?;  // don't do this
    offset += decoded.len();
    // ... manual classification ...
}
```

`forward_text` returns `TextForwardResult` which bundles:
- `output()` / `require()` / `get()` — hook cache access
- `encoding()` — `EncodingWithOffsets` with `tokens`, `ids`, `offsets`, `label_spans()`, `char_range_to_tokens()`, etc.
- `tokens()` — shortcut for raw BPE token strings (with space-prefix markers like `Ġ`)
- `seq_len()` — token count

For patching passes that reuse the same input tensor, extract it from the encoding:
```rust
let orig_input = candle_core::Tensor::new(&result.encoding().ids[..], model.device())?
    .unsqueeze(0)?;
```

## CLI Pattern

### Transformer/RWKV examples — `clap::Parser`

Use `clap::Parser` for examples with multiple flexible flags (analyses, sweeps, data-driven experiments). Standard arguments:

```rust
#[derive(Parser)]
#[command(name = "example_name")]
#[command(about = "Short description")]
struct Args {
    /// `HuggingFace` model ID
    #[arg(default_value = "meta-llama/Llama-3.2-1B")]
    model: String,

    /// Write structured JSON output to this file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Suppress per-item and total runtime reporting
    #[arg(long)]
    no_runtime: bool,

    /// Run only the first N items (for quick testing)
    #[arg(long)]
    limit: Option<usize>,
}
```

- Positional `model` argument with a sensible default. Alternatively, `model: Option<String>` when the example supports auto-discovery of all cached models (omit to run on all).
- `--output` for JSON serialization.
- `--no-runtime` to suppress timing (timing is ON by default). Standard for new examples; existing examples will be updated incrementally.
- `--limit` for quick iteration during development.
- `--data` when the example loads external data files (e.g., CounterFact prompt pairs).

### Auto-discovery examples — `env::args()`

Examples that run on all cached models (e.g., `generate`, `attention_patterns`, `rwkv_inference`) may use `env::args()` instead of `clap::Parser`. The pattern: no args → discover and run all cached models; one positional arg → run on that model only.

### Stoicheia examples — domain-specific flags

Stoicheia examples load from local `safetensors` files, not `HuggingFace` Hub. They use `clap::Parser` with domain-specific flags (`--task`, `--hidden-size`, `--seq-len`, `--weights`) instead of a `model` positional argument. The `--output` and `--no-runtime` conventions still apply where appropriate.

## Runtime Reporting

**Timing is on by default.** Use `--no-runtime` to suppress it.

Every example must measure:
- **Model load time** — `Instant::now()` around `from_pretrained`.
- **Per-item time** — for each prompt/pair/experiment, with breakdown of phases (e.g., capture vs sweep).
- **Total time** — wall clock for the full run.

```rust
let t0 = Instant::now();
let model = MIModel::from_pretrained(&args.model)?;
let load_time = t0.elapsed();
if !args.no_runtime {
    println!("  Load time: {load_time:.2?}");
}

// Per-item timing
let pair_start = Instant::now();
let capture_start = Instant::now();
// ... capture passes ...
let capture_time = capture_start.elapsed();
let sweep_start = Instant::now();
// ... sweep passes ...
let sweep_time = sweep_start.elapsed();
let pair_time = pair_start.elapsed();
if !args.no_runtime {
    println!("  Pair time: {pair_time:.2?} (capture: {capture_time:.2?}, sweep: {sweep_time:.2?})");
}

// Summary
if args.no_runtime {
    println!("  === Summary ({n} items, {patches} patches) ===");
} else {
    println!("  === Summary ({n} items, {patches} patches, {total_time:.1?}) ===");
}
```

**JSON always includes timing** regardless of `--no-runtime` — the flag only controls stdout display. Include `time_secs: f64` per item and `total_time_secs: f64` at the top level.

## Memory Reporting

Gate behind `#[cfg(feature = "memory")]`. Take snapshots before and after model loading:

```rust
#[cfg(feature = "memory")]
use candle_mi::{MemoryReport, MemorySnapshot};

// Before model load
#[cfg(feature = "memory")]
let mem_before = MemorySnapshot::now(
    &candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
)?;

let model = MIModel::from_pretrained(&args.model)?;

// After model load
#[cfg(feature = "memory")]
{
    let mem_after = MemorySnapshot::now(model.device())?;
    MemoryReport::new(mem_before, mem_after).print_before_after("Model load");
}
```

This shows GPU name, VRAM usage before/after, and RAM delta. Run with `--features memory`:
```bash
cargo run --release --features transformer,memory --example <name>
```

## JSON Output

Use `serde::Serialize` structs. Include timing, model metadata, and results:

```rust
#[derive(Serialize)]
struct JsonOutput {
    model_id: String,
    total_time_secs: f64,
    results: Vec<ItemResult>,
    summary: Summary,
}

#[derive(Serialize)]
struct ItemResult {
    // ... item-specific fields ...
    time_secs: f64,
}
```

Write with a helper:
```rust
fn write_json(path: &Path, output: &JsonOutput) -> candle_mi::Result<()> {
    let json = serde_json::to_string_pretty(output)
        .map_err(|e| candle_mi::MIError::Config(format!("JSON serialization failed: {e}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            candle_mi::MIError::Config(format!("failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, &json).map_err(|e| {
        candle_mi::MIError::Config(format!("failed to write {}: {e}", path.display()))
    })?;
    Ok(())
}
```

JSON files go to `examples/results/<example_name>/`.

## Model Loading

Use the `main` → `run` pattern for clean error reporting:

```rust
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    // ...
    Ok(())
}
```

Loading patterns vary by backend:

| Backend | Loading pattern |
|---------|----------------|
| Transformer | `MIModel::from_pretrained("model-id")?` — downloads from `HuggingFace` Hub |
| RWKV | `MIModel::from_pretrained("model-id")?` — same, with optional custom tokenizer |
| SAE | `SparseAutoencoder::from_pretrained_npz(repo, device)?` — specialized loader |
| Stoicheia | `StoicheiaRnn::load(config, "path/to.safetensors", &device)?` — local file, explicit config |

For transformer/RWKV forward passes, prefer `model.forward_text(prompt, &hooks)?` over manual `encode()` + `Tensor::new()` + `model.forward()` — see [Token Positions](#token-positions). Stoicheia examples operate on raw `f32` arrays (not text), so `forward_text` does not apply.

## Cargo.toml Entry

Every example needs:

```toml
[[example]]
name = "example_name"
required-features = ["transformer"]  # or ["rwkv"], ["clt", "transformer"], ["stoicheia"], etc.
```

## Run Commands

Always suggest `--features` explicitly. Always include `mmap` when the example might run on sharded models (7B+):

```bash
# Transformer — basic
cargo run --release --features transformer --example <name>

# Transformer — with memory reporting
cargo run --release --features transformer,memory --example <name>

# Transformer — with mmap for large models
cargo run --release --features transformer,mmap --example <name>

# Transformer — quick test
cargo run --release --features transformer --example <name> -- --limit 3

# Transformer — clean output (no timing)
cargo run --release --features transformer --example <name> -- --no-runtime

# Stoicheia — AlgZoo RNN analysis
cargo run --release --features stoicheia --example stoicheia_analysis -- \
    --weights path/to/weights.safetensors --hidden-size 2 --seq-len 2

# Stoicheia — AlgZoo inference
cargo run --release --features stoicheia --example stoicheia_inference -- \
    --task 2nd-argmax --hidden-size 16 --seq-len 10 --weights path/to/weights.safetensors
```

## Annotations (CONVENTIONS.md)

Examples follow the same annotation rules as library code:

- `// CAST:` on every `as` cast
- `// INDEX:` on every direct slice index with justification
- `// BORROW:` on `.chars().take()`, `.as_str()`, `.to_owned()` conversions
- `// PROMOTE:` on `.to_dtype(F32)` calls
- `// CONTIGUOUS:` before `.contiguous()` preceding matmul

## Checklist for New Examples

**Universal (all examples):**

1. SPDX header on line 1
2. Module doc with bash run command and paper reference (if replicating)
3. Standard `#![allow]` pragmas
4. `main` → `run` pattern for error handling
5. Per-item + total timing (on by default)
6. `[[example]]` entry in `Cargo.toml` with `required-features`
7. Entry in `examples/README.md` table + running commands
8. CHANGELOG bullet under `[Unreleased]`
9. All CONVENTIONS.md annotations applied as code is written

**Recommended for data-driven analyses:**

10. `clap::Parser` with `--output`, `--no-runtime`, `--limit`
11. JSON output with timing fields (`serde::Serialize` structs)
12. Memory reporting behind `#[cfg(feature = "memory")]`

**Transformer/RWKV examples only:**

13. `forward_text` + `label_spans` for position-aware work (otherwise bare `encode()` is fine)
