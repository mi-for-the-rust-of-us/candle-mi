// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests for the `hypomnesis`-backed memory measurement.
//!
//! The unit tests in `src/memory.rs` cover the delta/formatting logic on
//! synthetic [`MemorySnapshot`]s, but they cannot exercise the real
//! measurement path: `MemorySnapshot::now` flattens a
//! [`hypomnesis::Snapshot`], which is `#[non_exhaustive]` and has no public
//! constructor, so the flatten can only be validated against a **live**
//! measurement. These tests provide that external validation — the safety
//! check for the hypomnesis migration:
//!
//! - [`cpu_snapshot_is_ram_only`] (always runs): the CPU path reports RAM and
//!   leaves every VRAM field `None`.
//! - [`cuda_snapshot_is_sane`] (`#[ignore]`, needs CUDA): a live CUDA snapshot
//!   is internally consistent (`used <= total`, positive totals).
//! - [`cuda_allocation_is_visible_in_vram`] (`#[ignore]`, needs CUDA): a real
//!   ~512 MiB GPU allocation shows up in the VRAM delta — proving the
//!   measurement tracks actual device usage, not a stale/constant value.
//!
//! Run the live CUDA checks on a machine with a GPU:
//!   `cargo test --no-default-features --features memory,cuda --test validate_memory -- --ignored`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    missing_docs
)]

mod common;

use common::cuda_device;

use candle_core::{DType, Device, Tensor};
use candle_mi::{MemoryReport, MemorySnapshot};
use serial_test::serial;

#[test]
fn cpu_snapshot_is_ram_only() {
    let snap = MemorySnapshot::now(&Device::Cpu).unwrap();
    assert!(snap.ram_bytes > 0, "process RSS should be non-zero");
    assert!(snap.ram_mb() > 0.0, "RAM in MB should be positive");
    // A CPU device must not surface any VRAM data.
    assert!(snap.vram_bytes.is_none(), "CPU should report no VRAM used");
    assert!(
        snap.vram_total_bytes.is_none(),
        "CPU should report no VRAM total"
    );
    assert!(
        snap.vram_per_process.is_none(),
        "CPU should report no VRAM qualifier"
    );
    assert!(snap.gpu_name.is_none(), "CPU should report no GPU name");
    assert!(snap.vram_mb().is_none(), "CPU should report no VRAM MB");
}

#[test]
#[ignore = "requires a CUDA device; run with --features memory,cuda -- --ignored"]
#[serial]
fn cuda_snapshot_is_sane() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    let snap = MemorySnapshot::now(&device).unwrap();
    assert!(snap.ram_bytes > 0, "process RSS should be non-zero");

    // VRAM is best-effort; when a figure is present it must be internally
    // consistent. (We do not require VRAM to be present — a system could fall
    // through every backend — but if it is, the numbers must make sense.)
    if let Some(total) = snap.vram_total_bytes {
        assert!(total > 0, "device VRAM total should be positive");
        if let Some(used) = snap.vram_bytes {
            assert!(
                used <= total,
                "used VRAM {used} must not exceed total {total}"
            );
        }
        // Reserved (NVML v2 carve-out) is a subset of total when present.
        if let Some(reserved) = snap.vram_reserved_bytes {
            assert!(
                reserved < total,
                "reserved VRAM {reserved} must be a subset of total {total}"
            );
        }
    }

    println!(
        "CUDA snapshot: ram={:.0} MB, vram={:?} MB / total={:?} MB \
         (reserved={:?} MB), per_process={:?}, gpu={:?}",
        snap.ram_mb(),
        // CAST: u32 → u64, widening a VRAM reading in MiB to match the assertion's type;
        // lossless
        snap.vram_mb().map(|v| v as u64),
        snap.vram_total_bytes.map(|t| t / 1_048_576),
        snap.vram_reserved_bytes.map(|r| r / 1_048_576),
        snap.vram_per_process,
        snap.gpu_name,
    );
}

#[test]
#[ignore = "requires a CUDA device; run with --features memory,cuda -- --ignored"]
#[serial]
fn cuda_allocation_is_visible_in_vram() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };

    // Trim the pool first so a prior test's freed blocks don't mask the delta.
    candle_mi::sync_and_trim_gpu(&device);
    let before = MemorySnapshot::now(&device).unwrap();

    // Allocate ~512 MiB of F32 on the GPU and keep it alive across the
    // "after" snapshot. zeros() commits the pages, so the usage is real.
    let n_elems = 128 * 1024 * 1024; // 128 Mi × 4 B = 512 MiB
    let big = Tensor::zeros(n_elems, DType::F32, &device).unwrap();

    let after = MemorySnapshot::now(&device).unwrap();
    let report = MemoryReport::new(before, after);

    match report.vram_delta_mb() {
        Some(delta) => {
            // A 512 MiB allocation must be clearly visible. Use a generous
            // lower bound (256 MiB) to tolerate pool/measurement granularity.
            assert!(
                delta > 256.0,
                "expected >256 MB VRAM increase for a 512 MiB allocation, got {delta:.0} MB"
            );
            println!("VRAM delta for 512 MiB allocation: {delta:.0} MB");
        }
        None => {
            // No usable per-process VRAM backend on this system: nothing to
            // assert, but make the skip explicit rather than silently passing.
            eprintln!("SKIP assert: VRAM not measurable on this system");
        }
    }

    // Keep `big` alive until here, then release and trim.
    drop(big);
    candle_mi::sync_and_trim_gpu(&device);
}
