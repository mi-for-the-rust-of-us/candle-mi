// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process and GPU memory reporting.
//!
//! Provides [`MemorySnapshot`] to capture current RAM and VRAM usage,
//! and [`MemoryReport`] to measure deltas between two snapshots.
//!
//! # Measurement backend
//!
//! All platform measurement is delegated to the standalone
//! [`hypomnesis`](https://crates.io/crates/hypomnesis) crate, which candle-mi
//! flattens into its own [`MemorySnapshot`].  hypomnesis owns the unsafe FFI
//! (process `RSS`, `DXGI`, `NVML`, `nvidia-smi`, Apple `Metal`); candle-mi keeps
//! only the flat snapshot type, the delta/formatting helpers
//! ([`MemoryReport`]), and the CUDA pool-trim control ([`sync_and_trim_gpu`]).
//!
//! | Metric | Windows | Linux | macOS |
//! |--------|---------|-------|-------|
//! | RAM (`RSS`) | `K32GetProcessMemoryInfo` | `/proc/self/status` `VmRSS` | `task_info` `phys_footprint` |
//! | VRAM (per-process) | `DXGI` `IDXGIAdapter3` | `NVML` | Apple `Metal` ledger |
//! | VRAM (device total) | `NVML` / `DXGI` | `NVML` | `sysctl` + `Metal` |
//! | VRAM (reserved) | `NVML` v2 (R510+) | `NVML` v2 (R510+) | — |
//! | VRAM (fallback) | `nvidia-smi` (device-wide) | `nvidia-smi` (device-wide) | — |
//!
//! The reserved figure (driver/firmware carve-out) is a **subset** of the
//! device total — NVML reports `total = reserved + free + used` — so allocation
//! headroom is `total - reserved`.  It is `None` on non-`NVML` backends and on
//! pre-R510 drivers.
//!
//! # Feature gates
//!
//! - **`memory`**: Enables this module and pulls `hypomnesis` (lean feature set:
//!   `nvml`, `dxgi`, `nvidia-smi-fallback`, `metal`).  No unsafe code lives in
//!   candle-mi under this feature alone — the only remaining `unsafe` is the
//!   CUDA pool-trim in [`sync_and_trim_gpu`], gated behind `cuda`.
//! - **`memory-debug`** (implies `memory`): forwards `hypomnesis/debug-output`,
//!   printing raw `NVML` / `DXGI` / `nvidia-smi` / `Metal` values to stderr.

use crate::{MIError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Memory snapshot at a point in time.
///
/// Captures process RAM (resident set size) and optionally GPU VRAM.
/// Use [`MemorySnapshot::now`] to take a measurement, and
/// [`MemoryReport::new`] to compute deltas between two snapshots.
///
/// # Example
///
/// ```no_run
/// use candle_mi::MemorySnapshot;
///
/// let before = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// // ... load a model ...
/// let after = MemorySnapshot::now(&candle_core::Device::Cpu)?;
/// let report = candle_mi::MemoryReport::new(before, after);
/// println!("RAM delta: {:+.1} MB", report.ram_delta_mb());
/// # Ok::<(), candle_mi::MIError>(())
/// ```
// `#[non_exhaustive]`: obtained via [`MemorySnapshot::now`], not literal-
// constructed by downstream code — so new measurement fields (like
// `vram_reserved_bytes`) can be added without a breaking change.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Process resident set size (working set on Windows) in bytes.
    pub ram_bytes: u64,
    /// GPU memory used in bytes.
    /// Per-process when measured via `DXGI`/`NVML`, device-wide when via `nvidia-smi` fallback.
    /// `None` if no GPU is present or measurement failed.
    pub vram_bytes: Option<u64>,
    /// Total GPU memory on the active device in bytes.
    /// `None` if no GPU is present or measurement failed.
    pub vram_total_bytes: Option<u64>,
    /// Whether the VRAM measurement is per-process (`true`) or device-wide (`false`).
    /// `None` if no VRAM data is available.
    pub vram_per_process: Option<bool>,
    /// GPU adapter name (e.g., `NVIDIA GeForce RTX 5060 Ti`).
    /// `None` if not available (no GPU, or the backend did not report a name).
    pub gpu_name: Option<String>,
    /// GPU memory reserved for system use (driver/firmware) in bytes — the
    /// `NVML` v2 `reserved` carve-out (page tables, context/channel structures,
    /// ECC parity).  A **subset of** [`vram_total_bytes`](Self::vram_total_bytes)
    /// (NVML reports `total = reserved + free + used`), not an addition to it;
    /// allocation headroom is `total - reserved`.  `Some` only on the `NVML`
    /// path with an R510+ driver — `None` on older drivers and on the
    /// `DXGI`-only, `nvidia-smi`, and `Metal` paths.
    pub vram_reserved_bytes: Option<u64>,
}

/// Memory delta between two snapshots.
///
/// Computed from a `before` and `after` [`MemorySnapshot`].
/// Positive deltas mean memory increased; negative means freed.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MemoryReport {
    /// Snapshot taken before the operation.
    pub before: MemorySnapshot,
    /// Snapshot taken after the operation.
    pub after: MemorySnapshot,
}

impl MemorySnapshot {
    /// Capture current memory state.
    ///
    /// RAM is always measured (per-process `RSS`).  VRAM is measured only when
    /// `device` is a GPU (CUDA or Metal), by delegating to
    /// [`hypomnesis::Snapshot::now`] and flattening the result.  On CPU only
    /// RAM is reported (the VRAM fields are `None`).
    ///
    /// The GPU index queried is `0`: candle's [`Device`](candle_core::Device)
    /// exposes no per-device ordinal, and the historical implementation also
    /// queried GPU 0.  Multi-GPU correctness is a future enhancement gated on a
    /// candle ordinal accessor.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Memory`] if the underlying `RAM` query fails (platform
    /// API error).  VRAM measurement failures are non-fatal — the corresponding
    /// fields are set to `None`.
    pub fn now(device: &candle_core::Device) -> Result<Self> {
        if device.is_cuda() || device.is_metal() {
            let snapshot = hypomnesis::Snapshot::now(0)
                .map_err(|e| MIError::Memory(format!("failed to query GPU snapshot: {e}")))?;
            Ok(Self::from_hypomnesis(&snapshot))
        } else {
            let ram_bytes = hypomnesis::process_rss()
                .map_err(|e| MIError::Memory(format!("failed to query process RSS: {e}")))?;
            Ok(Self {
                ram_bytes,
                vram_bytes: None,
                vram_total_bytes: None,
                vram_per_process: None,
                gpu_name: None,
                vram_reserved_bytes: None,
            })
        }
    }

    /// Flatten a [`hypomnesis::Snapshot`] into candle-mi's flat snapshot.
    ///
    /// Maps `snapshot.gpu` (per-process used + per-process flag) and
    /// `snapshot.gpu_device` (device total + adapter name + reserved carve-out)
    /// onto the flat fields.
    fn from_hypomnesis(snapshot: &hypomnesis::Snapshot) -> Self {
        let (vram_bytes, vram_per_process) = snapshot.gpu.as_ref().map_or((None, None), |gpu| {
            (Some(gpu.used_bytes), Some(gpu.is_per_process))
        });
        let (vram_total_bytes, gpu_name, vram_reserved_bytes) = snapshot
            .gpu_device
            .as_ref()
            .map_or((None, None, None), |dev| {
                // BORROW: clone the adapter name out of the borrowed `Option<String>`
                (Some(dev.total_bytes), dev.name.clone(), dev.reserved_bytes)
            });
        Self {
            ram_bytes: snapshot.ram_bytes,
            vram_bytes,
            vram_total_bytes,
            vram_per_process,
            gpu_name,
            vram_reserved_bytes,
        }
    }

    /// Format RAM usage as megabytes.
    #[must_use]
    pub fn ram_mb(&self) -> f64 {
        // CAST: u64 → f64, value is memory in bytes — fits in f64 mantissa
        // for any realistic process size (< 2^53 bytes = 8 PB)
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        let mb = self.ram_bytes as f64 / 1_048_576.0;
        mb
    }

    /// Format VRAM usage as megabytes, if available.
    #[must_use]
    pub fn vram_mb(&self) -> Option<f64> {
        // CAST: u64 → f64, same justification as ram_mb
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        self.vram_bytes.map(|b| b as f64 / 1_048_576.0)
    }

    /// Format reserved VRAM (the driver/firmware carve-out) as megabytes, if
    /// available.  A subset of [`vram_mb`](Self::vram_mb)'s device total — see
    /// [`vram_reserved_bytes`](Self::vram_reserved_bytes).
    #[must_use]
    pub fn vram_reserved_mb(&self) -> Option<f64> {
        // CAST: u64 → f64, same justification as ram_mb
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        self.vram_reserved_bytes.map(|b| b as f64 / 1_048_576.0)
    }
}

/// Synchronize the CUDA device and trim its memory pool.
///
/// On a CUDA device this:
/// 1. Calls `cuCtxSynchronize` so all pending async frees complete.
/// 2. Calls `cuMemPoolTrimTo(pool, 0)` to release all unused reserved
///    VRAM back to the device.
///
/// cudarc's stream-ordered allocator (`malloc_async` / `free_async`)
/// keeps freed blocks in a pool for reuse. Over many forward passes
/// with varying tensor sizes the pool grows monotonically — `DXGI` and
/// `nvidia-smi` report this reserved memory as "in use", eventually
/// causing OOM even though no live tensors need it.
///
/// This function is a no-op on CPU and on Metal.
///
/// # Example
///
/// ```no_run
/// # use candle_mi::sync_and_trim_gpu;
/// # let device = candle_core::Device::Cpu;
/// // After dropping all GPU tensors from a forward pass:
/// sync_and_trim_gpu(&device);
/// ```
// Cannot be `const fn`: the `cuda` branch calls non-const FFI (cuDeviceGetDefaultMemPool,
// cuMemPoolTrimTo). Without `cuda` the body collapses to a no-op, which is why clippy
// suggests `const` — but `const fn` would break the cuda-enabled build.
#[allow(clippy::missing_const_for_fn)]
pub fn sync_and_trim_gpu(device: &candle_core::Device) {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(cuda_dev) = device {
        use candle_core::backend::BackendDevice;
        // Synchronize so all pending async frees complete.
        let _ = cuda_dev.synchronize();

        // Trim the default memory pool to release all unused reserved VRAM.
        #[allow(unsafe_code)]
        {
            use candle_core::cuda_backend::cudarc::driver::sys;

            let stream = cuda_dev.cuda_stream();
            // Allocate a zero-length slice just to access the CudaContext
            // (CudaStream.ctx is pub(crate), but CudaSlice.context() is pub).
            if let Ok(probe) = stream.null::<u8>() {
                let ctx = probe.context();
                let cu_device = ctx.cu_device();
                // SAFETY: cuDeviceGetDefaultMemPool and cuMemPoolTrimTo are
                // documented CUDA driver APIs for pool management. The CUdevice
                // handle comes from candle's CudaContext (valid after synchronize).
                // cuMemPoolTrimTo(pool, 0) releases all unused memory — it cannot
                // free memory that is still in use by live tensors.
                unsafe {
                    let mut pool = std::mem::zeroed();
                    let rc = sys::cuDeviceGetDefaultMemPool(&raw mut pool, cu_device);
                    if rc == sys::CUresult::CUDA_SUCCESS {
                        let _ = sys::cuMemPoolTrimTo(pool, 0);
                    }
                }
            }
        }
    }

    // Suppress unused-variable warning on non-CUDA builds.
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

impl MemoryReport {
    /// Create a report from two snapshots.
    #[must_use]
    pub const fn new(before: MemorySnapshot, after: MemorySnapshot) -> Self {
        Self { before, after }
    }

    /// RAM delta in megabytes (positive = increased).
    #[must_use]
    pub fn ram_delta_mb(&self) -> f64 {
        self.after.ram_mb() - self.before.ram_mb()
    }

    /// VRAM delta in megabytes (positive = increased).
    /// Returns `None` if either snapshot lacks VRAM data.
    #[must_use]
    pub fn vram_delta_mb(&self) -> Option<f64> {
        match (self.after.vram_mb(), self.before.vram_mb()) {
            (Some(after), Some(before)) => Some(after - before),
            (Some(_) | None, None) | (None, Some(_)) => None,
        }
    }

    /// Print a one-line summary of the delta.
    pub fn print_delta(&self, label: &str) {
        let ram = self.ram_delta_mb();
        print!("  {label}: RAM {ram:+.0} MB");
        if let Some(vram) = self.vram_delta_mb() {
            let qualifier = self.vram_qualifier();
            print!("  |  VRAM {vram:+.0} MB{qualifier}");
        }
        println!();
    }

    /// Print a two-line summary showing before → after for both RAM and VRAM.
    pub fn print_before_after(&self, label: &str) {
        println!(
            "  {label}: RAM {:.0} MB → {:.0} MB ({:+.0} MB)",
            self.before.ram_mb(),
            self.after.ram_mb(),
            self.ram_delta_mb(),
        );
        if let (Some(before), Some(after)) = (self.before.vram_mb(), self.after.vram_mb()) {
            // CAST: u64 → f64, same justification as ram_mb
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let total = self.after.vram_total_bytes.map_or(String::new(), |t| {
                format!(" / {:.0} MB", t as f64 / 1_048_576.0)
            });
            // Reserved is a subset of total (NVML v2: total = reserved + free + used).
            // CAST: u64 → f64, same justification as ram_mb
            #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
            let reserved = self.after.vram_reserved_bytes.map_or(String::new(), |r| {
                format!(", {:.0} MB reserved", r as f64 / 1_048_576.0)
            });
            let qualifier = self.vram_qualifier();
            let gpu = self
                .after
                .gpu_name
                .as_deref()
                .map_or(String::new(), |name| format!(" [{name}]"));
            println!(
                "  {label}: VRAM {before:.0} MB → {after:.0} MB ({:+.0} MB{total}{reserved}){qualifier}{gpu}",
                after - before,
            );
        }
    }

    /// Return a short qualifier string indicating VRAM measurement quality.
    #[must_use]
    const fn vram_qualifier(&self) -> &'static str {
        match self.after.vram_per_process {
            Some(true) => " [per-process]",
            Some(false) => " [device-wide]",
            None => "",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_cpu_has_ram() {
        let snap = MemorySnapshot::now(&candle_core::Device::Cpu).unwrap();
        // Process must be using > 0 bytes of RAM
        assert!(snap.ram_bytes > 0, "RAM should be non-zero");
        // CPU device should not have VRAM
        assert!(snap.vram_bytes.is_none(), "CPU should have no VRAM");
        assert!(
            snap.vram_per_process.is_none(),
            "CPU should have no VRAM qualifier"
        );
    }

    #[test]
    fn report_delta_positive_for_allocation() {
        let before = MemorySnapshot {
            ram_bytes: 100 * 1_048_576, // 100 MB
            vram_bytes: Some(500 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let after = MemorySnapshot {
            ram_bytes: 200 * 1_048_576, // 200 MB
            vram_bytes: Some(1_000 * 1_048_576),
            vram_total_bytes: Some(16_384 * 1_048_576),
            vram_per_process: Some(true),
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let report = MemoryReport::new(before, after);

        let ram_delta = report.ram_delta_mb();
        assert!(
            (ram_delta - 100.0).abs() < 0.01,
            "RAM delta should be ~100 MB, got {ram_delta}"
        );

        let vram_delta = report.vram_delta_mb().unwrap();
        assert!(
            (vram_delta - 500.0).abs() < 0.01,
            "VRAM delta should be ~500 MB, got {vram_delta}"
        );
    }

    #[test]
    fn report_delta_none_when_no_vram() {
        let before = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let after = MemorySnapshot {
            ram_bytes: 200,
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let report = MemoryReport::new(before, after);
        assert!(report.vram_delta_mb().is_none());
    }

    #[test]
    fn ram_mb_conversion() {
        let snap = MemorySnapshot {
            ram_bytes: 1_048_576, // exactly 1 MB
            vram_bytes: None,
            vram_total_bytes: None,
            vram_per_process: None,
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        assert!((snap.ram_mb() - 1.0).abs() < 0.001);
    }

    #[test]
    fn vram_reserved_mb_conversion() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500 * 1_048_576),
            vram_total_bytes: Some(16_311 * 1_048_576),
            vram_per_process: Some(true),
            gpu_name: None,
            vram_reserved_bytes: Some(259 * 1_048_576),
        };
        assert!((snap.vram_reserved_mb().unwrap() - 259.0).abs() < 0.001);
        // Reserved is a subset of total: total - reserved is the usable headroom.
        assert!(snap.vram_reserved_bytes.unwrap() < snap.vram_total_bytes.unwrap());
    }

    #[test]
    fn vram_reserved_mb_none_when_absent() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(true),
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        assert!(snap.vram_reserved_mb().is_none());
    }

    #[test]
    fn vram_qualifier_per_process() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(true),
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [per-process]");
    }

    #[test]
    fn vram_qualifier_device_wide() {
        let snap = MemorySnapshot {
            ram_bytes: 100,
            vram_bytes: Some(500),
            vram_total_bytes: Some(1000),
            vram_per_process: Some(false),
            gpu_name: None,
            vram_reserved_bytes: None,
        };
        let report = MemoryReport::new(snap.clone(), snap);
        assert_eq!(report.vram_qualifier(), " [device-wide]");
    }
}
