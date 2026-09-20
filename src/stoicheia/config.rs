// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration types for `AlgZoo` model backends (stoicheia).
//!
//! `AlgZoo` models are tiny (8–1,408 parameters) and solve algorithmic tasks.
//! Each task maps to a fixed architecture (RNN or attention-only transformer)
//! and output type (distribution over positions or scalar value).

use std::fmt;

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// Algorithmic task solved by an `AlgZoo` model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoicheiaTask {
    /// Find the position of the second-largest number in a sequence.
    SecondArgmax,
    /// Find the position of the median number in a sequence.
    Argmedian,
    /// Output the median value of a sequence.
    Median,
    /// Count the longest cycle in a permutation `f:{0..n-1} → {0..n-1}`.
    LongestCycle,
}

impl fmt::Display for StoicheiaTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SecondArgmax => write!(f, "2nd_argmax"),
            Self::Argmedian => write!(f, "argmedian"),
            Self::Median => write!(f, "median"),
            Self::LongestCycle => write!(f, "longest_cycle"),
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture
// ---------------------------------------------------------------------------

/// Model architecture for an `AlgZoo` model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoicheiaArch {
    /// Single-layer `ReLU` RNN (continuous input → distribution or scalar).
    Rnn,
    /// Attention-only transformer (discrete input → distribution or scalar).
    /// No MLP blocks, no layer normalization, no causal mask.
    Transformer,
}

impl fmt::Display for StoicheiaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rnn => write!(f, "rnn"),
            Self::Transformer => write!(f, "transformer"),
        }
    }
}

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Output type for an `AlgZoo` model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoicheiaOutput {
    /// Output is a distribution over positions (cross-entropy loss).
    /// Shape: `[batch, seq_len]`.
    Distribution,
    /// Output is a single scalar value (MSE loss).
    /// Shape: `[batch, 1]`.
    Scalar,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an `AlgZoo` model.
///
/// Use [`from_task`](Self::from_task) to construct with sensible defaults
/// derived from the task type, matching `AlgZoo`'s Python registry.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct StoicheiaConfig {
    /// Hidden dimension (`d_model` for transformer, RNN hidden size).
    pub hidden_size: usize,
    /// Input sequence length.
    pub seq_len: usize,
    /// Algorithmic task.
    pub task: StoicheiaTask,
    /// Model architecture (inferred from task).
    pub arch: StoicheiaArch,
    /// Output type (inferred from task).
    pub output: StoicheiaOutput,
    /// Number of attention layers (transformer only; always 1 for RNN).
    pub num_layers: usize,
    /// Number of attention heads (transformer only; 0 for RNN).
    pub num_heads: usize,
    /// Input range for discrete tasks (transformer only; equals `seq_len`).
    pub input_range: usize,
}

impl StoicheiaConfig {
    /// Create a configuration from a task, hidden size, and sequence length.
    ///
    /// Architecture, output type, number of layers, number of heads, and
    /// input range are inferred from the task, matching `AlgZoo`'s Python
    /// registry (`alg_zoo/tasks.py`).
    #[must_use]
    pub const fn from_task(task: StoicheiaTask, hidden_size: usize, seq_len: usize) -> Self {
        let (arch, output) = match task {
            StoicheiaTask::SecondArgmax | StoicheiaTask::Argmedian => {
                (StoicheiaArch::Rnn, StoicheiaOutput::Distribution)
            }
            StoicheiaTask::Median => (StoicheiaArch::Rnn, StoicheiaOutput::Scalar),
            StoicheiaTask::LongestCycle => {
                (StoicheiaArch::Transformer, StoicheiaOutput::Distribution)
            }
        };

        let num_layers = match arch {
            StoicheiaArch::Rnn => 1,
            // `AlgZoo` default: 2 attention layers
            StoicheiaArch::Transformer => 2,
        };

        let num_heads = match arch {
            StoicheiaArch::Rnn => 0,
            // `AlgZoo` default: single attention head
            StoicheiaArch::Transformer => 1,
        };

        let input_range = match arch {
            StoicheiaArch::Rnn => 0,
            // Discrete tasks: input integers in 0..seq_len
            StoicheiaArch::Transformer => seq_len,
        };

        Self {
            hidden_size,
            seq_len,
            task,
            arch,
            output,
            num_layers,
            num_heads,
            input_range,
        }
    }

    /// Output size: `seq_len` for distribution tasks, `1` for scalar tasks.
    #[must_use]
    pub const fn output_size(&self) -> usize {
        match self.output {
            StoicheiaOutput::Distribution => self.seq_len,
            StoicheiaOutput::Scalar => 1,
        }
    }

    /// Total parameter count for this model configuration.
    #[must_use]
    pub const fn param_count(&self) -> usize {
        match self.arch {
            // H*(1 + H + output_size)
            StoicheiaArch::Rnn => self.hidden_size * (1 + self.hidden_size + self.output_size()),
            // input_range*H + seq_len*H + n_layers*(4*H*H) + output_size*H
            StoicheiaArch::Transformer => {
                self.input_range * self.hidden_size
                    + self.seq_len * self.hidden_size
                    + self.num_layers * 4 * self.hidden_size * self.hidden_size
                    + self.output_size() * self.hidden_size
            }
        }
    }
}

impl fmt::Display for StoicheiaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(h={}, n={}, arch={}, {} params)",
            self.task,
            self.hidden_size,
            self.seq_len,
            self.arch,
            self.param_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_task_rnn_distribution() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 16, 10);
        assert_eq!(cfg.arch, StoicheiaArch::Rnn);
        assert_eq!(cfg.output, StoicheiaOutput::Distribution);
        assert_eq!(cfg.output_size(), 10);
        assert_eq!(cfg.num_layers, 1);
        assert_eq!(cfg.num_heads, 0);
        // H*(1 + H + output_size) = 16*(1+16+10) = 432
        assert_eq!(cfg.param_count(), 432);
    }

    #[test]
    fn from_task_rnn_scalar() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::Median, 4, 3);
        assert_eq!(cfg.arch, StoicheiaArch::Rnn);
        assert_eq!(cfg.output, StoicheiaOutput::Scalar);
        assert_eq!(cfg.output_size(), 1);
        // H*(1 + H + 1) = 4*(1+4+1) = 24
        assert_eq!(cfg.param_count(), 24);
    }

    #[test]
    fn from_task_transformer() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::LongestCycle, 4, 4);
        assert_eq!(cfg.arch, StoicheiaArch::Transformer);
        assert_eq!(cfg.output, StoicheiaOutput::Distribution);
        assert_eq!(cfg.num_layers, 2);
        assert_eq!(cfg.num_heads, 1);
        assert_eq!(cfg.input_range, 4);
        // 4*4 + 4*4 + 2*(4*16) + 4*4 = 16+16+128+16 = 176
        assert_eq!(cfg.param_count(), 176);
    }

    #[test]
    fn display_config() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 16, 10);
        let s = cfg.to_string();
        assert!(s.contains("2nd_argmax"));
        assert!(s.contains("432"));
    }

    #[test]
    fn blog_m2_2() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 2, 2);
        assert_eq!(cfg.param_count(), 10);
    }

    #[test]
    fn blog_m4_3() {
        let cfg = StoicheiaConfig::from_task(StoicheiaTask::SecondArgmax, 4, 3);
        assert_eq!(cfg.param_count(), 32);
    }
}
