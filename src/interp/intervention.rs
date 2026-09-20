// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention intervention for causal experiments.
//!
//! Enables causal intervention experiments by surgically modifying
//! attention edges and measuring impact on model outputs.
//!
//! ## Intervention Types
//!
//! - **Knockout**: Remove attention edges (pre-softmax, add `-inf`)
//! - **Scale**: Multiply attention by a factor (post-softmax, then renormalize)
//! - **`SetValue`**: Set attention to a specific value (post-softmax, then renormalize)
//!
//! ## Intervention Mechanism
//!
//! Knockout is implemented by adding negative infinity to specified attention
//! scores BEFORE softmax. After softmax, these edges become exactly 0,
//! completely removing their contribution to the output.
//!
//! Steering (Scale/SetValue) is applied AFTER softmax, modifying attention
//! weights and renormalizing rows to maintain valid probability distributions.

use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device, Tensor};

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract a 4D tensor to nested `Vec`s.
///
/// Candle doesn't provide `to_vec4()`, so we flatten and reshape manually.
///
/// # Shapes
///
/// - `tensor`: `[d0, d1, d2, d3]`
///
/// # Errors
///
/// Returns [`MIError::Intervention`] if the tensor is not 4D.
fn tensor_to_vec4(tensor: &Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let shape = tensor.dims();
    if shape.len() != 4 {
        return Err(MIError::Intervention(format!(
            "expected 4D tensor, got {}D",
            shape.len()
        )));
    }
    let s0 = shape.first().copied().unwrap_or(0);
    let s1 = shape.get(1).copied().unwrap_or(0);
    let s2 = shape.get(2).copied().unwrap_or(0);
    let s3 = shape.get(3).copied().unwrap_or(0);

    let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

    let mut result = Vec::with_capacity(s0);
    let mut iter = flat.into_iter();
    for _ in 0..s0 {
        let mut axis1 = Vec::with_capacity(s1);
        for _ in 0..s1 {
            let mut axis2 = Vec::with_capacity(s2);
            for _ in 0..s2 {
                let row: Vec<f32> = iter.by_ref().take(s3).collect();
                axis2.push(row);
            }
            axis1.push(axis2);
        }
        result.push(axis1);
    }

    Ok(result)
}

/// Convert logits to a probability distribution (softmax).
fn softmax_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    // PROMOTE: softmax needs f32 for numerical stability
    let logits_f32 = logits.to_dtype(DType::F32)?;
    // Terminal read-out (probability display): no gradient ever wanted, so
    // the fused kernel is used directly rather than `crate::nn_ops`.
    let probs = candle_nn::ops::softmax_last_dim(&logits_f32)?;
    Ok(probs.flatten_all()?.to_vec1()?)
}

/// Expand edge specifications, resolving sentinel values (`usize::MAX`).
///
/// - `(from, usize::MAX)` → all edges FROM `from` to every position
/// - `(usize::MAX, to)` → all edges TO `to` from every position
fn expand_edges(edges: &[AttentionEdge], seq_len: usize) -> Vec<AttentionEdge> {
    let mut expanded = Vec::new();

    for edge in edges {
        match (edge.from_pos, edge.to_pos) {
            (from, usize::MAX) if from != usize::MAX => {
                for to in 0..seq_len {
                    expanded.push(AttentionEdge::new(from, to));
                }
            }
            (usize::MAX, to) if to != usize::MAX => {
                for from in 0..seq_len {
                    expanded.push(AttentionEdge::new(from, to));
                }
            }
            (from, to) if from != usize::MAX && to != usize::MAX => {
                expanded.push(*edge);
            }
            _ => {} // Invalid sentinel combination, skip
        }
    }

    expanded
}

// ===========================================================================
// Part 1: Knockout Specification
// ===========================================================================

/// Specification for which layers to target.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LayerSpec {
    /// Apply to all layers.
    All,
    /// Apply to specific layers.
    Specific(Vec<usize>),
    /// Apply to a range of layers (inclusive).
    Range {
        /// First layer (inclusive).
        start: usize,
        /// Last layer (inclusive).
        end: usize,
    },
}

/// Specification for which heads to target.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum HeadSpec {
    /// Apply to all heads.
    All,
    /// Apply to specific heads.
    Specific(Vec<usize>),
}

/// A single attention edge from one position to another.
///
/// Uses `usize::MAX` as a sentinel for "all positions" (expanded at
/// mask creation time based on actual sequence length).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionEdge {
    /// Token position that is attending (row in attention matrix).
    pub from_pos: usize,
    /// Token position being attended to (column in attention matrix).
    pub to_pos: usize,
}

impl AttentionEdge {
    /// Create a new edge.
    #[must_use]
    pub const fn new(from_pos: usize, to_pos: usize) -> Self {
        Self { from_pos, to_pos }
    }
}

/// Specification for which attention edges to knock out.
///
/// An "edge" is attention from one token position to another.
/// Knockout removes the edge completely by setting its attention weight
/// to 0 (via pre-softmax `-inf` masking).
///
/// # Example
///
/// ```
/// use candle_mi::KnockoutSpec;
///
/// let spec = KnockoutSpec::new()
///     .layer(10)
///     .from_to_positions(5, &[0, 1, 2, 3]);
/// assert_eq!(spec.edges.len(), 4);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[must_use]
pub struct KnockoutSpec {
    /// Layer indices to apply intervention.
    pub layers: LayerSpec,
    /// Head indices to apply intervention.
    pub heads: HeadSpec,
    /// Attention edges to knock out: `(from_position, to_position)`.
    pub edges: Vec<AttentionEdge>,
}

impl KnockoutSpec {
    /// Create a new empty knockout specification (all layers, all heads).
    pub const fn new() -> Self {
        Self {
            layers: LayerSpec::All,
            heads: HeadSpec::All,
            edges: Vec::new(),
        }
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Target a single head.
    pub fn head(mut self, head: usize) -> Self {
        self.heads = HeadSpec::Specific(vec![head]);
        self
    }

    /// Target multiple specific heads.
    pub fn heads(mut self, heads: &[usize]) -> Self {
        self.heads = HeadSpec::Specific(heads.to_vec());
        self
    }

    /// Add a single edge to knock out.
    pub fn edge(mut self, from_pos: usize, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, to_pos));
        self
    }

    /// Knock out all attention FROM a specific position.
    pub fn from_position(mut self, from_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, usize::MAX));
        self
    }

    /// Knock out all attention TO a specific position.
    pub fn to_position(mut self, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(usize::MAX, to_pos));
        self
    }

    /// Add edges from one position to several positions.
    pub fn from_to_positions(mut self, from_pos: usize, to_positions: &[usize]) -> Self {
        for &to_pos in to_positions {
            self.edges.push(AttentionEdge::new(from_pos, to_pos));
        }
        self
    }

    /// Check if this layer should have intervention applied.
    #[must_use]
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Check if this head should have intervention applied.
    #[must_use]
    pub fn applies_to_head(&self, head: usize) -> bool {
        match &self.heads {
            HeadSpec::All => true,
            HeadSpec::Specific(heads) => heads.contains(&head),
        }
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if any layer, head, or edge
    /// position is out of range.
    pub fn validate(&self, n_layers: usize, n_heads: usize, seq_len: usize) -> Result<()> {
        validate_layers(&self.layers, n_layers)?;
        validate_heads(&self.heads, n_heads)?;
        validate_edges(&self.edges, seq_len)?;
        Ok(())
    }
}

impl Default for KnockoutSpec {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Part 2: Steering Specification
// ===========================================================================

/// Type of intervention to apply to attention weights.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InterventionType {
    /// Set attention to zero (pre-softmax: add `-inf`).
    #[default]
    Knockout,
    /// Multiply attention by factor (post-softmax, then renormalize).
    Scale(f32),
    /// Set attention to specific target value (post-softmax, then renormalize).
    SetValue(f32),
}

/// Specification for attention steering interventions.
///
/// Unlike knockout which removes edges, steering modifies attention weights
/// by scaling or setting values, then renormalizing to maintain valid
/// probability distributions.
///
/// # Example
///
/// ```
/// use candle_mi::SteeringSpec;
///
/// let spec = SteeringSpec::scale(3.0)
///     .layer(16)
///     .from_to_positions(5, &[0, 1, 2]);
/// assert_eq!(spec.edges.len(), 3);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[must_use]
pub struct SteeringSpec {
    /// Layer indices to apply intervention.
    pub layers: LayerSpec,
    /// Head indices to apply intervention.
    pub heads: HeadSpec,
    /// Attention edges to modify.
    pub edges: Vec<AttentionEdge>,
    /// Type of intervention to apply.
    pub intervention_type: InterventionType,
}

impl SteeringSpec {
    /// Create a new steering specification with the given intervention type.
    pub const fn new(intervention_type: InterventionType) -> Self {
        Self {
            layers: LayerSpec::All,
            heads: HeadSpec::All,
            edges: Vec::new(),
            intervention_type,
        }
    }

    /// Create a scaling intervention.
    pub const fn scale(factor: f32) -> Self {
        Self::new(InterventionType::Scale(factor))
    }

    /// Create a set-value intervention.
    pub const fn set_value(target: f32) -> Self {
        Self::new(InterventionType::SetValue(target))
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Target a single head.
    pub fn head(mut self, head: usize) -> Self {
        self.heads = HeadSpec::Specific(vec![head]);
        self
    }

    /// Target multiple specific heads.
    pub fn heads(mut self, heads: &[usize]) -> Self {
        self.heads = HeadSpec::Specific(heads.to_vec());
        self
    }

    /// Add a single edge to modify.
    pub fn edge(mut self, from_pos: usize, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, to_pos));
        self
    }

    /// Steer all attention FROM a specific position.
    pub fn from_position(mut self, from_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(from_pos, usize::MAX));
        self
    }

    /// Steer all attention TO a specific position.
    pub fn to_position(mut self, to_pos: usize) -> Self {
        self.edges.push(AttentionEdge::new(usize::MAX, to_pos));
        self
    }

    /// Add edges from one position to several positions.
    pub fn from_to_positions(mut self, from_pos: usize, to_positions: &[usize]) -> Self {
        for &to_pos in to_positions {
            self.edges.push(AttentionEdge::new(from_pos, to_pos));
        }
        self
    }

    /// Check if this layer should have intervention applied.
    #[must_use]
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Check if this head should have intervention applied.
    #[must_use]
    pub fn applies_to_head(&self, head: usize) -> bool {
        match &self.heads {
            HeadSpec::All => true,
            HeadSpec::Specific(heads) => heads.contains(&head),
        }
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if any layer, head, edge, or
    /// intervention parameter is out of range.
    pub fn validate(&self, n_layers: usize, n_heads: usize, seq_len: usize) -> Result<()> {
        validate_layers(&self.layers, n_layers)?;
        validate_heads(&self.heads, n_heads)?;
        validate_edges(&self.edges, seq_len)?;

        match self.intervention_type {
            InterventionType::Scale(factor) => {
                if factor < 0.0 {
                    return Err(MIError::Intervention(format!(
                        "scale factor must be non-negative, got {factor}"
                    )));
                }
            }
            InterventionType::SetValue(value) => {
                if !(0.0..=1.0).contains(&value) {
                    return Err(MIError::Intervention(format!(
                        "set value must be in [0, 1], got {value}"
                    )));
                }
            }
            InterventionType::Knockout => {}
        }

        Ok(())
    }

    /// Get the intervention type.
    #[must_use]
    pub const fn intervention_type(&self) -> InterventionType {
        self.intervention_type
    }

    /// Check if this is a knockout intervention.
    #[must_use]
    pub const fn is_knockout(&self) -> bool {
        matches!(self.intervention_type, InterventionType::Knockout)
    }

    /// Check if this is a post-softmax steering intervention.
    #[must_use]
    pub const fn is_steering(&self) -> bool {
        matches!(
            self.intervention_type,
            InterventionType::Scale(_) | InterventionType::SetValue(_)
        )
    }

    /// Check if steering only affects positions within the prompt.
    ///
    /// If all edges have `from_pos < prompt_len`, the steering can be
    /// applied once during prompt processing, cached, and reused for
    /// generation (no steering needed for generated tokens).
    #[must_use]
    pub fn is_prompt_only(&self, prompt_len: usize) -> bool {
        for edge in &self.edges {
            if edge.from_pos == usize::MAX {
                return false;
            }
            if edge.from_pos >= prompt_len {
                return false;
            }
        }
        true
    }

    /// Maximum `from_pos` among all edges (excluding sentinels).
    #[must_use]
    pub fn max_from_pos(&self) -> Option<usize> {
        self.edges
            .iter()
            .filter(|e| e.from_pos != usize::MAX)
            .map(|e| e.from_pos)
            .max()
    }

    /// Maximum `to_pos` among all edges (excluding sentinels).
    #[must_use]
    pub fn max_to_pos(&self) -> Option<usize> {
        self.edges
            .iter()
            .filter(|e| e.to_pos != usize::MAX)
            .map(|e| e.to_pos)
            .max()
    }
}

/// Convert a [`KnockoutSpec`] to a [`SteeringSpec`] for unified handling.
impl From<KnockoutSpec> for SteeringSpec {
    fn from(spec: KnockoutSpec) -> Self {
        Self {
            layers: spec.layers,
            heads: spec.heads,
            edges: spec.edges,
            intervention_type: InterventionType::Knockout,
        }
    }
}

// ===========================================================================
// Result types
// ===========================================================================

/// Result of an ablation (knockout) experiment.
///
/// Carries baseline and ablated logits so the caller can compute
/// KL divergence, logit diffs, and top-changed-token analyses.
#[non_exhaustive]
#[derive(Debug)]
pub struct AblationResult {
    /// Logits from baseline forward pass (no intervention).
    pub baseline_logits: Tensor,
    /// Logits from the knocked-out forward pass.
    pub ablated_logits: Tensor,
    /// The knockout specification used.
    pub spec: KnockoutSpec,
}

impl AblationResult {
    /// Create a new ablation result.
    #[must_use]
    pub const fn new(baseline_logits: Tensor, ablated_logits: Tensor, spec: KnockoutSpec) -> Self {
        Self {
            baseline_logits,
            ablated_logits,
            spec,
        }
    }

    /// KL divergence between baseline and ablated distributions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.ablated_logits)
    }

    /// Logit difference for a specific token (`baseline - ablated`).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if `token_id` is out of range.
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        logit_diff_impl(&self.baseline_logits, &self.ablated_logits, token_id)
    }

    /// Top-k tokens that changed most due to ablation.
    ///
    /// Returns `(token_id, baseline_prob, ablated_prob, abs_diff)`.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        top_changed_impl(&self.baseline_logits, &self.ablated_logits, k)
    }
}

/// Result of a steering experiment.
#[non_exhaustive]
#[derive(Debug)]
#[must_use]
pub struct SteeringResult {
    /// Logits from baseline forward pass (no intervention).
    pub baseline_logits: Tensor,
    /// Logits from the steered forward pass.
    pub steered_logits: Tensor,
    /// The steering specification used.
    pub spec: SteeringSpec,
    /// Mean attention to target edges before steering.
    pub baseline_attention_mean: Option<f32>,
    /// Mean attention to target edges after steering.
    pub steered_attention_mean: Option<f32>,
}

impl SteeringResult {
    /// Create a new steering result.
    pub const fn new(baseline_logits: Tensor, steered_logits: Tensor, spec: SteeringSpec) -> Self {
        Self {
            baseline_logits,
            steered_logits,
            spec,
            baseline_attention_mean: None,
            steered_attention_mean: None,
        }
    }

    /// Add attention measurements.
    pub const fn with_attention_measurements(
        mut self,
        baseline_mean: f32,
        steered_mean: f32,
    ) -> Self {
        self.baseline_attention_mean = Some(baseline_mean);
        self.steered_attention_mean = Some(steered_mean);
        self
    }

    /// KL divergence between baseline and steered distributions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.steered_logits)
    }

    /// Logit difference for a specific token (`baseline - steered`).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if `token_id` is out of range.
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        logit_diff_impl(&self.baseline_logits, &self.steered_logits, token_id)
    }

    /// Top-k tokens that changed most due to steering.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        top_changed_impl(&self.baseline_logits, &self.steered_logits, k)
    }

    /// Attention change ratio (`steered_mean / baseline_mean`).
    #[must_use]
    pub fn attention_ratio(&self) -> Option<f32> {
        match (self.baseline_attention_mean, self.steered_attention_mean) {
            (Some(base), Some(steered)) if base > 1e-10 => Some(steered / base),
            _ => None,
        }
    }
}

// ===========================================================================
// Shared result helpers
// ===========================================================================

/// Compute logit difference for a specific token.
fn logit_diff_impl(baseline: &Tensor, other: &Tensor, token_id: u32) -> Result<f32> {
    let baseline_f32 = baseline.to_dtype(DType::F32)?;
    let other_f32 = other.to_dtype(DType::F32)?;
    let baseline_vec: Vec<f32> = baseline_f32.flatten_all()?.to_vec1()?;
    let other_vec: Vec<f32> = other_f32.flatten_all()?.to_vec1()?;

    // CAST: u32 → usize, token ID used as Vec index
    #[allow(clippy::as_conversions)]
    let idx = token_id as usize;
    let b = baseline_vec
        .get(idx)
        .ok_or_else(|| MIError::Intervention(format!("token ID {token_id} out of range")))?;
    let o = other_vec
        .get(idx)
        .ok_or_else(|| MIError::Intervention(format!("token ID {token_id} out of range")))?;
    Ok(b - o)
}

/// Compute top-k changed tokens between two logit tensors.
// CAST: usize → u32, vocab index fits in u32
#[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
fn top_changed_impl(
    baseline: &Tensor,
    other: &Tensor,
    k: usize,
) -> Result<Vec<(u32, f32, f32, f32)>> {
    let baseline_probs = softmax_to_vec(baseline)?;
    let other_probs = softmax_to_vec(other)?;

    let mut changes: Vec<(u32, f32, f32, f32)> = baseline_probs
        .iter()
        .zip(other_probs.iter())
        .enumerate()
        .map(|(idx, (&base, &oth))| (idx as u32, base, oth, (base - oth).abs()))
        .collect();

    changes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    Ok(changes.into_iter().take(k).collect())
}

// ===========================================================================
// Shared validation helpers
// ===========================================================================

/// Validate layer specification against model dimensions.
fn validate_layers(layers: &LayerSpec, n_layers: usize) -> Result<()> {
    match layers {
        LayerSpec::Specific(ls) => {
            for &l in ls {
                if l >= n_layers {
                    return Err(MIError::Intervention(format!(
                        "layer {l} out of range (model has {n_layers} layers)"
                    )));
                }
            }
        }
        LayerSpec::Range { start, end } => {
            if *end >= n_layers {
                return Err(MIError::Intervention(format!(
                    "layer range end {end} out of range (model has {n_layers} layers)"
                )));
            }
            if start > end {
                return Err(MIError::Intervention(format!(
                    "invalid layer range: start {start} > end {end}"
                )));
            }
        }
        LayerSpec::All => {}
    }
    Ok(())
}

/// Validate head specification against model dimensions.
fn validate_heads(heads: &HeadSpec, n_heads: usize) -> Result<()> {
    if let HeadSpec::Specific(hs) = heads {
        for &h in hs {
            if h >= n_heads {
                return Err(MIError::Intervention(format!(
                    "head {h} out of range (model has {n_heads} heads)"
                )));
            }
        }
    }
    Ok(())
}

/// Validate edge positions against sequence length.
fn validate_edges(edges: &[AttentionEdge], seq_len: usize) -> Result<()> {
    for edge in edges {
        if edge.from_pos != usize::MAX && edge.from_pos >= seq_len {
            return Err(MIError::Intervention(format!(
                "edge from_pos {} out of range (seq_len is {seq_len})",
                edge.from_pos,
            )));
        }
        if edge.to_pos != usize::MAX && edge.to_pos >= seq_len {
            return Err(MIError::Intervention(format!(
                "edge to_pos {} out of range (seq_len is {seq_len})",
                edge.to_pos,
            )));
        }
    }
    Ok(())
}

// ===========================================================================
// Mask creation and steering application
// ===========================================================================

/// Create a knockout mask tensor for the given specification.
///
/// Returns a tensor of shape `[1, n_heads, seq_len, seq_len]` where:
/// - `0.0` = no knockout (attention allowed)
/// - `-inf` = knockout (attention blocked)
///
/// This mask is ADDED to the attention scores before softmax.
///
/// # Shapes
///
/// - returns: `[1, n_heads, seq_len, seq_len]`
///
/// # Errors
///
/// Returns [`MIError::Model`] if tensor creation fails.
#[allow(clippy::indexing_slicing)] // Bounds checked via edge.from_pos < seq_len
pub fn create_knockout_mask(
    spec: &KnockoutSpec,
    n_heads: usize,
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; n_heads * seq_len * seq_len];
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    for head in 0..n_heads {
        if !spec.applies_to_head(head) {
            continue;
        }

        for edge in &expanded_edges {
            if edge.from_pos < seq_len && edge.to_pos < seq_len {
                let idx = head * seq_len * seq_len + edge.from_pos * seq_len + edge.to_pos;
                mask_data[idx] = f32::NEG_INFINITY;
            }
        }
    }

    let mask = Tensor::from_vec(mask_data, (1, n_heads, seq_len, seq_len), device)?;
    Ok(mask.to_dtype(dtype)?)
}

/// Compute KL divergence between two logit tensors.
///
/// Returns `KL(P || Q)` where `P = softmax(baseline)`, `Q = softmax(other)`.
///
/// # Errors
///
/// Returns [`MIError::Model`] if tensor operations fail.
pub fn kl_divergence(baseline_logits: &Tensor, other_logits: &Tensor) -> Result<f32> {
    let p = softmax_to_vec(baseline_logits)?;
    let q = softmax_to_vec(other_logits)?;

    let kl: f32 = p
        .iter()
        .zip(q.iter())
        .filter(|&(&pi, &qi)| pi > 1e-10 && qi > 1e-10)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum();

    Ok(kl)
}

/// Apply steering intervention to attention weights (post-softmax).
///
/// Modifies attention weights according to the steering spec and
/// renormalizes rows to maintain valid probability distributions.
///
/// # Shapes
///
/// - `attn_weights`: `[batch, heads, seq, seq]`
/// - returns: `[batch, heads, seq, seq]`
///
/// # Errors
///
/// Returns [`MIError::Intervention`] if the spec uses knockout
/// (which should use [`create_knockout_mask`] instead).
pub fn apply_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    n_heads: usize,
    seq_len: usize,
) -> Result<Tensor> {
    match spec.intervention_type {
        InterventionType::Scale(factor) => {
            apply_scale_steering(attn_weights, spec, n_heads, seq_len, factor)
        }
        InterventionType::SetValue(target) => {
            apply_set_value_steering(attn_weights, spec, n_heads, seq_len, target)
        }
        InterventionType::Knockout => Err(MIError::Intervention(
            "knockout should use create_knockout_mask, not apply_steering".into(),
        )),
    }
}

/// Apply scaling to specified edges, then renormalize rows.
///
/// # Shapes
///
/// - `attn_weights`: `[batch, heads, seq, seq]`
/// - returns: `[batch, heads, seq, seq]`
///
/// # Errors
///
/// Returns [`MIError::Intervention`] on tensor extraction failures.
#[allow(clippy::indexing_slicing)] // Operating on extracted Vecs with validated bounds
pub fn apply_scale_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    _n_heads: usize,
    seq_len: usize,
    scale_factor: f32,
) -> Result<Tensor> {
    // PROMOTE: needs f32 for numerical manipulation
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let original_dtype = attn_weights.dtype();
    let device = attn_weights.device();

    let mut data = tensor_to_vec4(&attn_f32)?;
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    for batch_data in &mut data {
        for (h, head_data) in batch_data.iter_mut().enumerate() {
            if !spec.applies_to_head(h) {
                continue;
            }

            let mut rows_modified: HashSet<usize> = HashSet::new();

            for edge in &expanded_edges {
                if edge.from_pos < seq_len && edge.to_pos < seq_len {
                    head_data[edge.from_pos][edge.to_pos] *= scale_factor;
                    rows_modified.insert(edge.from_pos);
                }
            }

            for row in rows_modified {
                let row_sum: f32 = head_data[row].iter().sum();
                if row_sum > 1e-10 {
                    for val in &mut head_data[row] {
                        *val /= row_sum;
                    }
                }
            }
        }
    }

    let result = Tensor::new(data, device)?.to_dtype(original_dtype)?;
    Ok(result)
}

/// Set specified edges to a target value, redistributing mass.
///
/// # Shapes
///
/// - `attn_weights`: `[batch, heads, seq, seq]`
/// - returns: `[batch, heads, seq, seq]`
///
/// # Errors
///
/// Returns [`MIError::Intervention`] on tensor extraction failures.
// CAST: usize → f32, small counts (seq_len, edge count) fit in f32
#[allow(
    clippy::indexing_slicing, // Operating on extracted Vecs with validated bounds
    clippy::cast_precision_loss,
    clippy::as_conversions,
)]
pub fn apply_set_value_steering(
    attn_weights: &Tensor,
    spec: &SteeringSpec,
    _n_heads: usize,
    seq_len: usize,
    target_value: f32,
) -> Result<Tensor> {
    // PROMOTE: needs f32 for numerical manipulation
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let original_dtype = attn_weights.dtype();
    let device = attn_weights.device();

    let mut data = tensor_to_vec4(&attn_f32)?;
    let expanded_edges = expand_edges(&spec.edges, seq_len);

    // Group edges by row for efficient row-wise operations.
    let mut edges_by_row: HashMap<usize, Vec<usize>> = HashMap::new();
    for edge in &expanded_edges {
        if edge.from_pos < seq_len && edge.to_pos < seq_len {
            edges_by_row
                .entry(edge.from_pos)
                .or_default()
                .push(edge.to_pos);
        }
    }

    for batch_data in &mut data {
        for (h, head_data) in batch_data.iter_mut().enumerate() {
            if !spec.applies_to_head(h) {
                continue;
            }

            for (&row, target_cols) in &edges_by_row {
                let current_target_sum: f32 =
                    target_cols.iter().map(|&col| head_data[row][col]).sum();
                // CAST: usize → f32, target-column count is small and fits in f32
                let new_target_sum = target_value * target_cols.len() as f32;
                let delta = new_target_sum - current_target_sum;

                let non_target_cols: Vec<usize> =
                    (0..seq_len).filter(|i| !target_cols.contains(i)).collect();

                for &col in target_cols {
                    head_data[row][col] = target_value;
                }

                if !non_target_cols.is_empty() {
                    // CAST: usize → f32, non-target-column count is small and fits in f32
                    let adjustment = delta / non_target_cols.len() as f32;
                    for col in non_target_cols {
                        head_data[row][col] = (head_data[row][col] - adjustment).max(0.0);
                    }
                }

                let row_sum: f32 = head_data[row].iter().sum();
                if row_sum > 1e-10 {
                    for val in &mut head_data[row] {
                        *val /= row_sum;
                    }
                }
            }
        }
    }

    let result = Tensor::new(data, device)?.to_dtype(original_dtype)?;
    Ok(result)
}

/// Measure mean attention for specified edges in an attention tensor.
///
/// # Shapes
///
/// - `attn_weights`: `[batch, heads, seq, seq]`
///
/// # Errors
///
/// Returns [`MIError::Intervention`] if `from_pos` is out of range.
#[allow(clippy::indexing_slicing)] // Operating on extracted Vecs with validated bounds
pub fn measure_attention_to_targets(
    attn_weights: &Tensor,
    from_pos: usize,
    to_positions: &[usize],
) -> Result<f32> {
    let attn_f32 = attn_weights.to_dtype(DType::F32)?;
    let data = tensor_to_vec4(&attn_f32)?;

    let seq_len = data.first().and_then(|b| b.first()).map_or(0, Vec::len);

    if from_pos >= seq_len {
        return Err(MIError::Intervention(format!(
            "from_pos {from_pos} out of range (seq_len is {seq_len})"
        )));
    }

    let mut total = 0.0_f32;
    let mut count = 0_usize;

    for batch_data in &data {
        for head_data in batch_data {
            for &to_pos in to_positions {
                if to_pos < seq_len {
                    total += head_data[from_pos][to_pos];
                    count += 1;
                }
            }
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        // CAST: usize → f32, count is number of matching heads — fits in f32
        #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
        Ok(total / count as f32)
    }
}

// ===========================================================================
// Part 3: State Knockout (RWKV-6)
// ===========================================================================

/// Specification for RWKV-6 state knockout intervention.
///
/// State knockout makes specific token positions invisible to all future
/// tokens by skipping the recurrent state update at those positions.
/// This is the RNN analogue of all-edge attention knockout in transformers.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[must_use]
pub struct StateKnockoutSpec {
    /// Token positions where state update is skipped.
    pub positions: Vec<usize>,
    /// Which layers to apply knockout.
    pub layers: LayerSpec,
}

impl StateKnockoutSpec {
    /// Create a new empty spec (all layers, no positions yet).
    pub const fn new() -> Self {
        Self {
            positions: Vec::new(),
            layers: LayerSpec::All,
        }
    }

    /// Add a single position to knock out.
    pub fn position(mut self, pos: usize) -> Self {
        self.positions.push(pos);
        self
    }

    /// Add multiple positions to knock out.
    pub fn positions(mut self, positions: &[usize]) -> Self {
        self.positions.extend_from_slice(positions);
        self
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Check if knockout applies to this layer.
    #[must_use]
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Get knockout positions as a `HashSet` for O(1) lookup in the WKV loop.
    #[must_use]
    pub fn position_set(&self) -> HashSet<usize> {
        self.positions.iter().copied().collect()
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if positions, layers are out of
    /// range, or no positions are specified.
    pub fn validate(&self, n_layers: usize, seq_len: usize) -> Result<()> {
        validate_layers(&self.layers, n_layers)?;

        for &pos in &self.positions {
            if pos >= seq_len {
                return Err(MIError::Intervention(format!(
                    "position {pos} out of range (seq_len is {seq_len})"
                )));
            }
        }

        if self.positions.is_empty() {
            return Err(MIError::Intervention(
                "StateKnockoutSpec has no positions specified".into(),
            ));
        }

        Ok(())
    }
}

impl Default for StateKnockoutSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a state knockout ablation experiment (RWKV-6).
#[non_exhaustive]
#[derive(Debug)]
pub struct StateAblationResult {
    /// Logits from baseline forward pass (no intervention).
    pub baseline_logits: Tensor,
    /// Logits from state-knocked-out forward pass.
    pub ablated_logits: Tensor,
    /// The state knockout specification used.
    pub spec: StateKnockoutSpec,
}

impl StateAblationResult {
    /// Create a new state ablation result.
    #[must_use]
    pub const fn new(
        baseline_logits: Tensor,
        ablated_logits: Tensor,
        spec: StateKnockoutSpec,
    ) -> Self {
        Self {
            baseline_logits,
            ablated_logits,
            spec,
        }
    }

    /// KL divergence between baseline and ablated distributions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.ablated_logits)
    }

    /// Logit difference for a specific token.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if `token_id` is out of range.
    pub fn logit_diff(&self, token_id: u32) -> Result<f32> {
        logit_diff_impl(&self.baseline_logits, &self.ablated_logits, token_id)
    }

    /// Top-k tokens that changed most due to state knockout.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        top_changed_impl(&self.baseline_logits, &self.ablated_logits, k)
    }
}

// ===========================================================================
// Part 4: State Steering (RWKV-6)
// ===========================================================================

/// Specification for RWKV-6 state steering intervention.
///
/// State steering scales the kv write at specified positions, amplifying
/// or dampening the token's contribution to recurrent state.
///
/// - `scale = 0.0` → knockout (equivalent to [`StateKnockoutSpec`])
/// - `scale = 1.0` → no-op (normal forward pass)
/// - `scale > 1.0` → amplify the token's state write
/// - `scale < 1.0` → dampen the token's state write
#[non_exhaustive]
#[derive(Debug, Clone)]
#[must_use]
pub struct StateSteeringSpec {
    /// Token positions where state write is scaled.
    pub positions: Vec<usize>,
    /// Which layers to apply steering.
    pub layers: LayerSpec,
    /// Scale factor for kv write.
    pub scale: f32,
}

impl StateSteeringSpec {
    /// Create a new spec with the given scale factor (all layers, no positions).
    pub const fn new(scale: f32) -> Self {
        Self {
            positions: Vec::new(),
            layers: LayerSpec::All,
            scale,
        }
    }

    /// Add a single position to steer.
    pub fn position(mut self, pos: usize) -> Self {
        self.positions.push(pos);
        self
    }

    /// Add multiple positions to steer.
    pub fn positions(mut self, positions: &[usize]) -> Self {
        self.positions.extend_from_slice(positions);
        self
    }

    /// Target a single layer.
    pub fn layer(mut self, layer: usize) -> Self {
        self.layers = LayerSpec::Specific(vec![layer]);
        self
    }

    /// Target multiple specific layers.
    pub fn layers(mut self, layers: &[usize]) -> Self {
        self.layers = LayerSpec::Specific(layers.to_vec());
        self
    }

    /// Target a range of layers (inclusive).
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layers = LayerSpec::Range { start, end };
        self
    }

    /// Check if steering applies to this layer.
    #[must_use]
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        match &self.layers {
            LayerSpec::All => true,
            LayerSpec::Specific(layers) => layers.contains(&layer),
            LayerSpec::Range { start, end } => layer >= *start && layer <= *end,
        }
    }

    /// Get steering positions as a `HashSet` for O(1) lookup in the WKV loop.
    #[must_use]
    pub fn position_set(&self) -> HashSet<usize> {
        self.positions.iter().copied().collect()
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if positions or layers are out of
    /// range, or no positions are specified.
    pub fn validate(&self, n_layers: usize, seq_len: usize) -> Result<()> {
        validate_layers(&self.layers, n_layers)?;

        for &pos in &self.positions {
            if pos >= seq_len {
                return Err(MIError::Intervention(format!(
                    "position {pos} out of range (seq_len is {seq_len})"
                )));
            }
        }

        if self.positions.is_empty() {
            return Err(MIError::Intervention(
                "StateSteeringSpec has no positions specified".into(),
            ));
        }

        Ok(())
    }
}

/// Result of a state steering experiment (RWKV-6).
#[non_exhaustive]
#[derive(Debug)]
pub struct StateSteeringResult {
    /// Logits from baseline forward pass (no intervention).
    pub baseline_logits: Tensor,
    /// Logits from the steered forward pass.
    pub steered_logits: Tensor,
    /// The state steering specification used.
    pub spec: StateSteeringSpec,
}

impl StateSteeringResult {
    /// Create a new state steering result.
    #[must_use]
    pub const fn new(
        baseline_logits: Tensor,
        steered_logits: Tensor,
        spec: StateSteeringSpec,
    ) -> Self {
        Self {
            baseline_logits,
            steered_logits,
            spec,
        }
    }

    /// KL divergence between baseline and steered distributions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.steered_logits)
    }

    /// Top-k tokens that changed most due to state steering.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        top_changed_impl(&self.baseline_logits, &self.steered_logits, k)
    }
}

// ===========================================================================
// Part 5: CLT Injection (feature-gated)
// ===========================================================================

/// Pre-accumulated CLT injection vectors for per-layer residual stream injection.
///
/// Created by the CLT encoder's `prepare_injection()` method. The forward
/// pass adds each vector to the residual at the specified position after the
/// target layer completes.
#[cfg(feature = "clt")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CltInjectionSpec {
    /// Per-layer injection entries.
    pub injections: Vec<CltLayerInjection>,
}

/// A single CLT injection at one layer and position.
#[cfg(feature = "clt")]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CltLayerInjection {
    /// Target layer index (injection happens after this layer completes).
    pub target_layer: usize,
    /// Token position in the sequence to inject at.
    pub position: usize,
    /// Pre-accumulated and strength-scaled decoder vector, shape `[d_model]`.
    pub vector: Tensor,
}

#[cfg(feature = "clt")]
impl CltInjectionSpec {
    /// Create an empty injection spec.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            injections: Vec::new(),
        }
    }

    /// Add a single injection entry.
    pub fn add(&mut self, target_layer: usize, position: usize, vector: Tensor) {
        self.injections.push(CltLayerInjection {
            target_layer,
            position,
            vector,
        });
    }

    /// Check if any injection targets this layer.
    #[must_use]
    pub fn applies_to_layer(&self, layer: usize) -> bool {
        self.injections.iter().any(|inj| inj.target_layer == layer)
    }

    /// Get all injections targeting a specific layer.
    #[must_use]
    pub fn injections_for_layer(&self, layer: usize) -> Vec<&CltLayerInjection> {
        self.injections
            .iter()
            .filter(|inj| inj.target_layer == layer)
            .collect()
    }

    /// Validate the spec against model dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Intervention`] if any target layer, position,
    /// or vector dimension is out of range.
    pub fn validate(&self, n_layers: usize, seq_len: usize, d_model: usize) -> Result<()> {
        for inj in &self.injections {
            let target = inj.target_layer;
            if target >= n_layers {
                return Err(MIError::Intervention(format!(
                    "CLT injection target layer {target} out of range (model has {n_layers} layers)"
                )));
            }
            let pos = inj.position;
            if pos >= seq_len {
                return Err(MIError::Intervention(format!(
                    "CLT injection position {pos} out of range (seq_len={seq_len})"
                )));
            }
            let vec_dim = inj.vector.dim(0)?;
            if vec_dim != d_model {
                return Err(MIError::Intervention(format!(
                    "CLT injection vector dim {vec_dim} doesn't match model d_model={d_model}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "clt")]
impl Default for CltInjectionSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a CLT logit shift test (baseline vs. injected comparison).
#[cfg(feature = "clt")]
#[non_exhaustive]
#[derive(Debug)]
pub struct CltLogitShiftResult {
    /// Logits from baseline forward pass (no injection).
    pub baseline_logits: Tensor,
    /// Logits from CLT-injected forward pass.
    pub injected_logits: Tensor,
}

#[cfg(feature = "clt")]
impl CltLogitShiftResult {
    /// Create a new CLT logit shift result.
    #[must_use]
    pub const fn new(baseline_logits: Tensor, injected_logits: Tensor) -> Self {
        Self {
            baseline_logits,
            injected_logits,
        }
    }

    /// KL divergence between baseline and injected distributions.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn kl_divergence(&self) -> Result<f32> {
        kl_divergence(&self.baseline_logits, &self.injected_logits)
    }

    /// Top-k tokens that changed most due to CLT injection.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Model`] if tensor operations fail.
    pub fn top_changed_tokens(&self, k: usize) -> Result<Vec<(u32, f32, f32, f32)>> {
        top_changed_impl(&self.baseline_logits, &self.injected_logits, k)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    #[test]
    fn knockout_spec_builder() {
        let spec = KnockoutSpec::new()
            .layer(5)
            .head(2)
            .edge(3, 1)
            .from_to_positions(4, &[0, 1, 2]);

        assert!(matches!(spec.layers, LayerSpec::Specific(_)));
        assert!(matches!(spec.heads, HeadSpec::Specific(_)));
        assert_eq!(spec.edges.len(), 4); // 1 + 3
    }

    #[test]
    fn layer_spec_applies() {
        let spec = KnockoutSpec::new().layer_range(5, 10);

        assert!(!spec.applies_to_layer(4));
        assert!(spec.applies_to_layer(5));
        assert!(spec.applies_to_layer(7));
        assert!(spec.applies_to_layer(10));
        assert!(!spec.applies_to_layer(11));
    }

    #[test]
    fn expand_edges_sentinels() {
        let edges = vec![AttentionEdge::new(2, usize::MAX), AttentionEdge::new(1, 0)];

        let expanded = expand_edges(&edges, 4);
        assert_eq!(expanded.len(), 5); // 4 from sentinel + 1 specific
    }

    #[test]
    fn create_knockout_mask_correctness() {
        let spec = KnockoutSpec::new().head(0).edge(2, 1);

        let mask = create_knockout_mask(&spec, 2, 4, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(mask.dims(), &[1, 2, 4, 4]);

        let mask_vec: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Head 0, row 2, col 1 = index 0*16 + 2*4 + 1 = 9
        assert!(mask_vec[9].is_infinite() && mask_vec[9].is_sign_negative());

        // Head 1 should not be affected (index 1*16 + 2*4 + 1 = 25)
        assert_eq!(mask_vec[25], 0.0);
    }

    #[test]
    fn validation_catches_errors() {
        let spec = KnockoutSpec::new().layer(100).edge(50, 25);
        assert!(spec.validate(30, 16, 20).is_err());
    }

    #[test]
    fn validation_passes_valid() {
        let spec = KnockoutSpec::new().layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_ok());
    }

    #[test]
    fn steering_spec_builder() {
        let spec = SteeringSpec::scale(2.0)
            .layer(5)
            .head(2)
            .edge(3, 1)
            .from_to_positions(4, &[0, 1, 2]);

        assert!(matches!(spec.layers, LayerSpec::Specific(_)));
        assert!(matches!(spec.heads, HeadSpec::Specific(_)));
        assert_eq!(spec.edges.len(), 4);
        assert!(
            matches!(spec.intervention_type, InterventionType::Scale(f) if (f - 2.0).abs() < 1e-6)
        );
    }

    #[test]
    fn steering_validation() {
        let spec = SteeringSpec::scale(2.0).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_ok());

        let spec = SteeringSpec::scale(-1.0).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_err());

        let spec = SteeringSpec::set_value(0.09).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_ok());

        let spec = SteeringSpec::set_value(1.5).layer(10).edge(5, 3);
        assert!(spec.validate(30, 16, 20).is_err());
    }

    #[test]
    fn steering_is_methods() {
        let knockout = SteeringSpec::new(InterventionType::Knockout);
        assert!(knockout.is_knockout());
        assert!(!knockout.is_steering());

        let scale = SteeringSpec::scale(2.0);
        assert!(!scale.is_knockout());
        assert!(scale.is_steering());

        let set_value = SteeringSpec::set_value(0.1);
        assert!(!set_value.is_knockout());
        assert!(set_value.is_steering());
    }

    #[test]
    fn apply_scale_steering_correctness() {
        let data: Vec<f32> = vec![
            // Head 0: uniform attention (each row sums to 1.0)
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, // Head 1: same
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25,
        ];
        let tensor = Tensor::from_vec(data, (1, 2, 4, 4), &Device::Cpu).unwrap();

        let spec = SteeringSpec::scale(2.0).edge(2, 1);
        let result = apply_scale_steering(&tensor, &spec, 2, 4, 2.0).unwrap();
        let result_data = tensor_to_vec4(&result).unwrap();

        // Row 2: edge (2,1) scaled by 2, then renormalized
        // Before: [0.25, 0.25, 0.25, 0.25]
        // After scaling: [0.25, 0.50, 0.25, 0.25], sum = 1.25
        // After renorm: [0.20, 0.40, 0.20, 0.20]
        let row2 = &result_data[0][0][2];
        assert!((row2[0] - 0.20).abs() < 1e-5);
        assert!((row2[1] - 0.40).abs() < 1e-5);
        assert!((row2[2] - 0.20).abs() < 1e-5);
        assert!((row2[3] - 0.20).abs() < 1e-5);

        let row_sum: f32 = row2.iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn apply_set_value_steering_correctness() {
        let data: Vec<f32> = vec![
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25,
        ];
        let tensor = Tensor::from_vec(data, (1, 2, 4, 4), &Device::Cpu).unwrap();

        let spec = SteeringSpec::set_value(0.5).edge(2, 1);
        let result = apply_set_value_steering(&tensor, &spec, 2, 4, 0.5).unwrap();
        let result_data = tensor_to_vec4(&result).unwrap();

        let row2 = &result_data[0][0][2];
        let row_sum: f32 = row2.iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row sum should be 1.0, got {row_sum}"
        );

        // Edge (2,1) should be the largest value
        assert!(row2[1] > row2[0]);
        assert!(row2[1] > row2[2]);
        assert!(row2[1] > row2[3]);
    }

    #[test]
    fn knockout_to_steering_conversion() {
        let knockout = KnockoutSpec::new().layer(5).head(2).edge(3, 1);
        let steering: SteeringSpec = knockout.into();

        assert!(matches!(steering.layers, LayerSpec::Specific(ref v) if v == &[5]));
        assert!(matches!(steering.heads, HeadSpec::Specific(ref v) if v == &[2]));
        assert_eq!(steering.edges.len(), 1);
        assert!(steering.is_knockout());
    }

    #[test]
    fn is_prompt_only() {
        let spec = SteeringSpec::scale(2.0).edge(5, 2).edge(8, 3);
        assert!(spec.is_prompt_only(10));
        assert!(!spec.is_prompt_only(6));
    }

    #[test]
    fn is_prompt_only_with_sentinel() {
        let spec = SteeringSpec::scale(2.0).to_position(5);
        assert!(!spec.is_prompt_only(10));

        let spec2 = SteeringSpec::scale(2.0).from_position(5);
        assert!(spec2.is_prompt_only(10));
    }

    #[test]
    fn max_positions() {
        let spec = SteeringSpec::scale(2.0).edge(5, 2).edge(8, 3).edge(3, 7);
        assert_eq!(spec.max_from_pos(), Some(8));
        assert_eq!(spec.max_to_pos(), Some(7));
    }

    #[test]
    fn max_positions_empty() {
        let spec = SteeringSpec::scale(2.0);
        assert_eq!(spec.max_from_pos(), None);
        assert_eq!(spec.max_to_pos(), None);
    }

    // --- State knockout tests ---

    #[test]
    fn state_knockout_spec_builder() {
        let spec = StateKnockoutSpec::new().position(3).position(5).layer(10);
        assert_eq!(spec.positions, vec![3, 5]);
        assert!(matches!(spec.layers, LayerSpec::Specific(ref v) if v == &[10]));
    }

    #[test]
    fn state_knockout_validation() {
        assert!(
            StateKnockoutSpec::new()
                .position(5)
                .layer(10)
                .validate(24, 20)
                .is_ok()
        );
        assert!(
            StateKnockoutSpec::new()
                .position(25)
                .validate(24, 20)
                .is_err()
        );
        assert!(
            StateKnockoutSpec::new()
                .position(5)
                .layer(30)
                .validate(24, 20)
                .is_err()
        );
        assert!(StateKnockoutSpec::new().validate(24, 20).is_err()); // empty
    }

    #[test]
    fn state_knockout_position_set() {
        let spec = StateKnockoutSpec::new().position(3).position(5).position(3);
        let set = spec.position_set();
        assert_eq!(set.len(), 2); // deduplicated
        assert!(set.contains(&3));
        assert!(set.contains(&5));
    }

    #[test]
    fn state_knockout_layer_range() {
        let spec = StateKnockoutSpec::new().position(0).layer_range(5, 10);
        assert!(!spec.applies_to_layer(4));
        assert!(spec.applies_to_layer(5));
        assert!(spec.applies_to_layer(10));
        assert!(!spec.applies_to_layer(11));
    }
}
