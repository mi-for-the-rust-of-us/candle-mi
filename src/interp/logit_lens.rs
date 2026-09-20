// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logit lens: project hidden states to vocabulary at each layer.
//!
//! Projects activations from any layer through the final layer norm and
//! unembedding matrix to see what the model would predict at that layer.

/// Result of applying logit lens at a single layer.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogitLensResult {
    /// Layer index (0-indexed).
    pub layer: usize,
    /// Top-k token predictions with probabilities.
    pub predictions: Vec<TokenPrediction>,
}

impl LogitLensResult {
    /// Build a result for one layer.
    ///
    /// The type is `#[non_exhaustive]`, so a struct expression is rejected
    /// outside this crate; this is the construction path for callers that run
    /// their own lens and want to reuse the crate's reporting.
    #[must_use]
    pub const fn new(layer: usize, predictions: Vec<TokenPrediction>) -> Self {
        Self { layer, predictions }
    }
}

/// A single token prediction from logit lens analysis.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// Token ID in the vocabulary.
    pub token_id: u32,
    /// Decoded token string.
    pub token: String,
    /// Probability (0.0 to 1.0).
    pub probability: f32,
}

impl TokenPrediction {
    /// Build a single prediction.
    ///
    /// The type is `#[non_exhaustive]`, so a struct expression is rejected
    /// outside this crate; this is the construction path.
    #[must_use]
    pub const fn new(token_id: u32, token: String, probability: f32) -> Self {
        Self {
            token_id,
            token,
            probability,
        }
    }
}

/// Full logit lens analysis across all layers.
///
/// Collects per-layer predictions and provides summary methods
/// for identifying convergence and first appearance of tokens.
///
/// # Example
///
/// ```
/// use candle_mi::{LogitLensAnalysis, LogitLensResult, TokenPrediction};
///
/// let mut analysis = LogitLensAnalysis::new("fn main()".into(), 2);
/// analysis.push(LogitLensResult::new(
///     0,
///     vec![TokenPrediction::new(42, "main".into(), 0.8)],
/// ));
/// analysis.push(LogitLensResult::new(
///     1,
///     vec![TokenPrediction::new(42, "main".into(), 0.95)],
/// ));
/// let tops = analysis.top_predictions();
/// assert_eq!(tops.len(), 2);
/// ```
#[non_exhaustive]
#[derive(Debug)]
pub struct LogitLensAnalysis {
    /// Input text that was analyzed.
    pub input_text: String,
    /// Results for each layer.
    pub layer_results: Vec<LogitLensResult>,
    /// Number of layers analyzed.
    pub n_layers: usize,
}

impl LogitLensAnalysis {
    /// Create a new analysis with capacity for `n_layers` layers.
    #[must_use]
    pub fn new(input_text: String, n_layers: usize) -> Self {
        Self {
            input_text,
            layer_results: Vec::with_capacity(n_layers),
            n_layers,
        }
    }

    /// Add a layer's result.
    pub fn push(&mut self, result: LogitLensResult) {
        self.layer_results.push(result);
    }

    /// Get the top prediction at each layer.
    ///
    /// Returns `(token_str, probability)` for the highest-probability
    /// token at each analyzed layer.
    #[must_use]
    pub fn top_predictions(&self) -> Vec<(&str, f32)> {
        self.layer_results
            .iter()
            .filter_map(|r| r.predictions.first())
            // BORROW: explicit .as_str() — &str from String field
            .map(|p| (p.token.as_str(), p.probability))
            .collect()
    }

    /// Find at which layer a specific token first appears in top-k.
    ///
    /// Searches predictions using `contains()` on the token string.
    /// Returns `None` if the token never appears in the top-k at any layer.
    #[must_use]
    pub fn first_appearance(&self, token: &str, k: usize) -> Option<usize> {
        for result in &self.layer_results {
            let in_top_k = result
                .predictions
                .iter()
                .take(k)
                .any(|p| p.token.contains(token));
            if in_top_k {
                return Some(result.layer);
            }
        }
        None
    }

    /// Print a summary showing the top prediction at each layer.
    pub fn print_summary(&self) {
        println!("=== Logit Lens Analysis ===");
        println!("Input: {}", self.input_text);
        println!("\nTop prediction at each layer:");
        for result in &self.layer_results {
            if let Some(top) = result.predictions.first() {
                println!(
                    "  Layer {:2}: {:>12} ({})",
                    result.layer,
                    format!("\"{}\"", format_token(&top.token)),
                    format_probability(top.probability),
                );
            }
        }
    }

    /// Print detailed predictions for each layer (top-k per layer).
    pub fn print_detailed(&self, top_k: usize) {
        println!("=== Logit Lens Detailed Analysis ===");
        println!("Input: {}", self.input_text);
        for result in &self.layer_results {
            println!("\nLayer {}:", result.layer);
            for (i, pred) in result.predictions.iter().take(top_k).enumerate() {
                println!(
                    "  {}. {:>15} ({})",
                    i + 1,
                    format!("\"{}\"", format_token(&pred.token)),
                    format_probability(pred.probability),
                );
            }
        }
    }
}

/// Decode token IDs to [`TokenPrediction`] using a decode function.
///
/// Generic over any tokenizer — the caller provides a closure that
/// maps `token_id → String`.
///
/// # Example
///
/// ```
/// use candle_mi::interp::logit_lens::decode_predictions_with;
///
/// let preds = decode_predictions_with(&[(42, 0.7), (99, 0.2)], |id| {
///     format!("token_{id}")
/// });
/// assert_eq!(preds.len(), 2);
/// assert_eq!(preds[0].token, "token_42");
/// ```
#[must_use]
pub fn decode_predictions_with(
    predictions: &[(u32, f32)],
    decode_fn: impl Fn(u32) -> String,
) -> Vec<TokenPrediction> {
    predictions
        .iter()
        .map(|&(token_id, prob)| {
            let token = decode_fn(token_id);
            TokenPrediction {
                token_id,
                token,
                probability: prob,
            }
        })
        .collect()
}

/// Format a token for display, escaping whitespace characters.
#[must_use]
pub fn format_token(token: &str) -> String {
    token
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

/// Format a probability for display with adaptive precision.
///
/// - ≥ 1%: `"12.3%"` (1 decimal place)
/// - ≥ 0.01%: `"0.123%"` (3 decimal places)
/// - < 0.01%: `"1.2e-4%"` (scientific notation)
#[must_use]
pub fn format_probability(prob: f32) -> String {
    let pct = prob * 100.0;
    if pct >= 1.0 {
        format!("{pct:.1}%")
    } else if pct >= 0.01 {
        format!("{pct:.3}%")
    } else {
        format!("{pct:.1e}%")
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
    fn logit_lens_result_basic() {
        let result = LogitLensResult {
            layer: 0,
            predictions: vec![
                TokenPrediction {
                    token_id: 1,
                    token: "fn".to_string(),
                    probability: 0.5,
                },
                TokenPrediction {
                    token_id: 2,
                    token: "def".to_string(),
                    probability: 0.3,
                },
            ],
        };

        assert_eq!(result.layer, 0);
        assert_eq!(result.predictions.len(), 2);
        assert_eq!(result.predictions.first().unwrap().token, "fn");
    }

    #[test]
    fn first_appearance_found() {
        let mut analysis = LogitLensAnalysis::new("test".to_string(), 3);
        analysis.push(LogitLensResult {
            layer: 0,
            predictions: vec![TokenPrediction {
                token_id: 1,
                token: "a".to_string(),
                probability: 0.5,
            }],
        });
        analysis.push(LogitLensResult {
            layer: 1,
            predictions: vec![TokenPrediction {
                token_id: 2,
                token: "#[test]".to_string(),
                probability: 0.5,
            }],
        });

        assert_eq!(analysis.first_appearance("#[test]", 1), Some(1));
        assert_eq!(analysis.first_appearance("notfound", 1), None);
    }

    #[test]
    fn decode_predictions_with_custom_fn() {
        let preds = decode_predictions_with(&[(1, 0.5), (2, 0.3)], |id| format!("tok_{id}"));

        assert_eq!(preds.len(), 2);
        assert_eq!(preds.first().unwrap().token, "tok_1");
        assert_eq!(preds.first().unwrap().token_id, 1);
    }

    #[test]
    fn format_token_escapes() {
        assert_eq!(format_token("hello\nworld"), "hello\\nworld");
        assert_eq!(format_token("tab\there"), "tab\\there");
        assert_eq!(format_token("no_escapes"), "no_escapes");
    }

    #[test]
    fn top_predictions_empty() {
        let analysis = LogitLensAnalysis::new("test".to_string(), 0);
        assert!(analysis.top_predictions().is_empty());
    }
}
