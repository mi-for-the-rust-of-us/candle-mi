// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic character-based position handling.
//!
//! Provides universal position handling using character offsets instead of
//! model-specific token indices.  This enables:
//!
//! - **One corpus for all models**: No model-specific corpus files needed
//! - **Zero preprocessing**: Any new model works immediately
//! - **Guaranteed accuracy**: No offset heuristics, direct character mapping
//!
//! ## How It Works
//!
//! 1. Corpus stores character positions (byte offsets into the text)
//! 2. At runtime, tokenize with offset mapping
//! 3. Convert character positions to token indices using the offset map

/// Token with its character offset range.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TokenWithOffset {
    /// The token string.
    pub token: String,
    /// Start character position (byte offset).
    pub start: usize,
    /// End character position (byte offset, exclusive).
    pub end: usize,
}

/// Encoding result with tokens and their character offsets.
///
/// Produced by a tokenizer's `encode_with_offsets` method (or equivalent).
/// Used to map between character positions in source text and token indices.
///
/// # Example
///
/// ```
/// use candle_mi::EncodingWithOffsets;
///
/// let encoding = EncodingWithOffsets::new(
///     vec![1, 2, 3],
///     vec!["def".into(), " ".into(), "add".into()],
///     vec![(0, 3), (3, 4), (4, 7)],
/// );
///
/// // Character 4 ('a' in "add") is in token 2
/// assert_eq!(encoding.char_to_token(4), Some(2));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EncodingWithOffsets {
    /// Token IDs.
    pub ids: Vec<u32>,
    /// Token strings.
    pub tokens: Vec<String>,
    /// Character offset for each token: `(start, end)`.
    pub offsets: Vec<(usize, usize)>,
}

impl EncodingWithOffsets {
    /// Create a new encoding with offsets.
    #[must_use]
    pub const fn new(ids: Vec<u32>, tokens: Vec<String>, offsets: Vec<(usize, usize)>) -> Self {
        Self {
            ids,
            tokens,
            offsets,
        }
    }

    /// Get tokens with their character offsets.
    #[must_use]
    pub fn tokens_with_offsets(&self) -> Vec<TokenWithOffset> {
        self.tokens
            .iter()
            .zip(self.offsets.iter())
            .map(|(token, (start, end))| TokenWithOffset {
                token: token.clone(),
                start: *start,
                end: *end,
            })
            .collect()
    }

    /// Find the token index that contains the given character position.
    ///
    /// Returns `None` if no token spans that position.
    #[must_use]
    pub fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, end)| char_pos >= *start && char_pos < *end)
    }

    /// Find the token index for a character position, with fuzzy fallback.
    ///
    /// If the exact position isn't contained in any token, returns the
    /// index of the closest token by midpoint distance.
    #[must_use]
    pub fn char_to_token_fuzzy(&self, char_pos: usize) -> Option<usize> {
        // Try exact match first.
        if let Some(idx) = self.char_to_token(char_pos) {
            return Some(idx);
        }

        // Find closest token by midpoint distance.
        self.offsets
            .iter()
            .enumerate()
            .min_by_key(|(_, (start, end))| {
                let mid = usize::midpoint(*start, *end);
                char_pos.abs_diff(mid)
            })
            .map(|(idx, _)| idx)
    }

    /// Find the token index that starts at or after the given character position.
    #[must_use]
    pub fn char_to_token_start(&self, char_pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, _)| *start >= char_pos)
    }

    /// Find all token indices that overlap with the given character range.
    #[must_use]
    pub fn char_range_to_tokens(&self, start_char: usize, end_char: usize) -> Vec<usize> {
        self.offsets
            .iter()
            .enumerate()
            .filter_map(|(idx, (start, end))| {
                if *end > start_char && *start < end_char {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the character range for a token index.
    #[must_use]
    pub fn token_to_char_range(&self, token_idx: usize) -> Option<(usize, usize)> {
        self.offsets.get(token_idx).copied()
    }

    /// Number of tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the encoding is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Label each token by which named span it overlaps with.
    ///
    /// For each token, finds the first span (by input order) whose byte
    /// range overlaps the token's byte range. The last token matching
    /// each span label gets `"_final"` appended. Tokens matching no
    /// span receive `"other"`.
    ///
    /// # Example
    ///
    /// ```
    /// use candle_mi::EncodingWithOffsets;
    ///
    /// let enc = EncodingWithOffsets::new(
    ///     vec![1, 2, 3, 4],
    ///     vec!["The".into(), " Eiffel".into(), " Tower".into(), " is".into()],
    ///     vec![(0, 3), (3, 10), (10, 16), (16, 19)],
    /// );
    /// let labels = enc.label_spans(&[("subject", 0..16), ("relation", 16..19)]);
    /// assert_eq!(labels, vec!["subject", "subject", "subject_final", "relation_final"]);
    /// ```
    #[must_use]
    pub fn label_spans(&self, spans: &[(&str, std::ops::Range<usize>)]) -> Vec<String> {
        let mut labels: Vec<String> = self
            .offsets
            .iter()
            .map(|&(tok_start, tok_end)| {
                // BOS or special tokens with zero-width offsets match nothing
                if tok_start == tok_end {
                    return String::from("other");
                }
                // First span whose byte range overlaps this token wins
                for (name, range) in spans {
                    if tok_end > range.start && tok_start < range.end {
                        return String::from(*name);
                    }
                }
                String::from("other")
            })
            .collect();

        // For each unique span label, upgrade the last occurrence to "{label}_final"
        for (name, _) in spans {
            if let Some(last_idx) = labels.iter().rposition(|l| l.as_str() == *name) {
                // INDEX: last_idx came from rposition on labels, bounded by labels.len()
                #[allow(clippy::indexing_slicing)]
                {
                    labels[last_idx] = format!("{name}_final");
                }
            }
        }

        labels
    }
}

/// Result of converting a character position to a token index.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PositionConversion {
    /// Original character position.
    pub char_pos: usize,
    /// Converted token index (if found).
    pub token_idx: Option<usize>,
    /// The token at that position (if found).
    pub token: Option<String>,
    /// Whether this was an exact match or fuzzy.
    pub exact_match: bool,
}

/// Convert multiple character positions to token indices.
#[must_use]
pub fn convert_positions(
    encoding: &EncodingWithOffsets,
    char_positions: &[usize],
) -> Vec<PositionConversion> {
    char_positions
        .iter()
        .map(|&char_pos| {
            let exact = encoding.char_to_token(char_pos);
            let (token_idx, exact_match) = if exact.is_some() {
                (exact, true)
            } else {
                (encoding.char_to_token_fuzzy(char_pos), false)
            };

            let token = token_idx.and_then(|idx| encoding.tokens.get(idx).cloned());

            PositionConversion {
                char_pos,
                token_idx,
                token,
                exact_match,
            }
        })
        .collect()
}

/// Find the character position of a marker pattern in text.
///
/// Returns the byte offset of the first occurrence of `marker` in `text`,
/// or `None` if not found.
#[cfg(test)]
#[must_use]
fn find_marker_char_pos(text: &str, marker: &str) -> Option<usize> {
    text.find(marker)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample_encoding() -> EncodingWithOffsets {
        // Simulates tokenization of "def add(a, b):"
        EncodingWithOffsets::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![
                "def".into(),
                " ".into(),
                "add".into(),
                "(".into(),
                "a".into(),
                ",".into(),
                " ".into(),
                "b".into(),
            ],
            vec![
                (0, 3),
                (3, 4),
                (4, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
            ],
        )
    }

    #[test]
    fn char_to_token_exact() {
        let encoding = sample_encoding();

        // 'd' at position 0 → token 0 ("def")
        assert_eq!(encoding.char_to_token(0), Some(0));
        // 'a' in "add" at position 4 → token 2
        assert_eq!(encoding.char_to_token(4), Some(2));
        // Parameter 'a' at position 8 → token 4
        assert_eq!(encoding.char_to_token(8), Some(4));
        // Beyond all tokens
        assert_eq!(encoding.char_to_token(100), None);
    }

    #[test]
    fn char_to_token_fuzzy_fallback() {
        let encoding = sample_encoding();

        // Position 12 is beyond all tokens → fuzzy finds closest
        let result = encoding.char_to_token_fuzzy(12);
        assert!(result.is_some());
    }

    #[test]
    fn char_range_to_tokens_overlap() {
        let encoding = sample_encoding();

        // Characters 3..7 overlap tokens: " " (3,4), "add" (4,7)
        let tokens = encoding.char_range_to_tokens(3, 7);
        assert_eq!(tokens, vec![1, 2]);
    }

    #[test]
    fn token_to_char_range_roundtrip() {
        let encoding = sample_encoding();

        assert_eq!(encoding.token_to_char_range(2), Some((4, 7))); // "add"
        assert_eq!(encoding.token_to_char_range(100), None);
    }

    #[test]
    fn convert_positions_batch() {
        let encoding = sample_encoding();
        let results = convert_positions(&encoding, &[0, 4, 100]);

        assert_eq!(results.len(), 3);
        assert!(results[0].exact_match);
        assert_eq!(results[0].token_idx, Some(0));
        assert!(results[1].exact_match);
        assert_eq!(results[1].token_idx, Some(2));
        assert!(!results[2].exact_match); // fuzzy fallback
    }

    #[test]
    fn find_marker() {
        let code = "def add(a, b):\n    \"\"\"\n    >>> add(2, 3)\n    5\n    \"\"\"";
        assert!(find_marker_char_pos(code, ">>>").is_some());
        assert!(find_marker_char_pos(code, "zzz").is_none());
    }

    #[test]
    fn encoding_len_and_empty() {
        let encoding = sample_encoding();
        assert_eq!(encoding.len(), 8);
        assert!(!encoding.is_empty());

        let empty = EncodingWithOffsets::new(vec![], vec![], vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn label_spans_subject_relation() {
        // "The Eiffel Tower is located in"
        let enc = EncodingWithOffsets::new(
            vec![1, 2, 3, 4, 5, 6, 7],
            vec![
                "The".into(),
                " Eiffel".into(),
                " Tower".into(),
                " is".into(),
                " located".into(),
                " in".into(),
                " Paris".into(),
            ],
            vec![
                (0, 3),
                (3, 10),
                (10, 16),
                (16, 19),
                (19, 27),
                (27, 30),
                (30, 36),
            ],
        );
        let labels = enc.label_spans(&[("subject", 0..16), ("relation", 17..30)]);
        assert_eq!(
            labels,
            vec![
                "subject",
                "subject",
                "subject_final",
                "relation",
                "relation",
                "relation_final",
                "other",
            ]
        );
    }

    #[test]
    fn label_spans_bos_token() {
        // BOS has offset (0, 0) — should be "other"
        let enc = EncodingWithOffsets::new(
            vec![0, 1, 2],
            vec!["<bos>".into(), "Hello".into(), " world".into()],
            vec![(0, 0), (0, 5), (5, 11)],
        );
        let labels = enc.label_spans(&[("greeting", 0..5)]);
        assert_eq!(labels, vec!["other", "greeting_final", "other"]);
    }

    #[test]
    fn label_spans_no_spans() {
        let enc = EncodingWithOffsets::new(
            vec![1, 2],
            vec!["Hello".into(), " world".into()],
            vec![(0, 5), (5, 11)],
        );
        let labels = enc.label_spans(&[]);
        assert_eq!(labels, vec!["other", "other"]);
    }

    #[test]
    fn label_spans_first_span_wins() {
        // Two overlapping spans — first one wins
        let enc = EncodingWithOffsets::new(vec![1], vec!["overlap".into()], vec![(0, 7)]);
        let labels = enc.label_spans(&[("first", 0..5), ("second", 3..7)]);
        assert_eq!(labels, vec!["first_final"]);
    }
}
