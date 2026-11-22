// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizer abstraction: dispatch between `HuggingFace` and RWKV backends.
//!
//! [`MITokenizer`] provides a unified encode/decode interface regardless of
//! the underlying tokenizer implementation.

#[cfg(feature = "rwkv-tokenizer")]
mod rwkv;

use std::path::PathBuf;

use crate::error::{MIError, Result};
use crate::util::positioning::EncodingWithOffsets;

/// Unified tokenizer supporting multiple backends.
///
/// Most models use the `HuggingFace` `tokenizers` crate. RWKV-6 models
/// ship their own vocabulary format and require a custom trie-based
/// tokenizer, which is available behind the `rwkv-tokenizer` feature.
///
/// # Example
///
/// ```no_run
/// use candle_mi::MITokenizer;
///
/// # fn main() -> candle_mi::Result<()> {
/// let tok = MITokenizer::from_hf_path("tokenizer.json")?;
/// let ids = tok.encode("fn main()")?;
/// let text = tok.decode(&ids)?;
/// assert!(!ids.is_empty());
/// # Ok(())
/// # }
/// ```
#[non_exhaustive]
pub enum MITokenizer {
    /// `HuggingFace` `tokenizers` backend.
    HuggingFace(Box<tokenizers::Tokenizer>),
    /// RWKV World tokenizer (trie-based greedy longest-match).
    #[cfg(feature = "rwkv-tokenizer")]
    Rwkv(rwkv::RwkvTokenizer),
}

impl MITokenizer {
    /// Load a `HuggingFace` tokenizer from a `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the file cannot be loaded or parsed.
    pub fn from_hf_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            MIError::Tokenizer(format!(
                "failed to load HF tokenizer from {}: {e}",
                path.as_ref().display()
            ))
        })?;
        Ok(Self::HuggingFace(Box::new(tok)))
    }

    /// Wrap an already-loaded `HuggingFace` tokenizer.
    ///
    /// The argument is a `tokenizers::Tokenizer` from the same major line
    /// candle-mi depends on (`tokenizers` 0.23 since candle-mi 0.2.3, matching
    /// `candle-core` 0.11.1). A `Tokenizer` built with another line (0.22, say)
    /// is a different type to the compiler; depend on the same `tokenizers`
    /// version, or load from a file with [`from_hf_path`](Self::from_hf_path).
    #[must_use]
    pub fn from_hf(tokenizer: tokenizers::Tokenizer) -> Self {
        Self::HuggingFace(Box::new(tokenizer))
    }

    /// Load a `HuggingFace` tokenizer from the local Hub cache by repo id.
    ///
    /// Scans the `HuggingFace` cache (`$HF_HOME/hub` or
    /// `~/.cache/huggingface/hub`) for the first snapshot of `repo_id` that
    /// contains a `tokenizer.json` and loads it.  This does **not** download —
    /// pre-fetch with [`download_model`](crate::download_model) (or the `hf-fm`
    /// CLI) if the repo is not cached.  Handy for models whose *weight* repo
    /// ships no tokenizer (e.g. `MDLM`, which uses the `gpt2` tokenizer).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if no cached `tokenizer.json` is found for
    /// `repo_id`.
    pub fn from_hf_cache(repo_id: &str) -> Result<Self> {
        let path = Self::hf_cache_tokenizer_path(repo_id).ok_or_else(|| {
            MIError::Tokenizer(format!(
                "no cached tokenizer.json for `{repo_id}`; fetch it first, \
                 e.g. `hf-fm download-file {repo_id} tokenizer.json`"
            ))
        })?;
        Self::from_hf_path(path)
    }

    /// Resolve the `HuggingFace` Hub cache directory (`$HF_HOME/hub` or
    /// `~/.cache/huggingface/hub`), if it exists.
    fn hf_cache_dir() -> Option<PathBuf> {
        if let Ok(cache) = std::env::var("HF_HOME") {
            return Some(PathBuf::from(cache).join("hub"));
        }
        for var in ["USERPROFILE", "HOME"] {
            if let Ok(home) = std::env::var(var) {
                let dir = PathBuf::from(home)
                    .join(".cache")
                    .join("huggingface")
                    .join("hub");
                if dir.is_dir() {
                    return Some(dir);
                }
            }
        }
        None
    }

    /// Find a cached `tokenizer.json` for `repo_id` in the Hub cache.
    fn hf_cache_tokenizer_path(repo_id: &str) -> Option<PathBuf> {
        let snapshots = Self::hf_cache_dir()?
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots");
        for entry in std::fs::read_dir(&snapshots).ok()?.flatten() {
            let candidate = entry.path().join("tokenizer.json");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

    /// Load an RWKV World tokenizer from a vocabulary file.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the file cannot be loaded or parsed.
    #[cfg(feature = "rwkv-tokenizer")]
    pub fn from_rwkv_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tok = rwkv::RwkvTokenizer::from_file(path.as_ref())?;
        Ok(Self::Rwkv(tok))
    }

    /// Encode text into token IDs, adding special tokens (e.g. BOS for Gemma).
    ///
    /// Special tokens are added according to the tokenizer's configured
    /// post-processor, matching the `HuggingFace` convention for inference.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, true)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                Ok(encoding.get_ids().to_vec())
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.encode(text),
        }
    }

    /// Encode text into token IDs **without** adding special tokens.
    ///
    /// Useful for MI analyses that need raw tokenization without BOS/EOS.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails.
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, false)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                Ok(encoding.get_ids().to_vec())
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.encode(text),
        }
    }

    /// Encode text into token IDs with character offset mapping.
    ///
    /// Returns an [`EncodingWithOffsets`] containing token IDs, token strings,
    /// and byte-offset ranges for each token. Special tokens are added
    /// (e.g., BOS for Gemma); special tokens receive a `(0, 0)` offset.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails or if the backend
    /// does not support offset mapping (RWKV).
    pub fn encode_with_offsets(&self, text: &str) -> Result<EncodingWithOffsets> {
        self.encode_with_offsets_inner(text, true)
    }

    /// Encode text into token IDs with character offset mapping, **without**
    /// adding special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if encoding fails or if the backend
    /// does not support offset mapping (RWKV).
    pub fn encode_raw_with_offsets(&self, text: &str) -> Result<EncodingWithOffsets> {
        self.encode_with_offsets_inner(text, false)
    }

    /// Shared implementation for offset-bearing encode methods.
    fn encode_with_offsets_inner(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<EncodingWithOffsets> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, add_special_tokens)
                    .map_err(|e| MIError::Tokenizer(format!("HF encode failed: {e}")))?;
                let ids = encoding.get_ids().to_vec();
                let tokens: Vec<String> = encoding
                    .get_tokens()
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                let offsets = encoding.get_offsets().to_vec();
                Ok(EncodingWithOffsets::new(ids, tokens, offsets))
            }
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(_) => Err(MIError::Tokenizer(
                "RWKV tokenizer does not support offset mapping".into(),
            )),
        }
    }

    /// Decode token IDs back to a string.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if decoding fails.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::HuggingFace(tok) => tok
                .decode(ids, false)
                .map_err(|e| MIError::Tokenizer(format!("HF decode failed: {e}"))),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.decode(ids),
        }
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(tok) => tok.get_vocab_size(true),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => tok.vocab_size(),
        }
    }

    /// Find the token ID for a word, trying `" word"` (with leading space) first,
    /// then bare `"word"`.
    ///
    /// This handles BPE tokenizers that represent word-initial tokens with a
    /// leading space (e.g., `" cat"` → single token).
    ///
    /// Uses [`Self::encode_raw`] (no special tokens) so the result is
    /// independent of whether the tokenizer auto-prepends `BOS` (Llama, Gemma)
    /// or not (`Qwen2`, `Qwen3`).  Previously this method asserted
    /// `len == 2` assuming a `BOS` token was always present, which silently
    /// fell through to "last token" for `BOS`-free tokenizers and returned a
    /// sub-token (e.g. `" myself"` → `"self"` for Qwen3).
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if the word is multi-token in both the
    /// space-prefixed and bare forms — surfacing the genuine multi-token case
    /// to the caller rather than silently picking a sub-token.
    pub fn find_token_id(&self, word: &str) -> Result<u32> {
        // Try with leading space first (most BPE tokenizers).
        let with_space = format!(" {word}");
        let raw_ids = self.encode_raw(&with_space)?;
        if raw_ids.len() == 1 {
            // SAFE_INDEX: `.first()` cannot fail when len == 1.
            if let Some(&id) = raw_ids.first() {
                return Ok(id);
            }
        }

        // Try bare word.
        let bare_ids = self.encode_raw(word)?;
        if bare_ids.len() == 1 {
            // SAFE_INDEX: `.first()` cannot fail when len == 1.
            if let Some(&id) = bare_ids.first() {
                return Ok(id);
            }
        }

        Err(MIError::Tokenizer(format!(
            "\"{word}\" is multi-token in this vocabulary (\" {word}\" → {} \
             tokens, \"{word}\" → {} tokens); pick a synonym that is a single \
             token, or sum probabilities across the multi-token encoding",
            raw_ids.len(),
            bare_ids.len()
        )))
    }

    /// Decode a single token ID to its string representation.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Tokenizer`] if decoding fails.
    pub fn decode_token(&self, token_id: u32) -> Result<String> {
        self.decode(&[token_id])
    }
}

impl std::fmt::Debug for MITokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HuggingFace(_) => f.debug_tuple("HuggingFace").field(&"...").finish(),
            #[cfg(feature = "rwkv-tokenizer")]
            Self::Rwkv(tok) => f.debug_tuple("Rwkv").field(tok).finish(),
        }
    }
}
