// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration tests: load real models from the `HuggingFace` cache and
//! validate forward-pass outputs on both CPU and GPU.
//!
//! These tests require model weights in the local HF cache.
//!
//! Run CPU tests:
//!   `cargo test --test validate_models --no-default-features --features transformer`
//!
//! Run all (CPU + GPU):
//!   `cargo test --test validate_models --features transformer`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    clippy::missing_docs_in_private_items,
    clippy::missing_panics_doc,
    unsafe_code,
    missing_docs
)]

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_mi::{GenericTransformer, HookSpec, MIBackend, MITokenizer, TransformerConfig};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find the `HuggingFace` cache directory.
fn hf_cache_dir() -> std::path::PathBuf {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return std::path::PathBuf::from(cache).join("hub");
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    if let Ok(home) = std::env::var("HOME") {
        return std::path::PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    panic!("Cannot find HuggingFace cache directory");
}

/// Find the snapshot directory for a given model ID.
fn find_snapshot(model_id: &str) -> Option<std::path::PathBuf> {
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = hf_cache_dir().join(model_dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots_dir).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Get a CUDA device if available, or None.
fn cuda_device() -> Option<Device> {
    Device::cuda_if_available(0)
        .ok()
        .filter(candle_core::Device::is_cuda)
}

/// Collect safetensors paths for a model snapshot (single or sharded).
fn safetensors_paths(snapshot: &std::path::Path) -> Vec<std::path::PathBuf> {
    let single = snapshot.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }

    // Sharded: parse model.safetensors.index.json
    let index_path = snapshot.join("model.safetensors.index.json");
    let index_str = std::fs::read_to_string(&index_path).unwrap_or_else(|_| {
        panic!(
            "no model.safetensors or index.json in {}",
            snapshot.display()
        )
    });
    let index: serde_json::Value = serde_json::from_str(&index_str).unwrap();
    let weight_map = index["weight_map"].as_object().unwrap();

    let mut shard_names: Vec<String> = weight_map
        .values()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    shard_names.sort();
    shard_names.dedup();

    shard_names.iter().map(|name| snapshot.join(name)).collect()
}

/// Load a model from the local HF cache on the specified device.
///
/// Uses memory-mapped loading for sharded models (safe for test code).
fn load_model_on(
    model_id: &str,
    device: &Device,
) -> (GenericTransformer, MITokenizer, TransformerConfig) {
    let snapshot =
        find_snapshot(model_id).unwrap_or_else(|| panic!("{model_id} not found in HF cache"));

    // Parse config
    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    // F32 everywhere: research-grade precision matching Python/PyTorch.
    let dtype = DType::F32;

    // Resolve safetensors paths (single or sharded)
    let paths = safetensors_paths(&snapshot);

    // Load weights — use mmap (handles both single and multi-file)
    // SAFETY: safetensors files are not modified during test execution.
    let vb =
        unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&paths, dtype, device).unwrap() };

    // Build model
    let model = GenericTransformer::load(config.clone(), device, dtype, vb).unwrap();

    // Load tokenizer
    let tokenizer = MITokenizer::from_hf_path(snapshot.join("tokenizer.json")).unwrap();

    (model, tokenizer, config)
}

/// Run a forward pass and return top-k token strings for the last position.
fn top_k_last_token(
    model: &GenericTransformer,
    tokenizer: &MITokenizer,
    device: &Device,
    prompt: &str,
    k: usize,
) -> Vec<(String, f32)> {
    let token_ids = tokenizer.encode(prompt).unwrap();
    let seq_len = token_ids.len();

    let input = Tensor::new(&token_ids[..], device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let hooks = HookSpec::new();
    let result = model.forward(&input, &hooks).unwrap();

    let logits = result.output();
    let (batch, out_seq, _vocab) = logits.dims3().unwrap();
    assert_eq!(batch, 1);
    assert_eq!(out_seq, seq_len);

    // Move to CPU F32 for inspection
    let logits_cpu = logits
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Get logits for the last token position
    let last_logits: Vec<f32> = logits_cpu.i((0, seq_len - 1)).unwrap().to_vec1().unwrap();

    // Sort by logit value descending
    let mut indexed: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Decode top-k
    indexed
        .iter()
        .take(k)
        .map(|(idx, logit)| {
            let token = tokenizer.decode(&[*idx as u32]).unwrap();
            (token, *logit)
        })
        .collect()
}

/// Assert that a target substring appears in at least one of the top-k tokens.
fn assert_in_top_k(top_k: &[(String, f32)], target: &str, prompt: &str, model_name: &str) {
    let found = top_k
        .iter()
        .any(|(t, _)| t.to_lowercase().contains(&target.to_lowercase()));
    assert!(
        found,
        "[{model_name}] Expected '{target}' in top-{} for '{prompt}', got: {:?}",
        top_k.len(),
        top_k.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>()
    );
}

fn print_top_k(model_name: &str, device_name: &str, prompt: &str, top_k: &[(String, f32)]) {
    println!(
        "{model_name} ({device_name}) — Top {} for '{prompt}':",
        top_k.len()
    );
    for (rank, (token, logit)) in top_k.iter().enumerate() {
        println!("  {}: '{}' (logit={:.4})", rank + 1, token, logit);
    }
}

// ===========================================================================
// LLaMA 3.2 1B
// ===========================================================================

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn llama_3_2_1b_config_parse() {
    let Some(snapshot) = find_snapshot("meta-llama/Llama-3.2-1B") else {
        eprintln!("SKIP: meta-llama/Llama-3.2-1B not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_layers, 16);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 64);
    assert_eq!(config.intermediate_size, 8192);
    assert_eq!(config.vocab_size, 128_256);
    assert!(config.tie_word_embeddings);
    assert!(!config.qkv_bias);
    assert!(!config.mlp_bias);
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn llama_3_2_1b_forward_cpu() {
    if find_snapshot("meta-llama/Llama-3.2-1B").is_none() {
        eprintln!("SKIP: meta-llama/Llama-3.2-1B not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, config) = load_model_on("meta-llama/Llama-3.2-1B", &device);

    assert_eq!(model.num_layers(), 16);
    assert_eq!(model.hidden_size(), 2048);
    assert_eq!(model.vocab_size(), 128_256);
    assert_eq!(model.num_heads(), 32);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("LLaMA 3.2 1B", "CPU", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "LLaMA 3.2 1B CPU");

    // Output shape check
    let token_ids = tokenizer.encode(prompt).unwrap();
    let input = Tensor::new(&token_ids[..], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let result = model.forward(&input, &HookSpec::new()).unwrap();
    let (batch, seq, vocab) = result.output().dims3().unwrap();
    assert_eq!(batch, 1);
    assert_eq!(seq, token_ids.len());
    assert_eq!(vocab, config.vocab_size);
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn llama_3_2_1b_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if find_snapshot("meta-llama/Llama-3.2-1B").is_none() {
        eprintln!("SKIP: meta-llama/Llama-3.2-1B not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("meta-llama/Llama-3.2-1B", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("LLaMA 3.2 1B", "CUDA", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "LLaMA 3.2 1B CUDA");
}

// ===========================================================================
// Gemma 2 2B
// ===========================================================================

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn gemma_2_2b_config_parse() {
    let Some(snapshot) = find_snapshot("google/gemma-2-2b") else {
        eprintln!("SKIP: google/gemma-2-2b not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 2304);
    assert_eq!(config.num_layers, 26);
    assert_eq!(config.num_attention_heads, 8);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.vocab_size, 256_000);
    assert!(config.tie_word_embeddings);
    assert!(config.use_post_norms);
    assert!(config.attn_logit_softcapping.is_some());
    assert!(config.final_logit_softcapping.is_some());
    assert!(config.embedding_scale.is_some());
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn gemma_2_2b_forward_cpu() {
    if find_snapshot("google/gemma-2-2b").is_none() {
        eprintln!("SKIP: google/gemma-2-2b not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on("google/gemma-2-2b", &device);

    // Gemma 2 2B is a small model with logit softcapping (30.0) that flattens
    // the distribution.  "Paris" appears at rank 8 (verified against Python HF).
    let prompt = "The capital of France is";
    let top10 = top_k_last_token(&model, &tokenizer, &device, prompt, 10);
    print_top_k("Gemma 2 2B", "CPU", prompt, &top10);
    assert_in_top_k(&top10, "Paris", prompt, "Gemma 2 2B CPU");
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn gemma_2_2b_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if find_snapshot("google/gemma-2-2b").is_none() {
        eprintln!("SKIP: google/gemma-2-2b not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("google/gemma-2-2b", &device);

    // Gemma 2 2B: logit softcapping flattens the distribution (see CPU test).
    let prompt = "The capital of France is";
    let top10 = top_k_last_token(&model, &tokenizer, &device, prompt, 10);
    print_top_k("Gemma 2 2B", "CUDA", prompt, &top10);
    assert_in_top_k(&top10, "Paris", prompt, "Gemma 2 2B CUDA");
}

// ===========================================================================
// StarCoder2 3B
// ===========================================================================

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn starcoder2_3b_config_parse() {
    let Some(snapshot) = find_snapshot("bigcode/starcoder2-3b") else {
        eprintln!("SKIP: bigcode/starcoder2-3b not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 3072);
    assert_eq!(config.num_layers, 30);
    assert!(config.mlp_bias);
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn starcoder2_3b_forward_cpu() {
    if find_snapshot("bigcode/starcoder2-3b").is_none() {
        eprintln!("SKIP: bigcode/starcoder2-3b not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on("bigcode/starcoder2-3b", &device);

    let prompt = "def hello_world():\n    print(\"";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("StarCoder2 3B", "CPU", prompt, &top5);
    assert_in_top_k(&top5, "hello", prompt, "StarCoder2 3B CPU");
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn starcoder2_3b_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if find_snapshot("bigcode/starcoder2-3b").is_none() {
        eprintln!("SKIP: bigcode/starcoder2-3b not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("bigcode/starcoder2-3b", &device);

    let prompt = "def hello_world():\n    print(\"";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("StarCoder2 3B", "CUDA", prompt, &top5);
    assert_in_top_k(&top5, "hello", prompt, "StarCoder2 3B CUDA");
}

// ===========================================================================
// Qwen2.5 Coder 3B Instruct
// ===========================================================================

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn qwen2_5_coder_3b_config_parse() {
    let Some(snapshot) = find_snapshot("Qwen/Qwen2.5-Coder-3B-Instruct") else {
        eprintln!("SKIP: Qwen/Qwen2.5-Coder-3B-Instruct not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert!(config.qkv_bias); // Qwen2 has QKV bias
    assert_eq!(config.num_kv_heads, 2); // GQA with few KV heads
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn qwen2_5_coder_3b_forward_cpu() {
    if find_snapshot("Qwen/Qwen2.5-Coder-3B-Instruct").is_none() {
        eprintln!("SKIP: Qwen/Qwen2.5-Coder-3B-Instruct not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on("Qwen/Qwen2.5-Coder-3B-Instruct", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Qwen2.5-Coder-3B", "CPU", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Qwen2.5-Coder-3B CPU");
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn qwen2_5_coder_3b_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if find_snapshot("Qwen/Qwen2.5-Coder-3B-Instruct").is_none() {
        eprintln!("SKIP: Qwen/Qwen2.5-Coder-3B-Instruct not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("Qwen/Qwen2.5-Coder-3B-Instruct", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Qwen2.5-Coder-3B", "CUDA", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Qwen2.5-Coder-3B CUDA");
}

// ===========================================================================
// Phi-3 Mini 4K Instruct
// ===========================================================================

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn phi3_mini_config_parse() {
    let Some(snapshot) = find_snapshot("microsoft/Phi-3-mini-4k-instruct") else {
        eprintln!("SKIP: microsoft/Phi-3-mini-4k-instruct not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    // Phi-3 uses fused QKV and gated fused MLP
    assert_eq!(config.qkv_layout, candle_mi::QkvLayout::Fused);
    assert_eq!(config.mlp_layout, candle_mi::MlpLayout::GatedFused);
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn phi3_mini_forward_cpu() {
    if find_snapshot("microsoft/Phi-3-mini-4k-instruct").is_none() {
        eprintln!("SKIP: microsoft/Phi-3-mini-4k-instruct not in cache");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on("microsoft/Phi-3-mini-4k-instruct", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Phi-3 Mini", "CPU", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Phi-3 Mini CPU");
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn phi3_mini_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if find_snapshot("microsoft/Phi-3-mini-4k-instruct").is_none() {
        eprintln!("SKIP: microsoft/Phi-3-mini-4k-instruct not in cache");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("microsoft/Phi-3-mini-4k-instruct", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Phi-3 Mini", "CUDA", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Phi-3 Mini CUDA");
}

// ===========================================================================
// Mistral 7B v0.1
// ===========================================================================

/// Helper: ensure Mistral 7B v0.1 weights are in the local HF cache.
///
/// Uses `hf-fetch-model` to download if necessary (fast, parallel).
/// Returns the snapshot path, or None if download fails.
fn ensure_mistral_7b_cached() -> Option<std::path::PathBuf> {
    if let Some(snap) = find_snapshot("mistralai/Mistral-7B-v0.1") {
        // Check if weight files actually exist (not just metadata)
        let has_weights = snap.join("model.safetensors").exists()
            || snap.join("model.safetensors.index.json").exists()
                && safetensors_paths(&snap).iter().all(|p| p.exists());
        if has_weights {
            return Some(snap);
        }
    }

    // Trigger download via hf-fetch-model; go through the shared builder so
    // HF_TOKEN is picked up (Mistral 7B is a gated model).
    eprintln!("Downloading mistralai/Mistral-7B-v0.1 via hf-fetch-model...");
    let config = candle_mi::fetch_config_builder()
        .filter("*.safetensors")
        .filter("*.safetensors.index.json")
        .filter("*.json")
        .build()
        .ok()?;

    match hf_fetch_model::download_with_config_blocking(
        "mistralai/Mistral-7B-v0.1".to_owned(),
        &config,
    ) {
        Ok(_outcome) => find_snapshot("mistralai/Mistral-7B-v0.1"),
        Err(e) => {
            eprintln!("  FAILED to download: {e}");
            None
        }
    }
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
fn mistral_7b_config_parse() {
    let Some(snapshot) = find_snapshot("mistralai/Mistral-7B-v0.1") else {
        eprintln!("SKIP: mistralai/Mistral-7B-v0.1 not in cache");
        return;
    };

    let config_str = std::fs::read_to_string(snapshot.join("config.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let config = TransformerConfig::from_hf_config(&json).unwrap();

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_kv_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.vocab_size, 32000);
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.sliding_window, Some(4096));
    assert!(!config.alternating_sliding_window); // All layers, not alternating
}

#[test]
#[ignore = "7B F32 on CPU exceeds CI timeout; validated locally (41s, Paris at rank 4)"]
fn mistral_7b_forward_cpu() {
    if ensure_mistral_7b_cached().is_none() {
        eprintln!("SKIP: mistralai/Mistral-7B-v0.1 not available");
        return;
    }

    let device = Device::Cpu;
    let (model, tokenizer, _config) = load_model_on("mistralai/Mistral-7B-v0.1", &device);

    assert_eq!(model.num_layers(), 32);
    assert_eq!(model.hidden_size(), 4096);
    assert_eq!(model.vocab_size(), 32000);
    assert_eq!(model.num_heads(), 32);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Mistral 7B", "CPU", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Mistral 7B CPU");
}

#[ignore = "needs a cached HF model; a skip would otherwise report `ok`"]
#[test]
#[serial]
fn mistral_7b_forward_gpu() {
    let Some(device) = cuda_device() else {
        eprintln!("SKIP: no CUDA device available");
        return;
    };
    if ensure_mistral_7b_cached().is_none() {
        eprintln!("SKIP: mistralai/Mistral-7B-v0.1 not available");
        return;
    }

    let (model, tokenizer, _config) = load_model_on("mistralai/Mistral-7B-v0.1", &device);

    let prompt = "The capital of France is";
    let top5 = top_k_last_token(&model, &tokenizer, &device, prompt, 5);
    print_top_k("Mistral 7B", "CUDA", prompt, &top5);
    assert_in_top_k(&top5, "Paris", prompt, "Mistral 7B CUDA");
}
