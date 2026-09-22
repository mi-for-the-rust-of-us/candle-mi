#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate Gemma-1-2B forward-pass reference for Rust validation.

From-first-principles forward-pass oracle for the candle-mi Gemma **1** arm,
the one family in ``SUPPORTED_MODEL_TYPES`` that had no forward-parity entry
in ``RESURRECTION.md``.  Gemma 1 is deliberately the *inverse* of Gemma 2's
quirk set: it keeps ``GemmaRmsNorm``, ``sqrt(hidden_size)`` embedding scaling,
the GELU-tanh approximation and the required BOS token, but has **no** logit
soft-capping, **no** post-attention/post-feedforward norms and **no** sliding
window.  A Gemma 2 pass therefore says nothing about it.

``google/gemma-2b`` also exercises multi-query attention (a single KV head),
which none of the other validated families use, so this is the narrowest
test of the GQA path at its degenerate end.

Loads ``google/gemma-2b`` via HuggingFace ``transformers`` in ``F32`` on CPU,
runs ``forward()`` on a small set of fixed prompts with deterministic seeds,
and saves
**(a)** top-10 next-token logits + indices and
**(b)** the final-layer last-token residual (post-final-norm, pre-LM-head)
per prompt to JSON for cross-validation with the Rust implementation in
``src/transformer/``.

The methodology mirrors ``gemma2_validation.py``.  The reference JSON is
consumed by ``tests/validate_gemma_forward.rs``.  Acceptance bar:

- Detected ``model_type`` is ``"gemma"`` with null ``attn_logit_softcapping``
  and ``final_logit_softcapping``.
- ``(hidden_size, num_layers, vocab_size, head_dim, num_kv_heads)`` match the
  Python run.
- Per test case: top-10 logit indices match exactly, magnitudes within
  ``abs diff < 1e-3`` (``F32``, CPU vs CPU).

Dependencies: ``torch``, ``transformers``, ``safetensors``.

Usage:
    python scripts/gemma_validation.py

Output:
    scripts/gemma_forward_reference.json

Requires ``google/gemma-2b`` (gated; ~4.7 GiB) cached in the HF cache.  This
is a cache-only run - it does not re-download.
"""

import json
import os
import platform
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_REPO = "google/gemma-2b"
# The same three prompts as the Gemma 2 arm, so the two references are read
# side by side; the BOS Gemma requires is added by the tokenizer.
TEST_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Once upon a time, there was a",
]
TOP_K = 10


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    import transformers as hf_transformers

    print(f"Gemma 1 forward-pass reference generation for {MODEL_REPO}")
    print(f"  {len(TEST_PROMPTS)} prompts, top-{TOP_K} logits per prompt")
    print(f"  torch {torch.__version__}, transformers {hf_transformers.__version__}")
    print(f"  platform {platform.platform()}")
    print()

    # Eager attention, for a different reason than the Gemma 2 arm: Gemma 1 has
    # no soft-capping for `sdpa` to drop, but the SDPA backend PyTorch selects
    # varies by host and kernel availability, so its logits are not reproducible
    # across machines. Eager is one fixed arithmetic path, which is what an
    # oracle needs. It also matches what candle-mi's `transformer` arm computes.
    print("Loading model + tokenizer (attn_implementation=eager) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    # Pin the activation to the tanh GELU approximation, which is what Gemma 1
    # actually uses.  Without this pin transformers computes EXACT erf GELU for
    # this checkpoint, silently and with no warning.  The chain:
    #
    #   1. `google/gemma-2b/config.json` shipped in February 2024 with
    #      `"hidden_act": "gelu"` (`transformers_version: 4.38.0.dev0`).  That
    #      string was wrong on day one; Gemma is an approximate-GELU model.
    #   2. transformers PR #29402 added a `hidden_activation` field, and
    #      `GemmaMLP` guarded the legacy case:
    #          if config.hidden_activation is None:
    #              logger.warning_once("`config.hidden_act` is ignored ...")
    #              config.hidden_activation = "gelu_pytorch_tanh"
    #      Gemma 1.1 and later shipped configs carrying the new field; the
    #      original 1.0 repos never were updated and still rely on that guard.
    #   3. transformers PR #35235 ("All attention refactor", commit 2c47618c,
    #      2024-12-18, released in v4.48.0) replaced the whole guard with
    #      `self.act_fn = ACT2FN[config.hidden_act]`, deleting the warning in
    #      the same diff.  `gemma2/modeling_gemma2.py` kept
    #      `ACT2FN[config.hidden_activation]`, so only Gemma 1 lost its
    #      fallback.  `hidden_activation` then left `GemmaConfig` in v5.0.0.
    #
    # So every transformers >= 4.48.0 runs `google/gemma-2b`, `gemma-2b-it`,
    # `gemma-7b`, `gemma-7b-it` and `codegemma-2b` with the wrong activation,
    # while `GemmaConfig.hidden_act` still DEFAULTS to `"gelu_pytorch_tanh"`.
    # Configs that omit `hidden_act` (Gemma 1.1, codegemma-7b-it) are saved by
    # that default; only the ones that set it explicitly are affected.
    #
    # Left unpinned this shifts every logit by ~1e-2 against candle-mi's
    # `Activation::GeluApprox`, which is what first made
    # `tests/validate_gemma_forward.rs` fail.  candle-mi is the correct side;
    # the oracle has to be told so explicitly.  Verified 2026-09-22 against
    # torch 2.10.0+cu130 / transformers 5.1.0.
    config = AutoConfig.from_pretrained(MODEL_REPO)
    legacy_hidden_act = config.hidden_act
    config.hidden_act = "gelu_pytorch_tanh"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        config=config,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    assert model.config._attn_implementation == "eager", "oracle needs a fixed attention path"
    assert model.config.hidden_act == "gelu_pytorch_tanh", "activation pin did not take"
    act_fn = type(model.model.layers[0].mlp.act_fn).__name__
    assert act_fn == "GELUTanh", f"expected GELUTanh, got {act_fn}"
    print(f"  activation: config said {legacy_hidden_act!r}, pinned to 'gelu_pytorch_tanh' ({act_fn})")

    cfg = model.config
    assert cfg.model_type == "gemma", f"expected model_type 'gemma', got {cfg.model_type!r}"
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(
        f"  hidden_size={cfg.hidden_size}, num_layers={cfg.num_hidden_layers}, "
        f"vocab_size={cfg.vocab_size}, head_dim={head_dim}, "
        f"num_kv_heads={cfg.num_key_value_heads}"
    )
    print(
        f"  attn_softcap={getattr(cfg, 'attn_logit_softcapping', None)}, "
        f"final_softcap={getattr(cfg, 'final_logit_softcapping', None)} "
        f"(both expected None for Gemma 1)"
    )
    print()

    results: dict = {
        "model_repo": MODEL_REPO,
        "methodology": "from-first-principles forward-pass oracle "
        "(transformers.AutoModelForCausalLM, F32 CPU, eager attention); Gemma 1 arm",
        "torch_version": torch.__version__,
        "transformers_version": hf_transformers.__version__,
        "platform": platform.platform(),
        "model_type": cfg.model_type,
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_hidden_layers,
        "vocab_size": cfg.vocab_size,
        "head_dim": head_dim,
        "num_attention_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "attn_logit_softcapping": getattr(cfg, "attn_logit_softcapping", None),
        "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
        # Recorded so a future reader can see the pin was applied and why the
        # checkpoint's own config disagrees with the activation actually used.
        "hidden_act_in_checkpoint_config": legacy_hidden_act,
        "hidden_act_used": cfg.hidden_act,
        "test_cases": [],
    }

    with torch.no_grad():
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            tokens = input_ids[0].tolist()

            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            last_logits = outputs.logits[0, -1, :].float()
            top_vals, top_idx = last_logits.topk(TOP_K)
            final_hidden = outputs.hidden_states[-1]
            last_residual = final_hidden[0, -1, :].float().tolist()

            top_token_str = tokenizer.decode([int(top_idx[0])])
            print(
                f"  prompt={prompt!r}: {len(tokens)} tokens, "
                f"top1=({int(top_idx[0])}, {top_token_str!r}, {float(top_vals[0]):.4f})"
            )

            results["test_cases"].append(
                {
                    "prompt": prompt,
                    "tokens": tokens,
                    "top_10": [
                        {"index": int(idx), "logit": float(val)}
                        for idx, val in zip(top_idx, top_vals, strict=False)
                    ],
                    "last_residual_f32": last_residual,
                }
            )

    out_path = Path(__file__).parent / "gemma_forward_reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_cases = len(results["test_cases"])
    file_size = out_path.stat().st_size
    print(f"\nSaved {n_cases} test cases to {out_path} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
