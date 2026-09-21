#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Convert an OthelloMDLM ``.pt`` checkpoint to candle-mi safetensors.

The OthelloMDLM world model (a nanoGPT/minGPT-lineage GPT-2 backbone) is saved
as a ``torch.save`` dict with the state dict under ``ckpt["model"]`` and the
architecture config under ``ckpt["config"]``.  candle-mi's ``OthelloGpt`` loader
reads weight keys **verbatim** — its ``VarBuilder`` ``pp``-paths line up with the
PyTorch module paths exactly:

    tok_emb.weight                      pos_emb.weight
    blocks.{i}.ln1.{weight,bias}        blocks.{i}.ln2.{weight,bias}
    blocks.{i}.attn.qkv.{weight,bias}   blocks.{i}.attn.proj.{weight,bias}
    blocks.{i}.mlp.0.{weight,bias}      blocks.{i}.mlp.2.{weight,bias}
    ln_f.{weight,bias}                  head.weight

So **no name remap and no transpose** are needed (PyTorch ``nn.Linear`` stores
``weight`` as ``[out, in]``, the same convention as candle ``Linear``).  This
script simply lifts ``ckpt["model"]`` into a ``.safetensors`` file and writes a
companion ``config.json`` that ``OthelloGptConfig::from_hf_config`` can parse.

Dependencies: ``torch``, ``safetensors``.

Usage:
    python scripts/convert_othello_mdlm.py CHECKPOINT.pt OUTPUT_DIR

Output:
    OUTPUT_DIR/model.safetensors
    OUTPUT_DIR/config.json
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Keys candle-mi's OthelloGptConfig::from_hf_config reads.
CONFIG_KEYS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",
    "causal",
    # Omitting this writes the self_cond_emb weight tensor without the flag
    # that makes the loader create the table, so the model would load with
    # self-conditioning silently off and produce non-self-conditioned logits.
    "self_conditioning",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to the OthelloMDLM .pt file")
    parser.add_argument("out_dir", type=Path, help="output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # weights_only=False: the checkpoint carries a config object alongside the
    # state dict. Only load checkpoints you trust.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # Materialize contiguous CPU tensors (safetensors rejects shared storage).
    tensors = {k: v.contiguous().cpu() for k, v in state_dict.items()}

    weights_path = args.out_dir / "model.safetensors"
    save_file(tensors, str(weights_path))

    # Pull the architecture config; fall back to the released world-model
    # defaults for any key the checkpoint omits.
    raw_cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    cfg_obj = vars(raw_cfg) if raw_cfg is not None and not isinstance(raw_cfg, dict) else raw_cfg
    cfg_obj = cfg_obj or {}
    defaults = {
        "vocab_size": 62,
        "block_size": 60,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 512,
        "causal": False,
        "self_conditioning": False,
    }
    config = {k: cfg_obj.get(k, defaults[k]) for k in CONFIG_KEYS}

    # A checkpoint trained before the flag existed carries the table but not
    # the key. Trust the weights over the config in that direction only: a
    # present table with the flag off loads as a non-self-conditioned model,
    # which is a silent wrong answer rather than an error.
    if "self_cond_emb.weight" in tensors and not config["self_conditioning"]:
        print("note: self_cond_emb.weight present; setting self_conditioning=true")
        config["self_conditioning"] = True

    config_path = args.out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    n_tensors = len(tensors)
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {n_tensors} tensors ({size_mb:.1f} MB) to {weights_path}")
    print(f"Wrote config to {config_path}: {config}")


if __name__ == "__main__":
    main()
