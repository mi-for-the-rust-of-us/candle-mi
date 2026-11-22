"""Export Tracr-compiled models as a fixture for `tests/validate_plain_tracr.rs`.

The Tracr oracle of `design/plain-transformer.md` §7.4: compile a few RASP
programs from `tracr.compiler.lib`, export each compiled model's weights in
`PlainTransformer`'s layout, its inputs, and what Tracr computes on them (the
input embeddings, the residual stream after every attention and MLP sublayer,
and the output), so the Rust test can require `PlainTransformer` to reproduce
them. Tracr is JAX and `PlainTransformer` is candle: the comparison has a
tolerance (below), not bit-identity.

Environment (Tracr's last commit is 2024-02-05 and pins nothing; this is the
combination it was run with, verified 2026-11-22):

    uv python install 3.11
    uv venv --python 3.11 <venv>
    uv pip install --python <venv>/Scripts/python.exe \\
        "git+https://github.com/google-deepmind/tracr@9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd" \\
        safetensors

    resolved: Python 3.11.14, tracr 1.0.0 (9ce2b8c), jax 0.10.2, jaxlib 0.10.2,
    dm-haiku 0.0.17, numpy 2.4.6, chex 0.1.92, networkx 3.6.1, einops 0.8.2,
    safetensors 0.8.0

Run, from the candle-mi root (CPU, seconds):

    <venv>/Scripts/python.exe -I scripts/plain_tracr_validation.py

It writes `tests/fixtures/plain_tracr/tracr.safetensors` (weights and expected
tensors) and `tests/fixtures/plain_tracr/manifest.json` (configs, inputs,
expected outputs). The JSON records the versions above as `generator`.

Tolerance, fixed before the first run and applied by the Rust test, never
loosened after a failure:

- residual stream, every layer (the input embeddings, then Tracr's residual
  after each attention sublayer, `ResidMid`, and each MLP sublayer,
  `ResidPost`): `max |delta| <= 1e-5 * max(1, max |x_layer|)`, `x_layer` being
  Tracr's tensor at that layer. Relative, because numerical (discretising) MLPs
  use a large-number hyperparameter (`mlp_exactness`), so residual values far
  above 1 are normal;
- categorical outputs: the same argmax at every position but the first (the
  BOS position, which Tracr's decoder special-cases);
- numerical outputs: the output value, with the same relative tolerance.

Export choices (design §7.4): every Tracr linear has a (zero) bias, so models
are exported with `bias = true` and those biases, the faithful copy; inputs
carry no padding, because Tracr also masks padding tokens and `PlainTransformer`
has no padding mask; a causal fixture is compiled with `causal=True` and loaded
with `causal = true`. Tracr's attention scale is `1 / sqrt(key_size)`, which is
`PlainTransformer`'s default for `head_dim = key_size`.

Weight layout: haiku stores linear weights `[in, out]`; `PlainTransformer`
reads the `nn.Linear` layout `[out, in]`, so every linear weight is transposed
here. The unembedding is not a Tracr parameter (Tracr projects the residual
onto its output space, then takes the argmax), so it is rebuilt as the 0/1
matrix that picks the output space's residual directions, in the output
encoder's order.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tracr.compiler import compiling, lib
from tracr.rasp import rasp

OUT_DIR = Path("tests/fixtures/plain_tracr")
BOS = "BOS"

# (name, program builder, vocab, max_seq_len, causal, inputs). Inputs are
# token sequences without BOS (added here) and without padding.
PROGRAMS = [
    ("length", lambda: lib.make_length(), {1, 2, 3}, 5, False,
     [[1, 2, 3], [3, 3, 1, 2, 1], [2]]),
    ("length_causal", lambda: lib.make_length(), {1, 2, 3}, 5, True,
     [[1, 2, 3], [3, 3, 1, 2, 1]]),
    ("reverse", lambda: lib.make_reverse(rasp.tokens), {"a", "b", "c", "d"}, 5, False,
     [list("abcd"), list("dcbaa"), list("b")]),
    ("hist", lambda: lib.make_hist(), {"a", "b", "c"}, 5, False,
     [list("abcab"), list("ccc"), list("ba")]),
    ("sort_unique", lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens),
     {1, 2, 3, 4, 5}, 5, False, [[3, 1, 4, 2], [5, 2, 1], [1]]),
    ("frac_prevs_causal",
     lambda: lib.make_frac_prevs(rasp.numerical(rasp.tokens == "x")),
     {"x", "y"}, 5, True, [list("xyxx"), list("yyx"), list("x")]),
    ("shuffle_dyck", lambda: lib.make_shuffle_dyck(["()"]), {"(", ")"}, 6, False,
     [list("()()"), list("(())"), list("())("), list("((")]),
]


def export(name, build, vocab, max_seq_len, causal, inputs, tensors):
    """Compile one program, add its tensors to `tensors`, return its manifest entry."""
    program = build()
    model = compiling.compile_rasp_to_model(
        program, vocab=vocab, max_seq_len=max_seq_len, compiler_bos=BOS, causal=causal
    )
    cfg = model.model_config
    assert not cfg.layer_norm, "Tracr compiles without layer norm"
    assert cfg.causal == causal
    params = model.params
    labels = list(model.residual_labels)
    d_model = len(labels)

    def put(key, array):
        tensors[f"{name}/{key}"] = np.ascontiguousarray(np.asarray(array, dtype=np.float32))

    put("tok_emb.weight", params["token_embed"]["embeddings"])
    put("pos_emb.weight", params["pos_embed"]["embeddings"])
    for layer in range(cfg.num_layers):
        prefix = f"transformer/layer_{layer}"
        for tracr_name, plain_name in [
            ("attn/query", "attn.q"), ("attn/key", "attn.k"), ("attn/value", "attn.v"),
            ("attn/linear", "attn.o"), ("mlp/linear_1", "mlp.fc"), ("mlp/linear_2", "mlp.proj"),
        ]:
            p = params[f"{prefix}/{tracr_name}"]
            put(f"blocks.{layer}.{plain_name}.weight", np.asarray(p["w"]).T)
            put(f"blocks.{layer}.{plain_name}.bias", p["b"])

    # The output space: the residual directions named `<label>:<value>` for a
    # categorical output (in the output encoder's order), or `<label>` for a
    # numerical one.
    categorical = rasp.is_categorical(program)
    if categorical:
        enc = model.output_encoder.encoding_map
        rows = [labels.index(f"{program.label}:{value}")
                for value, _ in sorted(enc.items(), key=lambda kv: kv[1])]
    else:
        rows = [labels.index(program.label)]
    unembed = np.zeros((len(rows), d_model), dtype=np.float32)
    for out_index, residual_index in enumerate(rows):
        unembed[out_index, residual_index] = 1.0
    put("unembed.weight", unembed)

    input_map = model.input_encoder.encoding_map
    cases = []
    for case_index, seq in enumerate(inputs):
        tokens = [BOS, *seq]
        out = model.apply(tokens)
        key = f"case{case_index}"
        put(f"{key}/embed", out.input_embeddings[0])
        assert len(out.residuals) == 2 * cfg.num_layers
        for i, residual in enumerate(out.residuals):
            # Tracr appends one residual after each attention and each MLP.
            site = "resid_mid" if i % 2 == 0 else "resid_post"
            put(f"{key}/{site}_{i // 2}", residual[0])
        unembedded = np.asarray(out.unembedded[0])
        case = {
            "tokens": [str(t) for t in tokens],
            "ids": [int(input_map[t]) for t in tokens],
            "decoded": [str(t) for t in out.decoded],
        }
        if categorical:
            case["argmax"] = [int(v) for v in unembedded]
        else:
            put(f"{key}/output", unembedded)
        cases.append(case)

    return {
        "causal": causal,
        "output": "categorical" if categorical else "numerical",
        "vocab_size": int(params["token_embed"]["embeddings"].shape[0]),
        "output_size": len(rows),
        "max_positions": int(params["pos_embed"]["embeddings"].shape[0]),
        "n_layer": int(cfg.num_layers),
        "d_model": d_model,
        "n_head": int(cfg.num_heads),
        "head_dim": int(cfg.key_size),
        "mlp_hidden": int(cfg.mlp_hidden_size),
        "cases": cases,
    }


def main():
    if not Path("Cargo.toml").is_file():
        sys.exit("run from the candle-mi root")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tensors = {}
    manifest = {
        "generator": {
            "script": "scripts/plain_tracr_validation.py",
            "python": sys.version.split()[0],
            "tracr_commit": "9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd",
            **{pkg: importlib.metadata.version(pkg)
               for pkg in ["tracr", "jax", "jaxlib", "dm-haiku", "numpy", "safetensors"]},
        },
        "tolerance": "max|delta| <= 1e-5 * max(1, max|x_layer|) per layer; same argmax "
                     "(categorical, BOS position excluded); output value with the same "
                     "tolerance (numerical)",
        "programs": {},
    }
    for name, build, vocab, max_seq_len, causal, inputs in PROGRAMS:
        entry = export(name, build, vocab, max_seq_len, causal, inputs, tensors)
        manifest["programs"][name] = entry
        print(f"{name}: {entry['n_layer']} layers, {entry['n_head']} heads of "
              f"{entry['head_dim']}, d_model {entry['d_model']}, MLP {entry['mlp_hidden']}, "
              f"{entry['output']}, causal {causal}; "
              + "; ".join(" ".join(c["tokens"]) + " -> " + " ".join(c["decoded"])
                          for c in entry["cases"]))
    save_file(tensors, str(OUT_DIR / "tracr.safetensors"))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"wrote {len(tensors)} tensors to {OUT_DIR}")


if __name__ == "__main__":
    main()
