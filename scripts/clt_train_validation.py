# SPDX-License-Identifier: MIT OR Apache-2.0
"""Oracle fixture for `candle_mi::clt::train`: the PyTorch CLT of askesis, on one tiny fixed case.

The trainable cross-layer transcoder in `src/clt/train.rs` must reproduce askesis's reference
(`askesis/reference/scripts/clt.py`, the `CrossLayerTranscoder` its MDLM-owt CLTs were trained with):
TopK(ReLU) encode, strictly triangular decode, mean-MSE loss, and one Adam step. This script runs THAT
class -- imported from its file, not re-implemented -- on fixed random weights and inputs, and writes
every intermediate to `clt_train_reference.json`, read by `tests/validate_clt_train.rs` (bar 1e-4).

Run (CPU torch is enough), from the candle-mi root:
    <python with torch> scripts/clt_train_validation.py [path/to/askesis/reference/scripts/clt.py]
Doctests:
    <python with torch> -m doctest scripts/clt_train_validation.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

DEFAULT_CLT = Path(__file__).resolve().parents[2] / "askesis" / "reference" / "scripts" / "clt.py"
OUT = Path(__file__).resolve().parent / "clt_train_reference.json"
# Tiny, asymmetric sizes: a layer / feature / dimension mix-up cannot hide behind equal extents.
N_TOKENS, N_LAYERS, D_MODEL, N_FEATURES, TOP_K = 6, 3, 8, 16, 4
LR, BETAS, EPS = 1e-2, (0.9, 0.999), 1e-8


def nested(t: torch.Tensor) -> list:
    """A tensor as nested Python lists of floats (JSON-ready).

    >>> nested(torch.tensor([[1.0, 2.5]]))
    [[1.0, 2.5]]
    """
    return t.detach().to(torch.float64).tolist()


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("askesis_clt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    clt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLT
    ref = load_reference(clt_path)
    torch.manual_seed(20261013)
    clt = ref.CrossLayerTranscoder(n_layers=N_LAYERS, d_model=D_MODEL, features_per_layer=N_FEATURES, k=TOP_K)
    with torch.no_grad():  # non-zero biases, so both bias paths are exercised
        clt.b_enc.copy_(0.1 * torch.randn_like(clt.b_enc))
        clt.b_dec.copy_(0.1 * torch.randn_like(clt.b_dec))
    params = {"W_enc": clt.W_enc, "b_enc": clt.b_enc, "b_dec": clt.b_dec}
    params.update({f"dec_{l}": clt.dec[l].weight for l in range(N_LAYERS)})
    before = {name: nested(p) for name, p in params.items()}

    x = torch.randn(N_TOKENS, N_LAYERS, D_MODEL)
    y = torch.randn(N_TOKENS, N_LAYERS, D_MODEL)
    recon, acts = clt(x)
    loss = (recon - y).pow(2).mean()  # clt.py's reduction: the mean over (tokens, layers, d)
    opt = torch.optim.Adam(clt.parameters(), lr=LR, betas=BETAS, eps=EPS)
    opt.zero_grad()
    loss.backward()
    grads = {name: nested(p.grad) for name, p in params.items()}
    opt.step()
    after = {name: nested(p) for name, p in params.items()}

    OUT.write_text(json.dumps({
        "source": str(clt_path.name),
        "torch": torch.__version__,
        "config": {"n_tokens": N_TOKENS, "n_layers": N_LAYERS, "d_model": D_MODEL,
                   "n_features": N_FEATURES, "top_k": TOP_K},
        "adam": {"lr": LR, "beta1": BETAS[0], "beta2": BETAS[1], "eps": EPS},
        "layouts": {"W_enc": "[L, d, f]", "b_enc": "[L, f]", "b_dec": "[L, d]",
                    "dec_l": "torch Linear weight [(L - l) * d, f], output index (t - l) * d + j"},
        "params": before, "x": nested(x), "y": nested(y),
        "acts": nested(acts), "recon": nested(recon), "loss": loss.item(),
        "grads": grads, "after_step": after,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: loss {loss.item():.6f}, active per (token, layer) <= {TOP_K}, "
          f"{int((acts > 0).sum())} active of {acts.numel()}")


if __name__ == "__main__":
    main()
