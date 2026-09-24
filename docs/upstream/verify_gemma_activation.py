"""Verify the Gemma 1 activation regression (huggingface/transformers#49051) in one tree.

Runs against whichever `transformers` is importable and reports both the config-level and
the numeric consequences of the regression, so the same script can be pointed at a baseline
and at each candidate fix and the outputs tabulated side by side.

The numeric method loads the checkpoint as a user gets it today, forwards the prompts, then
swaps every `mlp.act_fn` to `gelu_pytorch_tanh` in place and forwards again. `act_fn` is
stateless, so the second pass is exactly what this model would have computed had the
activation been right, with nothing else different.

Two controls guard the comparison. The activation control checks that `gelu` and
`gelu_pytorch_tanh` themselves agree across trees, without which no cross-tree comparison
means anything. The determinism control forwards twice with nothing changed, so a
bit-identical verdict is known to be evidence of the fix rather than of a device that cannot
tell two runs apart.
"""

import argparse
import json
import logging
import os
import tempfile

import torch

REPO = "google/gemma-2b"
PROMPTS = (
  "The capital of France is",
  "Two plus two equals",
  "Once upon a time, there was a",
)
DTYPES = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
LEGACY_ACT = "gelu"
TRAINED_ACT = "gelu_pytorch_tanh"
GRID_POINTS = 160_001


class _WarningCapture(logging.Handler):
  """Collect the log records `transformers` emits.

  `contextlib.redirect_stderr` does not see them: the logging handler holds a reference to
  the original stderr, so the records have to be taken from the logger itself.
  """

  def __init__(self) -> None:
    super().__init__()
    self.records: list[str] = []

  def emit(self, record: logging.LogRecord) -> None:
    self.records.append(record.getMessage())


_CAPTURE = _WarningCapture()


def _reset_warnings() -> None:
  """Clear the capture and the `warning_once` cache.

  `warning_once` is `@functools.lru_cache(None)`, so a warning fired by an earlier probe in
  this process would otherwise silently suppress the same warning where we want to measure it.
  """
  try:
    logging.Logger.warning_once.cache_clear()
  except AttributeError:
    pass
  _CAPTURE.records.clear()


def _drain_warnings(needle: str = "gelu") -> list[str]:
  return [message for message in _CAPTURE.records if needle in message.lower()]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
  """Largest absolute elementwise difference, computed in F32.

  Promoting first matters: under bf16 the subtraction would otherwise round away the very
  difference being measured.

  >>> _max_abs(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))
  0.5
  """
  return float((left.float() - right.float()).abs().max())


def _resolve_device(requested: str) -> str:
  if requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit("cuda requested but torch.cuda.is_available() is False")
  return requested


def _probe_config(out: dict, act2fn: dict) -> None:
  """Record what the config says, what it persists, and what a config reader resolves to."""
  from transformers import AutoConfig

  _reset_warnings()
  config = AutoConfig.from_pretrained(REPO)
  warned = _drain_warnings()
  out["config_hidden_act"] = config.hidden_act
  out["config_load_warned"] = bool(warned)
  out["config_load_warning"] = warned[0][:200] if warned else ""

  # What a consumer reading config.json, rather than instantiating GemmaMLP, resolves to.
  # This is the position from which the bug was originally found.
  out["third_party_reading_config"] = type(act2fn[config.hidden_act]).__name__

  with tempfile.TemporaryDirectory() as tmp:
    config.save_pretrained(tmp)
    with open(os.path.join(tmp, "config.json"), encoding="utf-8") as handle:
      out["save_pretrained_writes"] = json.load(handle).get("hidden_act")


def _probe_scope(out: dict) -> None:
  """A fix must correct the legacy value without disturbing deliberate ones."""
  from transformers import GemmaConfig
  from transformers.models.gemma.modeling_gemma import GemmaMLP

  def one(value: str | None) -> dict:
    kwargs = {
      "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
      "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
    }
    if value is not None:
      kwargs["hidden_act"] = value
    _reset_warnings()
    config = GemmaConfig(**kwargs)
    return {
      "config": config.hidden_act,
      "act_fn": type(GemmaMLP(config).act_fn).__name__,
      "warned": bool(_drain_warnings()),
    }

  probes = (None, LEGACY_ACT, TRAINED_ACT, "gelu_new", "silu", "relu")
  out["scope"] = {value or "<default>": one(value) for value in probes}


def _probe_numeric(out: dict, act2fn: dict, device: str, dtype: torch.dtype) -> None:
  """Load the checkpoint, forward as shipped, then forward with the trained activation."""
  from transformers import AutoModelForCausalLM, AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(REPO)
  _reset_warnings()
  model = AutoModelForCausalLM.from_pretrained(
    REPO, dtype=dtype, device_map=device, attn_implementation="eager"
  ).eval()
  warned = _drain_warnings()
  out["model_load_warned"] = bool(warned)
  out["model_load_warning"] = warned[0][:200] if warned else ""
  out["model_config_hidden_act"] = model.config.hidden_act
  out["as_loaded_act_fn"] = type(model.model.layers[0].mlp.act_fn).__name__

  def forward(act_key: str | None) -> dict[str, torch.Tensor]:
    """Forward every prompt. `act_key=None` leaves the model exactly as loaded."""
    if act_key is not None:
      for layer in model.model.layers:
        layer.mlp.act_fn = act2fn[act_key]
    logits = {}
    with torch.no_grad():
      for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        logits[prompt] = model(input_ids=ids, use_cache=False).logits[0, -1, :]  # [vocab]
    return logits

  shipped = forward(None)
  # Determinism control: nothing changed between these two, so anything other than
  # bit-identical here would make the bit-identical verdict below unreadable.
  repeat = forward(None)
  out["determinism_control_bit_identical"] = all(
    torch.equal(shipped[prompt], repeat[prompt]) for prompt in PROMPTS
  )
  reference = forward(TRAINED_ACT)

  per_prompt = {}
  for prompt in PROMPTS:
    top_shipped = shipped[prompt].float().topk(10)
    top_reference = reference[prompt].float().topk(10)
    per_prompt[prompt] = {
      "top1_shipped": round(float(top_shipped.values[0]), 4),
      "top1_reference": round(float(top_reference.values[0]), 4),
      "top10_max_abs_diff": _max_abs(top_shipped.values, top_reference.values),
      "top10_index_mismatches": int((top_shipped.indices != top_reference.indices).sum()),
      "full_vocab_max_abs_diff": _max_abs(shipped[prompt], reference[prompt]),
    }

  out["per_prompt"] = per_prompt
  out["top10_max_abs_diff_overall"] = max(
    entry["top10_max_abs_diff"] for entry in per_prompt.values()
  )
  out["top10_index_mismatches_total"] = sum(
    entry["top10_index_mismatches"] for entry in per_prompt.values()
  )
  out["full_vocab_max_abs_diff_overall"] = max(
    entry["full_vocab_max_abs_diff"] for entry in per_prompt.values()
  )
  out["bit_identical_to_reference"] = all(
    torch.equal(shipped[prompt], reference[prompt]) for prompt in PROMPTS
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("label", help="free-text name for this tree, echoed into the output")
  parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
  parser.add_argument("--dtype", default="f32", choices=tuple(DTYPES))
  parser.add_argument("--skip-model", action="store_true",
                      help="config and scope probes only, no checkpoint load")
  args = parser.parse_args()

  device = _resolve_device(args.device)

  import transformers
  from transformers.activations import ACT2FN

  transformers.logging.set_verbosity_warning()
  logging.getLogger("transformers").addHandler(_CAPTURE)

  out = {
    "label": args.label,
    "transformers_version": transformers.__version__,
    "transformers_path": os.path.dirname(transformers.__file__),
    "torch_version": torch.__version__,
    "device": device,
    "dtype": args.dtype,
  }
  if torch.cuda.is_available():
    out["gpu"] = torch.cuda.get_device_name(0)

  # Activation control: if this moves between trees, nothing else is comparable.
  grid = torch.linspace(-8.0, 8.0, GRID_POINTS, dtype=torch.float32)  # [GRID_POINTS]
  with torch.no_grad():
    out["control_gelu_vs_tanh_max_abs"] = _max_abs(
      ACT2FN[LEGACY_ACT](grid), ACT2FN[TRAINED_ACT](grid)
    )

  _probe_config(out, ACT2FN)
  _probe_scope(out)
  if not args.skip_model:
    _probe_numeric(out, ACT2FN, device, DTYPES[args.dtype])

  print("===JSON===")
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
