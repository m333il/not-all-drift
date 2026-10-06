"""Loading a retrained router (gate) checkpoint over an already-built model.

``router.safetensors`` is a partial state dict, not a model: only the gate
matrices ``model.layers.N.mlp.gate.weight`` (plus an ``expert_bias`` buffer on
Ling, absent on Qwen). Everything else - experts, attention, embeddings - stays
as loaded.

Two properties matter and are enforced rather than assumed:

* it must be applied **after** the PEFT adapter is attached, because the wrapper
  changes module paths, and copied **in place** so the adapter keeps pointing at
  the same modules;
* every gate in the checkpoint must find its module and every gate in the model
  must be covered. A partially applied router is a silently different model, and
  this project has already lost days to interventions that quietly did nothing.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .masking import discover_gates

logger = logging.getLogger(__name__)

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.(weight|expert_bias)$")


@dataclass(frozen=True)
class RouterLoadReport:
    """What was replaced, for the run summary."""

    path: str
    sha256: str
    n_gates_in_checkpoint: int
    n_gates_replaced: int
    max_abs_delta: float

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_gates_in_checkpoint": self.n_gates_in_checkpoint,
            "n_gates_replaced": self.n_gates_replaced,
            "max_abs_delta": self.max_abs_delta,
        }


def load_router(model: nn.Module, path: str | Path) -> RouterLoadReport:
    """Copy a retrained gate checkpoint into ``model``, in place.

    Returns a report including the largest weight change: a checkpoint that
    loads cleanly but changes nothing is a mistake (usually the base router),
    and only a number can tell the difference.
    """
    import hashlib

    from safetensors.torch import load_file

    path = Path(path)
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    state = load_file(str(path))

    weights: dict[int, torch.Tensor] = {}
    biases: dict[int, torch.Tensor] = {}
    for key, tensor in state.items():
        match = LAYER_RE.search(key)
        if match is None:
            raise ValueError(f"{path.name}: unexpected key {key!r} in a router checkpoint")
        layer = int(match.group(1))
        (weights if match.group(2) == "weight" else biases)[layer] = tensor

    gates = {g.layer_idx: g for g in discover_gates(model)}
    missing_in_model = sorted(set(weights) - set(gates))
    missing_in_ckpt = sorted(set(gates) - set(weights))
    if missing_in_model:
        raise ValueError(f"{path.name}: gates for layers {missing_in_model} not found in the model")
    if missing_in_ckpt:
        raise ValueError(
            f"{path.name}: no gate weights for model layers {missing_in_ckpt}; "
            "a partially replaced router is a different model than the one intended"
        )

    max_delta = 0.0
    with torch.no_grad():
        for layer, tensor in weights.items():
            module = gates[layer].module
            new = tensor.to(module.weight.device, module.weight.dtype)
            if new.shape != module.weight.shape:
                raise ValueError(
                    f"{path.name}: layer {layer} gate is {tuple(module.weight.shape)}, "
                    f"checkpoint has {tuple(new.shape)}"
                )
            max_delta = max(max_delta, float((new - module.weight).abs().max()))
            module.weight.copy_(new)
        for layer, tensor in biases.items():
            module = gates[layer].module
            if not hasattr(module, "expert_bias"):
                raise ValueError(
                    f"{path.name}: checkpoint carries expert_bias for layer {layer} but this "
                    "gate has no such buffer (Ling has one, Qwen does not)"
                )
            module.expert_bias.copy_(tensor.to(module.expert_bias.device, module.expert_bias.dtype))

    if max_delta == 0.0:
        raise ValueError(
            f"{path.name}: loading changed no weight at all - this is the model's own router, "
            "not a retrained one"
        )
    logger.info(
        "router %s applied to %d gates, max |delta| %.4g", path.name, len(weights), max_delta
    )
    return RouterLoadReport(
        path=str(path),
        sha256=digest,
        n_gates_in_checkpoint=len(weights),
        n_gates_replaced=len(weights),
        max_abs_delta=max_delta,
    )
