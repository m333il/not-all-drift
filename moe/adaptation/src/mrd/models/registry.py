"""Dispatch a loaded model to its ``RouterAdapter``, and name the checkpoints
this project actually measures.

Adding a third architecture is one line in ``_ADAPTERS`` plus a new module -
nothing in ``routing.py``/``drift.py``/``measure_routing.py`` changes.
"""
from __future__ import annotations

import torch.nn as nn

from mrd.models.base import ModelSpec, RouterAdapter
from mrd.models.qwen3_moe import Qwen3MoeAdapter
from mrd.models.gpt_oss import GptOssAdapter

# Keyed by ``model.config.model_type`` - stable across checkpoints of the same
# architecture (base and instruct share it) and independent of repo naming.
_ADAPTERS: dict[str, type[RouterAdapter]] = {
    "qwen3_moe": Qwen3MoeAdapter,
    "gpt_oss": GptOssAdapter,
}


def build_adapter(model: nn.Module) -> RouterAdapter:
    model_type = getattr(model.config, "model_type", None)
    adapter_cls = _ADAPTERS.get(model_type)
    if adapter_cls is None:
        raise ValueError(
            f"no RouterAdapter registered for model_type={model_type!r} "
            f"(known: {sorted(_ADAPTERS)}). Add one in mrd/models/ and "
            f"register it in mrd/models/registry.py._ADAPTERS."
        )
    return adapter_cls(model)


# Named checkpoints this project measures. ``--model-spec <name>`` in the
# scripts looks these up; ``--model <repo_id>`` still works standalone for
# one-off runs that don't need a stage label.
MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-2507": ModelSpec(
        "Qwen/Qwen3-30B-A3B-Instruct-2507", stage="instruct",
        revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    ),
    "gpt-oss-20b": ModelSpec(
        "openai/gpt-oss-20b", stage="instruct",
        revision="6cee5e81ee83917806bbde320786a8fb61efebee",
    ),
}


def resolve_model_spec(name_or_repo_id: str) -> ModelSpec:
    """``"qwen3-2507"`` -> the named spec; anything else is treated as a bare
    repo id with ``stage="instruct"`` (the common case for one-off runs).
    """
    if name_or_repo_id in MODEL_SPECS:
        return MODEL_SPECS[name_or_repo_id]
    return ModelSpec(name_or_repo_id, stage="instruct")
