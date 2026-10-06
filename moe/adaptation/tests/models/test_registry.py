"""``build_adapter`` dispatches purely on ``model.config.model_type`` - these
tests use minimal fake models (just enough for each adapter's ``__init__`` to
succeed) rather than the fuller fixtures in the per-adapter test files, since
all that matters here is *which* adapter class gets picked, and that an
unknown ``model_type`` fails loudly instead of silently doing the wrong thing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from mrd.models.qwen3_moe import Qwen3MoeAdapter
from mrd.models.registry import MODEL_SPECS, build_adapter, resolve_model_spec


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts, self.top_k = 4, 2


def _fake_model(model_type: str, moe_module: nn.Module, path: str) -> nn.Module:
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    layer = nn.Module()
    setattr(layer, path, moe_module)
    model.model.layers.append(layer)
    return model


def test_dispatches_qwen3_moe_to_qwen3_adapter():
    gate_holder = nn.Module()
    gate_holder.gate = Qwen3MoeTopKRouter()
    model = _fake_model("qwen3_moe", gate_holder, "mlp")
    adapter = build_adapter(model)
    assert isinstance(adapter, Qwen3MoeAdapter)


def test_unknown_model_type_raises_with_a_clear_message():
    model = nn.Module()
    model.config = SimpleNamespace(model_type="some_future_architecture")
    with pytest.raises(ValueError, match="some_future_architecture"):
        build_adapter(model)


def test_resolve_model_spec_known_name():
    spec = resolve_model_spec("qwen3-2507")
    assert spec is MODEL_SPECS["qwen3-2507"]
    assert spec.revision is not None


def test_resolve_model_spec_bare_repo_id_defaults_to_instruct_stage():
    spec = resolve_model_spec("some-org/some-model")
    assert spec.repo_id == "some-org/some-model"
    assert spec.stage == "instruct"
