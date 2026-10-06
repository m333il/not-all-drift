"""Retrained-router loading: all gates or nothing."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402

from mrd_pruning.routers import load_router  # noqa: E402

N_EXPERTS, HIDDEN = 8, 4


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(HIDDEN, N_EXPERTS, bias=False)
        self.experts = nn.ModuleList(nn.Identity() for _ in range(N_EXPERTS))


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Block()


class _Model(nn.Module):
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(n_layers))


def write_router(path, layers, *, scale: float = 1.0, extra: dict | None = None):
    torch.manual_seed(0)
    state = {
        f"model.layers.{i}.mlp.gate.weight": torch.randn(N_EXPERTS, HIDDEN) * scale
        for i in layers
    }
    state.update(extra or {})
    save_file(state, str(path))
    return state


def test_loads_every_gate_and_reports_the_change(tmp_path) -> None:
    model = _Model()
    path = tmp_path / "router.safetensors"
    state = write_router(path, range(3))
    report = load_router(model, path)
    assert report.n_gates_replaced == 3
    assert report.max_abs_delta > 0
    assert len(report.sha256) == 64
    for i, layer in enumerate(model.layers):
        expected = state[f"model.layers.{i}.mlp.gate.weight"]
        assert torch.allclose(layer.mlp.gate.weight, expected)


def test_partial_checkpoint_is_refused(tmp_path) -> None:
    """Replacing 2 of 3 gates yields a model nobody meant to measure."""
    model = _Model()
    path = tmp_path / "router.safetensors"
    write_router(path, [0, 1])
    with pytest.raises(ValueError, match="no gate weights for model layers \\[2\\]"):
        load_router(model, path)


def test_checkpoint_for_missing_layer_is_refused(tmp_path) -> None:
    model = _Model(n_layers=2)
    path = tmp_path / "router.safetensors"
    write_router(path, [0, 1, 2])
    with pytest.raises(ValueError, match="not found in the model"):
        load_router(model, path)


def test_no_op_load_is_refused(tmp_path) -> None:
    """Loading the model's own router changes nothing and would silently make
    a 'retrained' cell identical to the frozen one."""
    model = _Model()
    path = tmp_path / "router.safetensors"
    save_file(
        {f"model.layers.{i}.mlp.gate.weight": layer.mlp.gate.weight.detach().clone()
         for i, layer in enumerate(model.layers)},
        str(path),
    )
    with pytest.raises(ValueError, match="changed no weight"):
        load_router(model, path)


def test_shape_mismatch_is_refused(tmp_path) -> None:
    model = _Model()
    path = tmp_path / "router.safetensors"
    save_file(
        {f"model.layers.{i}.mlp.gate.weight": torch.randn(N_EXPERTS, HIDDEN + 1)
         for i in range(3)},
        str(path),
    )
    with pytest.raises(ValueError, match="checkpoint has"):
        load_router(model, path)


def test_unexpected_key_is_refused(tmp_path) -> None:
    path = tmp_path / "router.safetensors"
    write_router(path, range(3), extra={"lm_head.weight": torch.randn(2, 2)})
    with pytest.raises(ValueError, match="unexpected key"):
        load_router(_Model(), path)


def test_expert_bias_without_a_buffer_is_refused(tmp_path) -> None:
    """Ling's checkpoints carry expert_bias, Qwen's gates have no such buffer."""
    path = tmp_path / "router.safetensors"
    write_router(path, range(3), extra={
        "model.layers.0.mlp.gate.expert_bias": torch.zeros(N_EXPERTS)
    })
    with pytest.raises(ValueError, match="no such buffer"):
        load_router(_Model(), path)
