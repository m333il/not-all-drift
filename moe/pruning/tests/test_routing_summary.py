"""Concentration must be per layer, not pooled across layers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

_spec = importlib.util.spec_from_file_location(
    "summarize_routing_grid",
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_routing_grid.py")
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _cell(counts: np.ndarray) -> dict:
    return {"stages": {"__all__": counts}, "meta": {"n_examples": 1}}


def test_pooling_across_layers_would_hide_the_imbalance() -> None:
    """Each layer routes independently and has its own hot experts. Summing the
    counts first lets an expert idle in one layer be covered by another, which
    turns a perfectly concentrated model into a perfectly uniform one."""
    n_layers, n_experts = 8, 8
    counts = np.zeros((n_layers, n_experts))
    for layer in range(n_layers):
        counts[layer, layer] = 100.0  # every layer uses exactly one expert

    st = mod.summarise(_cell(counts), None)

    # Per layer this is maximal concentration: one expert takes everything.
    assert st["gini"] == pytest.approx((n_experts - 1) / n_experts, abs=1e-9)
    assert st["dead_pct"] == pytest.approx(100.0 * (n_experts - 1) / n_experts)
    assert st["n_eff"] == pytest.approx(1.0)
    assert st["hottest"] == pytest.approx(n_experts)

    # Pooled first, the same counts look perfectly balanced - the bug this
    # guards against.
    pooled = counts.sum(axis=0)
    assert mod.gini(pooled / pooled.sum()) == pytest.approx(0.0, abs=1e-9)


def test_uniform_routing_scores_zero() -> None:
    counts = np.full((4, 16), 10.0)
    st = mod.summarise(_cell(counts), None)
    assert st["gini"] == pytest.approx(0.0, abs=1e-9)
    assert st["dead_pct"] == pytest.approx(0.0)
    assert st["n_eff"] == pytest.approx(16.0)
    assert st["hottest"] == pytest.approx(1.0)


def test_drift_is_bounded_by_two_and_scale_free() -> None:
    """L1 between shares is twice the total variation, so 128 experts and 32
    experts land on the same scale."""
    for n_experts in (32, 128):
        a = np.zeros((3, n_experts)); a[:, 0] = 1.0
        b = np.zeros((3, n_experts)); b[:, 1] = 1.0
        st = mod.summarise(_cell(a), _cell(b))
        assert st["drift_all"] == pytest.approx(2.0)


def test_drift_of_a_cell_against_itself_is_zero() -> None:
    counts = np.random.default_rng(0).random((5, 12)) + 0.1
    cell = _cell(counts)
    st = mod.summarise(cell, _cell(counts.copy()))
    assert st["drift_all"] == pytest.approx(0.0, abs=1e-12)
