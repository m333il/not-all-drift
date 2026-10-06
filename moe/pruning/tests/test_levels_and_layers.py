"""Control over how many experts go, and where."""
from __future__ import annotations

import numpy as np
import pytest

from mrd_pruning.frequency import (
    ExpertCounts, parse_layer_spec, resolve_level, resolve_levels, select_pruned,
)


def make_counts(n_layers: int = 4, n_experts: int = 8) -> ExpertCounts:
    # Expert e in layer l carries load (e + 1): the least used are the low ids.
    matrix = np.tile(np.arange(1, n_experts + 1, dtype=np.float64), (n_layers, 1))
    return ExpertCounts(matrix, tuple(range(n_layers)), "test", "comment", 10)


def test_absolute_percentage_and_fraction_levels_agree() -> None:
    assert resolve_level(8, 128) == 8
    assert resolve_level("8", 128) == 8
    assert resolve_level("25%", 128) == 32
    assert resolve_level(0.25, 128) == 32
    assert resolve_level("0.5", 128) == 64


def test_fractions_resolve_against_the_real_width() -> None:
    """The same sweep definition has to mean a quarter on Qwen's 128 experts
    and a quarter on Ling's 256, not a fixed 32."""
    assert resolve_level("25%", 256) == 64
    assert resolve_level("25%", 128) == 32


def test_levels_keep_order_and_drop_duplicates() -> None:
    assert resolve_levels(["0", "8", "6.25%", "16"], 128) == [0, 8, 16]


def test_bad_levels_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        resolve_level(-1, 128)
    with pytest.raises(ValueError, match="exceeds"):
        resolve_level(200, 128)
    with pytest.raises(ValueError, match="outside"):
        resolve_level("150%", 128)


def test_layer_spec_forms() -> None:
    available = list(range(48))
    assert parse_layer_spec("all", available) is None
    assert parse_layer_spec("", available) is None
    assert parse_layer_spec("0-3", available) == [0, 1, 2, 3]
    assert parse_layer_spec("0,3,7", available) == [0, 3, 7]
    assert parse_layer_spec("0-2,46-47", available) == [0, 1, 2, 46, 47]


def test_layer_spec_rejects_layers_that_do_not_exist() -> None:
    with pytest.raises(ValueError, match="do not exist"):
        parse_layer_spec("0,99", list(range(48)))


def test_layer_subset_leaves_other_layers_whole() -> None:
    counts = make_counts()
    pruned = select_pruned(counts, 2, top_k=2, layers=[1, 2])
    assert set(pruned) == {1, 2}
    assert pruned[1] == [0, 1]


def test_global_budget_scales_with_the_eligible_layers_only() -> None:
    counts = make_counts(n_layers=4, n_experts=8)
    pruned = select_pruned(counts, 2, top_k=2, selection="global", layers=[0, 1])
    assert sum(len(v) for v in pruned.values()) == 4  # 2 per eligible layer, 2 layers
    assert set(pruned) <= {0, 1}


def test_protected_experts_survive_global_selection_too() -> None:
    counts = make_counts()
    pruned = select_pruned(counts, 2, top_k=2, selection="global", protect=[0])
    assert all(0 not in experts for experts in pruned.values())


def test_unknown_layer_ids_are_rejected_before_any_work() -> None:
    with pytest.raises(ValueError, match="not present"):
        select_pruned(make_counts(), 1, top_k=2, layers=[99])
    with pytest.raises(ValueError, match="empty layer subset"):
        select_pruned(make_counts(), 1, top_k=2, layers=[])
