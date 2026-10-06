"""Frequency loading and pruned-set selection."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mrd_pruning.frequency import (
    ExpertCounts, load_counts_npz, pruned_set_stats, select_pruned,
)


def make_counts(matrix: list[list[float]]) -> ExpertCounts:
    array = np.asarray(matrix, dtype=np.float64)
    return ExpertCounts(
        counts=array,
        layer_ids=tuple(range(array.shape[0])),
        source="test",
        stage="comment",
        n_examples=10,
    )


def test_per_layer_selection_drops_the_least_used() -> None:
    counts = make_counts([[10, 1, 5, 2], [1, 10, 2, 5]])
    pruned = select_pruned(counts, 2, top_k=2)
    assert pruned == {0: [1, 3], 1: [0, 2]}


def test_ties_break_by_index_so_the_set_is_reproducible() -> None:
    counts = make_counts([[5, 5, 5, 5]])
    assert select_pruned(counts, 2, top_k=2) == {0: [0, 1]}


def test_zero_level_prunes_nothing() -> None:
    assert select_pruned(make_counts([[1, 2, 3, 4]]), 0, top_k=2) == {}


def test_protected_experts_survive_even_when_unused() -> None:
    counts = make_counts([[0, 1, 5, 9]])
    assert select_pruned(counts, 1, top_k=2, protect=[0]) == {0: [1]}


def test_global_selection_takes_from_the_flattest_layers_first() -> None:
    """Layer 1 spreads its load, layer 0 concentrates it: with a shared budget
    the weakest shares overall come from the concentrated layer's tail."""
    counts = make_counts([[100, 1, 1, 1], [25, 25, 25, 25]])
    pruned = select_pruned(counts, 1, top_k=2, selection="global")
    assert pruned == {0: [1, 2]}


def test_refuses_to_prune_below_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        select_pruned(make_counts([[1, 2, 3, 4]]), 3, top_k=2)


def test_pruned_mass_reports_what_the_cut_carried() -> None:
    counts = make_counts([[90, 10], [50, 50]])
    stats = pruned_set_stats({0: [1], 1: [1]}, counts)
    assert stats["mass_pruned_mean"] == pytest.approx(0.30)
    assert stats["mass_pruned_max"] == pytest.approx(0.50)
    assert stats["layers_pruned"] == 2


def test_counts_validation_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ExpertCounts(np.zeros(4), (0,), "s", "comment", 1)
    with pytest.raises(ValueError, match="negative"):
        ExpertCounts(np.array([[-1.0, 1.0]]), (0,), "s", "comment", 1)
    with pytest.raises(ValueError, match="layer_ids"):
        ExpertCounts(np.ones((2, 2)), (0,), "s", "comment", 1)


def test_load_counts_npz_reads_the_published_layout(tmp_path) -> None:
    meta = {
        "layer_ids": [0, 1],
        "num_experts": 3,
        "top_k": 2,
        "n_examples": 500,
        "entries": {"base|comment": {"total_assignments": 12.0}},
    }
    path = tmp_path / "expert_counts.npz"
    np.savez(
        path,
        _meta=json.dumps(meta),
        **{"base|comment": np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
           "gepa|comment": np.ones((2, 3))},
    )
    counts = load_counts_npz(path, arm="base", stage="comment")
    assert counts.layer_ids == (0, 1)
    assert counts.n_experts == 3
    assert counts.n_examples == 500
    assert counts.as_metadata()["source"].endswith("base|comment")


def test_load_counts_npz_lists_available_keys_on_a_miss(tmp_path) -> None:
    path = tmp_path / "expert_counts.npz"
    np.savez(path, **{"base|comment": np.ones((1, 2))})
    with pytest.raises(KeyError, match="base\\|comment"):
        load_counts_npz(path, arm="prompt_tuning", stage="comment")


def test_unknown_stage_is_rejected_before_touching_the_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        load_counts_npz(tmp_path / "missing.npz", arm="base", stage="virtualz")
