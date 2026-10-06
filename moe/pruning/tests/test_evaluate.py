"""Quality gates: a failed measurement must not turn into a result."""
from __future__ import annotations

import pytest

from mrd_pruning.evaluate import QualityGates, check_gates
from mrd_pruning.task import ScoreSummary


def summary(**kwargs) -> ScoreSummary:
    base = dict(n=500, f1_mean=0.33, exact_mean=0.1, empty_pred_rate=0.0,
                unparsable_rate=0.0, mixed_none_rate=0.0)
    base.update(kwargs)
    return ScoreSummary(**base)


def test_reasoning_mode_failure_stops_the_cell() -> None:
    """89-100% unparsable was the signature of the <think> budget bug, and it
    was read as a routing result for a day before anyone opened the responses."""
    with pytest.raises(RuntimeError, match="enable_thinking"):
        check_gates(summary(unparsable_rate=0.92), QualityGates(), context="cell")


def test_heavy_pruning_may_legitimately_empty_the_predictions() -> None:
    check_gates(summary(empty_pred_rate=0.9), QualityGates(), context="cell")


def test_strict_empty_turns_that_into_a_failure() -> None:
    with pytest.raises(RuntimeError, match="empty"):
        check_gates(summary(empty_pred_rate=0.99),
                    QualityGates(max_empty_pred_rate=0.5, strict_empty=True), context="cell")


def test_clean_cell_passes() -> None:
    check_gates(summary(f1_mean=0.72, unparsable_rate=0.01), QualityGates(), context="cell")


def test_pruned_model_may_produce_garbage() -> None:
    """The interesting end of the curve is where the model stops answering.

    An earlier version raised here, which deleted three GEPA cells at levels
    96, 112 and 120 - exactly where the effect being measured lives.
    """
    check_gates(summary(unparsable_rate=1.0), QualityGates(), context="cell",
                intervened=True)


def test_unmodified_model_still_must_answer() -> None:
    with pytest.raises(RuntimeError, match="unmodified model"):
        check_gates(summary(unparsable_rate=1.0), QualityGates(), context="cell",
                    intervened=False)
