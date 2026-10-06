"""Gini must equal its definition, not just itself."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mrd_pruning.routing_stats import (  # noqa: E402
    gini, gini_from_lorenz, layer_shares, layer_stats,
)


@pytest.mark.parametrize("n", [8, 32, 128, 256])
def test_uniform_is_zero_and_degenerate_is_the_finite_bound(n: int) -> None:
    """For a finite set of n experts the maximum is (n-1)/n, not 1 - the bound
    the project's metric reference states."""
    assert gini(np.ones(n)) == pytest.approx(0.0, abs=1e-12)
    one_hot = np.zeros(n)
    one_hot[0] = 1.0
    assert gini(one_hot) == pytest.approx((n - 1) / n)


def test_half_idle_half_equal_is_exactly_one_half() -> None:
    """The reading of the 0.5 threshold: half the experts share everything
    equally, half get nothing."""
    p = np.concatenate([np.ones(16), np.zeros(16)])
    assert gini(p) == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_the_lorenz_area(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for p in (rng.random(64), rng.pareto(1.5, 128), rng.random(32) ** 4):
        assert gini(p) == pytest.approx(gini_from_lorenz(p), abs=1e-12)


def test_gini_is_scale_invariant() -> None:
    """Counts and shares must give the same answer, so a table built on either
    is comparable with the other."""
    p = np.random.default_rng(0).random(64)
    assert gini(p) == pytest.approx(gini(p * 1234.5))


def test_negative_load_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gini(np.array([1.0, -1.0, 2.0]))


def test_empty_layer_is_nan_not_zero() -> None:
    """A layer that saw nothing is missing data, not perfect balance."""
    assert np.isnan(gini(np.zeros(16)))


def test_layer_shares_leave_empty_rows_alone() -> None:
    counts = np.array([[1.0, 3.0], [0.0, 0.0]])
    got = layer_shares(counts)
    assert got[0] == pytest.approx([0.25, 0.75])
    assert got[1] == pytest.approx([0.0, 0.0])


def test_layer_stats_on_a_known_distribution() -> None:
    p = np.array([0.5, 0.25, 0.25, 0.0])
    st = layer_stats(p)
    assert st["top1_share"] == pytest.approx(0.5)
    assert st["hottest_over_uniform"] == pytest.approx(2.0)
    assert st["n_eff"] == pytest.approx(1 / (0.25 + 0.0625 + 0.0625))
    assert st["untouched"] == pytest.approx(0.25)
