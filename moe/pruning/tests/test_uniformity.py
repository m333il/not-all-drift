"""Inequality measures, checked against cases whose answer is known by hand.

A metric that only agrees with its own previous output proves nothing. Every
case here has a closed-form value: a uniform load, a single-expert load, a
half-and-half load.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrd_pruning.uniformity import (
    METRIC_NAMES,
    dead_fraction,
    effective_experts,
    gini,
    max_over_mean,
    normalised_entropy,
    summarise,
    top_share,
)

N = 32
UNIFORM = np.ones(N)
DEGENERATE = np.eye(N)[0] * 1000.0
HALF = np.array([1.0] * 16 + [0.0] * 16)


def test_gini_is_zero_for_a_uniform_load():
    assert gini(UNIFORM) == pytest.approx(0.0, abs=1e-12)


def test_gini_approaches_one_when_a_single_expert_takes_everything():
    # Exactly (n-1)/n for a one-hot load.
    assert gini(DEGENERATE) == pytest.approx((N - 1) / N, abs=1e-12)


def test_gini_of_half_the_experts_sharing_equally():
    # Half idle, half equal: the Lorenz curve gives exactly 0.5.
    assert gini(HALF) == pytest.approx(0.5, abs=1e-12)


def test_entropy_is_one_for_uniform_and_zero_for_degenerate():
    assert normalised_entropy(UNIFORM) == pytest.approx(1.0)
    assert normalised_entropy(DEGENERATE) == pytest.approx(0.0)


def test_effective_experts_counts_the_experts_actually_used():
    assert effective_experts(UNIFORM) == pytest.approx(N)
    assert effective_experts(DEGENERATE) == pytest.approx(1.0)
    assert effective_experts(HALF) == pytest.approx(16.0)


def test_max_over_mean_is_the_imbalance_factor():
    assert max_over_mean(UNIFORM) == pytest.approx(1.0)
    assert max_over_mean(DEGENERATE) == pytest.approx(N)
    assert max_over_mean(HALF) == pytest.approx(2.0)


def test_top_share_adds_up():
    assert top_share(UNIFORM, 1) == pytest.approx(1 / N)
    assert top_share(UNIFORM, 4) == pytest.approx(4 / N)
    assert top_share(DEGENERATE, 1) == pytest.approx(1.0)
    assert top_share(HALF, 4) == pytest.approx(4 / 16)


def test_top_share_caps_at_the_number_of_experts():
    assert top_share(UNIFORM, k=1000) == pytest.approx(1.0)


def test_dead_fraction_counts_unused_experts():
    assert dead_fraction(UNIFORM) == pytest.approx(0.0)
    assert dead_fraction(HALF) == pytest.approx(0.5)
    assert dead_fraction(DEGENERATE) == pytest.approx((N - 1) / N)


def test_scale_invariance():
    """Counts, not shares, go in - so scaling the whole vector changes nothing."""
    for metric in (gini, normalised_entropy, max_over_mean):
        assert metric(UNIFORM * 7919) == pytest.approx(metric(UNIFORM))
        assert metric(HALF * 7919) == pytest.approx(metric(HALF))


def test_an_all_zero_layer_reads_as_uniform_rather_than_dividing_by_zero():
    zeros = np.zeros(N)
    assert gini(zeros) == pytest.approx(0.0, abs=1e-12)
    assert normalised_entropy(zeros) == pytest.approx(1.0)
    assert effective_experts(zeros) == pytest.approx(N)


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        gini(np.array([1.0, -1.0]))


def test_summarise_returns_every_named_metric():
    out = summarise(np.array([5.0, 3.0, 1.0, 1.0]))
    assert set(out) == set(METRIC_NAMES)
    assert all(math.isfinite(v) for v in out.values())


def test_ordering_matches_intuition():
    """Uniform < half-idle < degenerate on every inequality measure."""
    for metric in (gini, max_over_mean):
        assert metric(UNIFORM) < metric(HALF) < metric(DEGENERATE)
    for metric in (normalised_entropy, effective_experts):
        assert metric(UNIFORM) > metric(HALF) > metric(DEGENERATE)
