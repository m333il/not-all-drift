from __future__ import annotations

import math

import numpy as np
import pytest

from prompt_optimization.feature_similarity import (
    compare_feature_shifts,
    matched_feature_permutation,
    permutation_null,
)


def test_identical_shifts_have_unit_similarity() -> None:
    shifts = np.array(
        [
            [1.0, 0.0, -2.0, 0.5],
            [2.0, 1.0, -1.0, 0.0],
            [0.0, 2.0, -3.0, 1.0],
        ]
    )
    result = compare_feature_shifts(
        shifts,
        shifts.copy(),
        decoder_norms=np.array([1.0, 2.0, 1.0, 0.5]),
        top_ks=(2, 4),
    )

    assert result.aggregate["mean_shift_cosine"] == pytest.approx(1.0)
    assert result.aggregate["mean_shift_pearson"] == pytest.approx(1.0)
    assert result.aggregate["mean_shift_spearman"] == pytest.approx(1.0)
    assert result.aggregate["weighted_jaccard"] == pytest.approx(1.0)
    assert result.aggregate["mean_per_sample_cosine"] == pytest.approx(1.0)
    assert result.aggregate["valid_per_sample_cosine"] == 3
    assert all(row["overlap"] == pytest.approx(1.0) for row in result.topk)
    assert all(row["sign_agreement"] == pytest.approx(1.0) for row in result.topk)


def test_opposite_signed_shifts_keep_magnitude_overlap_but_flip_signs() -> None:
    left = np.array([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0]])
    result = compare_feature_shifts(
        left,
        -left,
        decoder_norms=np.ones(3),
        top_ks=(3,),
    )

    assert result.aggregate["mean_shift_cosine"] == pytest.approx(-1.0)
    assert result.aggregate["mean_shift_pearson"] == pytest.approx(-1.0)
    assert result.aggregate["mean_shift_spearman"] == pytest.approx(-1.0)
    assert result.aggregate["weighted_jaccard"] == pytest.approx(1.0)
    assert result.topk[0]["overlap"] == pytest.approx(1.0)
    assert result.topk[0]["sign_agreement"] == pytest.approx(0.0)


def test_topk_ties_are_broken_by_feature_index() -> None:
    left = np.array([[1.0, 1.0, 0.0, 0.0]])
    right = np.array([[0.0, 1.0, 1.0, 0.0]])
    result = compare_feature_shifts(
        left,
        right,
        decoder_norms=np.ones(4),
        top_ks=(1, 2),
    )

    assert result.topk[0]["top_features_a"] == (0,)
    assert result.topk[0]["top_features_b"] == (1,)
    assert result.topk[0]["overlap"] == pytest.approx(0.0)
    assert result.topk[1]["overlap"] == pytest.approx(0.5)


def test_zero_sample_directions_are_nan_and_counted_explicitly() -> None:
    left = np.array([[0.0, 0.0], [1.0, 0.0]])
    right = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = compare_feature_shifts(
        left,
        right,
        decoder_norms=np.ones(2),
        top_ks=(1,),
    )

    assert math.isnan(result.per_sample_cosine[0])
    assert result.per_sample_cosine[1] == pytest.approx(1.0)
    assert result.aggregate["valid_per_sample_cosine"] == 1
    assert result.aggregate["valid_per_sample_cosine_fraction"] == pytest.approx(0.5)


def test_popularity_matched_permutation_never_crosses_rank_bins() -> None:
    popularity = np.arange(12, dtype=float)
    permutation = matched_feature_permutation(
        popularity,
        bins=3,
        rng=np.random.default_rng(7),
    )

    assert sorted(permutation.tolist()) == list(range(12))
    for start in (0, 4, 8):
        assert set(permutation[start : start + 4]) == set(range(start, start + 4))


def test_permutation_null_is_deterministic_and_reports_both_null_types() -> None:
    left = np.array([[1.0, 0.0, 2.0, 0.0], [2.0, 1.0, 0.0, 0.0]])
    right = left.copy()
    kwargs = {
        "decoder_norms": np.ones(4),
        "popularity": np.array([0.1, 0.2, 0.8, 0.9]),
        "top_ks": (2,),
        "repeats": 8,
        "seed": 123,
        "popularity_bins": 2,
    }
    first = permutation_null(left, right, **kwargs)
    second = permutation_null(left, right, **kwargs)

    assert first == second
    assert {row["null_type"] for row in first} == {
        "unrestricted_permutation",
        "manual_frequency_matched_permutation",
    }
    assert {row["metric"] for row in first} >= {
        "mean_shift_cosine",
        "weighted_jaccard",
        "overlap_at_2",
        "sign_agreement_at_2",
    }
    assert all(row["repeats"] == 8 for row in first)
