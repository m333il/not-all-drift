"""Similarity metrics for Manual-relative shifts in a shared SAE dictionary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureSimilarityResult:
    """Aggregate, top-k, and paired per-sample feature-shift comparisons."""

    aggregate: dict[str, float | int]
    topk: list[dict[str, Any]]
    per_sample_cosine: np.ndarray
    per_sample_weighted_cosine: np.ndarray


def _validate_shifts(
    left: np.ndarray,
    right: np.ndarray,
    decoder_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    decoder_norms = np.asarray(decoder_norms, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Feature shifts must have shape [samples, features]")
    if left.shape != right.shape or left.shape[0] == 0 or left.shape[1] == 0:
        raise ValueError("Feature shifts must have equal, non-empty shapes")
    if decoder_norms.shape != (left.shape[1],):
        raise ValueError("decoder_norms must have shape [features]")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Feature shifts must contain only finite values")
    if not np.isfinite(decoder_norms).all() or (decoder_norms < 0).any():
        raise ValueError("decoder_norms must be finite and non-negative")
    return left, right, decoder_norms


def _cosine(left: np.ndarray, right: np.ndarray, *, eps: float) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= eps:
        return float("nan")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _row_cosine(left: np.ndarray, right: np.ndarray, *, eps: float) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    output = np.full(left.shape[0], np.nan, dtype=np.float64)
    valid = denominator > eps
    output[valid] = np.clip(
        np.einsum("ij,ij->i", left[valid], right[valid]) / denominator[valid],
        -1.0,
        1.0,
    )
    return output


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks, including exact ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray, *, eps: float) -> float:
    return _cosine(left - left.mean(), right - right.mean(), eps=eps)


def _spearman(left: np.ndarray, right: np.ndarray, *, eps: float) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right), eps=eps)


def _top_indices(weights: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("top-k values must be positive")
    count = min(k, len(weights))
    indices = np.arange(len(weights))
    if count == len(weights):
        return np.lexsort((indices, -weights))
    threshold = np.partition(weights, len(weights) - count)[len(weights) - count]
    greater = indices[weights > threshold]
    equal = indices[weights == threshold]
    selected = np.concatenate((greater, np.sort(equal)[: count - len(greater)]))
    return selected[np.lexsort((selected, -weights[selected]))]


def _compare_global_moments(
    mean_left: np.ndarray,
    mean_right: np.ndarray,
    mean_absolute_left: np.ndarray,
    mean_absolute_right: np.ndarray,
    *,
    decoder_norms: np.ndarray,
    top_ks: tuple[int, ...],
    eps: float,
    ranks_left: np.ndarray | None = None,
    ranks_right: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    importance_left = mean_absolute_left * decoder_norms
    importance_right = mean_absolute_right * decoder_norms
    maximum = np.maximum(importance_left, importance_right).sum()
    aggregate = {
        "mean_shift_cosine": _cosine(mean_left, mean_right, eps=eps),
        "mean_shift_pearson": _pearson(mean_left, mean_right, eps=eps),
        "mean_shift_spearman": (
            _pearson(ranks_left, ranks_right, eps=eps)
            if ranks_left is not None and ranks_right is not None
            else _spearman(mean_left, mean_right, eps=eps)
        ),
        "weighted_jaccard": (
            float(np.minimum(importance_left, importance_right).sum() / maximum)
            if maximum > eps
            else float("nan")
        ),
    }
    topk: list[dict[str, Any]] = []
    for requested_k in top_ks:
        left_indices = _top_indices(importance_left, requested_k)
        right_indices = _top_indices(importance_right, requested_k)
        overlap_indices = np.intersect1d(left_indices, right_indices, assume_unique=True)
        count = len(left_indices)
        left_sign = np.sign(mean_left[overlap_indices])
        right_sign = np.sign(mean_right[overlap_indices])
        sign_valid = (left_sign != 0) & (right_sign != 0)
        topk.append(
            {
                "requested_k": requested_k,
                "effective_k": count,
                "overlap_count": int(len(overlap_indices)),
                "overlap": float(len(overlap_indices) / count),
                "sign_agreement": (
                    float((left_sign[sign_valid] == right_sign[sign_valid]).mean())
                    if sign_valid.any()
                    else float("nan")
                ),
                "sign_valid_count": int(sign_valid.sum()),
                "top_features_a": tuple(int(value) for value in left_indices),
                "top_features_b": tuple(int(value) for value in right_indices),
            }
        )
    return aggregate, topk


def compare_feature_shift_summaries(
    mean_left: np.ndarray,
    mean_right: np.ndarray,
    mean_absolute_left: np.ndarray,
    mean_absolute_right: np.ndarray,
    *,
    decoder_norms: np.ndarray,
    top_ks: tuple[int, ...] = (10, 50, 100, 500),
    eps: float = 1e-12,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compare pre-aggregated shifts without retaining sample-by-feature arrays.

    This is the exact global/top-k half of :func:`compare_feature_shifts`. It lets
    GPU producers aggregate large candidate matrices on-device and transfer only
    four feature-width vectors to the CPU for deterministic ranks and top-k ties.
    """
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (mean_left, mean_right, mean_absolute_left, mean_absolute_right)
    )
    if any(value.ndim != 1 or len(value) == 0 for value in arrays):
        raise ValueError("Feature-shift summaries must be non-empty vectors")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("Feature-shift summaries must share one shape")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Feature-shift summaries must be finite")
    if (arrays[2] < 0).any() or (arrays[3] < 0).any():
        raise ValueError("Mean absolute feature shifts must be non-negative")
    decoder_norms = np.asarray(decoder_norms, dtype=np.float64)
    if decoder_norms.shape != arrays[0].shape:
        raise ValueError("decoder_norms must match the feature-shift summaries")
    if not np.isfinite(decoder_norms).all() or (decoder_norms < 0).any():
        raise ValueError("decoder_norms must be finite and non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not top_ks or len(set(top_ks)) != len(top_ks) or min(top_ks) <= 0:
        raise ValueError("top_ks must contain unique positive values")
    return _compare_global_moments(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        decoder_norms=decoder_norms,
        top_ks=top_ks,
        eps=eps,
    )


def compare_feature_shifts(
    left: np.ndarray,
    right: np.ndarray,
    *,
    decoder_norms: np.ndarray,
    top_ks: tuple[int, ...] = (10, 50, 100, 500),
    eps: float = 1e-12,
) -> FeatureSimilarityResult:
    """Compare paired shifts expressed in one layer's shared SAE coordinates.

    Feature importance is ``mean(abs(delta_z)) * ||decoder_direction||``.
    Direction metrics are undefined (NaN) when either vector has zero norm.
    Top-k ties are resolved by ascending feature index.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not top_ks or len(set(top_ks)) != len(top_ks):
        raise ValueError("top_ks must contain unique positive values")
    left, right, decoder_norms = _validate_shifts(left, right, decoder_norms)
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    global_metrics, topk = compare_feature_shift_summaries(
        mean_left,
        mean_right,
        np.abs(left).mean(axis=0),
        np.abs(right).mean(axis=0),
        decoder_norms=decoder_norms,
        top_ks=top_ks,
        eps=eps,
    )
    per_sample = _row_cosine(left, right, eps=eps)
    per_sample_weighted = _row_cosine(
        left * decoder_norms[None, :],
        right * decoder_norms[None, :],
        eps=eps,
    )
    valid = np.isfinite(per_sample)
    valid_weighted = np.isfinite(per_sample_weighted)
    aggregate: dict[str, float | int] = {
        "samples": int(left.shape[0]),
        "features": int(left.shape[1]),
        **global_metrics,
        "mean_per_sample_cosine": (
            float(per_sample[valid].mean()) if valid.any() else float("nan")
        ),
        "median_per_sample_cosine": (
            float(np.median(per_sample[valid])) if valid.any() else float("nan")
        ),
        "valid_per_sample_cosine": int(valid.sum()),
        "valid_per_sample_cosine_fraction": float(valid.mean()),
        "mean_per_sample_weighted_cosine": (
            float(per_sample_weighted[valid_weighted].mean())
            if valid_weighted.any()
            else float("nan")
        ),
        "median_per_sample_weighted_cosine": (
            float(np.median(per_sample_weighted[valid_weighted]))
            if valid_weighted.any()
            else float("nan")
        ),
        "valid_per_sample_weighted_cosine": int(valid_weighted.sum()),
        "valid_per_sample_weighted_cosine_fraction": float(valid_weighted.mean()),
    }
    return FeatureSimilarityResult(
        aggregate=aggregate,
        topk=topk,
        per_sample_cosine=per_sample,
        per_sample_weighted_cosine=per_sample_weighted,
    )


def matched_feature_permutation(
    popularity: np.ndarray,
    *,
    bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute feature identities within popularity-rank bins."""
    popularity = np.asarray(popularity, dtype=np.float64)
    if popularity.ndim != 1 or len(popularity) == 0:
        raise ValueError("popularity must be a non-empty vector")
    if not np.isfinite(popularity).all():
        raise ValueError("popularity must be finite")
    if bins <= 0 or bins > len(popularity):
        raise ValueError("bins must be between one and the feature count")
    feature_indices = np.arange(len(popularity))
    ranked = np.lexsort((feature_indices, popularity))
    rank_bins = np.array_split(ranked, bins)
    return _permutation_from_bins(len(popularity), rank_bins, rng=rng)


def _permutation_from_bins(
    features: int,
    rank_bins: list[np.ndarray],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    permutation = np.arange(features)
    for members in rank_bins:
        permutation[members] = rng.permutation(members)
    return permutation


def _finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "null_p05": float("nan"),
            "null_p95": float("nan"),
            "valid_repeats": 0,
        }
    return {
        "null_mean": float(finite.mean()),
        "null_std": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
        "null_p05": float(np.quantile(finite, 0.05)),
        "null_p95": float(np.quantile(finite, 0.95)),
        "valid_repeats": int(len(finite)),
    }


def permutation_null(
    left: np.ndarray,
    right: np.ndarray,
    *,
    decoder_norms: np.ndarray,
    popularity: np.ndarray,
    top_ks: tuple[int, ...] = (10, 50, 100, 500),
    repeats: int = 200,
    seed: int = 20260825,
    popularity_bins: int = 20,
    eps: float = 1e-12,
) -> list[dict[str, float | int | str]]:
    """Summarize unrestricted and Manual-frequency-matched feature nulls."""
    left, right, decoder_norms = _validate_shifts(left, right, decoder_norms)
    popularity = np.asarray(popularity, dtype=np.float64)
    if popularity.shape != (left.shape[1],) or not np.isfinite(popularity).all():
        raise ValueError("popularity must be a finite [features] vector")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if popularity_bins > left.shape[1]:
        raise ValueError("popularity_bins cannot exceed the feature count")
    mean_left = left.mean(axis=0)
    mean_right = right.mean(axis=0)
    mean_absolute_left = np.abs(left).mean(axis=0)
    mean_absolute_right = np.abs(right).mean(axis=0)
    ranks_left = _average_ranks(mean_left)
    ranks_right = _average_ranks(mean_right)
    popularity_order = np.lexsort((np.arange(len(popularity)), popularity))
    popularity_rank_bins = list(np.array_split(popularity_order, popularity_bins))
    observed_aggregate, observed_topk = _compare_global_moments(
        mean_left,
        mean_right,
        mean_absolute_left,
        mean_absolute_right,
        decoder_norms=decoder_norms,
        top_ks=top_ks,
        eps=eps,
    )
    observed_metrics: dict[str, float] = {
        key: float(value) for key, value in observed_aggregate.items()
    }
    for row in observed_topk:
        k = int(row["requested_k"])
        observed_metrics[f"overlap_at_{k}"] = float(row["overlap"])
        observed_metrics[f"sign_agreement_at_{k}"] = float(row["sign_agreement"])

    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        "unrestricted_permutation": {metric: [] for metric in observed_metrics},
        "manual_frequency_matched_permutation": {
            metric: [] for metric in observed_metrics
        },
    }
    for _ in range(repeats):
        permutations = {
            "unrestricted_permutation": rng.permutation(left.shape[1]),
            "manual_frequency_matched_permutation": _permutation_from_bins(
                len(popularity),
                popularity_rank_bins,
                rng=rng,
            ),
        }
        for null_type, permutation in permutations.items():
            aggregate, topk = _compare_global_moments(
                mean_left,
                mean_right[permutation],
                mean_absolute_left,
                mean_absolute_right[permutation],
                decoder_norms=decoder_norms,
                top_ks=top_ks,
                eps=eps,
                ranks_left=ranks_left,
                ranks_right=ranks_right[permutation],
            )
            current = {
                key: float(value) for key, value in aggregate.items()
            }
            for row in topk:
                k = int(row["requested_k"])
                current[f"overlap_at_{k}"] = float(row["overlap"])
                current[f"sign_agreement_at_{k}"] = float(row["sign_agreement"])
            for metric, value in current.items():
                values[null_type][metric].append(value)

    rows: list[dict[str, float | int | str]] = []
    for null_type, metrics in values.items():
        for metric, samples in metrics.items():
            finite_samples = np.asarray(samples, dtype=np.float64)
            finite_samples = finite_samples[np.isfinite(finite_samples)]
            observed_value = observed_metrics[metric]
            empirical_p = (
                float((1 + (finite_samples >= observed_value).sum()) / (1 + len(finite_samples)))
                if np.isfinite(observed_value) and len(finite_samples)
                else float("nan")
            )
            rows.append(
                {
                    "null_type": null_type,
                    "metric": metric,
                    "observed": observed_value,
                    "repeats": repeats,
                    "empirical_p_greater_equal": empirical_p,
                    **_finite_summary(samples),
                }
            )
    return rows
