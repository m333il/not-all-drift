"""Deterministic SAE feature sets and masked replacement vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def deterministic_topk(values: torch.Tensor, requested_k: int) -> list[int]:
    """Return a stable descending top-k, breaking exact ties by feature id."""
    flattened = values.detach().double().cpu().reshape(-1)
    if requested_k <= 0:
        raise ValueError("requested_k must be positive")
    if not len(flattened) or not bool(torch.isfinite(flattened).all()):
        raise ValueError("values must be a non-empty finite vector")
    k = min(int(requested_k), len(flattened))
    order = sorted(range(len(flattened)), key=lambda index: (-float(flattened[index]), index))
    return order[:k]


def build_pair_feature_sets(
    mean_method_shifts: Mapping[str, torch.Tensor],
    decoder_norm: torch.Tensor,
    *,
    top_ks: Sequence[int],
    sign_epsilon: float,
    random_seed: int,
    random_k: int = 64,
) -> dict[str, dict[str, list[int]]]:
    """Build pairwise sparse upper-bound, shared, unique, and random sets.

    Method shifts are SAE activation changes relative to Manual.  Shared sets
    require top-k membership in both methods and equal non-zero signs.  Direct
    sets rank the actual pairwise feature difference and therefore provide a
    sparse upper bound rather than evidence of a shared mechanism.
    """
    methods = tuple(sorted(map(str, mean_method_shifts)))
    if len(methods) < 2:
        raise ValueError("At least two method shifts are required")
    if sign_epsilon < 0:
        raise ValueError("sign_epsilon must be non-negative")
    requested = tuple(sorted(set(map(int, top_ks))))
    if not requested or min(requested) <= 0:
        raise ValueError("top_ks must contain positive values")
    norm = decoder_norm.detach().float().cpu().reshape(-1)
    if not len(norm) or not bool(torch.isfinite(norm).all()) or bool((norm < 0).any()):
        raise ValueError("decoder_norm must be a finite non-negative vector")
    shifts: dict[str, torch.Tensor] = {}
    rankings: dict[str, dict[int, list[int]]] = {}
    for method in methods:
        shift = mean_method_shifts[method].detach().float().cpu().reshape(-1)
        if shift.shape != norm.shape or not bool(torch.isfinite(shift).all()):
            raise ValueError("Method shifts must match decoder_norm")
        shifts[method] = shift
        importance = shift.abs() * norm
        rankings[method] = {k: deterministic_topk(importance, k) for k in requested}

    output: dict[str, dict[str, list[int]]] = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            pair_key = f"{left}__{right}"
            pair_shift = shifts[right] - shifts[left]
            pair_importance = pair_shift.abs() * norm
            sets: dict[str, list[int]] = {}
            for k in requested:
                left_top = set(rankings[left][k])
                right_top = set(rankings[right][k])
                intersection = left_top & right_top
                aligned = sorted(
                    feature
                    for feature in intersection
                    if abs(float(shifts[left][feature])) > sign_epsilon
                    and abs(float(shifts[right][feature])) > sign_epsilon
                    and bool(
                        torch.sign(shifts[left][feature])
                        == torch.sign(shifts[right][feature])
                    )
                )
                sets[f"direct_top{k}"] = deterministic_topk(pair_importance, k)
                sets[f"aligned_shared_top{k}"] = aligned
                sets[f"{left}_unique_top{k}"] = sorted(left_top - right_top)
                sets[f"{right}_unique_top{k}"] = sorted(right_top - left_top)

            direct_random_pool = torch.nonzero(pair_importance > 0, as_tuple=False).reshape(-1)
            excluded = set(sets[f"direct_top{max(requested)}"])
            pool = [int(index) for index in direct_random_pool.tolist() if int(index) not in excluded]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(random_seed))
            if pool:
                order = torch.randperm(len(pool), generator=generator).tolist()
                sets[f"random_top{random_k}"] = [
                    pool[index] for index in order[: min(random_k, len(pool))]
                ]
            else:
                sets[f"random_top{random_k}"] = []
            output[pair_key] = sets
    return output


def masked_feature_delta(
    feature_delta: torch.Tensor,
    decoder: torch.Tensor,
    feature_indices: Sequence[int],
) -> torch.Tensor:
    """Decode a selected feature delta without adding the decoder bias."""
    if feature_delta.ndim != 2 or decoder.ndim != 2:
        raise ValueError("feature_delta and decoder must be matrices")
    if feature_delta.shape[1] != decoder.shape[0]:
        raise ValueError("Feature width differs between activations and decoder")
    indices = [int(index) for index in feature_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("feature_indices must be unique")
    if any(index < 0 or index >= feature_delta.shape[1] for index in indices):
        raise ValueError("feature index is outside the SAE dictionary")
    if not indices:
        return torch.zeros(
            (feature_delta.shape[0], decoder.shape[1]),
            dtype=torch.float32,
            device="cpu",
        )
    selected_delta = feature_delta[:, indices].float()
    selected_decoder = decoder[indices].to(device=selected_delta.device, dtype=torch.float32)
    return (selected_delta @ selected_decoder).cpu()


def norm_match_rows(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale each vector to the corresponding reference norm."""
    if values.shape != reference.shape or values.ndim != 2:
        raise ValueError("values and reference must share shape [rows, hidden]")
    values = values.float()
    reference = reference.float()
    source_norm = values.norm(dim=-1, keepdim=True)
    target_norm = reference.norm(dim=-1, keepdim=True)
    scale = torch.where(source_norm > 1e-12, target_norm / source_norm, torch.zeros_like(source_norm))
    return values * scale
