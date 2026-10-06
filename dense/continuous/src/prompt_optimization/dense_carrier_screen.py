"""Pure helpers for dense prompt-carrier sufficiency screens."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import torch


DEFAULT_CARRIERS = (
    "answer_last",
    "trailing_4",
    "trailing_8",
    "trailing_16",
    "trailing_32",
    "all_fixed",
    "text_only",
    "all_common_real",
)

# These names deliberately distinguish the literal ``Answer:`` token inside
# the user message from the actual last rendered token used to predict the
# first generated token.  The distinction matters for Gemma's chat template,
# which appends ``<end_of_turn>\n<start_of_turn>model\n`` after ``Answer:``.
CROSS_METHOD_CARRIERS = (
    "generation_anchor",
    "generation_tail_4",
    "generation_tail_8",
    "chat_suffix",
    "task_first",
    "common_prefix_last",
    "text_marker_first",
    "text_marker_last",
    "text_last",
    "output_rules_first",
    "output_rules_last",
    "answer_first",
    "answer_last",
    "trailing_4",
    "trailing_8",
    "trailing_16",
    "trailing_32",
    "all_fixed",
    "all_common_real",
)

INTERNAL_CARRIERS = ("common_to_generation",)

_TRAILING = re.compile(r"trailing_([1-9][0-9]*)\Z")
_GENERATION_TRAILING = re.compile(r"generation_tail_([1-9][0-9]*)\Z")


def _validated_positions(values: Sequence[int], *, name: str) -> list[int]:
    positions = [int(value) for value in values]
    if not positions:
        raise ValueError(f"Carrier group {name!r} has no token positions")
    if any(right <= left for left, right in zip(positions, positions[1:], strict=False)):
        raise ValueError(f"Carrier group {name!r} positions must be strictly increasing")
    return positions


def carrier_positions(groups: Mapping[str, Sequence[int]], carrier: str) -> list[int]:
    """Resolve one semantic or trailing-window carrier from token-position groups."""
    if carrier in {"answer_last", "answer_literal_last"}:
        answer = _validated_positions(groups.get("answer", ()), name="answer")
        return answer[-1:]
    endpoint_groups = {
        "common_prefix_last": ("common_prefix", "last"),
        "text_marker_first": ("text_marker", "first"),
        "text_marker_last": ("text_marker", "last"),
        "text_last": ("text_only", "last"),
        "output_rules_first": ("output_rules_only", "first"),
        "output_rules_last": ("output_rules_only", "last"),
        "answer_first": ("answer", "first"),
    }
    if carrier in endpoint_groups:
        group_name, side = endpoint_groups[carrier]
        positions = _validated_positions(groups.get(group_name, ()), name=group_name)
        return positions[:1] if side == "first" else positions[-1:]
    valid = groups.get("valid_real", ())
    if carrier == "generation_anchor":
        return _validated_positions(valid, name="valid_real")[-1:]
    generation_trailing = _GENERATION_TRAILING.fullmatch(carrier)
    if generation_trailing is not None:
        rendered = _validated_positions(valid, name="valid_real")
        return rendered[-int(generation_trailing.group(1)) :]
    if carrier == "chat_suffix":
        rendered = _validated_positions(valid, name="valid_real")
        answer = _validated_positions(groups.get("answer", ()), name="answer")
        suffix = [position for position in rendered if position > answer[-1]]
        return _validated_positions(suffix, name="chat_suffix")
    if carrier == "task_first":
        common = _validated_positions(
            groups.get("all_common_real", ()), name="all_common_real"
        )
        return common[:1]
    if carrier == "common_to_generation":
        rendered = _validated_positions(valid, name="valid_real")
        common = _validated_positions(
            groups.get("all_common_real", ()), name="all_common_real"
        )
        suffix = [position for position in rendered if position >= common[0]]
        return _validated_positions(suffix, name="common_to_generation")
    trailing = _TRAILING.fullmatch(carrier)
    if trailing is not None:
        common = _validated_positions(
            groups.get("all_common_real", ()), name="all_common_real"
        )
        return common[-int(trailing.group(1)) :]
    if carrier not in {
        "all_fixed",
        "text_only",
        "all_common_real",
        "output_rules",
        "output_rules_only",
    }:
        raise ValueError(f"Unknown dense carrier: {carrier}")
    return _validated_positions(groups.get(carrier, ()), name=carrier)


def indices_within_superset(
    superset_rows: Sequence[Sequence[int]],
    subset_rows: Sequence[Sequence[int]],
) -> list[int]:
    """Map row-wise token positions to flattened row-major capture indices."""
    if len(superset_rows) != len(subset_rows):
        raise ValueError("Superset and subset must have the same number of rows")
    indices: list[int] = []
    offset = 0
    for row_index, (superset, subset) in enumerate(
        zip(superset_rows, subset_rows, strict=True)
    ):
        full = _validated_positions(superset, name=f"superset row {row_index}")
        selected = _validated_positions(subset, name=f"subset row {row_index}")
        lookup = {position: offset + index for index, position in enumerate(full)}
        missing = [position for position in selected if position not in lookup]
        if missing:
            raise ValueError(
                f"Subset row {row_index} contains positions outside the superset: {missing}"
            )
        indices.extend(lookup[position] for position in selected)
        offset += len(full)
    return indices


def aggregate_recovery(
    baseline_distance: torch.Tensor,
    patched_distance: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute ratio-of-sums recovery and denominator diagnostics."""
    if baseline_distance.shape != patched_distance.shape or baseline_distance.ndim != 1:
        raise ValueError("Distances must have the same one-dimensional shape")
    if not len(baseline_distance):
        raise ValueError("Distances must contain at least one sample")
    if eps <= 0:
        raise ValueError("eps must be positive")
    baseline = baseline_distance.float()
    patched = patched_distance.float()
    if not bool(torch.isfinite(baseline).all() and torch.isfinite(patched).all()):
        raise ValueError("Distances must be finite")
    denominator_sum = float(baseline.sum())
    if denominator_sum <= eps:
        recovery = float("nan")
    else:
        recovery = 1.0 - float(patched.sum()) / denominator_sum
    stable = baseline > eps
    per_sample = 1.0 - patched[stable] / baseline[stable]
    return {
        "recovery": recovery,
        "denominator_sum": denominator_sum,
        "denominator_mean": float(baseline.mean()),
        "denominator_median": float(baseline.median()),
        "patched_sum": float(patched.sum()),
        "stable_fraction": float(stable.float().mean()),
        "mean_stable_sample_recovery": (
            float(per_sample.mean()) if len(per_sample) else math.nan
        ),
    }


def replacement_delta(host_states: torch.Tensor, goal_states: torch.Tensor) -> torch.Tensor:
    """Return the direct state delta that turns a host carrier into a goal carrier."""
    if host_states.shape != goal_states.shape or host_states.ndim != 2:
        raise ValueError("Host and goal states must share shape [selected_tokens, hidden]")
    if not len(host_states):
        raise ValueError("Replacement states must not be empty")
    return goal_states.float() - host_states.float()


def replacement_sae_components(
    host_states: torch.Tensor,
    goal_states: torch.Tensor,
    host_reconstruction: torch.Tensor,
    goal_reconstruction: torch.Tensor,
    *,
    shuffle_seed: int,
) -> dict[str, torch.Tensor]:
    """Partition a directed replacement into SAE-decoded and residual deltas."""
    tensors = (host_states, goal_states, host_reconstruction, goal_reconstruction)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("Replacement component states must have shape [states, hidden]")
    if len({tuple(tensor.shape) for tensor in tensors}) != 1 or not len(host_states):
        raise ValueError("Replacement component states must share a non-empty shape")
    dense = replacement_delta(host_states, goal_states)
    sae = goal_reconstruction.float() - host_reconstruction.float()
    residual = dense - sae
    return {
        "dense": dense,
        "sae": sae,
        "residual": residual,
        "shuffled_dense": shuffled_rows(dense, seed=int(shuffle_seed)),
        "shuffled_sae": shuffled_rows(sae, seed=int(shuffle_seed) + 1),
    }


def shuffled_rows(values: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return a deterministic row derangement for a token-state control."""
    if values.ndim != 2:
        raise ValueError("values must have shape [rows, hidden]")
    if len(values) < 2:
        raise ValueError("At least two rows are required for a shuffled control")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    identity = torch.arange(len(values))
    for _ in range(10_000):
        permutation = torch.randperm(len(values), generator=generator)
        if bool((permutation != identity).all()):
            return values[permutation.to(device=values.device)]
    return values[torch.roll(identity, shifts=1).to(device=values.device)]
