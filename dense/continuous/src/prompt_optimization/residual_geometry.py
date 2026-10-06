"""Pure helpers for per-example residual-stream geometry comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


BEHAVIOR_GROUPS = (
    "both_rescue",
    "prompt_only_rescue",
    "prefix_only_rescue",
    "both_still_wrong",
    "both_preserve",
    "prompt_only_breaks",
    "prefix_only_breaks",
    "both_break",
)


def behavior_group(
    manual_correct: bool,
    prompt_correct: bool,
    prefix_correct: bool,
) -> str:
    """Name one of the eight possible exact-match transition patterns."""
    key = (bool(manual_correct), bool(prompt_correct), bool(prefix_correct))
    groups = {
        (False, True, True): "both_rescue",
        (False, True, False): "prompt_only_rescue",
        (False, False, True): "prefix_only_rescue",
        (False, False, False): "both_still_wrong",
        (True, True, True): "both_preserve",
        (True, False, True): "prompt_only_breaks",
        (True, True, False): "prefix_only_breaks",
        (True, False, False): "both_break",
    }
    return groups[key]


def transition_name(before: bool, after: bool) -> str:
    """Return a compact correctness transition label."""
    return {
        (False, False): "wrong_to_wrong",
        (False, True): "wrong_to_correct",
        (True, False): "correct_to_wrong",
        (True, True): "correct_to_correct",
    }[(bool(before), bool(after))]


def cardinality_bucket(labels: Sequence[str]) -> str:
    """Bucket a multilabel target by number of positive labels."""
    count = len(labels)
    return str(count) if count < 3 else "3+"


def hidden_state_index_for_block(block: int, num_hidden_states: int) -> int:
    """Map a zero-based transformer block to its post-block hidden-state index.

    Hugging Face returns the embedding output at index 0 and post-block states at
    indices 1..num_blocks.
    """
    if num_hidden_states < 2:
        raise ValueError("num_hidden_states must include embeddings and at least one block")
    if block < 0 or block >= num_hidden_states - 1:
        raise ValueError(
            f"block {block} is outside 0..{num_hidden_states - 2} for "
            f"{num_hidden_states} hidden states"
        )
    return block + 1


def safe_cosine(left: np.ndarray, right: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity, returning NaN for a near-zero vector."""
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= eps:
        return float("nan")
    return float(np.clip(np.dot(left64, right64) / denominator, -1.0, 1.0))


def cosine_angle_degrees(cosine: float) -> float:
    """Convert a cosine similarity to an angle in degrees."""
    if np.isnan(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class GeometryResult:
    method_rows: list[dict[str, object]]
    cross_method_rows: list[dict[str, object]]
    deltas: dict[tuple[str, str, int], np.ndarray]


def compute_geometry(
    manual_states: np.ndarray,
    prompt_states: np.ndarray,
    prefix_states: np.ndarray,
    *,
    ids: Sequence[str],
    labels: Sequence[Sequence[str]],
    behaviors: Mapping[str, str],
    blocks: Sequence[int],
    seed: int,
) -> GeometryResult:
    """Compute per-example geometry at selected post-transformer-block states."""
    arrays = {
        "manual": np.asarray(manual_states),
        "prompt_tuning": np.asarray(prompt_states),
        "prefix_tuning": np.asarray(prefix_states),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Activation shapes do not match: {shapes}")
    if arrays["manual"].ndim != 3:
        raise ValueError("States must have shape [samples, hidden_states, hidden_size]")
    num_samples, num_hidden_states, _ = arrays["manual"].shape
    if len(ids) != num_samples or len(labels) != num_samples:
        raise ValueError("IDs and labels must align with the activation sample dimension")
    block_to_index = {
        int(block): hidden_state_index_for_block(int(block), num_hidden_states)
        for block in blocks
    }

    method_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    deltas: dict[tuple[str, str, int], np.ndarray] = {}
    for sample_index, sample_id in enumerate(ids):
        common = {
            "seed": int(seed),
            "id": sample_id,
            "gold_labels": ", ".join(labels[sample_index]) or "NONE",
            "gold_cardinality": len(labels[sample_index]),
            "cardinality_bucket": cardinality_bucket(labels[sample_index]),
            "behavior_group": behaviors[sample_id],
        }
        for block, hidden_index in block_to_index.items():
            manual = arrays["manual"][sample_index, hidden_index].astype(np.float32)
            method_deltas: dict[str, np.ndarray] = {}
            method_norms: dict[str, float] = {}
            manual_norm = float(np.linalg.norm(manual))
            for method in ("prompt_tuning", "prefix_tuning"):
                tuned = arrays[method][sample_index, hidden_index].astype(np.float32)
                delta = tuned - manual
                delta_norm = float(np.linalg.norm(delta))
                tuned_norm = float(np.linalg.norm(tuned))
                raw_cosine = safe_cosine(manual, tuned)
                method_deltas[method] = delta
                method_norms[method] = delta_norm
                deltas[(method, sample_id, block)] = delta
                method_rows.append(
                    {
                        **common,
                        "block": block,
                        "hidden_state_index": hidden_index,
                        "method": method,
                        "manual_norm": manual_norm,
                        "tuned_norm": tuned_norm,
                        "norm_ratio": tuned_norm / manual_norm if manual_norm else np.nan,
                        "delta_norm": delta_norm,
                        "relative_delta_norm": delta_norm / manual_norm if manual_norm else np.nan,
                        "raw_state_cosine": raw_cosine,
                        "raw_state_angle_deg": cosine_angle_degrees(raw_cosine),
                    }
                )
            delta_cosine = safe_cosine(
                method_deltas["prompt_tuning"], method_deltas["prefix_tuning"]
            )
            prefix_norm = method_norms["prefix_tuning"]
            cross_rows.append(
                {
                    **common,
                    "block": block,
                    "hidden_state_index": hidden_index,
                    "prompt_delta_norm": method_norms["prompt_tuning"],
                    "prefix_delta_norm": prefix_norm,
                    "prompt_to_prefix_delta_norm_ratio": (
                        method_norms["prompt_tuning"] / prefix_norm
                        if prefix_norm
                        else np.nan
                    ),
                    "delta_cosine": delta_cosine,
                    "delta_angle_deg": cosine_angle_degrees(delta_cosine),
                }
            )
    return GeometryResult(method_rows, cross_rows, deltas)


def select_examples(
    records: Iterable[Mapping[str, object]],
    *,
    label_names: Sequence[str],
    samples_per_group: int,
    selection_seed: int,
) -> list[str]:
    """Select deterministic behavior-, label-, and cardinality-diverse examples."""
    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    rows = [dict(record) for record in records]
    rng = np.random.default_rng(selection_seed)
    order = rng.permutation(len(rows))
    rows = [rows[index] for index in order]
    selected: list[str] = []
    selected_set: set[str] = set()
    covered_labels: set[str] = set()
    covered_cardinalities: set[str] = set()

    def row_labels(row: Mapping[str, object]) -> set[str]:
        labels = {str(label) for label in row["labels"]}  # type: ignore[arg-type]
        return labels or {"NONE"}

    def add(row: Mapping[str, object]) -> None:
        sample_id = str(row["id"])
        if sample_id in selected_set:
            return
        selected.append(sample_id)
        selected_set.add(sample_id)
        covered_labels.update(row_labels(row))
        covered_cardinalities.add(str(row["cardinality_bucket"]))

    for group in BEHAVIOR_GROUPS:
        candidates = [row for row in rows if row["behavior_group"] == group]
        for _ in range(min(samples_per_group, len(candidates))):
            remaining = [row for row in candidates if str(row["id"]) not in selected_set]
            if not remaining:
                break
            best = max(
                remaining,
                key=lambda row: (
                    len(row_labels(row) - covered_labels),
                    str(row["cardinality_bucket"]) not in covered_cardinalities,
                    len(row_labels(row)),
                ),
            )
            add(best)

    required_labels = [*label_names, "NONE"]
    for label in required_labels:
        if label in covered_labels:
            continue
        candidates = [
            row
            for row in rows
            if label in row_labels(row) and str(row["id"]) not in selected_set
        ]
        if candidates:
            add(candidates[0])

    for cardinality in ("0", "1", "2", "3+"):
        if cardinality in covered_cardinalities:
            continue
        candidates = [
            row
            for row in rows
            if row["cardinality_bucket"] == cardinality
            and str(row["id"]) not in selected_set
        ]
        if candidates:
            add(candidates[0])
    return selected
