"""Expert usage frequencies and the pruned sets derived from them.

Frequencies come from one of two places:

* a calibration pass in this repo (:func:`collect_counts`), which is what a
  clean experiment should use, or
* the published ``expert_counts.npz`` of an earlier routing run
  (:func:`load_counts_npz`), which is cheap but ties the pruning decision to a
  different sample and a different generation setting.

Whichever source is used, the choice is recorded in the sweep summary, because
"whose frequencies" is the independent variable of the prune-then-PEFT question:
pruning by the base model's frequencies and pruning by the arm's own are two
different experiments that look identical in a results table.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

Selection = Literal["per_layer", "global"]
STAGES = ("__all__", "system", "virtual", "comment", "question", "template",
          "reasoning", "answer")


@dataclass(frozen=True)
class ExpertCounts:
    """Per-layer expert usage counts, ``[n_layers, n_experts]``."""

    counts: np.ndarray
    layer_ids: tuple[int, ...]
    source: str
    stage: str
    n_examples: int

    def __post_init__(self) -> None:
        if self.counts.ndim != 2:
            raise ValueError(f"counts must be 2-D, got shape {self.counts.shape}")
        if len(self.layer_ids) != self.counts.shape[0]:
            raise ValueError("layer_ids length does not match counts rows")
        if not np.isfinite(self.counts).all():
            raise ValueError("counts contain non-finite values")
        if (self.counts < 0).any():
            raise ValueError("counts contain negative values")

    @property
    def n_layers(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_experts(self) -> int:
        return int(self.counts.shape[1])

    def as_metadata(self) -> dict[str, object]:
        return {
            "source": self.source,
            "stage": self.stage,
            "n_examples": self.n_examples,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "total_assignments": float(self.counts.sum()),
        }


def validate_stage_spec(stage: str) -> None:
    """Reject a malformed stage spec without reading the file."""
    if stage == "__text__":
        return
    parts = [p.strip() for p in stage.split("+") if p.strip()]
    unknown = [p for p in parts if p not in STAGES]
    if not parts or unknown:
        raise ValueError(
            f"unknown stage {stage!r}; expected '__text__', one of {STAGES}, "
            "or a '+'-joined sum of them"
        )


def resolve_stage_parts(stage: str, available: Sequence[str]) -> list[str]:
    """Turn a stage spec into the list of stage matrices to sum.

    Beyond a single stage name, two composite forms are accepted:

    * ``"a+b"`` - the sum of those stages;
    * ``"__text__"`` - every stage present except ``virtual``.

    ``__text__`` exists because ``__all__`` is not a like-for-like axis across
    arms. A prompt-tuning arm routes its own virtual tokens, and they dominate
    the count - 32 % of all assignments at m=100, 49 % at m=200, 70 % at m=500 -
    while a prefix-projected arm contributes no virtual positions at all, since
    its prefix enters through the KV cache rather than as input positions. So
    pruning both by ``__all__`` picks one arm's least-used experts mostly from
    what its trained embeddings do, and the other's purely from text.
    """
    present = [s for s in available if s != "__all__"]
    if stage == "__text__":
        parts = [s for s in present if s != "virtual"]
        if not parts:
            raise ValueError(f"__text__ needs a non-virtual stage; have {sorted(present)}")
        return parts
    parts = [p.strip() for p in stage.split("+") if p.strip()]
    unknown = [p for p in parts if p not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected from {STAGES}")
    return parts


def load_counts_npz(path: str | Path, arm: str, stage: str = "__all__") -> ExpertCounts:
    """Read one arm/stage matrix out of a published ``expert_counts.npz``.

    The published files hold two arms per cell (``base`` plus the cell's arm)
    and one matrix per stage under ``"<arm>|<stage>"``, with a ``_meta`` JSON
    blob alongside. Both are validated here rather than trusted.

    ``stage`` may also be a composite spec - see :func:`resolve_stage_parts`.
    """
    path = Path(path)
    validate_stage_spec(stage)  # before opening the file, so a typo fails fast
    with np.load(path, allow_pickle=True) as bundle:
        stored = sorted(
            k.split("|", 1)[1] for k in bundle.files
            if k != "_meta" and k.startswith(f"{arm}|")
        )
        if not stored:
            available = sorted(k for k in bundle.files if k != "_meta")
            raise KeyError(f"no stage of arm {arm!r} in {path.name}; available: {available}")
        parts = resolve_stage_parts(stage, stored)
        missing = [p for p in parts if f"{arm}|{p}" not in bundle.files]
        if missing:
            raise KeyError(
                f"{arm!r} has no stage(s) {missing} in {path.name}; available: {stored}"
            )
        key = f"{arm}|{stage}"
        counts = np.sum(
            [np.asarray(bundle[f"{arm}|{p}"], dtype=np.float64) for p in parts], axis=0
        )
        meta = json.loads(str(bundle["_meta"])) if "_meta" in bundle.files else {}
    layer_ids = tuple(int(i) for i in meta.get("layer_ids", range(counts.shape[0])))
    entry = meta.get("entries", {}).get(key, {})
    n_examples = int(meta.get("n_examples", 0))
    logger.info(
        "loaded %s from %s: %d layers x %d experts, %s assignments over n=%d",
        key, path.name, counts.shape[0], counts.shape[1],
        f"{entry.get('total_assignments', counts.sum()):.0f}", n_examples,
    )
    return ExpertCounts(
        counts=counts,
        layer_ids=layer_ids,
        source=f"{path.name}:{key}",
        stage=stage,
        n_examples=n_examples,
    )


def resolve_level(level: str | int | float, n_experts: int) -> int:
    """Turn one level spec into an absolute count of experts per layer.

    Accepts ``8`` (eight experts), ``"25%"`` and ``0.25`` (a quarter of the
    layer's experts, rounded down). Fractions are resolved against the real
    ``n_experts`` of the loaded counts, never against an assumed 128, so the
    same sweep definition means the same thing on Qwen (128) and Ling (256).
    """
    if isinstance(level, str):
        text = level.strip()
        if text.endswith("%"):
            fraction = float(text[:-1]) / 100.0
            return _fraction_to_count(fraction, n_experts)
        value: float | int = float(text) if "." in text else int(text)
    else:
        value = level
    if isinstance(value, float) and not float(value).is_integer():
        return _fraction_to_count(value, n_experts)
    count = int(value)
    if count < 0:
        raise ValueError(f"level {level!r} is negative")
    if count > n_experts:
        raise ValueError(f"level {level!r} exceeds {n_experts} experts")
    return count


def _fraction_to_count(fraction: float, n_experts: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction {fraction} outside [0, 1]")
    return int(np.floor(fraction * n_experts))


def resolve_levels(levels: Sequence[str | int | float], n_experts: int) -> list[int]:
    """Resolve a whole sweep's levels, keeping order and dropping duplicates."""
    seen: dict[int, None] = {}
    for level in levels:
        seen.setdefault(resolve_level(level, n_experts), None)
    return list(seen)


def select_pruned(
    counts: ExpertCounts,
    n_prune: int,
    *,
    top_k: int,
    selection: Selection = "per_layer",
    protect: Sequence[int] = (),
    layers: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    """Choose the experts to drop: the least used ones, ties broken by index.

    ``per_layer`` removes ``n_prune`` experts from every eligible layer, which
    keeps the surviving width uniform and makes the compute saving comparable
    across layers. ``global`` ranks every (layer, expert) pair by its share of
    that layer's load and removes the ``n_prune * n_eligible_layers`` weakest
    overall, so layers that concentrate their load lose more experts than flat
    layers do.

    ``protect`` is never pruned in any layer, for pinning a shared expert.
    ``layers`` restricts pruning to a subset - the layers left out keep every
    expert, which is how "prune only the first third" is expressed and how the
    depth of the damage becomes a variable rather than a constant.
    """
    if n_prune < 0:
        raise ValueError(f"n_prune must be non-negative, got {n_prune}")
    if n_prune == 0:
        return {}
    protected = {int(e) for e in protect}
    keepable = counts.n_experts - len(protected)
    if counts.n_experts - n_prune < top_k:
        raise ValueError(
            f"pruning {n_prune} of {counts.n_experts} leaves fewer than top_k={top_k}"
        )
    if n_prune > keepable:
        raise ValueError(f"cannot prune {n_prune} experts with {len(protected)} protected")

    eligible_rows = _eligible_rows(counts, layers)
    shares = counts.counts / np.maximum(counts.counts.sum(axis=1, keepdims=True), 1.0)
    if protected:
        shares = shares.copy()
        shares[:, sorted(protected)] = np.inf  # never among the least used

    if selection == "per_layer":
        order = np.argsort(shares, axis=1, kind="stable")
        return {
            int(counts.layer_ids[row]): sorted(int(e) for e in order[row, :n_prune])
            for row in eligible_rows
        }

    if selection == "global":
        budget = n_prune * len(eligible_rows)
        eligible = np.asarray(eligible_rows, dtype=int)
        sub = shares[eligible].reshape(-1)
        order = np.argsort(sub, kind="stable")[:budget]
        out: dict[int, list[int]] = {int(counts.layer_ids[row]): [] for row in eligible_rows}
        per_layer_cap = counts.n_experts - top_k
        for flat_idx in order:
            sub_row, col = divmod(int(flat_idx), counts.n_experts)
            layer = int(counts.layer_ids[eligible[sub_row]])
            if len(out[layer]) >= per_layer_cap or col in protected:
                continue
            out[layer].append(int(col))
        return {layer: sorted(experts) for layer, experts in out.items() if experts}

    raise ValueError(f"unknown selection {selection!r}")


def _eligible_rows(counts: ExpertCounts, layers: Sequence[int] | None) -> list[int]:
    if layers is None:
        return list(range(counts.n_layers))
    wanted = {int(layer) for layer in layers}
    index = {int(layer): row for row, layer in enumerate(counts.layer_ids)}
    unknown = sorted(wanted - set(index))
    if unknown:
        raise ValueError(f"layers not present in the counts: {unknown}")
    if not wanted:
        raise ValueError("empty layer subset - pass None to prune every layer")
    return [index[layer] for layer in sorted(wanted)]


def parse_layer_spec(spec: str, available: Sequence[int]) -> list[int] | None:
    """Parse ``"all"``, ``"0-15"``, ``"32-47"`` or ``"0,3,7"`` into layer ids."""
    text = spec.strip().lower()
    if text in ("", "all"):
        return None
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            chosen.update(range(int(lo), int(hi) + 1))
        elif part:
            chosen.add(int(part))
    unknown = sorted(chosen - set(int(layer) for layer in available))
    if unknown:
        raise ValueError(f"layer spec {spec!r} names layers that do not exist: {unknown}")
    return sorted(chosen)


def pruned_set_stats(pruned: Mapping[int, Sequence[int]], counts: ExpertCounts) -> dict[str, float]:
    """How much routed mass the pruned set carried, per the frequencies used.

    This is the number that says whether a pruning level is trivial or brutal,
    and it belongs in every summary next to F1: dropping 32 of 128 experts that
    carried 2% of the load is a different experiment from dropping 32 that
    carried 20%.
    """
    shares = counts.counts / np.maximum(counts.counts.sum(axis=1, keepdims=True), 1.0)
    index = {int(layer): row for row, layer in enumerate(counts.layer_ids)}
    lost: list[float] = []
    for layer, experts in pruned.items():
        row = index.get(int(layer))
        if row is None:
            continue
        lost.append(float(shares[row, list(experts)].sum()))
    if not lost:
        return {"mass_pruned_mean": 0.0, "mass_pruned_max": 0.0, "layers_pruned": 0}
    return {
        "mass_pruned_mean": float(np.mean(lost)),
        "mass_pruned_max": float(np.max(lost)),
        "layers_pruned": len(lost),
    }
