"""Export the compact JSON behind the appendix geometry figures.

Reads the paired activation stores and the outputs of ``e3_rank_analysis.py`` and
``civil_shift_cosine_control.py`` under ``--e3-root``. Prompt and prefix tuning use 500
virtual tokens; all conditions share the same 1000 probe examples.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportConfig:
    e3_root: Path
    output: Path
    split_seeds: tuple[int, ...] = (42, 43, 44)
    hidden_indices: tuple[int, ...] = (1, 7, 14, 21, 26)
    block_labels: tuple[int, ...] = (0, 6, 13, 20, 25)


def _store_path(root: Path, condition: str) -> Path:
    candidates = list((root / "activations" / condition).glob("*/*/acts.zarr"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"{condition}: expected one activation store, found {len(candidates)}"
        )
    return candidates[0]


def _selected_states(
    root: Path, condition: str, hidden_indices: tuple[int, ...]
) -> np.ndarray:
    store = zarr.open(str(_store_path(root, condition)), mode="r")
    return np.asarray(store["last_prompt"][:, list(hidden_indices), :], dtype=np.float32)


def _rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("...d,...d->...", left, right)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _cosine_distributions(cfg: ExportConfig) -> dict[str, Any]:
    by_pair: dict[str, list[list[float]]] = {
        "prefix_prompt": [[] for _ in cfg.hidden_indices],
        "gepa_prompt": [[] for _ in cfg.hidden_indices],
        "gepa_prefix": [[] for _ in cfg.hidden_indices],
    }
    condition_templates = {
        "gepa": "C_adapt_s{seed}_n1000",
        "prompt": "C_prompt_vt500_n1000_s{seed}",
        "prefix": "C_prefix_vt500_n1000_s{seed}",
    }
    pair_names = {
        "prefix_prompt": ("prefix", "prompt"),
        "gepa_prompt": ("gepa", "prompt"),
        "gepa_prefix": ("gepa", "prefix"),
    }

    seed_states = _selected_states(cfg.e3_root, "C_seed", cfg.hidden_indices)
    for split_seed in cfg.split_seeds:
        shifts = {
            method: _selected_states(
                cfg.e3_root,
                template.format(seed=split_seed),
                cfg.hidden_indices,
            )
            - seed_states
            for method, template in condition_templates.items()
        }
        for pair, (left, right) in pair_names.items():
            values = _rowwise_cosine(shifts[left], shifts[right])
            for layer_index in range(values.shape[1]):
                by_pair[pair][layer_index].extend(values[:, layer_index].tolist())

    return {
        "hidden_indices": list(cfg.hidden_indices),
        "block_labels": list(cfg.block_labels),
        "n_examples_per_split": int(seed_states.shape[0]),
        "split_seeds": list(cfg.split_seeds),
        "pairs": by_pair,
    }


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rank_curves(cfg: ExportConfig, payload: dict[str, Any]) -> dict[str, Any]:
    conditions = payload["conditions"]
    templates: dict[str, tuple[str, ...]] = {
        "gepa": tuple(f"C_adapt_s{seed}_n1000" for seed in cfg.split_seeds),
        "prompt": tuple(
            f"C_prompt_vt500_n1000_s{seed}" for seed in cfg.split_seeds
        ),
        "prefix": tuple(
            f"C_prefix_vt500_n1000_s{seed}" for seed in cfg.split_seeds
        ),
        "padding": tuple(f"C_seed_pad_s{seed}" for seed in cfg.split_seeds),
        "bland": ("C_bland",),
    }
    layers = list(range(1, len(next(iter(conditions.values())))))
    result: dict[str, Any] = {"layers": layers, "conditions": {}}
    for name, condition_names in templates.items():
        result["conditions"][name] = {}
        for metric in ("eta", "erank_centred"):
            result["conditions"][name][metric] = [
                _aggregate([conditions[condition][layer][metric] for condition in condition_names])
                for layer in layers
            ]
    return result


def _movement_geometry(cfg: ExportConfig, payload: dict[str, Any]) -> dict[str, Any]:
    templates: dict[str, tuple[str, ...]] = {
        "gepa": tuple(f"C_adapt_s{seed}_n1000" for seed in cfg.split_seeds),
        "prompt": tuple(
            f"C_prompt_vt500_n1000_s{seed}" for seed in cfg.split_seeds
        ),
        "prefix": tuple(
            f"C_prefix_vt500_n1000_s{seed}" for seed in cfg.split_seeds
        ),
        "padding": tuple(f"C_seed_pad_s{seed}" for seed in cfg.split_seeds),
        "bland": ("C_bland",),
    }
    layer = 18
    conditions = {
        name: _aggregate(
            [1.0 - payload[condition]["cos_centred"][layer] for condition in names]
        )
        for name, names in templates.items()
    }
    return {"layer": layer, "conditions": conditions}


def export(cfg: ExportConfig) -> None:
    rank_payload = json.loads((cfg.e3_root / "rank_analysis.json").read_text())
    cosine_payload = json.loads(
        (cfg.e3_root / "shift_cosine_control.json").read_text()
    )
    output = {
        "source": str(cfg.e3_root),
        "cosine_distributions": _cosine_distributions(cfg),
        "rank_geometry": _rank_curves(cfg, rank_payload),
        "movement_geometry": _movement_geometry(cfg, cosine_payload),
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("wrote %s", cfg.output)


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return ExportConfig(e3_root=args.e3_root, output=args.output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export(parse_args())


if __name__ == "__main__":
    main()
