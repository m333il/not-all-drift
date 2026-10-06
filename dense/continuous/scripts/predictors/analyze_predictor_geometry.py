#!/usr/bin/env python3
"""Compare affine residual predictors across methods, objectives, and seeds.

The script has two independent parts.  Matrix geometry only needs predictor
checkpoints.  Sample geometry additionally needs aligned teacher caches whose
baseline and adapted states were captured on the same trajectories.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


@dataclass(frozen=True)
class RunSpec:
    method: str
    objective: str
    seed: int
    directory: Path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run(value: str) -> RunSpec:
    key, separator, raw_path = value.partition("=")
    fields = key.split(":")
    if not separator or len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "Run must be METHOD:OBJECTIVE:SEED=/path/to/fit"
        )
    try:
        seed = int(fields[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed in {value!r}") from error
    return RunSpec(fields[0], fields[1], seed, Path(raw_path))


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("Expected NAME=/path")
    return name, Path(raw_path)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator > 0 else math.nan


def mean_and_sample_sd(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "sample_sd": float(tensor.std(unbiased=True)) if len(tensor) > 1 else 0.0,
        "count": len(tensor),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_affine(run: RunSpec) -> dict[str, torch.Tensor]:
    path = run.directory / "predictor.safetensors"
    state = load_file(str(path), device="cpu")
    if set(state) != {"weight", "bias"}:
        raise ValueError(f"Expected full affine checkpoint at {path}, got {sorted(state)}")
    width = state["bias"].numel()
    if state["weight"].shape != (width, width):
        raise ValueError(f"Expected square predictor matrix at {path}")
    return {key: value.float() for key, value in state.items()}


def matrix_geometry(
    runs: list[RunSpec],
    states: dict[tuple[str, str, int], dict[str, torch.Tensor]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_objective: dict[str, list[RunSpec]] = {}
    for run in runs:
        by_objective.setdefault(run.objective, []).append(run)
    rows: list[dict[str, Any]] = []
    for objective, selected in sorted(by_objective.items()):
        methods = sorted({run.method for run in selected})
        seeds = sorted({run.seed for run in selected})
        for left, right in itertools.combinations(methods, 2):
            for seed in seeds:
                left_key, right_key = (left, objective, seed), (right, objective, seed)
                if left_key not in states or right_key not in states:
                    continue
                rows.append(
                    {
                        "comparison": "cross_method_same_seed",
                        "objective": objective,
                        "left_method": left,
                        "right_method": right,
                        "left_seed": seed,
                        "right_seed": seed,
                        "weight_cosine": cosine(
                            states[left_key]["weight"], states[right_key]["weight"]
                        ),
                        "bias_cosine": cosine(
                            states[left_key]["bias"], states[right_key]["bias"]
                        ),
                    }
                )
        for method in methods:
            available = sorted(run.seed for run in selected if run.method == method)
            for left_seed, right_seed in itertools.combinations(available, 2):
                left_key = (method, objective, left_seed)
                right_key = (method, objective, right_seed)
                rows.append(
                    {
                        "comparison": "within_method_cross_seed",
                        "objective": objective,
                        "left_method": method,
                        "right_method": method,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "weight_cosine": cosine(
                            states[left_key]["weight"], states[right_key]["weight"]
                        ),
                        "bias_cosine": cosine(
                            states[left_key]["bias"], states[right_key]["bias"]
                        ),
                    }
                )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["comparison"],
            row["objective"],
            row["left_method"],
            row["right_method"],
        )
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        summary.append(
            {
                "comparison": key[0],
                "objective": key[1],
                "left_method": key[2],
                "right_method": key[3],
                "weight_cosine": mean_and_sample_sd(
                    [float(row["weight_cosine"]) for row in values]
                ),
                "bias_cosine": mean_and_sample_sd(
                    [float(row["bias_cosine"]) for row in values]
                ),
            }
        )
    return rows, summary


def cache_tensors(directory: Path, split: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    contract = read_json(directory / "contract.json")
    entry = contract["splits"][split]
    data = load_file(str(directory / entry["path"]), device="cpu")
    baseline_key = "baseline" if "baseline" in data else "manual"
    adapted_key = "adapted" if "adapted" in data else "target"
    required = {baseline_key, adapted_key, "offsets"}
    if not required.issubset(data):
        raise ValueError(f"Cache {directory} is missing {sorted(required - set(data))}")
    baseline, adapted, offsets = data[baseline_key], data[adapted_key], data["offsets"].long()
    if baseline.shape != adapted.shape or offsets.ndim != 1 or int(offsets[-1]) != len(baseline):
        raise ValueError(f"Malformed aligned trajectory cache: {directory}")
    ids = list(map(str, entry.get("ids", contract.get("samples", {}).get(split, []))))
    if ids and len(ids) != len(offsets) - 1:
        raise ValueError(f"Cache IDs do not match offsets: {directory}")
    return {
        "baseline": baseline.float(),
        "adapted": adapted.float(),
        "offsets": offsets,
    }, ids


def sample_geometry(
    caches: dict[str, Path],
    models: dict[str, dict[str, torch.Tensor]],
    split: str,
) -> dict[str, Any]:
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    ids_by_method: dict[str, list[str]] = {}
    for method, directory in caches.items():
        loaded[method], ids_by_method[method] = cache_tensors(directory, split)
    populated_ids = [ids for ids in ids_by_method.values() if ids]
    if populated_ids and any(ids != populated_ids[0] for ids in populated_ids[1:]):
        raise ValueError("Sample order differs across method caches")
    row_counts = {len(value["offsets"]) - 1 for value in loaded.values()}
    if len(row_counts) != 1:
        raise ValueError("Trajectory row count differs across method caches")

    method_rows: list[dict[str, Any]] = []
    true_mean: dict[str, torch.Tensor] = {}
    true_anchor: dict[str, torch.Tensor] = {}
    predicted_anchor: dict[str, torch.Tensor] = {}
    for method, data in sorted(loaded.items()):
        state = models[method]
        baseline = data["baseline"]
        true_delta = data["adapted"] - baseline
        predicted_delta = F.linear(baseline, state["weight"], state["bias"])
        per_position_cosine = F.cosine_similarity(predicted_delta, true_delta, dim=-1)
        mean = true_delta.mean(0)
        total_energy = float(true_delta.square().sum())
        eta = len(true_delta) * float(mean.square().sum()) / total_energy
        input_term = F.linear(baseline, state["weight"], None)
        anchors = data["offsets"][:-1]
        true_mean[method] = mean
        true_anchor[method] = true_delta[anchors]
        predicted_anchor[method] = predicted_delta[anchors]
        method_rows.append(
            {
                "method": method,
                "positions": len(baseline),
                "trajectories": len(anchors),
                "mean_prediction_to_truth_cosine": float(per_position_cosine.mean()),
                "prediction_rmse": float((predicted_delta - true_delta).square().mean().sqrt()),
                "mean_direction_energy_fraction": eta,
                "mean_input_dependent_norm": float(input_term.norm(dim=-1).mean()),
                "bias_norm": float(state["bias"].norm()),
            }
        )

    pair_rows = []
    for left, right in itertools.combinations(sorted(loaded), 2):
        true_cosines = F.cosine_similarity(true_anchor[left], true_anchor[right], dim=-1)
        predicted_cosines = F.cosine_similarity(
            predicted_anchor[left], predicted_anchor[right], dim=-1
        )
        pair_rows.append(
            {
                "left_method": left,
                "right_method": right,
                "mean_direction_cosine": cosine(true_mean[left], true_mean[right]),
                "median_first_step_true_shift_cosine": float(true_cosines.median()),
                "median_first_step_predicted_shift_cosine": float(predicted_cosines.median()),
                "absolute_median_reproduction_error": abs(
                    float(predicted_cosines.median() - true_cosines.median())
                ),
            }
        )
    return {"split": split, "methods": method_rows, "pairs": pair_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--cache", action="append", type=parse_named_path, default=[])
    parser.add_argument("--cache-split", default="test")
    parser.add_argument("--sample-objective", default="dense")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    keys = [(run.method, run.objective, run.seed) for run in args.run]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate METHOD:OBJECTIVE:SEED run")
    states = {key: load_affine(run) for key, run in zip(keys, args.run, strict=True)}
    matrix_rows, matrix_summary = matrix_geometry(args.run, states)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "matrix_pairwise.csv", matrix_rows)
    sample_report = None
    caches = dict(args.cache)
    if caches:
        selected = {}
        for method in caches:
            key = (method, args.sample_objective, args.sample_seed)
            if key not in states:
                raise ValueError(f"No predictor run for sample cache {key}")
            selected[method] = states[key]
        sample_report = sample_geometry(caches, selected, args.cache_split)
        write_csv(args.output_dir / "sample_method_geometry.csv", sample_report["methods"])
        write_csv(args.output_dir / "sample_pair_geometry.csv", sample_report["pairs"])
    payload = {
        "status": "done",
        "matrix_geometry": matrix_summary,
        "sample_geometry": sample_report,
        "definitions": {
            "matrix_cosine": "cosine between flattened affine matrices W",
            "mean_direction_energy_fraction": "N * ||mean(delta)||^2 / sum_i ||delta_i||^2",
            "input_dependent_term": "W h; bias excluded",
            "first_step": "the trajectory position at offsets[row]",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
