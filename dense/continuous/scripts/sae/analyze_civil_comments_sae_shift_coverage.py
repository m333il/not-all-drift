#!/usr/bin/env python3
"""Measure how much prompt-induced residual drift is expressed by shared SAE codes."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from prompt_optimization.civil_comments import compute_multilabel_metrics, sha256_file
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU
from prompt_optimization.sae_shift import (
    compute_shift_coverage,
    decode_feature_shift,
    per_state_shift_metrics,
)


METHODS = ("prompt", "prefix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--manual-activation-dir", type=Path)
    parser.add_argument("--method-activation-dir", type=Path)
    parser.add_argument("--manual-predictions-file", type=Path)
    parser.add_argument("--method-predictions-file", type=Path)
    parser.add_argument("--method-name", choices=("prompt", "prefix", "gepa"))
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--sae-chunk-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def canonical_sae_entries(
    manifest: dict[str, Any],
    layers: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        layer = int(entry["layer"])
        if layer in layers and "canonical_depth" in entry["roles"]:
            if layer in selected:
                raise ValueError(f"Multiple canonical SAEs for layer {layer}")
            selected[layer] = entry
    missing = set(layers) - set(selected)
    if missing:
        raise ValueError(f"Missing canonical SAEs for layers {sorted(missing)}")
    return selected


def load_condition(
    activation_root: Path,
    condition: str,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    directory = activation_root / condition
    summary = read_json(directory / "summary.json")
    metadata = read_json(directory / "metadata.json")
    if summary.get("status") != "done" or summary.get("condition") != condition:
        raise ValueError(f"Invalid activation summary for {condition}")
    rows = metadata.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"Activation metadata rows missing for {condition}")
    states = load_file(str(directory / "states.safetensors"), device="cpu")["states"]
    if list(states.shape) != summary.get("shape") or len(rows) != len(states):
        raise ValueError(f"Activation shape/metadata mismatch for {condition}")
    return states, summary, rows


def load_condition_directory(
    directory: Path,
    *,
    expected_condition: str,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(directory / "summary.json")
    metadata = read_json(directory / "metadata.json")
    if summary.get("status") != "done" or summary.get("condition") != expected_condition:
        raise ValueError(
            f"Invalid activation summary in {directory}: expected {expected_condition}"
        )
    rows = metadata.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"Activation metadata rows missing in {directory}")
    states = load_file(str(directory / "states.safetensors"), device="cpu")["states"]
    if list(states.shape) != summary.get("shape") or len(rows) != len(states):
        raise ValueError(f"Activation shape/metadata mismatch in {directory}")
    return states, summary, rows


def fixed_subset_metrics(
    rows_by_id: dict[str, dict[str, Any]],
    sample_ids: list[str],
    *,
    labels: tuple[str, ...],
) -> dict[str, float]:
    selected = [rows_by_id[sample_id] for sample_id in sample_ids]
    targets = [tuple(map(str, row["target"])) for row in selected]
    predictions = [
        None if row.get("prediction") is None else tuple(map(str, row["prediction"]))
        for row in selected
    ]
    return compute_multilabel_metrics(targets, predictions, labels=labels)


def predictions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected prediction list in {path}")
    output: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict) or "id" not in row:
            raise TypeError(f"Malformed prediction row in {path}")
        sample_id = str(row["id"])
        if sample_id in output:
            raise ValueError(f"Duplicate prediction id {sample_id} in {path}")
        output[sample_id] = row
    return output


def set_correct(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    target = row.get("target")
    if not isinstance(target, list) or prediction is None:
        return False
    if not isinstance(prediction, list):
        raise TypeError("Multilabel prediction must be a list or null")
    return set(map(str, prediction)) == set(map(str, target))


def transition(before: bool, after: bool) -> str:
    if before and after:
        return "correct_to_correct"
    if before and not after:
        return "correct_to_wrong"
    if not before and after:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def encode_in_chunks(
    sae: GemmaScopeJumpReLU,
    states: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    chunks = [
        sae.encode(states[start : start + chunk_size])
        for start in range(0, len(states), chunk_size)
    ]
    return torch.cat(chunks, dim=0)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def aggregate_row(
    dense: torch.Tensor,
    decoded: torch.Tensor,
    manual_l0: torch.Tensor,
    method_l0: torch.Tensor,
    *,
    method: str,
    layer: int,
    group_type: str,
    group: str,
    indices: torch.Tensor,
) -> dict[str, Any]:
    selected_dense = dense[indices]
    selected_decoded = decoded[indices]
    metrics = compute_shift_coverage(selected_dense, selected_decoded).as_dict()
    metrics.update(
        {
            "method": method,
            "layer": layer,
            "group_type": group_type,
            "group": group,
            "mean_manual_l0": float(manual_l0[indices].float().mean().item()),
            "mean_method_l0": float(method_l0[indices].float().mean().item()),
            "mean_l0_delta": float(
                (method_l0[indices] - manual_l0[indices]).float().mean().item()
            ),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    pair_mode = args.method_name is not None
    pair_paths = (
        args.manual_activation_dir,
        args.method_activation_dir,
        args.manual_predictions_file,
        args.method_predictions_file,
    )
    if pair_mode:
        if any(path is None for path in pair_paths):
            raise ValueError(
                "Pair mode requires both activation directories and prediction files"
            )
        if args.activation_root is not None or args.predictions_root is not None:
            raise ValueError("Do not mix pair-mode paths with root-mode paths")
        methods = (str(args.method_name),)
        activation_directories = {
            "manual": args.manual_activation_dir,
            methods[0]: args.method_activation_dir,
        }
        prediction_files = {
            "manual": args.manual_predictions_file,
            methods[0]: args.method_predictions_file,
        }
    else:
        if any(path is not None for path in pair_paths):
            raise ValueError("Pair-mode paths require --method-name")
        if args.activation_root is None or args.predictions_root is None:
            raise ValueError("Root mode requires --activation-root and --predictions-root")
        methods = METHODS
        activation_directories = {
            condition: args.activation_root / condition
            for condition in ("manual", *methods)
        }
        prediction_files = {
            condition: args.predictions_root / condition / "original_predictions.json"
            for condition in ("manual", *methods)
        }
    layers = tuple(args.layers)
    if not layers or min(layers) < 0 or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique non-negative indices")
    if layers != tuple(sorted(layers)):
        raise ValueError("--layers must be sorted")
    for name in ("sae_chunk_size", "cpu_threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("SAE shift analysis requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        args.output_dir / "summary.json",
        args.output_dir / "aggregate_metrics.csv",
        args.output_dir / "per_sample_metrics.csv",
    )
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Refusing to overwrite artifacts in {args.output_dir}")

    states: dict[str, torch.Tensor] = {}
    summaries: dict[str, dict[str, Any]] = {}
    metadata_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in ("manual", *methods):
        states[condition], summaries[condition], metadata_rows[condition] = (
            load_condition_directory(
                activation_directories[condition],
                expected_condition=condition,
            )
        )
    manual_ids = [str(row["id"]) for row in metadata_rows["manual"]]
    manual_labels = [tuple(map(str, row["labels"])) for row in metadata_rows["manual"]]
    for condition in methods:
        if [str(row["id"]) for row in metadata_rows[condition]] != manual_ids:
            raise ValueError(f"Sample order differs for {condition}")
        if [tuple(map(str, row["labels"])) for row in metadata_rows[condition]] != manual_labels:
            raise ValueError(f"Gold labels differ for {condition}")
        if summaries[condition]["layers"] != list(layers):
            raise ValueError(f"Layer contract differs for {condition}")
        if states[condition].shape != states["manual"].shape:
            raise ValueError(f"State shape differs for {condition}")
    if summaries["manual"]["layers"] != list(layers):
        raise ValueError("Manual layer contract differs from requested layers")

    predictions = {
        condition: predictions_by_id(prediction_files[condition])
        for condition in ("manual", *methods)
    }
    for condition, rows in predictions.items():
        if not set(manual_ids).issubset(rows):
            raise ValueError(f"Prediction cache misses activation IDs for {condition}")
    manual_correct = [set_correct(predictions["manual"][sample_id]) for sample_id in manual_ids]
    method_correct = {
        method: [set_correct(predictions[method][sample_id]) for sample_id in manual_ids]
        for method in methods
    }
    transitions = {
        method: [
            transition(before, after)
            for before, after in zip(manual_correct, method_correct[method], strict=True)
        ]
        for method in methods
    }

    method_configs: dict[str, dict[str, Any]] = {}
    fixed_metrics: dict[str, dict[str, float]] = {}
    if pair_mode:
        method = methods[0]
        reference_run = Path(summaries[method]["reference_run"])
        config = read_json(reference_run / "config.json")
        labels = tuple(map(str, config["labels"]))
        method_configs[method] = {
            "method_family": (
                "frozen_text_prompt" if method == "gepa" else str(config["method"])
            ),
            "train_samples": int(config["train_samples"]),
            "split_seed": int(config["split_seed"]),
            "training_seed": int(config["training_seed"]),
            "num_virtual_tokens": (
                0 if method == "gepa" else int(config["num_virtual_tokens"])
            ),
            "reference_run": str(reference_run.resolve()),
            "frozen_prompt_sha256": summaries[method].get("frozen_prompt_sha256"),
        }
        fixed_metrics = {
            condition: fixed_subset_metrics(
                predictions[condition],
                manual_ids,
                labels=labels,
            )
            for condition in ("manual", method)
        }

    manifest = read_json(args.sae_manifest)
    entries = canonical_sae_entries(manifest, layers)
    device = torch.device("cuda:0")
    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    max_decoder_bias_cancellation_error = 0.0
    torch.cuda.reset_peak_memory_stats()
    started_at = time.time()
    with torch.inference_mode():
        for layer_index, layer in enumerate(layers):
            entry = entries[layer]
            sae_path = args.sae_snapshot / entry["path"]
            if not sae_path.is_file():
                raise FileNotFoundError(sae_path)
            sae = GemmaScopeJumpReLU.from_npz(
                sae_path,
                device=device,
                dtype=torch.float32,
            ).eval()
            manual_state = states["manual"][:, layer_index].to(device=device, dtype=torch.float32)
            manual_features = encode_in_chunks(
                sae,
                manual_state,
                chunk_size=args.sae_chunk_size,
            )
            manual_l0 = (manual_features != 0).sum(dim=-1).cpu()
            for method in methods:
                method_state = states[method][:, layer_index].to(
                    device=device,
                    dtype=torch.float32,
                )
                method_features = encode_in_chunks(
                    sae,
                    method_state,
                    chunk_size=args.sae_chunk_size,
                )
                decoded = decode_feature_shift(
                    manual_features,
                    method_features,
                    sae.W_dec,
                )
                explicit = sae.decode(method_features[:1]) - sae.decode(manual_features[:1])
                bias_cancellation_error = float(
                    (decoded[:1] - explicit).abs().max().item()
                )
                max_decoder_bias_cancellation_error = max(
                    max_decoder_bias_cancellation_error,
                    bias_cancellation_error,
                )
                if bias_cancellation_error > 1e-4:
                    raise RuntimeError(
                        "Decoder-bias cancellation check failed: "
                        f"max_abs={bias_cancellation_error:.6g}"
                    )
                dense_cpu = (method_state - manual_state).cpu()
                decoded_cpu = decoded.cpu()
                method_l0 = (method_features != 0).sum(dim=-1).cpu()
                per_state = per_state_shift_metrics(dense_cpu, decoded_cpu)
                all_indices = torch.arange(len(dense_cpu))
                overall_row = aggregate_row(
                        dense_cpu,
                        decoded_cpu,
                        manual_l0,
                        method_l0,
                        method=method,
                        layer=layer,
                        group_type="overall",
                        group="all",
                        indices=all_indices,
                    )
                if pair_mode:
                    overall_row.update(method_configs[method])
                    overall_row.update(
                        {
                            "manual_samples_f1": fixed_metrics["manual"]["samples_f1"],
                            "method_samples_f1": fixed_metrics[method]["samples_f1"],
                            "method_macro_f1": fixed_metrics[method]["macro_f1"],
                            "method_subset_accuracy": fixed_metrics[method]["subset_accuracy"],
                            "method_invalid_rate": fixed_metrics[method]["invalid_rate"],
                        }
                    )
                aggregate_rows.append(overall_row)
                group_values = {
                    "transition": transitions[method],
                    "label_cardinality": [str(len(labels)) for labels in manual_labels],
                }
                for group_type, values in group_values.items():
                    for group in sorted(set(values)):
                        indices = torch.tensor(
                            [index for index, value in enumerate(values) if value == group],
                            dtype=torch.long,
                        )
                        grouped_row = aggregate_row(
                                dense_cpu,
                                decoded_cpu,
                                manual_l0,
                                method_l0,
                                method=method,
                                layer=layer,
                                group_type=group_type,
                                group=group,
                                indices=indices,
                            )
                        if pair_mode:
                            grouped_row.update(method_configs[method])
                        aggregate_rows.append(grouped_row)
                for sample_index, sample_id in enumerate(manual_ids):
                    per_sample_row = {
                            "id": sample_id,
                            "method": method,
                            "layer": layer,
                            "labels": ",".join(manual_labels[sample_index]),
                            "label_cardinality": len(manual_labels[sample_index]),
                            "manual_correct": manual_correct[sample_index],
                            "method_correct": method_correct[method][sample_index],
                            "transition": transitions[method][sample_index],
                            "manual_l0": int(manual_l0[sample_index].item()),
                            "method_l0": int(method_l0[sample_index].item()),
                            **{
                                key: finite_or_none(float(values[sample_index].item()))
                                for key, values in per_state.items()
                                if key != "valid_direction"
                            },
                            "valid_direction": bool(
                                per_state["valid_direction"][sample_index].item()
                            ),
                    }
                    if pair_mode:
                        per_sample_row.update(
                            {
                                key: value
                                for key, value in method_configs[method].items()
                                if key != "reference_run"
                            }
                        )
                    per_sample_rows.append(per_sample_row)
                del method_state, method_features, decoded, dense_cpu, decoded_cpu
            del sae, manual_state, manual_features
            torch.cuda.empty_cache()

    max_identity_error = 0.0
    for row in aggregate_rows:
        if row["centered_shift_r_squared"] is not None:
            identity_error = abs(
                row["centered_shift_r_squared"]
                - (row["explained_variance"] - row["mean_bias_fraction"])
            )
            max_identity_error = max(max_identity_error, identity_error)
    if max_identity_error > 1e-6:
        raise RuntimeError(f"Shift metric identity failed: {max_identity_error}")

    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(args.output_dir / "per_sample_metrics.csv", per_sample_rows)
    overall = [row for row in aggregate_rows if row["group_type"] == "overall"]
    payload = {
        "status": "done",
        "scope": "collective_sae_shift_coverage",
        "baseline": "manual",
        "methods": list(methods),
        "split": summaries["manual"]["split"],
        "samples": len(manual_ids),
        "layers": list(layers),
        "anchor": summaries["manual"]["anchor"],
        "sae": {
            "manifest": str(args.sae_manifest.resolve()),
            "manifest_sha256": sha256_file(args.sae_manifest),
            "snapshot": str(args.sae_snapshot.resolve()),
            "revision": manifest["revision"],
            "entries": {str(layer): entries[layer] for layer in layers},
        },
        "activation_summaries": {
            condition: str((activation_directories[condition] / "summary.json").resolve())
            for condition in ("manual", *methods)
        },
        "prediction_files": {
            condition: str(prediction_files[condition].resolve())
            for condition in ("manual", *methods)
        },
        "method_configs": method_configs,
        "fixed_subset_metrics": fixed_metrics,
        "metrics": {
            "shift_energy_recovered": "1 - SSE(delta_h - W_dec delta_z) / SSE(delta_h)",
            "centered_shift_r_squared": "1 - SSE(delta_h - W_dec delta_z) / SST(delta_h)",
            "decoder_bias": "cancelled",
            "max_decoder_bias_cancellation_error": (
                max_decoder_bias_cancellation_error
            ),
            "max_r2_identity_error": max_identity_error,
        },
        "overall": overall,
        "artifacts": {
            "aggregate_metrics": "aggregate_metrics.csv",
            "per_sample_metrics": "per_sample_metrics.csv",
        },
        "elapsed_seconds": time.time() - started_at,
        "resource_usage": {
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated())
        },
        "environment": {
            "git_revision": git_revision(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "visible_devices": visible,
            "safetensors": importlib.metadata.version("safetensors"),
        },
    }
    write_json(args.output_dir / "summary.json", payload)
    print(json.dumps({"status": "done", "runs": len(overall), "samples": len(manual_ids)}))


if __name__ == "__main__":
    main()
