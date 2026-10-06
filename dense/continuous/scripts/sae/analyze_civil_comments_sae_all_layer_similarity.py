#!/usr/bin/env python3
"""Compare Prompt, Prefix, and GEPA shifts in shared Gemma Scope coordinates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from safetensors.torch import load_file, save_file

from prompt_optimization.civil_comments import sha256_file
from prompt_optimization.feature_similarity import (
    FeatureSimilarityResult,
    compare_feature_shifts,
    permutation_null,
)
from prompt_optimization.gemma_scope import GemmaScopeJumpReLU


DEFAULT_DATASET_MARKER = "civil_comments_splits_v2_multilabel"
METHOD_ORDER = {"prompt": 0, "prefix": 1, "gepa": 2}
METHOD_LABELS = {"prompt": "Prompt", "prefix": "Prefix", "gepa": "GEPA"}
PAIR_COLORS = {
    "Prompt–Prefix": "#6A3D9A",
    "Prompt–GEPA": "#1B9E77",
    "Prefix–GEPA": "#D95F02",
    "Prompt within seeds": "#0072B2",
    "Prefix within seeds": "#E69F00",
    "GEPA within seeds": "#009E73",
}


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    directory: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.method, self.seed


@dataclass(frozen=True)
class ActivationArtifact:
    spec: RunSpec | None
    states: torch.Tensor
    states_path: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    config: dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    comparison_type: str
    left: RunSpec
    right: RunSpec

    @property
    def label(self) -> str:
        if self.comparison_type == "cross_method":
            return f"{METHOD_LABELS[self.left.method]}–{METHOD_LABELS[self.right.method]}"
        return f"{METHOD_LABELS[self.left.method]} within seeds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="METHOD:SEED=DIR",
        help="Repeat for every aligned Prompt, Prefix, or GEPA activation directory.",
    )
    parser.add_argument("--sae-manifest", type=Path, required=True)
    parser.add_argument("--sae-snapshot", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=tuple(range(26)))
    parser.add_argument("--top-k", type=int, nargs="+", default=(10, 50, 100, 500))
    parser.add_argument("--selected-layers", type=int, nargs="+", default=(0, 6, 13, 20, 25))
    parser.add_argument("--sae-chunk-size", type=int, default=64)
    parser.add_argument("--minimum-group-samples", type=int, default=1)
    parser.add_argument(
        "--allow-train-sample-mismatch",
        action="store_true",
        help=(
            "Allow compared runs to use different train_samples while still requiring "
            "the same model revision and virtual-token count."
        ),
    )
    parser.add_argument(
        "--allow-virtual-token-mismatch",
        action="store_true",
        help=(
            "Allow compared runs to use different num_virtual_tokens for m ablations."
        ),
    )
    parser.add_argument(
        "--allow-layer-subset",
        action="store_true",
        help="Select --layers from activation artifacts that contain a strict layer superset.",
    )
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--null-seed", type=int, default=20260825)
    parser.add_argument("--popularity-bins", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-dataset-marker", default=DEFAULT_DATASET_MARKER)
    parser.add_argument(
        "--activation-layout",
        choices=("anchor", "split_files", "auto"),
        default="anchor",
        help=(
            "anchor uses states.safetensors; split_files uses CONDITION_SPLIT files; "
            "auto resolves the layout independently for each artifact."
        ),
    )
    parser.add_argument("--activation-split", default="test")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], *, compressed: bool = False) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    frame = pd.DataFrame.from_records(rows)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if compressed else None)
    temporary.replace(path)


def parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise ValueError(f"Run spec must be METHOD:SEED=DIR: {value}")
    identity, raw_path = value.split("=", 1)
    parts = identity.split(":")
    if len(parts) != 2:
        raise ValueError(f"Run identity must be METHOD:SEED: {identity}")
    method = parts[0].lower()
    if method not in METHOD_ORDER:
        raise ValueError(f"Unsupported method {method!r}")
    return RunSpec(method=method, seed=int(parts[1]), directory=Path(raw_path))


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_artifact(
    directory: Path,
    spec: RunSpec | None,
    *,
    expected_dataset_marker: str,
    activation_layout: str,
    activation_split: str,
) -> ActivationArtifact:
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    expected_condition = "manual" if spec is None else spec.method
    resolved_layout = activation_layout
    if resolved_layout == "auto":
        if (directory / "metadata.json").is_file() and (
            directory / "states.safetensors"
        ).is_file():
            resolved_layout = "anchor"
        else:
            resolved_layout = "split_files"
    source_condition = (
        expected_condition
        if resolved_layout == "anchor"
        else (
            "manual"
            if spec is None
            else "adapted"
        )
    )
    if summary.get("status") != "done" or summary.get("condition") != source_condition:
        raise ValueError(f"Invalid {expected_condition} activation summary in {directory}")
    if expected_dataset_marker not in json.dumps(summary.get("split_contract", {})):
        raise ValueError(
            f"Activation is not tied to {expected_dataset_marker}: {directory}"
        )
    if resolved_layout == "anchor":
        metadata_path = directory / "metadata.json"
        states_path = directory / "states.safetensors"
    else:
        metadata_path = directory / f"{source_condition}_{activation_split}.json"
        states_path = directory / f"{source_condition}_{activation_split}.safetensors"
        summary = dict(summary)
        summary["condition"] = expected_condition
        summary["layers"] = list(summary["blocks"])
        summary["shape"] = list(summary["splits"][activation_split]["shape"])
        summary["split"] = activation_split
        summary["anchor"] = "last_common_textual_prompt_token"
    metadata = read_json(metadata_path)
    if summary.get("anchor") != "last_common_textual_prompt_token":
        raise ValueError(f"Unsupported activation anchor in {directory}")
    rows = metadata.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"Malformed metadata rows in {directory}")
    states = load_file(str(states_path), device="cpu").get("states")
    if states is None or states.ndim != 3:
        raise ValueError(f"Expected [samples,layers,d_model] states in {directory}")
    if list(states.shape) != summary.get("shape") or len(states) != len(rows):
        raise ValueError(f"State shape and metadata disagree in {directory}")
    config_path = Path(str(summary.get("reference_run", ""))) / "config.json"
    if config_path.is_file():
        config = read_json(config_path)
    else:
        raise FileNotFoundError(f"Reference-run config is required: {config_path}")
    return ActivationArtifact(
        spec=spec,
        states=states,
        states_path=states_path,
        rows=rows,
        summary=summary,
        config=config,
    )


def select_layer_subset(
    artifact: ActivationArtifact, layers: tuple[int, ...]
) -> ActivationArtifact:
    available = tuple(map(int, artifact.summary.get("layers", [])))
    if available == layers:
        return artifact
    missing = sorted(set(layers) - set(available))
    if missing:
        raise ValueError(f"Activation artifact misses requested layers {missing}")
    indices = torch.tensor([available.index(layer) for layer in layers], dtype=torch.long)
    states = artifact.states.index_select(1, indices)
    summary = dict(artifact.summary)
    summary["source_layers"] = list(available)
    summary["layers"] = list(layers)
    summary["shape"] = list(states.shape)
    return ActivationArtifact(
        spec=artifact.spec,
        states=states,
        states_path=artifact.states_path,
        rows=artifact.rows,
        summary=summary,
        config=artifact.config,
    )


def validate_artifacts(
    manual: ActivationArtifact,
    artifacts: dict[tuple[str, int], ActivationArtifact],
    layers: tuple[int, ...],
    *,
    expected_dataset_marker: str,
    allow_train_sample_mismatch: bool = False,
    allow_virtual_token_mismatch: bool = False,
) -> None:
    if not artifacts:
        raise ValueError("At least one method activation is required")
    manual_ids = [str(row.get("id")) for row in manual.rows]
    if len(set(manual_ids)) != len(manual_ids):
        raise ValueError("Manual metadata contains duplicate sample IDs")
    manual_labels = [tuple(map(str, row.get("labels", []))) for row in manual.rows]
    if tuple(map(int, manual.summary.get("layers", []))) != layers:
        raise ValueError("Manual activation layers differ from --layers")
    matched_config_fields = ["model_name", "model_revision"]
    if not allow_virtual_token_mismatch:
        matched_config_fields.append("num_virtual_tokens")
    if not allow_train_sample_mismatch:
        matched_config_fields.append("train_samples")
    expected_config = tuple(manual.config.get(field) for field in matched_config_fields)
    for key, artifact in artifacts.items():
        if [str(row.get("id")) for row in artifact.rows] != manual_ids:
            raise ValueError(f"Sample IDs/order differ for {key}")
        if [tuple(map(str, row.get("labels", []))) for row in artifact.rows] != manual_labels:
            raise ValueError(f"Gold labels differ for {key}")
        if tuple(map(int, artifact.summary.get("layers", []))) != layers:
            raise ValueError(f"Layer contract differs for {key}")
        if artifact.states.shape != manual.states.shape:
            raise ValueError(f"Activation shape differs for {key}")
        if artifact.summary.get("split") != manual.summary.get("split"):
            raise ValueError(f"Split differs for {key}")
        if tuple(artifact.config.get(field) for field in matched_config_fields) != expected_config:
            raise ValueError(f"Matched model/N/m config differs for {key}")
        if int(artifact.config.get("training_seed", -1)) != key[1]:
            raise ValueError(f"Reference-run training seed differs for {key}")
        if expected_dataset_marker not in str(artifact.config.get("split_root", "")):
            raise ValueError(f"Reference-run dataset contract differs for {key}")


def canonical_sae_entries(
    manifest: dict[str, Any], layers: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    entries: dict[int, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        layer = int(entry["layer"])
        if layer in layers and "canonical_depth" in entry.get("roles", []):
            if layer in entries:
                raise ValueError(f"Multiple canonical SAEs for layer {layer}")
            entries[layer] = entry
    missing = sorted(set(layers) - set(entries))
    if missing:
        raise ValueError(f"Canonical SAE manifest misses layers {missing}")
    return entries


def build_comparisons(specs: list[RunSpec]) -> list[Comparison]:
    by_seed: dict[int, list[RunSpec]] = {}
    by_method: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_seed.setdefault(spec.seed, []).append(spec)
        by_method.setdefault(spec.method, []).append(spec)
    comparisons: list[Comparison] = []
    for seed, members in sorted(by_seed.items()):
        ordered = sorted(members, key=lambda item: METHOD_ORDER[item.method])
        for left, right in combinations(ordered, 2):
            if left.method != right.method:
                comparisons.append(Comparison("cross_method", left, right))
    for method in sorted(by_method, key=METHOD_ORDER.get):
        members = sorted(by_method[method], key=lambda item: item.seed)
        for left, right in combinations(members, 2):
            comparisons.append(Comparison("within_method", left, right))
    if not comparisons:
        raise ValueError("Inputs yield no cross-method or cross-seed comparisons")
    return comparisons


def group_masks(rows: list[dict[str, Any]], *, minimum_samples: int) -> list[tuple[str, str, np.ndarray]]:
    labels = [tuple(map(str, row.get("labels", []))) for row in rows]
    groups: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(rows), dtype=bool))
    ]
    for label in sorted({label for sample_labels in labels for label in sample_labels}):
        mask = np.array([label in sample_labels for sample_labels in labels], dtype=bool)
        if int(mask.sum()) >= minimum_samples:
            groups.append(("gold_label", label, mask))
    cardinalities = np.array([len(sample_labels) for sample_labels in labels])
    for name, mask in (
        ("0", cardinalities == 0),
        ("1", cardinalities == 1),
        ("2plus", cardinalities >= 2),
    ):
        if int(mask.sum()) >= minimum_samples:
            groups.append(("gold_cardinality", name, mask))
    return groups


def encode_in_chunks(
    sae: GemmaScopeJumpReLU,
    states: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(states), chunk_size):
            current = states[start : start + chunk_size].to(device=device, dtype=torch.float32)
            chunks.append(sae.encode(current).cpu())
    return torch.cat(chunks).numpy()


def comparison_base(
    comparison: Comparison,
    artifacts: dict[tuple[str, int], ActivationArtifact],
    *,
    layer: int,
) -> dict[str, Any]:
    left_config = artifacts[comparison.left.key].config
    right_config = artifacts[comparison.right.key].config
    return {
        "comparison_type": comparison.comparison_type,
        "comparison": comparison.label,
        "method_a": comparison.left.method,
        "method_b": comparison.right.method,
        "seed_a": comparison.left.seed,
        "seed_b": comparison.right.seed,
        "train_samples_a": left_config.get("train_samples"),
        "train_samples_b": right_config.get("train_samples"),
        "num_virtual_tokens_a": left_config.get("num_virtual_tokens"),
        "num_virtual_tokens_b": right_config.get("num_virtual_tokens"),
        "layer": layer,
    }


def append_result(
    result: FeatureSimilarityResult,
    *,
    base: dict[str, Any],
    group_type: str,
    group: str,
    sample_ids: list[str],
    labels: list[tuple[str, ...]],
    mask: np.ndarray,
    aggregate_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
) -> None:
    aggregate_rows.append(
        {**base, "group_type": group_type, "group": group, **result.aggregate}
    )
    for row in result.topk:
        topk_rows.append(
            {
                **base,
                "group_type": group_type,
                "group": group,
                **row,
                "top_features_a": json.dumps(row["top_features_a"]),
                "top_features_b": json.dumps(row["top_features_b"]),
            }
        )
    if group_type != "overall":
        return
    selected_indices = np.flatnonzero(mask)
    for offset, sample_index in enumerate(selected_indices):
        cosine = float(result.per_sample_cosine[offset])
        weighted = float(result.per_sample_weighted_cosine[offset])
        per_sample_rows.append(
            {
                **base,
                "sample_id": sample_ids[sample_index],
                "labels": "|".join(labels[sample_index]) or "NONE",
                "label_cardinality": len(labels[sample_index]),
                "cosine": cosine if np.isfinite(cosine) else None,
                "decoder_weighted_cosine": weighted if np.isfinite(weighted) else None,
                "valid_cosine": bool(np.isfinite(cosine)),
                "valid_decoder_weighted_cosine": bool(np.isfinite(weighted)),
            }
        )


def stable_null_seed(base_seed: int, comparison: Comparison, layer: int) -> int:
    identity = (
        f"{base_seed}|{comparison.comparison_type}|{comparison.left.method}|"
        f"{comparison.left.seed}|{comparison.right.method}|{comparison.right.seed}|{layer}"
    )
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")


def _save_figure(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_layer_trajectories(
    aggregate: pd.DataFrame,
    topk: pd.DataFrame,
    output_dir: Path,
) -> None:
    overall = aggregate[aggregate["group_type"] == "overall"].copy()
    overlap = topk[
        (topk["group_type"] == "overall") & (topk["requested_k"] == 100)
    ][["comparison_type", "comparison", "layer", "overlap"]]
    overlap = overlap.groupby(["comparison_type", "comparison", "layer"], as_index=False).mean()
    metrics = (
        ("mean_shift_cosine", "Signed mean-shift cosine"),
        ("weighted_jaccard", "Weighted Jaccard"),
        ("mean_per_sample_weighted_cosine", "Mean paired cosine (decoder-weighted)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)
    fig.patch.set_facecolor("#f4f0e6")
    for axis, (metric, title) in zip(axes.flat[:3], metrics, strict=True):
        grouped = overall.groupby(
            ["comparison_type", "comparison", "layer"], as_index=False
        )[metric].agg(["mean", "std"]).reset_index()
        for (comparison_type, label), frame in grouped.groupby(
            ["comparison_type", "comparison"], sort=False
        ):
            frame = frame.sort_values("layer")
            axis.plot(
                frame["layer"],
                frame["mean"],
                color=PAIR_COLORS.get(label, "#666666"),
                linestyle="-" if comparison_type == "cross_method" else "--",
                marker="o" if comparison_type == "cross_method" else None,
                linewidth=2,
                label=label,
            )
            if frame["std"].notna().any():
                std = frame["std"].fillna(0)
                axis.fill_between(frame["layer"], frame["mean"] - std, frame["mean"] + std, alpha=0.1)
        axis.set_title(title)
    axis = axes.flat[3]
    for (comparison_type, label), frame in overlap.groupby(
        ["comparison_type", "comparison"], sort=False
    ):
        axis.plot(
            frame["layer"],
            frame["overlap"],
            color=PAIR_COLORS.get(label, "#666666"),
            linestyle="-" if comparison_type == "cross_method" else "--",
            marker="o" if comparison_type == "cross_method" else None,
            linewidth=2,
            label=label,
        )
    axis.set_title("Overlap@100")
    for axis in axes.flat:
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Decoder block")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "All-layer SAE feature similarity · solid=cross-method, dashed=seed ceiling",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_figure(fig, output_dir / "all_layer_similarity")


def plot_selected_distributions(
    per_sample: pd.DataFrame,
    *,
    selected_layers: tuple[int, ...],
    output_dir: Path,
) -> None:
    selected = per_sample[
        (per_sample["comparison_type"] == "cross_method")
        & per_sample["layer"].isin(selected_layers)
        & per_sample["decoder_weighted_cosine"].notna()
    ]
    if selected.empty:
        return
    fig, axes = plt.subplots(1, len(selected_layers), figsize=(4.2 * len(selected_layers), 4.4), sharey=True)
    axes_array = np.atleast_1d(axes)
    for axis, layer in zip(axes_array, selected_layers, strict=True):
        layer_frame = selected[selected["layer"] == layer]
        for label, frame in layer_frame.groupby("comparison", sort=False):
            values = frame["decoder_weighted_cosine"].to_numpy(dtype=float)
            if len(values) > 1 and np.ptp(values) > 1e-12:
                sns.kdeplot(
                    x=values,
                    ax=axis,
                    color=PAIR_COLORS.get(label, "#666666"),
                    linewidth=2,
                    label=label,
                    common_norm=False,
                )
            else:
                axis.axvline(values[0], color=PAIR_COLORS.get(label, "#666666"), label=label)
        axis.axvline(0, color="#444444", linewidth=0.8)
        axis.set_title(f"Block {layer}")
        axis.set_xlabel("Paired Δz cosine (decoder-weighted)")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes_array[0].set_ylabel("KDE density")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Cross-method per-sample feature-shift distributions", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(fig, output_dir / "selected_layer_per_sample_kde")


def plot_conditional_heatmaps(aggregate: pd.DataFrame, output_dir: Path) -> None:
    cross = aggregate[aggregate["comparison_type"] == "cross_method"]
    for group_type, stem in (
        ("gold_label", "similarity_by_gold_label"),
        ("gold_cardinality", "similarity_by_gold_cardinality"),
    ):
        selected = cross[cross["group_type"] == group_type]
        if selected.empty:
            continue
        comparisons = list(dict.fromkeys(selected["comparison"]))
        fig, axes = plt.subplots(1, len(comparisons), figsize=(5.2 * len(comparisons), 4.8), squeeze=False)
        for axis, label in zip(axes[0], comparisons, strict=True):
            frame = selected[selected["comparison"] == label]
            matrix = frame.pivot_table(
                index="group", columns="layer", values="mean_shift_cosine", aggfunc="mean"
            )
            sns.heatmap(matrix, ax=axis, cmap="vlag", center=0, vmin=-1, vmax=1, cbar=axis is axes[0, -1])
            axis.set_title(label)
            axis.set_xlabel("Decoder block")
            axis.set_ylabel(group_type.replace("gold_", "Gold "))
        fig.suptitle("Signed mean SAE-shift cosine", y=1.02)
        fig.tight_layout()
        _save_figure(fig, output_dir / stem)


def main() -> None:
    args = parse_args()
    layers = tuple(args.layers)
    top_ks = tuple(args.top_k)
    selected_layers = tuple(layer for layer in args.selected_layers if layer in layers)
    if layers != tuple(sorted(set(layers))) or not layers or min(layers) < 0:
        raise ValueError("--layers must be unique, sorted, and non-negative")
    if top_ks != tuple(sorted(set(top_ks))) or min(top_ks) <= 0:
        raise ValueError("--top-k must be unique, sorted, and positive")
    for name in ("sae_chunk_size", "minimum_group_samples", "cpu_threads", "popularity_bins"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.null_repeats < 0:
        raise ValueError("--null-repeats must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires an available GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.device == "cuda" and (visible is None or "," in visible):
        raise RuntimeError("Expose exactly one GPU through CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(args.cpu_threads)
    specs = [parse_run_spec(value) for value in args.run]
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("Duplicate METHOD:SEED run specification")
    comparisons = build_comparisons(specs)
    load_options = {
        "expected_dataset_marker": args.expected_dataset_marker,
        "activation_layout": args.activation_layout,
        "activation_split": args.activation_split,
    }
    manual = load_artifact(args.manual_dir, None, **load_options)
    artifacts = {
        spec.key: load_artifact(spec.directory, spec, **load_options) for spec in specs
    }
    if args.allow_layer_subset:
        manual = select_layer_subset(manual, layers)
        artifacts = {
            key: select_layer_subset(artifact, layers)
            for key, artifact in artifacts.items()
        }
    validate_artifacts(
        manual,
        artifacts,
        layers,
        expected_dataset_marker=args.expected_dataset_marker,
        allow_train_sample_mismatch=args.allow_train_sample_mismatch,
        allow_virtual_token_mismatch=args.allow_virtual_token_mismatch,
    )
    manifest = read_json(args.sae_manifest)
    sae_entries = canonical_sae_entries(manifest, layers)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir()

    sample_ids = [str(row["id"]) for row in manual.rows]
    labels = [tuple(map(str, row.get("labels", []))) for row in manual.rows]
    masks = group_masks(manual.rows, minimum_samples=args.minimum_group_samples)
    device = torch.device(args.device)
    aggregate_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    signature_tensors: dict[str, torch.Tensor] = {}
    peak_cuda_bytes = 0
    started = time.time()
    for layer_position, layer in enumerate(layers):
        print(json.dumps({"stage": "layer_start", "layer": layer}), flush=True)
        entry = sae_entries[layer]
        sae_path = args.sae_snapshot / entry["path"]
        if not sae_path.is_file() or sha256_file(sae_path) != entry["sha256"]:
            raise ValueError(f"Missing or checksum-mismatched SAE: {sae_path}")
        sae = GemmaScopeJumpReLU.from_npz(sae_path, device=device, dtype=torch.float32).eval()
        manual_features = encode_in_chunks(
            sae,
            manual.states[:, layer_position],
            device=device,
            chunk_size=args.sae_chunk_size,
        )
        popularity = (manual_features != 0).mean(axis=0, dtype=np.float64)
        decoder_norms = sae.W_dec.detach().float().norm(dim=1).cpu().numpy()
        signature_tensors[f"layer_{layer:02d}/manual_firing_frequency"] = torch.from_numpy(
            popularity.astype(np.float32)
        )
        signature_tensors[f"layer_{layer:02d}/decoder_norm"] = torch.from_numpy(
            decoder_norms.astype(np.float32)
        )
        shifts: dict[tuple[str, int], np.ndarray] = {}
        for spec in specs:
            method_features = encode_in_chunks(
                sae,
                artifacts[spec.key].states[:, layer_position],
                device=device,
                chunk_size=args.sae_chunk_size,
            )
            shift = method_features - manual_features
            shifts[spec.key] = shift
            key = f"{spec.method}_seed{spec.seed}/layer_{layer:02d}"
            signature_tensors[f"{key}/mean_delta"] = torch.from_numpy(
                shift.mean(axis=0, dtype=np.float64).astype(np.float32)
            )
            signature_tensors[f"{key}/importance"] = torch.from_numpy(
                (np.abs(shift).mean(axis=0, dtype=np.float64) * decoder_norms).astype(np.float32)
            )

        for comparison in comparisons:
            base = comparison_base(comparison, artifacts, layer=layer)
            left = shifts[comparison.left.key]
            right = shifts[comparison.right.key]
            for group_type, group, mask in masks:
                result = compare_feature_shifts(
                    left[mask],
                    right[mask],
                    decoder_norms=decoder_norms,
                    top_ks=top_ks,
                )
                append_result(
                    result,
                    base=base,
                    group_type=group_type,
                    group=group,
                    sample_ids=sample_ids,
                    labels=labels,
                    mask=mask,
                    aggregate_rows=aggregate_rows,
                    topk_rows=topk_rows,
                    per_sample_rows=per_sample_rows,
                )
            if args.null_repeats and comparison.comparison_type == "cross_method":
                for row in permutation_null(
                    left,
                    right,
                    decoder_norms=decoder_norms,
                    popularity=popularity,
                    top_ks=top_ks,
                    repeats=args.null_repeats,
                    seed=stable_null_seed(args.null_seed, comparison, layer),
                    popularity_bins=args.popularity_bins,
                ):
                    null_rows.append({**base, **row})
        if args.device == "cuda":
            peak_cuda_bytes = max(peak_cuda_bytes, int(torch.cuda.max_memory_allocated()))
        del sae, manual_features, shifts
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "layer_done", "layer": layer}), flush=True)

    aggregate_path = args.output_dir / "aggregate_similarity.csv"
    topk_path = args.output_dir / "topk_similarity.csv"
    per_sample_path = args.output_dir / "per_sample_cosine.csv.gz"
    signatures_path = args.output_dir / "feature_signatures.safetensors"
    write_csv(aggregate_path, aggregate_rows)
    write_csv(topk_path, topk_rows)
    write_csv(per_sample_path, per_sample_rows, compressed=True)
    if null_rows:
        write_csv(args.output_dir / "null_summary.csv", null_rows)
    save_file(signature_tensors, str(signatures_path))
    aggregate_frame = pd.DataFrame.from_records(aggregate_rows)
    topk_frame = pd.DataFrame.from_records(topk_rows)
    per_sample_frame = pd.DataFrame.from_records(per_sample_rows)
    plot_layer_trajectories(aggregate_frame, topk_frame, figures_dir)
    plot_selected_distributions(
        per_sample_frame,
        selected_layers=selected_layers,
        output_dir=figures_dir,
    )
    plot_conditional_heatmaps(aggregate_frame, figures_dir)

    input_artifacts = {
        "manual": {
            "directory": str(args.manual_dir.resolve()),
            "states_sha256": sha256_file(manual.states_path),
            "reference_run": manual.summary.get("reference_run"),
            "reference_config": manual.config,
        },
        "runs": [
            {
                "method": spec.method,
                "seed": spec.seed,
                "directory": str(spec.directory.resolve()),
                "states_sha256": sha256_file(artifacts[spec.key].states_path),
                "reference_run": artifacts[spec.key].summary.get("reference_run"),
                "reference_config": artifacts[spec.key].config,
            }
            for spec in specs
        ],
    }
    write_json(
        args.output_dir / "summary.json",
        {
            "status": "done",
            "scope": "civil_comments_sae_all_layer_feature_similarity",
            "dataset_version": args.expected_dataset_marker,
            "split": manual.summary["split"],
            "anchor": manual.summary["anchor"],
            "samples": len(sample_ids),
            "layers": list(layers),
            "methods": sorted({spec.method for spec in specs}, key=METHOD_ORDER.get),
            "training_seeds": sorted({spec.seed for spec in specs}),
            "comparison_contract": {
                "allow_train_sample_mismatch": args.allow_train_sample_mismatch,
                "allow_virtual_token_mismatch": args.allow_virtual_token_mismatch,
                "allow_layer_subset": args.allow_layer_subset,
                "train_samples_by_run": {
                    f"{spec.method}:{spec.seed}": artifacts[spec.key].config.get(
                        "train_samples"
                    )
                    for spec in specs
                },
                "virtual_tokens_by_run": {
                    f"{spec.method}:{spec.seed}": artifacts[spec.key].config.get(
                        "num_virtual_tokens"
                    )
                    for spec in specs
                },
            },
            "comparisons": {
                "cross_method": sum(c.comparison_type == "cross_method" for c in comparisons),
                "within_method_seed_ceiling": sum(
                    c.comparison_type == "within_method" for c in comparisons
                ),
            },
            "definitions": {
                "delta_z": "z_method(x,l)-z_manual(x,l) in the same layer SAE",
                "signed_signature": "mean_x delta_z",
                "importance": "mean_x abs(delta_z_j) * L2_norm(W_dec[j])",
                "weighted_jaccard": "sum_j min(w_a,w_b) / sum_j max(w_a,w_b)",
                "overlap_at_k": "intersection size of deterministic importance top-k divided by effective k",
                "sign_agreement_at_k": "equal nonzero signs of mean delta_z on top-k intersection",
                "per_sample_cosine": "cosine(delta_z_a(x),delta_z_b(x)); zero directions are NaN",
                "decoder_weighted_cosine": "cosine(delta_z_a(x)*||W_dec||,delta_z_b(x)*||W_dec||)",
            },
            "nulls": {
                "repeats": args.null_repeats,
                "base_seed": args.null_seed,
                "unrestricted": "permute method-B feature identities over the full dictionary",
                "matched": (
                    "permute method-B feature identities within Manual SAE firing-frequency "
                    f"rank bins; bins={args.popularity_bins}"
                ),
                "scope": "overall cross-method global signatures only",
            },
            "interpretation_boundary": (
                "All 26 layers are descriptive. Mechanistic feature claims require the "
                "separate functional R_SAE|dense gate and later causal transfer."
            ),
            "sae": {
                "manifest": str(args.sae_manifest.resolve()),
                "manifest_sha256": sha256_file(args.sae_manifest),
                "snapshot": str(args.sae_snapshot.resolve()),
                "revision": manifest.get("revision"),
                "entries": {str(layer): sae_entries[layer] for layer in layers},
            },
            "inputs": input_artifacts,
            "artifacts": {
                "aggregate_similarity": aggregate_path.name,
                "topk_similarity": topk_path.name,
                "per_sample_cosine": per_sample_path.name,
                "null_summary": "null_summary.csv" if null_rows else None,
                "feature_signatures": signatures_path.name,
                "figures": "figures/",
            },
            "elapsed_seconds": time.time() - started,
            "resource_usage": {"peak_cuda_memory_bytes": peak_cuda_bytes},
            "environment": {
                "git_revision": git_revision(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "seaborn": sns.__version__,
                "safetensors": importlib.metadata.version("safetensors"),
                "cuda": torch.version.cuda,
                "visible_devices": visible,
                "device": args.device,
                "cpu_threads": args.cpu_threads,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "done",
                "samples": len(sample_ids),
                "layers": len(layers),
                "comparisons": len(comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
