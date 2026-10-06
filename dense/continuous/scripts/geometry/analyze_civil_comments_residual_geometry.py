#!/usr/bin/env python3
"""Analyze per-example residual shifts for Prompt and Prefix Tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from safetensors.torch import load_file

from prompt_optimization.residual_geometry import (
    BEHAVIOR_GROUPS,
    behavior_group,
    compute_geometry,
    safe_cosine,
    select_examples,
    transition_name,
)


BEHAVIOR_LABELS = {
    "both_rescue": "both rescue",
    "prompt_only_rescue": "Prompt only rescues",
    "prefix_only_rescue": "Prefix only rescues",
    "both_still_wrong": "both remain wrong",
    "both_preserve": "both preserve",
    "prompt_only_breaks": "Prompt only breaks",
    "prefix_only_breaks": "Prefix only breaks",
    "both_break": "both break",
}
METHOD_LABELS = {
    "prompt_tuning": "Prompt Tuning",
    "prefix_tuning": "Prefix Tuning",
}
METHOD_STYLES = {"prompt_tuning": "-", "prefix_tuning": (0, (6, 3))}
METHOD_COLORS = {"prompt_tuning": "#0072B2", "prefix_tuning": "#E69F00"}
CARDINALITY_COLORS = {0: "#7f7f7f", 1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
DEFAULT_LABEL_ORDER = (
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--prompt-activations-root", type=Path, required=True)
    parser.add_argument("--prefix-activations-root", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--num-virtual-tokens", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--blocks", type=int, nargs="+", default=(0, 6, 13, 20, 25))
    parser.add_argument("--samples-per-group", type=int, default=3)
    parser.add_argument(
        "--prompt-run-template",
        default="prompt_tuning_n{n}_split{seed}_train{seed}_vt{m}",
    )
    parser.add_argument(
        "--prefix-run-template",
        default="prefix_tuning_n{n}_split{seed}_train{seed}_vt{m}",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use a new versioned directory."
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    (path / "tables").mkdir(exist_ok=True)


def activation_run_dir(
    root: Path,
    template: str,
    *,
    method: str,
    n: int,
    m: int,
    seed: int,
) -> Path:
    relative = template.format(method=method, n=n, m=m, seed=seed)
    path = root / relative
    if not path.is_dir():
        raise FileNotFoundError(f"Missing activation directory: {path}")
    return path


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return result


def metadata_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(f"Invalid activation metadata: {path}")
    return rows


def load_states(path: Path) -> np.ndarray:
    tensor = load_file(path)["states"]
    return tensor.float().numpy()


def validate_id_order(reference: list[str], candidate: list[str], source: Path) -> None:
    if reference != candidate:
        raise ValueError(f"Sample order mismatch in {source}")


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def behavior_count_plot(counts: pd.DataFrame, figures: Path) -> None:
    pivot = counts.pivot(index="seed", columns="behavior_group", values="count")
    pivot = pivot.reindex(columns=BEHAVIOR_GROUPS).fillna(0).astype(int)
    pivot.columns = [BEHAVIOR_LABELS[column] for column in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 3.6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Test examples by behavior group")
    ax.set_xlabel("")
    ax.set_ylabel("training seed")
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, figures, "01_behavior_counts")


def label_behavior_plot(
    assignments: pd.DataFrame,
    figures: Path,
    *,
    label_names: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(assignments["seed"].unique()), figsize=(19, 5), sharey=True)
    axes = np.atleast_1d(axes)
    labels = [*label_names, "NONE"]
    for ax, (seed, frame) in zip(axes, assignments.groupby("seed"), strict=True):
        long_rows: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            active = str(row.gold_labels).split(", ") if row.gold_labels != "NONE" else ["NONE"]
            long_rows.extend(
                {"behavior_group": row.behavior_group, "label": label}
                for label in active
            )
        counts = pd.DataFrame(long_rows).value_counts().rename("count").reset_index()
        pivot = counts.pivot(index="behavior_group", columns="label", values="count")
        pivot = pivot.reindex(index=BEHAVIOR_GROUPS, columns=labels).fillna(0).astype(int)
        pivot.index = [BEHAVIOR_LABELS[index] for index in pivot.index]
        sns.heatmap(pivot, annot=True, fmt="d", cmap="mako", cbar=False, ax=ax)
        ax.set_title(f"seed={seed}")
        ax.set_xlabel("gold label")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Gold label by behavior group on the full test split", y=1.02)
    fig.tight_layout()
    save_figure(fig, figures, "02_label_by_behavior_counts")


def behavior_facets(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    seed: int,
    y: str,
    ylabel: str,
    title: str,
    figures: Path,
    stem: str,
    method_dimension: bool,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    axes = axes.ravel()
    selected_seed = selected[selected["seed"] == seed]
    selected_ids = set(selected_seed["id"])
    subset = frame[(frame["seed"] == seed) & frame["id"].isin(selected_ids)]
    for ax, group in zip(axes, BEHAVIOR_GROUPS, strict=True):
        group_frame = subset[subset["behavior_group"] == group]
        sample_rows = selected_seed[selected_seed["behavior_group"] == group]
        palette = (
            sns.color_palette("colorblind", n_colors=len(sample_rows))
            if len(sample_rows)
            else []
        )
        sample_handles: list[Line2D] = []
        for color, row in zip(palette, sample_rows.itertuples(index=False), strict=True):
            sample = group_frame[group_frame["id"] == row.id]
            label = f"{str(row.id)[:7]} | k={row.gold_cardinality} | {row.gold_labels}"
            if method_dimension:
                for method in ("prompt_tuning", "prefix_tuning"):
                    method_frame = sample[sample["method"] == method].sort_values("block")
                    ax.plot(
                        method_frame["block"],
                        method_frame[y],
                        linestyle=METHOD_STYLES[method],
                        marker="o",
                        color=color,
                        linewidth=1.8,
                    )
                sample_handles.append(
                    Line2D(
                        [], [], color=color, marker="o", linestyle="-",
                        linewidth=1.8, label=label,
                    )
                )
            else:
                sample = sample.sort_values("block")
                ax.plot(sample["block"], sample[y], marker="o", color=color, label=label)
        ax.set_title(BEHAVIOR_LABELS[group])
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(frame["block"].unique()))
        if group_frame.empty:
            ax.text(0.5, 0.5, "no selected examples", ha="center", va="center", transform=ax.transAxes)
        else:
            if method_dimension:
                sample_legend = ax.legend(
                    handles=sample_handles,
                    title="Example (color)",
                    fontsize=6,
                    title_fontsize=7,
                    loc="upper left",
                )
                ax.add_artist(sample_legend)
                method_handles = [
                    Line2D(
                        [], [], color="black", linestyle=METHOD_STYLES[method],
                        linewidth=2.2, label=METHOD_LABELS[method],
                    )
                    for method in ("prompt_tuning", "prefix_tuning")
                ]
                ax.legend(
                    handles=method_handles,
                    title="Method (line)",
                    fontsize=7,
                    title_fontsize=7,
                    loc="lower right",
                )
            else:
                ax.legend(fontsize=7, loc="best")
    for ax in axes[-2:]:
        ax.set_xlabel("transformer block")
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    fig.suptitle(f"{title}; seed={seed}; one line per example", y=0.995)
    fig.tight_layout()
    save_figure(fig, figures, f"{stem}_seed{seed}")


def per_label_cross_cosine_plot(
    cross: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    seed: int,
    figures: Path,
    label_names: list[str],
) -> None:
    labels = [*label_names, "NONE"]
    selected_seed = selected[selected["seed"] == seed]
    selected_ids = set(selected_seed["id"])
    subset = cross[(cross["seed"] == seed) & cross["id"].isin(selected_ids)]
    behavior_colors = dict(
        zip(BEHAVIOR_GROUPS, sns.color_palette("colorblind", len(BEHAVIOR_GROUPS)), strict=True)
    )
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), labels, strict=False):
        for row in selected_seed.itertuples(index=False):
            active = str(row.gold_labels).split(", ") if row.gold_labels != "NONE" else ["NONE"]
            if label not in active:
                continue
            sample = subset[subset["id"] == row.id].sort_values("block")
            ax.plot(
                sample["block"],
                sample["delta_cosine"],
                marker="o",
                color=behavior_colors[row.behavior_group],
                label=f"{str(row.id)[:7]} | {BEHAVIOR_LABELS[row.behavior_group]}",
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(label)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(cross["block"].unique()))
        if ax.lines:
            ax.legend(fontsize=7, loc="best")
    axes.ravel()[-1].axis("off")
    for ax in axes[-1, :1]:
        ax.set_xlabel("transformer block")
    for ax in axes[:, 0]:
        ax.set_ylabel("cos(ΔPrompt, ΔPrefix)")
    fig.suptitle(f"Shift directions by gold label; seed={seed}; no sample averaging", y=0.995)
    fig.tight_layout()
    save_figure(fig, figures, f"06_cross_method_cosine_by_label_seed{seed}")


def pairwise_rows_and_plot(
    deltas: dict[tuple[int, str, str, int], np.ndarray],
    selected: pd.DataFrame,
    *,
    seeds: list[int],
    blocks: list[int],
    figures: Path,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for seed in seeds:
        selected_seed = selected[selected["seed"] == seed]
        ids = list(selected_seed["id"])
        short = [str(sample_id)[:7] for sample_id in ids]
        fig, axes = plt.subplots(len(blocks), 2, figsize=(16, 4 * len(blocks)))
        for row_index, block in enumerate(blocks):
            for column_index, method in enumerate(("prompt_tuning", "prefix_tuning")):
                matrix = np.empty((len(ids), len(ids)), dtype=np.float64)
                for left_index, left_id in enumerate(ids):
                    for right_index, right_id in enumerate(ids):
                        cosine = safe_cosine(
                            deltas[(seed, method, left_id, block)],
                            deltas[(seed, method, right_id, block)],
                        )
                        matrix[left_index, right_index] = cosine
                        output.append(
                            {
                                "seed": seed,
                                "method": method,
                                "block": block,
                                "left_id": left_id,
                                "right_id": right_id,
                                "delta_cosine": cosine,
                            }
                        )
                ax = axes[row_index, column_index]
                sns.heatmap(
                    matrix,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap="vlag",
                    xticklabels=short,
                    yticklabels=short,
                    cbar=column_index == 1,
                    ax=ax,
                )
                ax.set_title(f"{METHOD_LABELS[method]}, block={block}")
                ax.tick_params(axis="x", rotation=70, labelsize=6)
                ax.tick_params(axis="y", rotation=0, labelsize=6)
        fig.suptitle(
            f"Pairwise cosine between example-specific shifts; seed={seed}",
            y=0.998,
        )
        fig.tight_layout()
        save_figure(fig, figures, f"07_pairwise_sample_delta_cosine_seed{seed}")
    return output


def main() -> None:
    args = parse_args()
    require_empty_output(args.output_dir)
    sns.set_theme(style="whitegrid", context="notebook")

    test_rows = read_jsonl(args.test_jsonl)
    test_map = {str(row["id"]): row for row in test_rows}
    manual_metadata_path = args.manual_dir / "manual_test.json"
    manual_summary = read_json(args.manual_dir / "manual_summary.json")
    manual_tensor_path = args.manual_dir / "manual_test.safetensors"
    manual_predictions_path = args.manual_dir / "manual_test_predictions.json"
    manual_metadata = metadata_rows(manual_metadata_path)
    ids = [str(row["id"]) for row in manual_metadata]
    labels = [list(row["labels"]) for row in manual_metadata]
    observed_labels = {str(label) for values in labels for label in values}
    label_names = [
        str(label)
        for label in manual_summary.get("labels", DEFAULT_LABEL_ORDER)
        if str(label) in observed_labels
    ]
    label_names.extend(sorted(observed_labels.difference(label_names)))
    if set(ids) != set(test_map):
        raise ValueError("Manual activation IDs do not exactly match test JSONL")
    manual_predictions = prediction_map(manual_predictions_path)
    manual_states = load_states(manual_tensor_path)

    assignment_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    all_deltas: dict[tuple[int, str, str, int], np.ndarray] = {}
    input_runs: list[dict[str, object]] = []

    for seed in args.seeds:
        prompt_dir = activation_run_dir(
            args.prompt_activations_root,
            args.prompt_run_template,
            method="prompt_tuning",
            n=args.train_samples,
            m=args.num_virtual_tokens,
            seed=seed,
        )
        prefix_dir = activation_run_dir(
            args.prefix_activations_root,
            args.prefix_run_template,
            method="prefix_tuning",
            n=args.train_samples,
            m=args.num_virtual_tokens,
            seed=seed,
        )
        prompt_meta_path = prompt_dir / "adapted_test.json"
        prefix_meta_path = prefix_dir / "adapted_test.json"
        prompt_meta = metadata_rows(prompt_meta_path)
        prefix_meta = metadata_rows(prefix_meta_path)
        validate_id_order(ids, [str(row["id"]) for row in prompt_meta], prompt_meta_path)
        validate_id_order(ids, [str(row["id"]) for row in prefix_meta], prefix_meta_path)
        prompt_predictions = prediction_map(prompt_dir / "adapted_test_predictions.json")
        prefix_predictions = prediction_map(prefix_dir / "adapted_test_predictions.json")
        if set(prompt_predictions) != set(ids) or set(prefix_predictions) != set(ids):
            raise ValueError(f"Prediction IDs do not match activation IDs for seed={seed}")

        behaviors: dict[str, str] = {}
        for sample_id, sample_labels in zip(ids, labels, strict=True):
            manual_correct = bool(manual_predictions[sample_id]["exact_match"])
            prompt_correct = bool(prompt_predictions[sample_id]["exact_match"])
            prefix_correct = bool(prefix_predictions[sample_id]["exact_match"])
            group = behavior_group(manual_correct, prompt_correct, prefix_correct)
            behaviors[sample_id] = group
            assignment_rows.append(
                {
                    "seed": seed,
                    "id": sample_id,
                    "gold_labels": ", ".join(sample_labels) or "NONE",
                    "gold_cardinality": len(sample_labels),
                    "cardinality_bucket": str(len(sample_labels)) if len(sample_labels) < 3 else "3+",
                    "manual_correct": manual_correct,
                    "prompt_correct": prompt_correct,
                    "prefix_correct": prefix_correct,
                    "prompt_transition": transition_name(manual_correct, prompt_correct),
                    "prefix_transition": transition_name(manual_correct, prefix_correct),
                    "behavior_group": group,
                    "manual_prediction": manual_predictions[sample_id].get("raw_output"),
                    "prompt_prediction": prompt_predictions[sample_id].get("raw_output"),
                    "prefix_prediction": prefix_predictions[sample_id].get("raw_output"),
                    "text": test_map[sample_id]["text"],
                }
            )

        geometry = compute_geometry(
            manual_states,
            load_states(prompt_dir / "adapted_test.safetensors"),
            load_states(prefix_dir / "adapted_test.safetensors"),
            ids=ids,
            labels=labels,
            behaviors=behaviors,
            blocks=args.blocks,
            seed=seed,
        )
        method_rows.extend(geometry.method_rows)
        cross_rows.extend(geometry.cross_method_rows)
        all_deltas.update(
            {
                (seed, method, sample_id, block): delta
                for (method, sample_id, block), delta in geometry.deltas.items()
            }
        )
        input_runs.extend(
            [
                {"seed": seed, "method": "prompt_tuning", "activation_dir": str(prompt_dir.resolve())},
                {"seed": seed, "method": "prefix_tuning", "activation_dir": str(prefix_dir.resolve())},
            ]
        )

    assignments = pd.DataFrame.from_records(assignment_rows)
    methods = pd.DataFrame.from_records(method_rows)
    cross = pd.DataFrame.from_records(cross_rows)
    selected_parts: list[pd.DataFrame] = []
    for seed in args.seeds:
        seed_assignments = assignments[assignments["seed"] == seed]
        records = [
            {
                "id": row.id,
                "labels": [] if row.gold_labels == "NONE" else str(row.gold_labels).split(", "),
                "behavior_group": row.behavior_group,
                "cardinality_bucket": row.cardinality_bucket,
            }
            for row in seed_assignments.itertuples(index=False)
        ]
        chosen = select_examples(
            records,
            label_names=label_names,
            samples_per_group=args.samples_per_group,
            selection_seed=seed,
        )
        selected_parts.append(seed_assignments[seed_assignments["id"].isin(chosen)].copy())
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["selection_order"] = selected.groupby("seed").cumcount()

    tables = args.output_dir / "tables"
    figures = args.output_dir / "figures"
    counts = (
        assignments.groupby(["seed", "behavior_group"])
        .size()
        .rename("count")
        .reset_index()
    )
    label_count_rows: list[dict[str, object]] = []
    for row in assignments.itertuples(index=False):
        active = str(row.gold_labels).split(", ") if row.gold_labels != "NONE" else ["NONE"]
        label_count_rows.extend(
            {"seed": row.seed, "behavior_group": row.behavior_group, "label": label}
            for label in active
        )
    label_counts = (
        pd.DataFrame(label_count_rows)
        .groupby(["seed", "behavior_group", "label"])
        .size()
        .rename("count")
        .reset_index()
    )
    assignments.to_csv(tables / "behavior_assignments.csv", index=False)
    counts.to_csv(tables / "behavior_counts.csv", index=False)
    label_counts.to_csv(tables / "behavior_label_counts.csv", index=False)
    selected.to_csv(tables / "selected_samples.csv", index=False)
    methods.to_csv(tables / "all_method_geometry.csv", index=False)
    cross.to_csv(tables / "all_cross_method_geometry.csv", index=False)
    selected_keys = pd.MultiIndex.from_frame(selected[["seed", "id"]])
    method_keys = pd.MultiIndex.from_frame(methods[["seed", "id"]])
    cross_keys = pd.MultiIndex.from_frame(cross[["seed", "id"]])
    methods[method_keys.isin(selected_keys)].to_csv(tables / "selected_method_geometry.csv", index=False)
    cross[cross_keys.isin(selected_keys)].to_csv(tables / "selected_cross_method_geometry.csv", index=False)

    behavior_count_plot(counts, figures)
    label_behavior_plot(assignments, figures, label_names=label_names)
    for seed in args.seeds:
        behavior_facets(
            cross,
            selected,
            seed=seed,
            y="delta_cosine",
            ylabel="cos(ΔPrompt, ΔPrefix)",
            title="Residual-shift direction agreement",
            figures=figures,
            stem="03_cross_method_delta_cosine",
            method_dimension=False,
        )
        behavior_facets(
            methods,
            selected,
            seed=seed,
            y="relative_delta_norm",
            ylabel="||Δ|| / ||h_manual||",
            title="Relative residual-shift magnitude",
            figures=figures,
            stem="04_relative_delta_norm",
            method_dimension=True,
        )
        behavior_facets(
            methods,
            selected,
            seed=seed,
            y="raw_state_angle_deg",
            ylabel="angle(h_manual, h_tuned), degrees",
            title="Angle between tuned and manual states",
            figures=figures,
            stem="05_raw_state_angle",
            method_dimension=True,
        )
        per_label_cross_cosine_plot(
            cross,
            selected,
            seed=seed,
            figures=figures,
            label_names=label_names,
        )
    pairwise = pairwise_rows_and_plot(
        all_deltas,
        selected,
        seeds=list(args.seeds),
        blocks=list(args.blocks),
        figures=figures,
    )
    pd.DataFrame.from_records(pairwise).to_csv(
        tables / "selected_pairwise_sample_delta_cosine.csv", index=False
    )

    manifest = {
        "analysis": "civil_comments_residual_geometry",
        "git": git_revision(),
        "parameters": {
            "train_samples": args.train_samples,
            "num_virtual_tokens": args.num_virtual_tokens,
            "seeds": list(args.seeds),
            "blocks": list(args.blocks),
            "hidden_state_indices": [block + 1 for block in args.blocks],
            "samples_per_group": args.samples_per_group,
            "prompt_run_template": args.prompt_run_template,
            "prefix_run_template": args.prefix_run_template,
        },
        "dataset": {
            "path": str(args.test_jsonl.resolve()),
            "sha256": sha256_file(args.test_jsonl),
            "samples": len(test_rows),
        },
        "manual_dir": str(args.manual_dir.resolve()),
        "input_runs": input_runs,
        "script_sha256": sha256_file(Path(__file__)),
        "row_counts": {
            "assignments": len(assignments),
            "selected": len(selected),
            "method_geometry": len(methods),
            "cross_method_geometry": len(cross),
            "pairwise": len(pairwise),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
if __name__ == "__main__":
    main()
