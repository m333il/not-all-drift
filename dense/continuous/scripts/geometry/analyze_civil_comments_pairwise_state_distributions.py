#!/usr/bin/env python3
"""Analyze pairwise cosine and angle distributions across prompt conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file


METHODS = ("Manual", "Prompt", "Prefix projected")
RAW_PAIRS = tuple(combinations(METHODS, 2))
DELTA_METHODS = ("Prompt", "Prefix projected")
DELTA_PAIRS = tuple(combinations(DELTA_METHODS, 2))
PAIR_COLORS = {
    "Manual vs Prompt": "#0072B2",
    "Manual vs Prefix projected": "#E69F00",
    "Prompt vs Prefix projected": "#CC79A7",
}
SEED_COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--projected-prefix-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=500)
    parser.add_argument("--num-virtual-tokens", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--blocks", type=int, nargs="+", default=(0, 6, 13, 20, 25))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument(
        "--prompt-template",
        default="prompt_tuning_n{n}_split{seed}_train{seed}_vt{m}",
    )
    parser.add_argument(
        "--prefix-template",
        default="prefix_projection__prefix_tuning_n{n}_split{seed}_train{seed}_vt{m}",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
        ).splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}


def require_empty_output(path: Path) -> tuple[Path, Path]:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    figures = path / "figures"
    tables = path / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def metadata_ids(path: Path) -> list[str]:
    rows = read_json(path)["rows"]
    return [str(row["id"]) for row in rows]


def condition_paths(args: argparse.Namespace, seed: int) -> dict[str, tuple[Path, Path]]:
    n = args.train_samples
    m = args.num_virtual_tokens
    prompt = args.prompt_root / args.prompt_template.format(n=n, m=m, seed=seed)
    prefix = args.projected_prefix_root / args.prefix_template.format(n=n, m=m, seed=seed)
    return {
        "Manual": (
            args.manual_dir / "manual_test.safetensors",
            args.manual_dir / "manual_test.json",
        ),
        "Prompt": (prompt / "adapted_test.safetensors", prompt / "adapted_test.json"),
        "Prefix projected": (
            prefix / "adapted_test.safetensors",
            prefix / "adapted_test.json",
        ),
    }


def load_selected_states(
    path: Path,
    hidden_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    states = load_file(path, device=str(device))["states"]
    if states.ndim != 3:
        raise ValueError(f"Expected [samples,layers,width], got {states.shape} in {path}")
    if max(hidden_indices) >= states.shape[1]:
        raise ValueError(f"Hidden-state index out of range in {path}: {hidden_indices}")
    return states[:, hidden_indices].float()


def cosine_angle_torch(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-wise cosine and arccos angle in degrees."""
    if first.shape != second.shape:
        raise ValueError(f"Shape mismatch: {first.shape} versus {second.shape}")
    first_norm = torch.linalg.vector_norm(first, dim=-1)
    second_norm = torch.linalg.vector_norm(second, dim=-1)
    denominator = first_norm * second_norm
    cosine = torch.full_like(denominator, torch.nan)
    valid = denominator > eps
    cosine[valid] = (
        (first[valid] * second[valid]).sum(dim=-1) / denominator[valid]
    ).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return cosine, angle


def pair_name(first: str, second: str) -> str:
    return f"{first} vs {second}"


def collect_distributions(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    hidden_indices = [block + 1 for block in args.blocks]
    manual_ids = metadata_ids(args.manual_dir / "manual_test.json")
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []

    for seed in args.seeds:
        paths = condition_paths(args, seed)
        states: dict[str, torch.Tensor] = {}
        for method, (state_path, metadata_path) in paths.items():
            if not state_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing {method} input for seed={seed}: {state_path}")
            ids = metadata_ids(metadata_path)
            if ids != manual_ids:
                raise ValueError(f"Test ID order mismatch for {method}, seed={seed}")
            states[method] = load_selected_states(state_path, hidden_indices, device)
            if states[method].shape[:2] != (len(manual_ids), len(args.blocks)):
                raise ValueError(f"Unexpected selected-state shape: {states[method].shape}")
            inputs.append(
                {
                    "seed": seed,
                    "method": method,
                    "states": str(state_path.resolve()),
                    "metadata": str(metadata_path.resolve()),
                    "state_bytes": state_path.stat().st_size,
                    "metadata_sha256": sha256_file(metadata_path),
                }
            )

        for first_method, second_method in RAW_PAIRS:
            cosine, angle = cosine_angle_torch(
                states[first_method],
                states[second_method],
            )
            cosine_array = cosine.cpu().numpy()
            angle_array = angle.cpu().numpy()
            name = pair_name(first_method, second_method)
            for layer_position, block in enumerate(args.blocks):
                for sample_index, sample_id in enumerate(manual_ids):
                    records.append(
                        {
                            "geometry": "raw_state",
                            "seed": seed,
                            "id": sample_id,
                            "block": block,
                            "pair": name,
                            "method_a": first_method,
                            "method_b": second_method,
                            "cosine": float(cosine_array[sample_index, layer_position]),
                            "angle_deg": float(angle_array[sample_index, layer_position]),
                        }
                    )

        deltas = {
            method: states[method] - states["Manual"]
            for method in DELTA_METHODS
        }
        for first_method, second_method in DELTA_PAIRS:
            cosine, angle = cosine_angle_torch(
                deltas[first_method],
                deltas[second_method],
            )
            cosine_array = cosine.cpu().numpy()
            angle_array = angle.cpu().numpy()
            name = pair_name(first_method, second_method)
            for layer_position, block in enumerate(args.blocks):
                for sample_index, sample_id in enumerate(manual_ids):
                    records.append(
                        {
                            "geometry": "manual_centered_delta",
                            "seed": seed,
                            "id": sample_id,
                            "block": block,
                            "pair": name,
                            "method_a": first_method,
                            "method_b": second_method,
                            "cosine": float(cosine_array[sample_index, layer_position]),
                            "angle_deg": float(angle_array[sample_index, layer_position]),
                        }
                    )
        del states, deltas
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame.from_records(records)
    if frame[["cosine", "angle_deg"]].isna().any().any():
        invalid = frame[frame[["cosine", "angle_deg"]].isna().any(axis=1)]
        raise ValueError(f"Undefined geometry rows: {len(invalid)}")
    return frame, inputs, manual_ids


def per_sample_seed_average(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            ["geometry", "id", "block", "pair", "method_a", "method_b"],
            as_index=False,
        )[["cosine", "angle_deg"]]
        .mean()
        .rename(
            columns={
                "cosine": "cosine_seed_mean",
                "angle_deg": "angle_deg_seed_mean",
            }
        )
    )
    seed_counts = frame.groupby(
        ["geometry", "id", "block", "pair"],
        as_index=False,
    )["seed"].nunique()
    if not seed_counts["seed"].eq(frame["seed"].nunique()).all():
        raise ValueError("Incomplete seed coverage in per-sample average")
    return grouped


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def summarize_distributions(
    frame: pd.DataFrame,
    seed_mean: pd.DataFrame,
    *,
    replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_map = {
        "cosine": "cosine_seed_mean",
        "angle_deg": "angle_deg_seed_mean",
    }
    rng = np.random.default_rng(bootstrap_seed)
    records = []
    for (geometry, pair, block), group in seed_mean.groupby(
        ["geometry", "pair", "block"],
        sort=True,
    ):
        source = frame[
            (frame["geometry"] == geometry)
            & (frame["pair"] == pair)
            & (frame["block"] == block)
        ]
        for metric, averaged_metric in metric_map.items():
            values = group[averaged_metric].to_numpy(dtype=np.float64)
            seed_medians = source.groupby("seed")[metric].median().to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_median_ci(
                values,
                replicates=replicates,
                rng=rng,
            )
            records.append(
                {
                    "geometry": geometry,
                    "pair": pair,
                    "block": int(block),
                    "metric": metric,
                    "n_test_samples": len(values),
                    "n_seeds": len(seed_medians),
                    "sample_mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)),
                    "sample_median": float(np.median(values)),
                    "sample_q05": float(np.quantile(values, 0.05)),
                    "sample_q25": float(np.quantile(values, 0.25)),
                    "sample_q75": float(np.quantile(values, 0.75)),
                    "sample_q95": float(np.quantile(values, 0.95)),
                    "sample_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                    "bootstrap_median_ci_low": ci_low,
                    "bootstrap_median_ci_high": ci_high,
                    "seed_median_mean": float(np.mean(seed_medians)),
                    "seed_median_std": float(np.std(seed_medians, ddof=1)),
                    "seed_median_min": float(np.min(seed_medians)),
                    "seed_median_max": float(np.max(seed_medians)),
                }
            )

    per_seed = (
        frame.groupby(["geometry", "pair", "block", "seed"])
        .agg(
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            cosine_median=("cosine", "median"),
            cosine_q05=("cosine", lambda values: values.quantile(0.05)),
            cosine_q25=("cosine", lambda values: values.quantile(0.25)),
            cosine_q75=("cosine", lambda values: values.quantile(0.75)),
            cosine_q95=("cosine", lambda values: values.quantile(0.95)),
            angle_mean=("angle_deg", "mean"),
            angle_std=("angle_deg", "std"),
            angle_median=("angle_deg", "median"),
            angle_q05=("angle_deg", lambda values: values.quantile(0.05)),
            angle_q25=("angle_deg", lambda values: values.quantile(0.25)),
            angle_q75=("angle_deg", lambda values: values.quantile(0.75)),
            angle_q95=("angle_deg", lambda values: values.quantile(0.95)),
            n_test_samples=("id", "nunique"),
        )
        .reset_index()
    )
    return pd.DataFrame.from_records(records), per_seed


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    for suffix, kwargs in (
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ):
        fig.savefig(figures / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def ecdf(ax: plt.Axes, values: np.ndarray, **kwargs: Any) -> None:
    finite = np.sort(values[np.isfinite(values)])
    probabilities = np.arange(1, len(finite) + 1) / len(finite)
    ax.plot(finite, probabilities, **kwargs)


def pair_slug(pair: str) -> str:
    return (
        pair.lower()
        .replace(" projected", "")
        .replace(" ", "_")
        .replace("vs", "vs")
    )


def geometry_pairs(geometry: str) -> tuple[tuple[str, str], ...]:
    return RAW_PAIRS if geometry == "raw_state" else DELTA_PAIRS


def plot_overview_ecdf(
    seed_mean: pd.DataFrame,
    *,
    geometry: str,
    metric: str,
    blocks: list[int],
    figures: Path,
    stem: str,
) -> None:
    value_column = f"{metric}_seed_mean"
    subset = seed_mean[seed_mean["geometry"] == geometry]
    pairs = [pair_name(*pair) for pair in geometry_pairs(geometry)]
    fig, axes = plt.subplots(1, len(blocks), figsize=(20, 4.2), sharey=True)
    for ax, block in zip(axes, blocks, strict=True):
        part = subset[subset["block"] == block]
        for pair in pairs:
            values = part[part["pair"] == pair][value_column].to_numpy()
            ecdf(
                ax,
                values,
                label=pair,
                color=PAIR_COLORS[pair],
                linewidth=1.7,
            )
        ax.set_title(f"block {block}")
        ax.set_xlabel("Cosine" if metric == "cosine" else "Angle, degrees")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("ECDF over 500 matched test samples")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    save_figure(fig, figures, stem)


def plot_pair_seed_ecdfs(
    frame: pd.DataFrame,
    seed_mean: pd.DataFrame,
    *,
    geometry: str,
    blocks: list[int],
    figures: Path,
) -> None:
    pairs = [pair_name(*pair) for pair in geometry_pairs(geometry)]
    for pair in pairs:
        fig, axes = plt.subplots(2, len(blocks), figsize=(20, 7.4), sharey="row")
        for column, block in enumerate(blocks):
            raw_part = frame[
                (frame["geometry"] == geometry)
                & (frame["pair"] == pair)
                & (frame["block"] == block)
            ]
            mean_part = seed_mean[
                (seed_mean["geometry"] == geometry)
                & (seed_mean["pair"] == pair)
                & (seed_mean["block"] == block)
            ]
            for row, (metric, averaged_metric, xlabel) in enumerate(
                (
                    ("cosine", "cosine_seed_mean", "Cosine"),
                    ("angle_deg", "angle_deg_seed_mean", "Angle, degrees"),
                )
            ):
                ax = axes[row, column]
                for seed, group in raw_part.groupby("seed"):
                    ecdf(
                        ax,
                        group[metric].to_numpy(),
                        color=SEED_COLORS.get(int(seed)),
                        linewidth=1.0,
                        alpha=0.62,
                        label=f"seed {seed}",
                    )
                ecdf(
                    ax,
                    mean_part[averaged_metric].to_numpy(),
                    color="black",
                    linewidth=2.0,
                    linestyle="--",
                    label="per-sample seed mean",
                )
                ax.set_xlabel(xlabel)
                ax.grid(alpha=0.2)
                if row == 0:
                    ax.set_title(f"block {block}")
        axes[0, 0].set_ylabel("ECDF")
        axes[1, 0].set_ylabel("ECDF")
        axes[0, -1].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        save_figure(
            fig,
            figures,
            f"pair_{geometry}_{pair_slug(pair)}_seed_ecdf",
        )


def plot_angle_violins(
    seed_mean: pd.DataFrame,
    *,
    geometry: str,
    blocks: list[int],
    figures: Path,
) -> None:
    subset = seed_mean[seed_mean["geometry"] == geometry]
    pairs = [pair_name(*pair) for pair in geometry_pairs(geometry)]
    fig, axes = plt.subplots(1, len(blocks), figsize=(21, 5.3), sharey=True)
    for ax, block in zip(axes, blocks, strict=True):
        part = subset[subset["block"] == block]
        datasets = [
            part[part["pair"] == pair]["angle_deg_seed_mean"].to_numpy()
            for pair in pairs
        ]
        violin = ax.violinplot(
            datasets,
            positions=np.arange(len(pairs)),
            showmedians=True,
            showextrema=False,
            widths=0.8,
        )
        for body, pair in zip(violin["bodies"], pairs, strict=True):
            body.set_facecolor(PAIR_COLORS[pair])
            body.set_edgecolor("#333333")
            body.set_alpha(0.72)
        violin["cmedians"].set_color("black")
        violin["cmedians"].set_linewidth(1.2)
        ax.set_xticks(
            np.arange(len(pairs)),
            [pair.replace(" projected", "") for pair in pairs],
            rotation=38,
            ha="right",
            fontsize=7,
        )
        ax.set_title(f"block {block}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Per-sample seed-mean angle, degrees")
    fig.tight_layout()
    save_figure(fig, figures, f"overview_{geometry}_angle_violins")


def plot_quantile_heatmaps(
    summary: pd.DataFrame,
    *,
    geometry: str,
    blocks: list[int],
    figures: Path,
) -> None:
    pairs = [pair_name(*pair) for pair in geometry_pairs(geometry)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.72 * len(pairs) + 2.4))
    configs = (
        ("cosine", "sample_median", "Median cosine", "viridis", None, None),
        ("angle_deg", "sample_median", "Median angle, degrees", "magma", 0.0, None),
    )
    for ax, (metric, value, label, cmap, vmin, vmax) in zip(
        axes,
        configs,
        strict=True,
    ):
        part = summary[
            (summary["geometry"] == geometry)
            & (summary["metric"] == metric)
        ]
        matrix = (
            part.pivot(index="pair", columns="block", values=value)
            .reindex(index=pairs, columns=blocks)
        )
        image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(blocks)), blocks)
        ax.set_yticks(
            np.arange(len(pairs)),
            [pair.replace(" projected", "") for pair in pairs],
        )
        ax.set_xlabel("Transformer block")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                number = matrix.iloc[row, column]
                text = f"{number:.4f}" if metric == "cosine" else f"{number:.1f}°"
                ax.text(column, row, text, ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(image, ax=ax, label=label, shrink=0.82)
    fig.tight_layout()
    save_figure(fig, figures, f"heatmap_{geometry}_median_cosine_angle")


def plot_seed_median_angles(
    per_seed: pd.DataFrame,
    *,
    geometry: str,
    blocks: list[int],
    figures: Path,
) -> None:
    pairs = [pair_name(*pair) for pair in geometry_pairs(geometry)]
    columns = 3
    rows = int(np.ceil(len(pairs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, 3.5 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, pair in zip(axes, pairs, strict=False):
        part = per_seed[
            (per_seed["geometry"] == geometry)
            & (per_seed["pair"] == pair)
        ]
        for seed, group in part.groupby("seed"):
            group = group.set_index("block").reindex(blocks)
            ax.plot(
                blocks,
                group["angle_median"],
                marker="o",
                color=SEED_COLORS.get(int(seed)),
                linewidth=1.2,
                label=f"seed {seed}",
            )
        average = part.groupby("block")["angle_median"].mean().reindex(blocks)
        ax.plot(
            blocks,
            average,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="mean of seed medians",
        )
        ax.set_title(pair.replace(" projected", ""), fontsize=10)
        ax.set_xticks(blocks)
        ax.set_ylabel("Median angle, degrees")
        ax.grid(alpha=0.2)
    for ax in axes[len(pairs):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures, f"seed_stability_{geometry}_median_angle")


def build_figures(
    frame: pd.DataFrame,
    seed_mean: pd.DataFrame,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    *,
    blocks: list[int],
    figures: Path,
) -> None:
    for geometry in ("raw_state", "manual_centered_delta"):
        plot_overview_ecdf(
            seed_mean,
            geometry=geometry,
            metric="cosine",
            blocks=blocks,
            figures=figures,
            stem=f"overview_{geometry}_cosine_ecdf",
        )
        plot_overview_ecdf(
            seed_mean,
            geometry=geometry,
            metric="angle_deg",
            blocks=blocks,
            figures=figures,
            stem=f"overview_{geometry}_angle_ecdf",
        )
        plot_pair_seed_ecdfs(
            frame,
            seed_mean,
            geometry=geometry,
            blocks=blocks,
            figures=figures,
        )
        plot_angle_violins(
            seed_mean,
            geometry=geometry,
            blocks=blocks,
            figures=figures,
        )
        plot_quantile_heatmaps(
            summary,
            geometry=geometry,
            blocks=blocks,
            figures=figures,
        )
        plot_seed_median_angles(
            per_seed,
            geometry=geometry,
            blocks=blocks,
            figures=figures,
        )


def main() -> None:
    args = parse_args()
    figures, tables = require_empty_output(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    frame, inputs, ids = collect_distributions(args, device)
    seed_mean = per_sample_seed_average(frame)
    summary, per_seed = summarize_distributions(
        frame,
        seed_mean,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    frame.to_csv(tables / "per_seed_per_sample_geometry.csv.gz", index=False, compression="gzip")
    seed_mean.to_csv(tables / "per_sample_seed_mean_geometry.csv.gz", index=False, compression="gzip")
    summary.to_csv(tables / "distribution_summary.csv", index=False)
    per_seed.to_csv(tables / "per_seed_summary.csv", index=False)

    build_figures(
        frame,
        seed_mean,
        summary,
        per_seed,
        blocks=list(args.blocks),
        figures=figures,
    )
    manifest = {
        "analysis": "civil_comments_pairwise_state_distributions",
        "status": "done",
        "git": git_revision(),
        "parameters": {
            "train_samples": args.train_samples,
            "num_virtual_tokens": args.num_virtual_tokens,
            "seeds": list(args.seeds),
            "blocks": list(args.blocks),
            "hidden_state_indices": [block + 1 for block in args.blocks],
            "device": str(device),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "seed_average_definition": "arithmetic mean of each metric across matched seeds within test ID",
        },
        "representation": "residual stream at the last textual prompt token",
        "prefix_variant": "projected (prefix_projection=true)",
        "inputs": inputs,
        "row_counts": {
            "per_seed_per_sample": len(frame),
            "per_sample_seed_mean": len(seed_mean),
            "distribution_summary": len(summary),
            "per_seed_summary": len(per_seed),
        },
        "figure_stem_count": len(list(figures.glob("*.png"))),
        "script_sha256": sha256_file(Path(__file__)),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
