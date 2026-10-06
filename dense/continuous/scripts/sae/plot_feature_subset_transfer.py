#!/usr/bin/env python3
"""Plot dense, full-SAE, and Top-K causal method-replacement recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {"gepa": "#27AE60", "prompt": "#F2994A", "prefix": "#2F80ED"}
COMPONENTS = {
    "dense": ("Dense shift", "D", -0.23),
    "full_sae": ("Full SAE", "s", 0.0),
    "direct_top128": ("Top-128 features", "o", 0.23),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.65,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(True, linestyle=(0, (3, 2)), linewidth=0.45, color="#D8CFC2", alpha=0.78)
    for spine in axis.spines.values():
        spine.set_color("#B9B1A8")
        spine.set_linewidth(0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--carrier", default="generation_anchor")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--transition", nargs="+")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="sae_feature_subset_transfer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in args.metrics]
    data = pd.concat(frames, ignore_index=True)
    top_component = f"direct_top{args.top_k}"
    components = ("dense", "full_sae", top_component)
    required = {"carrier", "layer", "transition", "component", "total_recovery", "seed"}
    if not required.issubset(data.columns):
        raise ValueError(f"Metrics miss columns: {sorted(required - set(data.columns))}")
    keep = data[
        (data["carrier"] == args.carrier)
        & (data["layer"] == args.layer)
        & data["component"].isin(components)
    ].copy()
    transitions = args.transition or sorted(keep["transition"].unique())
    keep = keep[keep["transition"].isin(transitions)]
    if keep.empty:
        raise ValueError("No rows match the requested carrier/layer/transitions")
    expected = {(transition, component) for transition in transitions for component in components}
    observed = set(zip(keep["transition"], keep["component"], strict=False))
    missing = expected - observed
    if missing:
        raise ValueError(f"Missing transition/component cells: {sorted(missing)}")
    keep.sort_values(["transition", "component", "seed"]).to_csv(
        args.output_dir / f"{args.stem}_points.csv", index=False
    )
    component_specs = {
        "dense": COMPONENTS["dense"],
        "full_sae": COMPONENTS["full_sae"],
        top_component: (f"Top-{args.top_k} features", "o", 0.23),
    }
    configure_style()
    width = max(5.0, 1.25 * len(transitions) + 1.5)
    fig, axis = plt.subplots(figsize=(width, 2.75))
    style_axes(axis)
    axis.axhline(0, color="#7D746C", lw=0.7)
    axis.axhline(1, color="#7D746C", lw=0.7, ls=(0, (4, 2)))
    for component in components:
        label, marker, offset = component_specs[component]
        for xpos, transition in enumerate(transitions):
            current = keep[
                (keep["transition"] == transition) & (keep["component"] == component)
            ].sort_values("seed")
            values = current["total_recovery"].to_numpy(dtype=float)
            goal = str(current.iloc[0].get("goal_condition", transition.split("->")[-1])).lower()
            color = METHOD_COLORS.get(goal, "#4D4D4D")
            jitter = np.linspace(-0.04, 0.04, len(values))
            axis.scatter(
                xpos + offset + jitter,
                values,
                s=22,
                marker=marker,
                facecolor="white",
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            axis.errorbar(
                xpos + offset,
                mean,
                yerr=sd,
                fmt=marker,
                ms=4.4,
                color=color,
                markerfacecolor=color,
                markeredgecolor="0.15",
                capsize=2.3,
                elinewidth=0.9,
                zorder=4,
            )
    handles = [
        mpl.lines.Line2D(
            [], [], marker=component_specs[name][1], ls="", color="0.3",
            markerfacecolor="white", label=component_specs[name][0], markersize=4.8
        )
        for name in components
    ]
    legend = axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=len(handles),
        frameon=True,
        fancybox=False,
    )
    legend.get_frame().set_facecolor("#FFFBFB")
    legend.get_frame().set_edgecolor("#D0D0D0")
    axis.set_xticks(range(len(transitions)), [value.replace("->", "→").title() for value in transitions])
    axis.set_xlim(-0.55, len(transitions) - 0.45)
    axis.set_ylabel(r"KL recovery $R_{\mathrm{KL}}$")
    fig.tight_layout(pad=0.25)
    fig.savefig(args.output_dir / f"{args.stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        args.output_dir / f"{args.stem}.png", bbox_inches="tight", pad_inches=0.02, dpi=300
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
