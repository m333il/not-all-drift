#!/usr/bin/env python3
"""Routing displacement by depth (the displacement figure of the MoE section).

Reads the routing maps written by ``moe/pruning/scripts/measure_routing_map.py``,
laid out as ``<maps>/<model>/<cell>/expert_counts.npz``, and plots the total
variation distance between each arm's expert shares and the frozen base's,
layer by layer. The default stage is ``comment``, the tokens every arm shares.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = {"qwen": "Qwen3-30B-A3B (128 experts, top-8)",
          "gpt-oss": "gpt-oss-20b (32 experts, top-4)"}


@dataclass(frozen=True)
class Arm:
    cell: str
    label: str
    color: str
    style: str


ARMS = {
    "qwen": [
        Arm("gepa-qwen3-2507-n1000-s42", "GEPA", "#E69F00", "--"),
        Arm("prompt-m500-s42", "prompt tuning", "#0072B2", "-"),
        Arm("prefix-projected-m500-s42", "prefix tuning", "#009E73", "-."),
    ],
    "gpt-oss": [
        Arm("gepa-gpt-oss-20b-n1000-s42", "GEPA", "#E69F00", "--"),
        Arm("prompt-m500-s42", "prompt tuning", "#0072B2", "-"),
        Arm("prefix-projected-m500-s42", "prefix tuning", "#009E73", "-."),
    ],
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def shares(maps: Path, model: str, cell: str, stage: str) -> np.ndarray:
    """Per-layer share of the top-k assignments each expert receives."""
    z = np.load(maps / model / cell / "expert_counts.npz", allow_pickle=True)
    key = next(k for k in z.files if k != "_meta" and k.endswith(f"|{stage}"))
    counts = np.asarray(z[key], dtype=np.float64)
    total = counts.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return counts / total


def total_variation(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=1)


def fig_displacement(maps: Path, out: Path, stage: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1))
    for col, model in enumerate(MODELS):
        ax = axes[col]
        base = shares(maps, model, "base", stage)
        for arm in ARMS[model]:
            tv = total_variation(shares(maps, model, arm.cell, stage), base)
            depth = np.arange(tv.size) / (tv.size - 1)
            ax.plot(depth, tv, color=arm.color, ls=arm.style, label=arm.label)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_title(MODELS[model])
        ax.set_xlabel("relative depth (first $\\rightarrow$ last MoE layer)")
        if col == 0:
            ax.set_ylabel("routing displacement\n$\\mathrm{TV}$ from frozen base")
            ax.legend(loc="upper right", handlelength=1.6)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    path = out / "moe_displacement.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True,
                        help="Directory of <model>/<cell>/expert_counts.npz")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--stage", default="comment")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(fig_displacement(args.maps, args.out, args.stage))


if __name__ == "__main__":
    main()
