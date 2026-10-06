#!/usr/bin/env python3
"""Plot the verified five-arm contribution smoke summaries."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    with (args.analysis / "group-summary.csv").open() as stream:
        rows = {(r["arm"], int(r["layer"]), r["group"]): r for r in csv.DictReader(stream)}
    arms = ["base", "prefix-m500-init", "prefix-m500-s42", "prefix-m500-best", "prompt-m500-s42"]
    labels = ["Base", "Prefix: step 0", "Prefix: LR 1e-4", "Prefix: LR 3e-5", "Prompt: LR 0.1"]
    mass = np.array([[100 * float(rows[a, l, "virtual"]["attention_mass"]) for l in range(6)] for a in arms])
    early = np.array([[100 * sum(float(rows[a, l, g]["attention_mass"]) for g in ("real_0", "real_1", "real_2")) for l in range(6)] for a in arms])
    rms = np.array([[float(rows[a, l, "virtual"]["rms_l2"]) for l in range(6)] for a in arms])
    loo = np.array([[100 * float(rows[a, l, "virtual"]["loo_constant_energy_fraction"] or "nan") for l in range(6)] for a in arms])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout="constrained")
    panels = [(mass, "Attention to virtual keys (%)", "Blues", 0, 100),
              (early, "Attention to real keys 0, 1, 2 (%)", "Blues", 0, 100),
              (rms, "Virtual contribution after W_O: RMS L2", "YlOrBr", 0, float(rms.max())),
              (loo, "Virtual contribution: LOO constant energy score (%)", "viridis", 0, 100)]
    for ax, (values, title, cmap, low, high) in zip(axes.flat, panels):
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(6), labels=[f"L{i}" for i in range(6)])
        ax.set_yticks(range(5), labels=labels)
        for i in range(5):
            for j in range(6):
                value = values[i, j]
                text = "n/a" if np.isnan(value) else f"{value:.2f}" if values is rms else f"{value:.1f}"
                rgb = im.cmap(im.norm(value))[:3] if np.isfinite(value) else (1, 1, 1)
                color = "black" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.5 else "white"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle("Qwen3-30B-A3B: four matched validation inputs, prefill only\n"
                 "Equal weight per example; layers are zero-based; no causal intervention", fontsize=13)
    fig.savefig(args.analysis / "contributions-overview.png", dpi=180)
    fig.savefig(args.analysis / "contributions-overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
