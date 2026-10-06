#!/usr/bin/env python3
"""Read routing maps as a starting point for analysis.

Each checkpoint stores expert_counts.npz under routing_maps_final/<model>/<cell>.
Keys have the form <arm>|<stage>; values count token assignments in a
layers-by-experts matrix. The __all__ stage sums every prompt stage and is used
for aggregate Gini statistics and frequency-based pruning.

    python scripts/read_maps.py
    python scripts/read_maps.py qwen prompt-m500
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MAPS = Path(os.environ.get("MRD_MAPS",
                           Path(__file__).resolve().parent.parent / "routing_maps_final"))


def gini(x: np.ndarray) -> float:
    """0 - all layer experts load the same, (n−1) /n - all traffic for one."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def overview() -> None:
    for f in sorted(MAPS.glob("*/*/expert_counts.npz")):
        z = np.load(f, allow_pickle=True)
        stages = [k.split("|", 1)[1] for k in z.files if k != "_meta"]
        arm = next(k.split("|", 1)[0] for k in z.files if k != "_meta")
        shape = z[f"{arm}|__all__"].shape
        print(f"{f.parts[-3]:8s} {f.parts[-2]:28s} arm={arm:14s} "
              f"layers x experts={shape[0]}×{shape[1]}  stages: {', '.join(stages)}")


def detail(model: str, cell: str) -> None:
    hits = list(MAPS.glob(f"{model}/*{cell}*/expert_counts.npz"))
    if not hits:
        print(f"not found: {model}/{cell}")
        return
    z = np.load(hits[0], allow_pickle=True)
    arm = next(k.split("|", 1)[0] for k in z.files if k != "_meta")
    counts = np.asarray(z[f"{arm}|__all__"], dtype=np.float64)

    print(f"{hits[0].parts[-3]}/{hits[0].parts[-2]}, arm {arm}")
    print(f"layers {counts.shape[0]}, experts {counts.shape[1]}, "
          f"assignments {counts.sum():,.0f}")

    per_layer = np.array([gini(row) for row in counts])
    print(f"Gini: median {np.median(per_layer):.3f}, "
          f"highest-Gini layer {per_layer.argmax()} ({per_layer.max():.3f})")

    dead = (counts <= 0).sum(axis=1)
    print(f"dead experts per layer: the median {np.median(dead):.0f}, "
          f"maximum {dead.max()}")

    # So much traffic is removed by a mask, knocking out half the experts of each layer.
    half = counts.shape[1] // 2
    lost = [np.sort(row)[:half].sum() / row.sum() for row in counts if row.sum() > 0]
    print(f"the least-used half of each layer carries {100 * np.mean(lost):.1f}% of traffic")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", help="Backbone folder, e.g. qwen")
    parser.add_argument("cell", nargs="?", help="Cell folder, e.g. prompt-m500-s42")
    parser.add_argument("--maps", type=Path, default=MAPS)
    args = parser.parse_args()
    MAPS = args.maps
    if args.model and args.cell:
        detail(args.model, args.cell)
    else:
        overview()
