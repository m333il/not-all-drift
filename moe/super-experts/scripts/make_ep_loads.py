#!/usr/bin/env python3
"""Collect per-arm expert shares from routing maps into the input of ``ep_bench.py``.

Reads ``<maps>/<model>/<cell>/expert_counts.npz`` as written by
``moe/pruning/scripts/measure_routing_map.py`` and stores, for every cell, the
share of each layer's assignments that each expert receives over all routed
positions (stage ``__all__``). Keys are ``<model>|<cell>``. Every model needs a
``base`` cell, which ``ep_bench.py`` uses for the layer count of the uniform load.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def shares(path: Path, stage: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    key = next(k for k in z.files if k != "_meta" and k.endswith(f"|{stage}"))
    counts = np.asarray(z[key], dtype=np.float64)
    total = counts.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError(f"{path}: a layer has no assignments in stage {stage}")
    return counts / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--models", default="qwen,gpt-oss")
    parser.add_argument("--stage", default="__all__")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loads = {}
    for model in args.models.split(","):
        cells = sorted(p.parent.name for p in (args.maps / model).glob("*/expert_counts.npz"))
        if "base" not in cells:
            raise FileNotFoundError(f"{args.maps / model}: no base cell")
        for cell in cells:
            loads[f"{model}|{cell}"] = shares(args.maps / model / cell / "expert_counts.npz", args.stage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **loads)
    print(f"{args.out}: {len(loads)} cells")


if __name__ == "__main__":
    main()
