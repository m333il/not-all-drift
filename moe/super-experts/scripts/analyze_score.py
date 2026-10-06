#!/usr/bin/env python3
"""Paired uncertainty for a finished task-pruning run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.run / "score" / "rows.jsonl").read_text().splitlines()]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["arm"], row["condition"]), {})[row["key"]] = row["score"]

    results = []
    for arm in sorted({arm for arm, _condition in grouped}):
        intact = grouped.get((arm, "intact"))
        if not intact:
            continue
        for condition in sorted(condition for other, condition in grouped if other == arm and condition != "intact"):
            other = grouped[(arm, condition)]
            keys = sorted(set(intact) & set(other))
            deltas = [other[key] - intact[key] for key in keys]
            mean = sum(deltas) / len(deltas)
            rng = random.Random(42)
            means = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                           for _ in range(args.bootstrap))
            lo = means[int(0.025 * len(means))]
            hi = means[int(0.975 * len(means))]
            results.append({
                "arm": arm, "condition": condition, "n": len(deltas), "paired_delta": mean,
                "bootstrap_95": [lo, hi],
                "improved": sum(delta > 0 for delta in deltas),
                "unchanged": sum(delta == 0 for delta in deltas),
                "worsened": sum(delta < 0 for delta in deltas),
            })
    print("SCORE_STATS=" + json.dumps(results, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
