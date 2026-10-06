"""Inequality metrics for every measured routing map.

Three views of the same counts:

* **overall** - the whole model, one number per metric per stage (counts summed
  over layers first, so a layer with more traffic weighs more);
* **per layer** - the same metrics for each layer separately, which is where the
  depth profile lives;
* **per stage** - every stage the map carries, because prompt tokens, an arm's
  own virtual tokens and the answer do not load the router the same way.

Writes a tidy CSV (one row per arm × stage × layer, with layer ``-1`` meaning
"whole model") plus a Markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mrd_pruning.uniformity import METRIC_NAMES, summarise  # noqa: E402

logger = logging.getLogger("analyze_maps")

FIELDS = ("model", "arm", "stage", "layer", "n_experts", "assignments") + METRIC_NAMES


def read_map(path: Path) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    """Return the arm key, its per-stage matrices, and the metadata block."""
    with np.load(path, allow_pickle=True) as bundle:
        meta = json.loads(str(bundle["_meta"])) if "_meta" in bundle.files else {}
        keys = [k for k in bundle.files if k != "_meta"]
        arms = sorted({k.split("|", 1)[0] for k in keys})
        if len(arms) != 1:
            raise ValueError(f"{path}: expected one arm, found {arms}")
        arm = arms[0]
        stages = {
            k.split("|", 1)[1]: np.asarray(bundle[k], dtype=np.float64) for k in keys
        }
    return arm, stages, meta


def rows_for_map(model: str, cell: str, stages: dict[str, np.ndarray]) -> list[dict]:
    """One row per stage (layer -1) and per stage×layer."""
    out: list[dict] = []
    for stage, matrix in sorted(stages.items()):
        if matrix.ndim != 2:
            raise ValueError(f"{model}/{cell}/{stage}: expected [layers, experts]")
        whole = matrix.sum(axis=0)
        out.append({
            "model": model, "arm": cell, "stage": stage, "layer": -1,
            "n_experts": matrix.shape[1], "assignments": float(whole.sum()),
            **summarise(whole),
        })
        for layer in range(matrix.shape[0]):
            out.append({
                "model": model, "arm": cell, "stage": stage, "layer": layer,
                "n_experts": matrix.shape[1], "assignments": float(matrix[layer].sum()),
                **summarise(matrix[layer]),
            })
    return out


def overall_table(rows: list[dict], stage: str = "__all__") -> str:
    """Markdown: one line per arm, whole-model metrics on one stage."""
    picked = [r for r in rows if r["stage"] == stage and r["layer"] == -1]
    if not picked:
        return f"(stages) {stage} Not in any maps."
    head = ("| model | arm | Gini | entropy | eff. experts | max/mean | "
            "Top 1 | Top 4 | Dead |")
    lines = [head, "|" + "---|" * 9]
    for r in sorted(picked, key=lambda x: (x["model"], x["arm"])):
        lines.append(
            f"| {r['model']} | {r['arm']} | {r['gini']:.3f} | {r['entropy_norm']:.3f} | "
            f"{r['effective_experts']:.1f} from {r['n_experts']} | {r['max_over_mean']:.2f} | "
            f"{r['top1_share']:.3f} | {r['top4_share']:.3f} | {r['dead_fraction']:.3f} |"
        )
    return "\n".join(lines)


def per_stage_table(rows: list[dict], model: str, cell: str) -> str:
    """Markdown: one line per stage of a single arm, whole-model metrics."""
    picked = [r for r in rows
              if r["model"] == model and r["arm"] == cell and r["layer"] == -1]
    lines = ["| stage | assignments | Gini | entropy | eff. experts | max/mean | top-1 | dead |",
             "|" + "---|" * 8]
    for r in sorted(picked, key=lambda x: (x["stage"] != "__all__", x["stage"])):
        lines.append(
            f"| {r['stage']} | {r['assignments']:,.0f} | {r['gini']:.3f} | "
            f"{r['entropy_norm']:.3f} | {r['effective_experts']:.1f} | "
            f"{r['max_over_mean']:.2f} | {r['top1_share']:.3f} | {r['dead_fraction']:.3f} |"
        )
    return "\n".join(lines).replace(",", " ")


def depth_table(rows: list[dict], model: str, cell: str, stage: str,
                every: int = 1) -> str:
    """Markdown: Gini and effective experts down the layers."""
    picked = sorted(
        (r for r in rows if r["model"] == model and r["arm"] == cell
         and r["stage"] == stage and r["layer"] >= 0),
        key=lambda x: x["layer"],
    )
    lines = ["| layer | Gini | entropy | eff. experts | max/mean | top-1 |",
             "|" + "---|" * 6]
    for r in picked:
        if r["layer"] % every:
            continue
        lines.append(
            f"| {r['layer']} | {r['gini']:.3f} | {r['entropy_norm']:.3f} | "
            f"{r['effective_experts']:.1f} | {r['max_over_mean']:.2f} | "
            f"{r['top1_share']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True,
                        help="routing_maps_s42_native")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--stage", default="__all__",
                        help="stage for the overall table")
    parser.add_argument("--depth-every", type=int, default=4,
                        help="print every Nth layer in the depth tables")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows: list[dict] = []
    cells: list[tuple[str, str]] = []
    for path in sorted(args.maps.glob("*/*/expert_counts.npz")):
        model, cell = path.parent.parent.name, path.parent.name
        _, stages, _ = read_map(path)
        rows.extend(rows_for_map(model, cell, stages))
        cells.append((model, cell))
        logger.info("%s / %s: stages %d", model, cell, len(stages))

    if not rows:
        raise SystemExit(f"into {args.maps} no cards")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("CSV: %s (%d lines)", args.csv, len(rows))

    parts = [f"## Summary of all maps (stage){args.stage}`)", "",
             overall_table(rows, args.stage), ""]
    for model, cell in cells:
        parts += [f"### {model}/{cell}", "", "**By stages, the whole model:**", "",
                  per_stage_table(rows, model, cell), "",
                  f"** By layers, stage *{args.stage}`:**", "",
                  depth_table(rows, model, cell, args.stage, args.depth_every), ""]
    text = "\n".join(parts)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
        logger.info("Markdown: %s", args.markdown)
    print(text)


if __name__ == "__main__":
    main()
