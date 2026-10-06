"""Collect the own-frequency pruning sweep into one arm x level table.

Each arm was pruned by the experts *it* used least, so the levels are only
comparable down a column within one arm: the unpruned cell of that same arm is
the baseline, and the drop from it is the quantity of interest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FIELDS = (
    "model", "arm", "method", "m", "level", "pct_experts", "f1", "exact",
    "empty_pred_rate", "unparsable_rate", "mass_removed", "mass_removed_max_layer", "n",
)


def parse_arm(name: str) -> tuple[str, int | None]:
    """'prefix-projected-m200-s44' -> ('prefix-projected', 200)."""
    parts = name.split("-")
    for index, part in enumerate(parts):
        if part.startswith("m") and part[1:].isdigit():
            return "-".join(parts[:index]), int(part[1:])
    return name, None


def read_cells(grid: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(grid.glob("*/*/*_prune*/summary.json")):
        summary = json.loads(summary_path.read_text())
        arm_dir = summary_path.parent.parent
        method, m = parse_arm(arm_dir.name)
        mass = summary.get("pruned_mass") or {}
        level = summary.get("prune_level")
        if level is None:  # older cells carry it only in the directory name
            level = int(summary_path.parent.name.rsplit("prune", 1)[-1])
        rows.append(
            {
                "model": arm_dir.parent.name,
                "arm": arm_dir.name,
                "method": method,
                "m": m,
                "level": level,
                "pct_experts": summary.get("prune_level_pct"),
                "f1": summary.get("f1_mean"),
                "exact": summary.get("exact_mean"),
                "empty_pred_rate": summary.get("empty_pred_rate"),
                "unparsable_rate": summary.get("unparsable_rate"),
                # The comparable axis across arms: experts removed is a count,
                # this is the share of routed traffic those experts carried.
                "mass_removed": mass.get("mass_pruned_mean"),
                "mass_removed_max_layer": mass.get("mass_pruned_max"),
                "n": summary.get("n"),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]]) -> str:
    """One row per arm, one column per pruning level, plus the drop at 75%."""
    levels = sorted({r["pct_experts"] for r in rows if r["pct_experts"] is not None})
    by_arm: dict[tuple[str, str], dict[float, dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault((row["model"], row["arm"]), {})[row["pct_experts"]] = row

    head = "| model | arm | " + " | ".join(f"{lvl:g}%" for lvl in levels) + " | fall |"
    sep = "|" + "---|" * (len(levels) + 3)
    lines = [head, sep]
    for (model, arm), cells in sorted(by_arm.items()):
        values = []
        for lvl in levels:
            cell = cells.get(lvl)
            if not cell or cell["f1"] is None:
                values.append(" - ")
                continue
            mass = cell.get("mass_removed")
            # F1 with the share of routed traffic that level actually removed:
            # the same percentage of experts is a different intervention on
            # each arm, and the mass is what makes the arms comparable.
            values.append(
                f"{cell['f1']:.4f}" + (f" ({100 * mass:.1f}%)" if mass else "")
            )
        base = cells.get(0.0)
        last = cells.get(levels[-1]) if levels else None
        if base and last and base["f1"] is not None and last["f1"] is not None:
            drop = f"{last['f1'] - base['f1']:+.4f}"
        else:
            drop = " - "
        lines.append(f"| {model} | {arm} | " + " | ".join(values) + f" | {drop} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True,
                        help="Directory of finished pruning cells, e.g. results/prune")
    parser.add_argument("--out", type=Path, default=None, help="markdown table")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = read_cells(args.grid)
    if not rows:
        raise SystemExit(f"into {args.grid} no counted cells")

    table = markdown(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table + "\n")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    armed = len({(r["model"], r["arm"]) for r in rows})
    logger.info("%d cells over %d arms", len(rows), armed)
    print(table)


if __name__ == "__main__":
    main()
