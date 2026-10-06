#!/usr/bin/env python3
"""Collect the quality grid into one table.

Every cell is the same measurement: the pruning sweep at level 0, which prunes
nothing and is therefore a plain evaluation, on the same 2000 test examples with
the same parser. That is the whole point of running it rather than quoting the
archive's own numbers - those were 200 validation examples scored by each run's
own parser, and are not comparable across arms.

Cells still running are simply absent. The table says so rather than leaving a
gap that reads like a zero.

    uv run scripts/collect_quality_grid.py --grid results/quality_grid \
        --out results-files/quality_grid.md
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import statistics
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mrd_pruning.task import f1_sample, parse_response, predicted_labels  # noqa: E402

logger = logging.getLogger("collect_quality")

CELL = re.compile(r"^(prompt|prefix-projected)-m(\d+)-s(\d+)$")
MODEL_LABEL = {"qwen": "Qwen3-30B-A3B", "gpt-oss": "gpt-oss-20b"}
METHOD_LABEL = {"prompt": "prompt tuning", "prefix-projected": "prefix (projected)",
                "base": "base"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=Path, default=Path("results/quality_grid"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--csv", type=Path)
    return p.parse_args(argv)


def read_cell(path: Path) -> dict | None:
    """One cell's summary.

    The sweep names the level-0 directory after the arm, so it is
    `base_prune000` for a base cell and `prompt_tuning_prune000` for an adapter
    one. Globbing rather than hard-coding: the base name silently skipped every
    adapter cell.
    """
    candidates = sorted(path.glob("*_prune000/summary.json")) + [path / "summary.json"]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size:
            return json.loads(candidate.read_text())
    return None


def metric(summary: dict, name: str) -> float | None:
    value = summary.get(name)
    return None if value is None else float(value)


def rescore(cell: Path, none_policy: str = "lenient") -> dict | None:
    """Score a cell from its stored responses with the *current* parser.

    Each cell's own ``summary.json`` was written by whatever parser version ran
    at the time, and this project has changed that parser twice in one day. The
    responses themselves are stored, so re-scoring makes every cell comparable
    without re-running a single model.

    Two F1 numbers, because they answer different questions. ``f1`` counts every
    example, including those whose generation budget ran out before the model
    answered. ``f1_completed`` counts only answered ones. The gap is the size of
    the budget artifact, and it is not symmetric across arms: a tuned arm reasons
    longer than the base and so gets truncated more often.
    """
    path = next(iter(sorted(cell.glob("*_prune000/results.jsonl"))), None)
    if path is None or not path.stat().st_size:
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        return None
    f1_all: list[float] = []
    f1_done: list[float] = []
    exact = unparsable = truncated = mixed = empty_pred = 0
    for row in rows:
        response = row["response"]
        parsed = parse_response(response)
        pred = list(predicted_labels(parsed, none_policy))
        gold = row["true_labels"]
        if isinstance(gold, str):
            gold = ast.literal_eval(gold)
        score = f1_sample(pred, gold)
        f1_all.append(score)
        # Empty means the harmony budget ran out before the `final` channel.
        if response.strip():
            f1_done.append(score)
        else:
            truncated += 1
        exact += sorted(pred) == sorted(gold)
        unparsable += parsed.unparsable
        mixed += parsed.mixed_none
        empty_pred += not pred
    return {
        "f1": statistics.mean(f1_all),
        "f1_completed": statistics.mean(f1_done) if f1_done else None,
        "exact": exact / len(rows),
        "unparsable": unparsable / len(rows),
        "truncated": truncated / len(rows),
        "mixed_none": mixed / len(rows),
        "empty": empty_pred / len(rows),
        "n": len(rows),
    }


def collect(grid: Path) -> list[dict]:
    rows: list[dict] = []
    for model_dir in sorted(p for p in grid.glob("*") if p.is_dir()):
        for cell in sorted(p for p in model_dir.glob("*") if p.is_dir()):
            scored = rescore(cell)
            if scored is None:
                continue
            m = CELL.match(cell.name)
            rows.append({
                "model": model_dir.name,
                "method": m.group(1) if m else cell.name,
                "m": int(m.group(2)) if m else None,
                "seed": int(m.group(3)) if m else None,
                **scored,
            })
    return rows


def fmt(value: float | None, digits: int = 4) -> str:
    return " - " if value is None else f"{value:.{digits}f}"


def markdown(rows: list[dict], expected: int) -> str:
    out = [
        "All cells are measured the same way: the pruning sweep at level 0, that is, nothing pruned.",
        "It doesn't cut off and it's a normal score, on the same 2,000 test samples and the same",
        "Numbers from the archive (200 examples of validation, each has its own parser)",
        "They are not mixed here - they are not comparable between arms.",
        "",
        "All cells are recalculated from the saved answers by the current parser, so",
        "They are comparable regardless of when they were counted.",
        "“Stripped” is the share of answers where the generation budget ran out before the final channel.",
        "These are zeroed in F1, and the column \"F1 without crops\" shows the price of this.",
        "",
        f"Cells ready: **{len(rows)} from {expected}**.",
        "",
    ]
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        out.append(f"#### {MODEL_LABEL.get(model, model)}\n")
        out.append("| arm | m |side | F1 | F1 uncircumcised | exact | cropped | not dissected | impurity NONE | n |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda r: (r["method"] != "base", r["method"],
                                            r["m"] or 0, r["seed"] or 0)):
            name = METHOD_LABEL.get(r["method"], r["method"])
            out.append(
                f"| {name} | {r['m'] or ' - '} | {r['seed'] or ' - '} | "
                f"**{fmt(r['f1'])}** | {fmt(r['f1_completed'])} | {fmt(r['exact'])} | "
                f"{fmt(r['truncated'], 3)} | {fmt(r['unparsable'], 3)} | "
                f"{fmt(r['mixed_none'], 3)} | {int(r['n'])} |"
            )
        out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)
    rows = collect(args.grid)
    logger.info("collected %d cells from %s", len(rows), args.grid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown(rows, expected=38) + "\n")
    logger.info("wrote %s", args.out)
    if args.csv and rows:
        with args.csv.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
