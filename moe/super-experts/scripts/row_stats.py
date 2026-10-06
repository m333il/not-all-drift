#!/usr/bin/env python3
"""Exact match and empty share of finished scoring runs, read from their rows.

An unreadable answer counts as the empty label set, as in the pruning tables. An
answer is looping when it has no end marker and its text ends in one phrase of at
most 60 characters repeated at least three times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def looping(text: str) -> bool:
    tail = text.rstrip()
    for period in range(1, 61):
        if len(tail) >= 3 * period and tail[-period:] * 3 == tail[-3 * period:]:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True, help="test rows with gold labels")
    args = parser.parse_args()
    gold = {row["id"]: set(row["labels"]) for row in map(json.loads, args.rows.read_text().splitlines()) if row}
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "score" / "rows.jsonl").read_text().splitlines() if line]
        exact = empty = loops = 0
        for row in rows:
            predicted = set(row["prediction"]) if row["valid"] else set()
            exact += predicted == gold[row["key"]]
            empty += not predicted
            loops += (not row["finished"]) and looping(row.get("text", "") + row.get("analysis", ""))
        print("ROWSTATS " + json.dumps({"run": run.name, "n": len(rows), "exact": exact / len(rows),
                                        "empty": empty / len(rows), "looping": loops / len(rows),
                                        "no_end": sum(not row["finished"] for row in rows) / len(rows)}),
              flush=True)


if __name__ == "__main__":
    main()
