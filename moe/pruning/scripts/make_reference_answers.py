"""Turn a sweep's per-example results into the answer file the router map wants.

``measure_routing_map.py`` needs one ``{"comment", "answer"}`` row per data row,
in the same order, so that the answer tokens it counts are the ones the arm
actually produced on that example.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="jsonl with 'comment'")
    parser.add_argument("--results", type=Path, required=True, help="sweep results.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", action="store_true",
                        help="use the full decode (reasoning included) instead of the "
                             "parsed answer - required for a reasoning-mode map")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    results = [
        json.loads(line) for line in args.results.read_text().splitlines() if line.strip()
    ]
    if len(results) != len(rows):
        raise SystemExit(
            f"Not evenly defined: data {len(rows)}results {len(results)}"
        )

    empty = 0
    with args.out.open("w") as handle:
        for row, result in zip(rows, results):
            answer = (result.get("raw_response") if args.raw else None)
            if answer is None:
                if args.raw:
                    raise SystemExit(
                        "--raw was asked for but results.jsonl has no 'raw_response'; "
                        "that cell predates the field and must be recomputed"
                    )
                answer = result.get("response", "")
            empty += int(not answer.strip())
            handle.write(
                json.dumps(
                    {"comment": row.get("comment", row.get("text")), "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("written %d answers (%d empty) → %s", len(rows), empty, args.out)


if __name__ == "__main__":
    main()
