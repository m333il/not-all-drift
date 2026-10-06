"""Build the two splits the own-frequency pruning run needs.

The frequency map must be measured on data the quality number is *not* taken
from, otherwise the pruning decision has seen the evaluation set. This writes a
calibration split drawn from the held-out validation file and an evaluation
split drawn from the canonical test, and refuses to write anything if the two
overlap.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def normalise(row: dict[str, Any]) -> dict[str, Any]:
    """Both files carry the text under a different key; the sweep wants 'comment'."""
    comment = row.get("comment", row.get("text"))
    if comment is None:
        raise KeyError(f"row without text: {sorted(row)}")
    return {"id": row["id"], "comment": comment, "labels": list(row["labels"])}


def describe(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(rows),
        "empty_label_rate": round(sum(1 for r in rows if not r["labels"]) / len(rows), 4),
        "mean_labels": round(sum(len(r["labels"]) for r in rows) / len(rows), 4),
    }


def write_split(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--n-calibration", type=int, default=500)
    parser.add_argument("--n-evaluation", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    calibration = [normalise(r) for r in read_rows(args.validation, args.n_calibration)]
    evaluation = [normalise(r) for r in read_rows(args.test, args.n_evaluation)]
    for name, rows, want in (
        ("caliberation", calibration, args.n_calibration),
        ("score", evaluation, args.n_evaluation),
    ):
        if len(rows) < want:
            raise SystemExit(f"{name}received {len(rows)} lines requested {want}")

    overlap = {r["id"] for r in calibration} & {r["id"] for r in evaluation}
    if overlap:
        raise SystemExit(
            f"calibration and evaluation overlap {len(overlap)} Examples: "
            "Frequency map would look at the score set."
        )

    calib_path = args.out_dir / f"calib_val_n{len(calibration)}.jsonl"
    eval_path = args.out_dir / f"prune_eval_n{len(evaluation)}.jsonl"
    write_split(calibration, calib_path)
    write_split(evaluation, eval_path)

    provenance = {
        "calibration": {
            "path": str(calib_path),
            "source": str(args.validation),
            **describe(calibration),
        },
        "evaluation": {
            "path": str(eval_path),
            "source": str(args.test),
            **describe(evaluation),
        },
        "overlap_ids": 0,
    }
    prov_path = args.out_dir / "prune_splits.provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))

    logger.info("Calibration by %s (%s)", calib_path, describe(calibration))
    logger.info("%s (%s)", eval_path, describe(evaluation))
    logger.info("intersection: 0 | provenance: %s", prov_path)


if __name__ == "__main__":
    main()
