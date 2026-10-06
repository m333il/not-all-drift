#!/usr/bin/env python3
"""Build the project's canonical multilabel test file.

Every earlier quality number in this project was measured on the **first 2000
rows** of ``data/civil_multilabel/test_stratified.jsonl`` (3000 rows, md5
``23176d599186994abe954e34110cc171``), with ``--max-new-tokens 24``. A number
measured on any other sample is not on that scale.

The file this repo had been using, ``test_n2000_norm.jsonl``, is a legitimate
subset of the same pool - all 2000 ids are in it and every text and label set
matches - but it is a *different* 2000: 1993 of them fall in the canonical first
2000, and the order differs. That is a small difference, and small differences
between protocols are exactly what makes two tables disagree for reasons nobody
can later reconstruct.

Field names are normalised (``text`` -> ``comment``) because that is what the
sweep reads; nothing else is touched, and the source digest is recorded.

    uv run scripts/build_canonical_test.py \
        --source ~/moe-routing-drift/data/civil_multilabel/test_stratified.jsonl \
        --out data/test_canonical_n2000.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("build_canonical_test")

SOURCE_MD5 = "23176d599186994abe954e34110cc171"
LABELS = ("toxicity", "obscene", "threat", "insult", "identity_attack")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--allow-digest-mismatch", action="store_true",
                   help="proceed even if the source is not the recorded file")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)

    blob = args.source.read_bytes()
    digest = hashlib.md5(blob).hexdigest()
    if digest != SOURCE_MD5:
        message = (f"source md5 {digest} != the recorded {SOURCE_MD5}; this is not "
                   "the file every earlier number was measured on")
        if not args.allow_digest_mismatch:
            raise SystemExit(message)
        logger.warning("%s", message)

    rows = [json.loads(line) for line in blob.decode().splitlines() if line.strip()]
    logger.info("source: %d rows, md5 %s", len(rows), digest)
    if len(rows) < args.n:
        raise SystemExit(f"source has {len(rows)} rows, need {args.n}")

    unknown: set[str] = set()
    out_rows = []
    for row in rows[: args.n]:
        labels = list(row["labels"])
        unknown |= {label for label in labels if label not in LABELS}
        out_rows.append({"id": row["id"], "comment": row["text"], "labels": labels})
    if unknown:
        raise SystemExit(f"labels outside the task's vocabulary: {sorted(unknown)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    provenance = {
        "source": str(args.source),
        "source_md5": digest,
        "n": args.n,
        "selection": "first N rows, original order",
        "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    n_empty = sum(not r["labels"] for r in out_rows)
    logger.info("wrote %s: %d rows, %d with no label (%.1f%%)",
                args.out, len(out_rows), n_empty, 100 * n_empty / len(out_rows))
    logger.info("sha256 %s", provenance["out_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
