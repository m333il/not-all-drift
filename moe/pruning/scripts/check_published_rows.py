"""Re-score the training run's held-out test answers with the pruning parser.

The intact column of the pruning table comes from the training run in
`moe/adaptation`, the pruned columns from this package. `compare_scorers.py` showed the two parsers agree on *our* generations;
this closes the other half - it reads *their* saved generations, scores them
here, and compares row by row against the score they recorded.

What it can prove: that the intact numbers are reproducible by this harness,
so the drop column is not an artefact of scoring. What it still cannot cover:
their run used its own decoding (ceiling 64 tokens, their batching), and a
generation difference would not show up here because the text is theirs.

    python scripts/check_published_rows.py --repo <hf-dataset-with-test-rows>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.task import (  # noqa: E402
    answer_text, f1_sample, parse_response, predicted_labels,
)

ROWS = (
    "heldout-test-best-lr-20260918/test-best-prompt-20260918T150607Z--rows.jsonl",
    "heldout-test-best-lr-20260918/test-best-prefix-20260918T150609Z--rows.jsonl",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="Hugging Face dataset holding the saved test rows")
    ap.add_argument("--test", type=Path, default=Path("data/test_canonical_n2000.jsonl"))
    ap.add_argument("--none-policy", default="lenient")
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    gold = {}
    for line in args.test.open():
        row = json.loads(line)
        gold[row["id"]] = set(row["labels"])

    per_arm = defaultdict(lambda: {"ours": 0.0, "theirs": 0.0, "n": 0,
                                   "differ": 0, "missing": 0})
    shown = 0
    for name in ROWS:
        path = hf_hub_download(args.repo, name)
        for line in Path(path).open():
            row = json.loads(line)
            arm = row["arm"]
            acc = per_arm[arm]
            key = row.get("key")
            if key not in gold:
                acc["missing"] += 1
                continue
            true = gold[key]
            raw = row.get("text") or ""
            # Their rows hold the answer span already; `answer_text` is a no-op
            # on text without channel markers and the right thing on text with.
            ours = set(predicted_labels(parse_response(answer_text(raw, raw)),
                                        args.none_policy))
            acc["ours"] += f1_sample(ours, true)
            acc["theirs"] += float(row["score"])
            acc["n"] += 1
            theirs_pred = set(row.get("prediction") or ())
            if ours != theirs_pred:
                acc["differ"] += 1
                if shown < args.examples:
                    shown += 1
                    print(f"    {arm}:our={sorted(ours) or 'NONE'} "
                          f"their{sorted(theirs_pred) or 'NONE'} "
                          f"truth{sorted(true) or 'NONE'} | {raw[:80]!r}")

    print(f"\n{'arm':22} {'rows':>6} {'F1':>8} {'our F1':>8} {'difference':>9} "
          f"{'tagged':>16} {'not in the test':>12}")
    for arm in sorted(per_arm):
        a = per_arm[arm]
        n = max(a["n"], 1)
        print(f"{arm:22} {a['n']:6} {a['theirs'] / n:8.4f} {a['ours'] / n:8.4f} "
              f"{(a['ours'] - a['theirs']) / n:+9.4f} {a['differ']:16} {a['missing']:12}")


if __name__ == "__main__":
    main()
