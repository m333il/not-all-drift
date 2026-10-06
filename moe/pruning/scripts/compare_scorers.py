"""Score the same answers with the pruning parser and with the training-run parser.

The pruning table subtracts two numbers produced by two different harnesses:
the intact F1 comes from `arm-provenance.json`, written by the training run in
`moe/adaptation`, the pruned F1 from this package. If the two scorers disagree on the same text, part
column is the harness, not the pruning.

This measures that disagreement directly: one set of saved answers, two
scorers, same metric. The metric itself is provably identical - theirs is
2PR/(P+R) with P=tp/|pred|, R=tp/|true|, which reduces to 2|p∩t|/(|p|+|t|) -
so anything that moves is the parser.

Their parser, transcribed from `mrd/civil_multilabel_task.py` (01-09-2026):

  * reads the whole response, not the first line;
  * keeps only what follows the last `answer:`;
  * turns newlines into commas and splits on commas;
  * `NONE` *terminates* the list - tokens before it count, tokens after do not;
  * unknown tokens are dropped silently.

Ours reads the first line, treats a roster echo as unparsable, and applies a
NONE policy. Both choices are defensible; what matters is that they are not the
same, and the table currently mixes them.

    python scripts/compare_scorers.py --results results/prune_final_5075
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.task import (  # noqa: E402
    LABELS, answer_text, f1_sample, parse_response, predicted_labels,
)

NONE_TOKEN = "NONE"


def parse_theirs(response: str) -> set[str]:
    """`mrd.civil_multilabel_task.parse_response`, transcribed verbatim."""
    text = response.strip().lower()
    if not text:
        return set()
    if "answer:" in text:
        text = text.rsplit("answer:", 1)[1]
    pieces = [p.strip(" \t\n\r.,;:!\"'`*-") for p in text.replace("\n", ",").split(",")]
    found: list[str] = []
    for piece in pieces:
        if piece == NONE_TOKEN.lower():
            break
        if piece in set(LABELS):
            found.append(piece)
    return set(found)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--none-policy", default="lenient")
    ap.add_argument("--examples", type=int, default=0,
                    help="Show so many lines where the two parsers split")
    args = ap.parse_args()

    print(f"{'level':46} {'our F1':>8} {'F1':>8} {'difference':>9} {'rows changed':>16}")
    shown = 0
    for path in sorted(args.results.glob("*/*/*/results.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        if not rows:
            continue
        ours = theirs = 0.0
        differ = 0
        for row in rows:
            true = set(row["true_labels"])
            raw = row.get("raw_response") or ""
            span = answer_text(raw, row.get("response") or "")
            p_ours = set(predicted_labels(parse_response(span), args.none_policy))
            # Theirs never saw harmony channels - it was written for a model
            # that has none - so it is given the same span ours reads. Handing
            # it the raw text would measure the channel split, not the parser.
            p_theirs = parse_theirs(span)
            ours += f1_sample(p_ours, true)
            theirs += f1_sample(p_theirs, true)
            if p_ours != p_theirs:
                differ += 1
                if shown < args.examples:
                    shown += 1
                    print(f"    ours{sorted(p_ours) or 'NONE'} their{sorted(p_theirs) or 'NONE'} "
                          f"truth{sorted(true) or 'NONE'} | {span[:90]!r}")
        n = len(rows)
        cell = "/".join(path.parts[-3:-1]).replace("prefix-projected", "prefix").replace("-s42", "")
        print(f"{cell:46} {ours / n:8.4f} {theirs / n:8.4f} "
              f"{(ours - theirs) / n:+9.4f} {differ:16}")


if __name__ == "__main__":
    main()
