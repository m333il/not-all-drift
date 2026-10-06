"""Look at what the pruned models actually said, not just what they scored.

A level is summarised by one number, and that number lies in the same two ways
every time: a silent model reads 0.330 because a third of the test carries no
labels, and a model cut off mid-sentence reads as if it had nothing to say. Both
are visible in a handful of raw answers and invisible in the summary.

So this prints, per level: the share of answers that are empty, that carry no
end-of-turn marker, that repeat themselves, and the length they run to - then a
few actual rows, whole, including the reasoning channel when there is one.

    python scripts/peek_answers.py --results results/prune_final_5075 --rows 2
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

END_MARKERS = ("<|im_end|>", "<|return|>", "<|endoftext|>", "<|end|>")
# Padding after the end marker is not something the model "said": counting it
# as text makes every finished answer look like it rambled.
SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def repeats(text: str, run: int = 4) -> bool:
    """True when some 4-gram of words occurs more than four times.

    Degenerate decoding loops on a phrase; that is what separates 'needed more
    room' from 'would never have stopped'.
    """
    words = re.findall(r"\w+", SPECIAL.sub(" ", text).lower())
    if len(words) < run * 5:
        return False
    grams = [tuple(words[i:i + run]) for i in range(len(words) - run)]
    if not grams:
        return False
    top = max(grams.count(g) for g in set(grams[:400]))
    return top > 4


def said(row: dict) -> str:
    """What the model produced, with the padding that follows the end stripped."""
    raw = row.get("raw_response") or row.get("response") or ""
    for marker in ("<|return|>", "<|im_end|>", "<|end|>"):
        if marker in raw:
            return raw.split(marker)[0] + marker
    return raw


def summarise(rows: list[dict]) -> dict:
    raw = [row.get("raw_response") or row.get("response") or "" for row in rows]
    body = [said(row) for row in rows]
    # The sweep's own field. Reading a name it does not use makes every row look
    # empty, which is how this script first reported 100% silence on a level
    # that scored 0.736.
    parsed = [row.get("predicted_labels") for row in rows]
    lens = np.array([len(t) for t in body])
    return {
        "n": len(rows),
        "empty": sum(1 for p in parsed if not p) / len(rows),
        "rawly": sum(1 for row in rows if row.get("unparsable")) / len(rows),
        "endless": sum(1 for t in raw if not any(m in t for m in END_MARKERS)) / len(rows),
        "looped": sum(1 for t in body[:300] if repeats(t)) / min(300, len(rows)),
        "symbolism": float(np.median(lens)),
        "symbol: p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--rows", type=int, default=2, help="How many answers do I have?")
    p.add_argument("--chars", type=int, default=400, help="length")
    p.add_argument("--only", default=None, help="liner")
    args = p.parse_args()

    for path in sorted(args.results.glob("*/*/*/results.jsonl")):
        cell = f"{path.parent.parent.parent.name}/{path.parent.parent.name}/{path.parent.name}"
        if args.only and args.only not in cell:
            continue
        rows = [json.loads(l) for l in path.open()]
        if not rows:
            print(f"\n### {cell}empty")
            continue
        stats = summarise(rows)
        head = ", ".join(
            f"{k} {v:.2%}" if isinstance(v, float) and v <= 1 and "symbolism" not in k
            else f"{k} {v:.0f}"
            for k, v in stats.items() if k != "n")
        print(f"\n### {cell}  (n={stats['n']})")
        print(f"    {head}")
        # The longest answers are where the trouble is, so show those, not the head.
        order = sorted(range(len(rows)), key=lambda i: -len(said(rows[i])))
        for i in order[:args.rows]:
            r = rows[i]
            text = said(r).replace("\n", " ⏎ ")
            print(f"    [{i}] predicted={r.get('predicted_labels')} "
                  f"truth{r.get('true_labels')} f1={r.get('f1')} | {len(text)} Simv.")
            print(f"        {text[:args.chars]}")


if __name__ == "__main__":
    main()
