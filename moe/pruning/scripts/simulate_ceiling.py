"""What a smaller token ceiling would have cost, measured on answers already on disk.

The queue's whole cost is (number of batches) x (ceiling), because under pruning
most rows run the budget out and one such row makes its batch pay in full. So
the ceiling is the one free parameter left - but lowering it is only free if the
rows it cuts were producing nothing anyway.

That is answerable without a GPU. Decoding is greedy, so a row that finished
before the cut would have produced exactly the same text under any wider budget;
a row that did not finish becomes truncated, which the scorer already treats as
unparsable. Simulating a ceiling is therefore just: re-score, counting every row
longer than the cut as cut off.

    python scripts/simulate_ceiling.py --results results/prune_final_5075 \\
        --model gpt-oss --ceilings 512,1024,2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.task import answer_text, parse_response, predicted_labels  # noqa: E402

END_MARKERS = ("<|im_end|>", "<|return|>", "<|endoftext|>", "<|end|>")


def f1_sample(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    hit = len(pred & true)
    if not hit:
        return 0.0
    p, r = hit / len(pred), hit / len(true)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--model", default="gpt-oss")
    ap.add_argument("--ceilings", default="512,1024,2048")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer id; omitted, lengths are counted in "
                         "characters at four per token, which is close enough "
                         "for a go/no-go and needs no download")
    args = ap.parse_args()
    cuts = [int(c) for c in args.ceilings.split(",")]

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def n_tokens(text: str) -> int:
        if tok is None:
            return len(text) // 4
        return len(tok(text, add_special_tokens=False)["input_ids"])

    head = f"{'level':46} {'F1 as is.':>11}" + "".join(f"{f'F1 @{c}':>10}" for c in cuts)
    print(head)
    print(f"{'':46} {'':>11}" + "".join(f"{'(condemns)':>10}" for _ in cuts))
    for path in sorted((args.results / args.model).glob("*/*/results.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        if not rows:
            continue
        lens = [n_tokens(r.get("raw_response") or "") for r in rows]
        base_f1 = 0.0
        per_cut = {c: 0.0 for c in cuts}
        cut_count = {c: 0 for c in cuts}
        for row, ln in zip(rows, lens):
            true = set(row["true_labels"])
            raw = row.get("raw_response") or ""
            text = answer_text(raw, row.get("response") or "")
            pred = set(predicted_labels(parse_response(text), "lenient"))
            base_f1 += f1_sample(pred, true)
            for c in cuts:
                # Cut here means: the row never reached its end marker, so the
                # scorer sees no answer at all - the same thing it sees today
                # for a row that ran out of budget.
                if ln > c:
                    cut_count[c] += 1
                    per_cut[c] += f1_sample(set(), true)
                else:
                    per_cut[c] += f1_sample(pred, true)
        n = len(rows)
        cell = "/".join(path.parts[-3:-1]).replace("prefix-projected", "prefix").replace("-s42", "")
        line = f"{cell:46} {base_f1 / n:11.4f}"
        for c in cuts:
            line += f"{per_cut[c] / n:10.4f}"
        print(line)
        print(f"{'':46} {'':>11}" + "".join(f"{100 * cut_count[c] / n:9.0f}%" for c in cuts))


if __name__ == "__main__":
    main()
