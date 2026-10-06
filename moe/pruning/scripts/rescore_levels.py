"""Re-score finished levels from the stored responses, without a GPU.

Every level kept `response` and `raw_response` for all 2000 rows, so a parser
fix can be applied to work that already ran: the tokens the model produced do
not change, only how they are read. That covers the echoed-roster bug and any
later reading fix.

What it cannot repair is a response that was never finished. When the token
budget ran out mid-answer the missing text does not exist anywhere, and the
only honest thing to do is to count those rows and say so - `truncated_rate`
below - rather than to score a cut-off answer as a wrong one.

The original `summary.json` is never touched. The corrected numbers land in
`summary_rescored.json` beside it, so the raw record of what the run did stays
exactly as the run wrote it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from mrd_pruning.task import answer_text, parse_response, predicted_labels  # noqa: E402

# Every end-of-turn marker the two backbones use. A response with none of them
# ran into the budget.
END_MARKERS = ("<|im_end|>", "<|return|>", "<|endoftext|>", "<|end|>")


def f1_sample(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    hit = len(pred & true)
    if not hit:
        return 0.0
    precision, recall = hit / len(pred), hit / len(true)
    return 2 * precision * recall / (precision + recall)


def rescore(level_dir: Path, none_policy: str) -> dict | None:
    summary_path = level_dir / "summary.json"
    records_path = level_dir / "results.jsonl"
    if not (summary_path.is_file() and records_path.is_file()):
        return None
    summary = json.loads(summary_path.read_text())
    records = [json.loads(line) for line in records_path.open()]
    if not records:
        return None

    f1_total = exact_total = 0.0
    empty = unparsable = mixed = truncated = changed = 0
    tp = fp = fn = 0
    for rec in records:
        true = set(rec["true_labels"])
        # From the raw response, not from `response`. `response` is what the
        # run's own `answer_text` already extracted, so a fix to *which span is
        # the answer* - the channel-header case - cannot reach it: the text was
        # thrown away before it was ever written down. `raw_response` is the
        # model's own output and is the only record that survives a parser fix.
        raw = rec.get("raw_response") or ""
        text = answer_text(raw, rec.get("response") or "") if raw else rec.get("response") or ""
        parsed = parse_response(text)
        pred = set(predicted_labels(parsed, none_policy))

        f1_total += f1_sample(pred, true)
        exact_total += pred == true
        empty += not pred
        unparsable += bool(parsed.unparsable)
        mixed += bool(parsed.mixed_none)
        changed += pred != set(rec["predicted_labels"])
        truncated += not any(marker in raw for marker in END_MARKERS)
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)

    n = len(records)
    return {
        "n": n,
        "f1_mean": f1_total / n,
        "exact_mean": exact_total / n,
        "empty_pred_rate": empty / n,
        "unparsable_rate": unparsable / n,
        "mixed_none_rate": mixed / n,
        "truncated_rate": truncated / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "rows_changed": changed,
        "f1_mean_as_run": summary.get("f1_mean"),
        "none_policy": none_policy,
        "source": "rescored from results.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/prune_eval_s42")
    parser.add_argument("--none-policy", default="lenient", choices=["lenient", "strict"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = Path(args.results)
    written = moved = 0
    for summary_path in sorted(results.glob("*/*/*_prune*/summary.json")):
        level_dir = summary_path.parent
        fresh = rescore(level_dir, args.none_policy)
        if fresh is None:
            continue
        delta = fresh["f1_mean"] - (fresh["f1_mean_as_run"] or 0.0)
        if abs(delta) > 0.0005:
            moved += 1
            print(f"{level_dir.relative_to(results)}: "
                  f"{fresh['f1_mean_as_run']:.3f} -> {fresh['f1_mean']:.3f} "
                  f"({delta:+.3f}), lines have changed {fresh['rows_changed']}")
        if not args.dry_run:
            (level_dir / "summary_rescored.json").write_text(
                json.dumps(fresh, ensure_ascii=False, indent=2) + "\n")
            written += 1

    print(f"rescored levels: {written}, scores changed: {moved}")


if __name__ == "__main__":
    main()
