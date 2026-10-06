#!/usr/bin/env python3
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from mrd.civil_multilabel_task import LABELS, parse_response, score_example, target_text
from mrd.jsonl import read_jsonl


def score_arm(examples, rows):
    by_key = {row["key"]: row for row in examples}
    if len(by_key) != len(examples):
        raise ValueError("Duplicate example IDs")
    if len(rows) != len(by_key) or {row["key"] for row in rows} != by_key.keys():
        raise ValueError("Expected exactly one generation per example ID")
    scored = []
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    for row in rows:
        example = by_key[row["key"]]
        predicted = parse_response(row["final"])
        truth = set(example["labels"])
        f1 = score_example(example, row["final"])[0]
        scored.append({"key": row["key"], "label_f1": f1,
                       "score": f1 if row["finished"] else 0.0,
                       "valid": predicted is not None, "finished": row["finished"],
                       "exact_label_set": predicted == truth,
                       "exact_output_protocol": row["final"].strip() == target_text(example["labels"]),
                       "predicted_labels": sorted(predicted) if predicted is not None else None,
                       "true_labels": example["labels"]})
        for label in LABELS:
            positive = row["finished"] and predicted is not None and label in predicted
            counts[label]["tp"] += positive and label in truth
            counts[label]["fp"] += positive and label not in truth
            counts[label]["fn"] += not positive and label in truth
    for count in counts.values():
        denominator = 2 * count["tp"] + count["fp"] + count["fn"]
        count["f1"] = 2 * count["tp"] / denominator if denominator else 0.0
    return {"n": len(rows),
            **{name: sum(row[name] for row in scored) / len(scored)
               for name in ["label_f1", "score", "valid", "finished", "exact_label_set", "exact_output_protocol"]},
            "per_label": counts, "macro_label_f1": sum(c["f1"] for c in counts.values()) / len(LABELS),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "generation_seconds": sum(row["elapsed_seconds"] for row in rows), "per_example": scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = read_jsonl(args.examples)
    groups = defaultdict(list)
    for row in read_jsonl(args.generations):
        if "final" in row:
            groups[row["arm"]].append(row)
    report = {"examples_sha256": hashlib.sha256(args.examples.read_bytes()).hexdigest(),
              "generations_sha256": hashlib.sha256(args.generations.read_bytes()).hexdigest(),
              "primary": "mean completed-response label-set F1; invalid responses zero",
              "per_label_convention": "invalid or unfinished responses treated as no positive predictions; undefined label F1 zero",
              "arms": {name: score_arm(examples, rows) for name, rows in groups.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in arm.items() if k not in {"per_example", "per_label"}}
                      for name, arm in report["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
