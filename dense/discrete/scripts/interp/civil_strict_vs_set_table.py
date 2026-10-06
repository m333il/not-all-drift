"""Strict and set samples-F1 on the test split for GEPA and the continuous methods.

Both branches are re-parsed from their raw responses with the same parser. GEPA runs are
read from ``predictions.parquet`` under ``--eval-root``; prompt/prefix tuning runs from
``predictions.jsonl`` under ``--continuous-root``. No model is run.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interpretability_gepa.datasets import CIVIL_LABELS
from interpretability_gepa.metrics import empty_aware_sample_f1_rows, multilabel_matrix
from interpretability_gepa.prompts import parse_labels


def _parse(text: str | None) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool, bool]:
    """Return (strict, set, strict_ok, set_ok, order_violation)."""
    if text is None:
        return (), (), False, False, False
    try:
        lenient = parse_labels(text, CIVIL_LABELS, enforce_order=False)
        lenient_ok = True
    except Exception:
        lenient, lenient_ok = (), False
    try:
        strict = parse_labels(text, CIVIL_LABELS)
        return strict, lenient, True, lenient_ok, False
    except Exception:
        return (), lenient, False, lenient_ok, lenient_ok


def _metrics(
    y_true: np.ndarray, predicted: Sequence[Sequence[str]], ok: Sequence[bool], prefix: str
) -> dict[str, float]:
    y_pred = multilabel_matrix(predicted, CIVIL_LABELS)
    mask = np.asarray(ok, dtype=bool)
    rows = empty_aware_sample_f1_rows(y_true, y_pred)
    rows[~mask] = 0.0
    empty = y_true.sum(axis=1) == 0
    positive_rows = empty_aware_sample_f1_rows(y_true[~empty], y_pred[~empty])
    positive_rows[~mask[~empty]] = 0.0
    return {
        f"f1_{prefix}": float(rows.mean()),
        f"f1_pos_{prefix}": float(positive_rows.mean()),
        f"none_acc_{prefix}": float(
            np.logical_and(y_pred[empty].sum(axis=1) == 0, mask[empty]).mean()
        ),
        f"avg_pred_{prefix}": float(y_pred.sum(axis=1).mean()),
        f"fail_{prefix}": float(1.0 - mask.mean()),
    }


def _row(name: str, branch: str, y_true: np.ndarray, texts: Sequence[str | None]) -> dict[str, Any]:
    strict, lenient, strict_ok, set_ok, violated = [], [], [], [], []
    for text in texts:
        parsed, as_set, so, lo, v = _parse(text)
        strict.append(parsed)
        lenient.append(as_set)
        strict_ok.append(so)
        set_ok.append(lo)
        violated.append(v)
    record: dict[str, Any] = {"branch": branch, "condition": name, "n": int(len(texts))}
    record.update(_metrics(y_true, strict, strict_ok, "strict"))
    record.update(_metrics(y_true, lenient, set_ok, "set"))
    record["order_violations"] = int(sum(violated))
    record["order_violation_rate"] = float(np.mean(violated))
    record["delta_set_minus_strict"] = record["f1_set"] - record["f1_strict"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--controls-root", type=Path)
    parser.add_argument(
        "--continuous-root",
        type=Path,
        help="root of prompt/prefix tuning runs laid out as <method>/vt*/n*/s*/r*/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--virtual-tokens", type=int, nargs="+", default=[100, 200, 500])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gold_rows = [tuple(json.loads(line)["labels"]) for line in args.split_file.open()]
    y_true = multilabel_matrix(gold_rows, CIVIL_LABELS)

    records: list[dict[str, Any]] = []
    for root, branch in ((args.eval_root, "discrete"), (args.controls_root, "control")):
        if root is None or not root.exists():
            continue
        for run_dir in sorted(root.glob("*/")):
            predictions = run_dir / "predictions.parquet"
            if not predictions.exists():
                continue
            frame = pd.read_parquet(predictions)
            if len(frame) != len(gold_rows):
                print(f"SKIP {run_dir.name}: {len(frame)} rows")
                continue
            records.append(
                _row(str(frame["condition"].iloc[0]), branch, y_true, list(frame["raw_response"]))
            )

    per_cell: list[dict[str, Any]] = []
    continuous = [] if args.continuous_root is None else args.continuous_root.glob(
        "*/vt*/n*/s*/r*/run_config.json"
    )
    for config_path in sorted(continuous):
        config = json.loads(config_path.read_text())
        if config["num_virtual_tokens"] not in args.virtual_tokens:
            continue
        predictions = config_path.parent / "predictions.jsonl"
        if not predictions.exists():
            continue
        raw = [json.loads(line) for line in predictions.open(encoding="utf-8")]
        if len(raw) != len(gold_rows):
            print(f"SKIP {config_path.parent}: {len(raw)} rows")
            continue
        name = (
            f"{config['method']} vt{config['num_virtual_tokens']} "
            f"n{config['n_samples']} s{config['sample_idx']}"
        )
        record = _row(name, config["method"], y_true, [item.get("pred_text") for item in raw])
        record["virtual_tokens"] = config["num_virtual_tokens"]
        record["train_size"] = config["n_samples"]
        record["seed"] = config["sample_idx"]
        per_cell.append(record)

    cells = pd.DataFrame(per_cell)
    if not cells.empty:
        cells.to_csv(args.output / "strict_vs_set_continuous_cells.csv", index=False)
        numeric = [
            column
            for column in cells.select_dtypes(include="number").columns
            if column not in {"train_size", "seed", "virtual_tokens"}
        ]
        grouped = cells.groupby(["branch", "train_size"])[numeric].mean().reset_index()
        grouped["condition"] = grouped.apply(
            lambda r: f"{r['branch']} n{int(r['train_size'])} (mean vt/seed)", axis=1
        )
        records.extend(grouped.drop(columns=["train_size"]).to_dict("records"))

    table = pd.DataFrame(records)
    order = [
        "branch",
        "condition",
        "n",
        "f1_strict",
        "f1_set",
        "delta_set_minus_strict",
        "fail_strict",
        "fail_set",
        "order_violations",
        "order_violation_rate",
        "f1_pos_strict",
        "f1_pos_set",
        "none_acc_strict",
        "none_acc_set",
        "avg_pred_strict",
        "avg_pred_set",
    ]
    table = table[[c for c in order if c in table.columns]]
    table.to_csv(args.output / "strict_vs_set_table.csv", index=False)
    print(f"WROTE strict_vs_set_table.csv rows={len(table)}")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
