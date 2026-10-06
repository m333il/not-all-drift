"""Multi-label Civil Comments: five labels, any subset, possibly none.

A target is a comma-separated list of up to five label names, and a response
is scored by set-level F1 against the true labels.

Data: ``data/civil_multilabel/``, the v2 splits built by
``dense/discrete/src/interpretability_gepa/civil_splits_v2`` (``manifest.json``
records their contract). Train sizes 200/500/1000/10000/20000 for seeds
42/43/44, a shared 3000-row test set, 33% of rows with no label.

Class frequencies in test are very uneven (toxicity 65.4%, insult 47.6%,
identity_attack 5.8%, obscene 4.4%, threat 3.4%), which is why the score is a
per-example F1 rather than exact-set-match: exact match would be dominated by
whether the model gets `toxicity` right and would give the optimizer almost no
signal on the rare labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from mrd.config import REPO_ROOT

__all__ = ["LABELS", "DATA_DIR", "Example", "build_prompt", "target_text",
           "score_example", "load_splits", "splits_summary"]

DATA_DIR = REPO_ROOT / "data" / "civil_multilabel"

# Fixed order, and the order the model is asked to answer in. Taken from the
# bundle's manifest so prompt, target and scoring cannot drift apart.
LABELS = ("toxicity", "obscene", "threat", "insult", "identity_attack")

NONE_TOKEN = "NONE"


class Example(TypedDict):
    source: str
    prompt: str
    labels: list[str]        # ground-truth subset, possibly empty
    target: str              # the same subset rendered as the model should answer
    id: str


def build_prompt(text: str) -> str:
    """The user turn. Fixed across all arms and never optimized.

    GEPA mutates the *system* block instead, because routing drift is measured
    by aligning base and arm on their longest common token suffix: editing the
    user turn would split the sequences mid-way and confound drift with a
    change of input. (A neighbouring team optimizes the first line of this very
    prompt instead, which is fine for them -- they do not measure routing.)
    """
    return (
        f"Text:\n{text}\n\n"
        f"Labels (use exact names): {', '.join(LABELS)}\n"
        f"Return every applicable label in the order listed, separated by commas. "
        f"If no label applies, return exactly {NONE_TOKEN}. Return no other text.\n"
        f"Answer:"
    )


def target_text(labels: list[str]) -> str:
    """Ground truth as the model should emit it: canonical order, or NONE."""
    ordered = [l for l in LABELS if l in set(labels)]
    return ", ".join(ordered) if ordered else NONE_TOKEN


def parse_response(response: str) -> set[str] | None:
    """Return labels, or None for an invalid answer; only explicit NONE is empty."""
    text = response.strip().lower()
    if text.startswith("answer:"):
        text = text[len("answer:"):].strip()
    pieces = [p.strip(" \t\n\r.,;:!\"'`*-") for p in text.replace("\n", ",").split(",")]
    if pieces == [NONE_TOKEN.lower()]:
        return set()
    if not pieces or any(p not in LABELS for p in pieces):
        return None
    return set(pieces)


def score_example(example: Example, response: str) -> tuple[float, str]:
    """Per-example F1 between predicted and true label sets.

    Both empty scores 1.0: predicting "no labels apply" on a clean comment is a
    correct answer, and 33% of the data is exactly that by construction. One
    side empty and the other not scores 0.0, which falls out of the F1
    definition and needs no special case beyond the both-empty branch.

    F1 rather than exact-set-match because the label frequencies span 65.4%
    down to 3.4%: under exact match nearly all of the signal would come from
    `toxicity`, and a model that found `threat` but missed `insult` would be
    scored the same as one that produced nothing.
    """
    pred = parse_response(response)
    if pred is None:
        return 0.0, "invalid answer: use only the listed labels, or NONE alone"
    true = set(example["labels"])
    if not pred and not true:
        return 1.0, "both empty (correct)"
    tp = len(pred & true)
    if tp == 0:
        return 0.0, f"predicted={sorted(pred) or 'NONE'} true={sorted(true) or 'NONE'} (no overlap)"
    precision = tp / len(pred)
    recall = tp / len(true)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, (f"predicted={sorted(pred)} true={sorted(true)} "
                f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}")


def exact_match(example: Example, response: str) -> float:
    """Exact label-set match with the same format tolerance as F1."""
    return float(parse_response(response) == set(example["labels"]))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing split file: {path}")
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _to_example(row: dict) -> Example:
    labels = list(row.get("labels") or [])
    return {
        "source": "civil_comments_multilabel",
        "prompt": build_prompt(row["text"]),
        "labels": labels,
        "target": target_text(labels),
        "id": row.get("id", row.get("source_id", "")),
        # carried through for the segment report, same as the binary task
        "text": row["text"],
    }


def _resolve(explicit: str | Path | None, data_dir: Path, default: str) -> Path:
    if explicit is None:
        return data_dir / default
    path = Path(explicit)
    return path if len(path.parts) > 1 else data_dir / path


def load_splits(
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    seed: int = 42,
    data_dir: Path = DATA_DIR,
    train_file: str | Path | None = None,
    val_file: str | Path | None = None,
    test_file: str | Path | None = None,
    **_ignored: Any,
) -> dict[str, list[Example]]:
    """``{"train","val","test"}``. ``0`` means "whole pool", which is the normal
    call: these are frozen files, and truncation happens downstream.

    Defaults follow the bundle's own naming: ``optimizer_train_seed<seed>_n<N>``
    with the train size chosen by ``train_file``, and a single shared
    ``test.jsonl``. ``**_ignored`` accepts and ignores ``attribute``: there is
    no per-attribute split here, the five labels live in every row.
    """
    train_path = _resolve(train_file, data_dir, f"optimizer_train_seed{seed}_n1000.jsonl")
    val_path = _resolve(val_file, data_dir, f"optimizer_val_seed{seed}_n200.jsonl")
    test_path = _resolve(test_file, data_dir, "test.jsonl")

    def take(rows: list[dict], n: int) -> list[dict]:
        return rows if n <= 0 else rows[:n]

    return {
        "train": [_to_example(r) for r in take(_read_jsonl(train_path), n_train)],
        "val": [_to_example(r) for r in take(_read_jsonl(val_path), n_val)],
        "test": [_to_example(r) for r in take(_read_jsonl(test_path), n_test)],
    }


def splits_summary(splits: dict[str, list[Example]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, rows in splits.items():
        if not rows:
            continue
        per_label = {l: sum(1 for r in rows if l in r["labels"]) for l in LABELS}
        out[name] = {
            "n": len(rows),
            "empty": sum(1 for r in rows if not r["labels"]),
            "labels_per_row_mean": round(
                sum(len(r["labels"]) for r in rows) / len(rows), 3),
            "per_label": per_label,
        }
    return out
