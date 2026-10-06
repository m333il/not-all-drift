"""Fixed Civil Comments experiment contract and multilabel helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CIVIL_COMMENTS_LABELS = (
    "toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)
BINARY_LABELS = ("safe", "toxic")
BINARY_OUTPUT_FORMATS = ("safe-toxic", "yes-no")
SETUPS = ("multilabel", "binary")

CIVIL_COMMENTS_PROMPT_PREFIX = """You are a text classifier. Classify the text using the provided labels. More than one label may apply.
Labels (use exact names): {label_names}
Text:
{{text}}
Return every applicable label in the order listed, separated by commas. If no label applies, return exactly NONE. Return no other text.
Answer:"""

CIVIL_COMMENTS_PROMPT_TEMPLATE = CIVIL_COMMENTS_PROMPT_PREFIX.format(
    label_names=", ".join(CIVIL_COMMENTS_LABELS)
)
CIVIL_COMMENTS_PROMPT_INIT_TEXT = CIVIL_COMMENTS_PROMPT_TEMPLATE.format(text="")

BINARY_PROMPT_TEMPLATE = """You are a binary text classifier. Classify the text as safe or toxic.
Labels (use exact names): safe, toxic
Text:
{text}
Return exactly one label: safe or toxic. Return no other text.
Answer:"""

BINARY_YES_NO_PROMPT_TEMPLATE = """You are a text classifier. Decide whether the given attribute applies to the text.
Attribute: toxicity
Text:
{text}
Answer exactly Yes if the attribute applies to the text, or exactly No if it does not. Return no other text.
Answer:"""


def validate_setup(setup: str) -> str:
    if setup not in SETUPS:
        raise ValueError(f"Unsupported Civil Comments setup: {setup}")
    return setup


def validate_binary_output_format(output_format: str) -> str:
    if output_format not in BINARY_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported binary output format: {output_format}")
    return output_format


def validate_label_contract(labels: Sequence[str]) -> tuple[str, ...]:
    """Validate a manifest label list against the supported canonical taxonomy."""
    supplied = tuple(labels)
    if not supplied:
        raise ValueError("Civil Comments label contract must not be empty")
    if len(set(supplied)) != len(supplied):
        raise ValueError("Civil Comments label contract must not contain duplicates")
    unknown = sorted(set(supplied) - set(CIVIL_COMMENTS_LABELS))
    if unknown:
        raise ValueError(f"Unknown Civil Comments labels: {unknown}")
    canonical = tuple(label for label in CIVIL_COMMENTS_LABELS if label in supplied)
    if canonical != supplied:
        raise ValueError("Civil Comments manifest labels must use canonical order")
    return supplied


def prompt_template(
    labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
    *,
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> str:
    """Render the fixed classifier prompt for an explicit manifest label set."""
    if validate_setup(setup) == "binary":
        output_format = validate_binary_output_format(binary_output_format)
        return (
            BINARY_YES_NO_PROMPT_TEMPLATE
            if output_format == "yes-no"
            else BINARY_PROMPT_TEMPLATE
        )
    canonical = validate_label_contract(labels)
    return CIVIL_COMMENTS_PROMPT_PREFIX.format(label_names=", ".join(canonical))


def prompt_init_text(
    labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
    *,
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> str:
    return prompt_template(
        labels,
        setup=setup,
        binary_output_format=binary_output_format,
    ).format(text="")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash relative names and contents of every regular file in a directory."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {path}")
    for file_path in files:
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def prompt_sha256(
    labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
    *,
    setup: str = "multilabel",
    binary_output_format: str = "safe-toxic",
) -> str:
    return hashlib.sha256(
        prompt_template(
            labels,
            setup=setup,
            binary_output_format=binary_output_format,
        ).encode("utf-8")
    ).hexdigest()


def canonicalize_labels(
    labels: Iterable[str],
    *,
    allowed_labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> tuple[str, ...]:
    """Validate and return labels in the prompt's fixed canonical order."""
    supplied = tuple(labels)
    canonical_labels = validate_label_contract(allowed_labels)
    unknown = sorted(set(supplied) - set(canonical_labels))
    if unknown:
        raise ValueError(f"Unknown Civil Comments labels: {unknown}")
    if len(set(supplied)) != len(supplied):
        raise ValueError("Civil Comments labels must not contain duplicates")
    selected = set(supplied)
    return tuple(label for label in canonical_labels if label in selected)


def canonical_target(
    labels: Iterable[str],
    *,
    allowed_labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> str:
    normalized = canonicalize_labels(labels, allowed_labels=allowed_labels)
    return ", ".join(normalized) if normalized else "NONE"


def parse_prediction(
    generated_text: str,
    *,
    allowed_labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> tuple[str, ...] | None:
    """Parse a prediction as an order-invariant label set.

    The original training verifier compares sets, so the returned tuple is put
    back into canonical label order. Extra text and unknown labels remain
    invalid. Use :func:`parse_prediction_strict` to evaluate answer
    construction as well as label selection.
    """
    stripped = generated_text.strip()
    if stripped == "NONE":
        return ()
    if not stripped:
        return None
    parts = tuple(part.strip() for part in stripped.split(","))
    if any(not part for part in parts):
        return None
    try:
        canonical = canonicalize_labels(set(parts), allowed_labels=allowed_labels)
    except ValueError:
        return None
    return canonical


def parse_prediction_strict(
    generated_text: str,
    *,
    allowed_labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> tuple[str, ...] | None:
    """Parse the fixed output contract while enforcing canonical label order."""
    stripped = generated_text.strip()
    if stripped == "NONE":
        return ()
    if not stripped:
        return None
    parts = tuple(part.strip() for part in stripped.split(","))
    if any(not part for part in parts) or len(parts) != len(set(parts)):
        return None
    try:
        canonical = canonicalize_labels(parts, allowed_labels=allowed_labels)
    except ValueError:
        return None
    return canonical if parts == canonical else None


def binary_target(example: dict[str, Any]) -> str:
    target = example.get("binary_label")
    if target not in BINARY_LABELS:
        raise ValueError(f"Invalid binary_label: {target!r}")
    return str(target)


def binary_target_text(
    example: dict[str, Any],
    *,
    output_format: str = "safe-toxic",
) -> str:
    target = binary_target(example)
    if validate_binary_output_format(output_format) == "yes-no":
        return "Yes" if target == "toxic" else "No"
    return target


def parse_binary_prediction(
    generated_text: str,
    *,
    output_format: str = "safe-toxic",
) -> str | None:
    stripped = generated_text.strip()
    if validate_binary_output_format(output_format) == "yes-no":
        return {"Yes": "toxic", "No": "safe"}.get(stripped)
    return stripped if stripped in BINARY_LABELS else None


def compute_binary_metrics(
    targets: Sequence[str],
    predictions: Sequence[str | None],
) -> dict[str, float]:
    """Score the strict safe/toxic textual contract."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    if any(target not in BINARY_LABELS for target in targets):
        raise ValueError("Binary targets must be safe or toxic")
    scored = [prediction if prediction is not None else "safe" for prediction in predictions]
    metrics = {
        "accuracy": float(accuracy_score(targets, scored)),
        "macro_f1": float(f1_score(targets, scored, labels=BINARY_LABELS, average="macro")),
        "micro_f1": float(f1_score(targets, scored, labels=BINARY_LABELS, average="micro")),
        "samples_f1": float(f1_score(targets, scored, pos_label="toxic")),
        "precision_toxic": float(
            precision_score(targets, scored, pos_label="toxic", zero_division=0)
        ),
        "recall_toxic": float(
            recall_score(targets, scored, pos_label="toxic", zero_division=0)
        ),
        "f1_safe": float(f1_score(targets, scored, pos_label="safe", zero_division=0)),
        "f1_toxic": float(f1_score(targets, scored, pos_label="toxic", zero_division=0)),
        "invalid_rate": float(
            sum(prediction is None for prediction in predictions) / max(len(predictions), 1)
        ),
    }
    return metrics


def labels_to_matrix(
    rows: Sequence[Sequence[str]],
    *,
    labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int64)
    for row_index, row in enumerate(rows):
        for label in row:
            try:
                matrix[row_index, label_to_index[label]] = 1
            except KeyError as error:
                raise ValueError(f"Unknown label in target rows: {label}") from error
    return matrix


def compute_multilabel_metrics(
    targets: Sequence[Sequence[str]],
    predictions: Sequence[tuple[str, ...] | None],
    *,
    labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
) -> dict[str, float]:
    """Score strict textual predictions as a multilabel classification task."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length")
    canonical_labels = validate_label_contract(labels)
    target_matrix = labels_to_matrix(targets, labels=canonical_labels)
    predicted_rows = [prediction if prediction is not None else () for prediction in predictions]
    prediction_matrix = labels_to_matrix(predicted_rows, labels=canonical_labels)
    per_label = f1_score(
        target_matrix,
        prediction_matrix,
        average=None,
        zero_division=0,
    )
    target_empty = target_matrix.sum(axis=1) == 0
    predicted_empty = prediction_matrix.sum(axis=1) == 0
    metrics = {
        "macro_f1": float(
            f1_score(target_matrix, prediction_matrix, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(target_matrix, prediction_matrix, average="micro", zero_division=0)
        ),
        "samples_f1": float(
            f1_score(target_matrix, prediction_matrix, average="samples", zero_division=1)
        ),
        "subset_accuracy": float(accuracy_score(target_matrix, prediction_matrix)),
        "none_f1": float(f1_score(target_empty, predicted_empty, zero_division=0)),
        "invalid_rate": float(
            sum(prediction is None for prediction in predictions) / max(len(predictions), 1)
        ),
        "mean_target_cardinality": float(target_matrix.sum(axis=1).mean()),
        "mean_prediction_cardinality": float(prediction_matrix.sum(axis=1).mean()),
    }
    metrics.update(
        {
            f"f1_{label}": float(score)
            for label, score in zip(canonical_labels, per_label, strict=True)
        }
    )
    return metrics


def split_file_name(
    kind: str, *, split_seed: int | None = None, train_samples: int | None = None
) -> str:
    if kind == "optimizer_train":
        if split_seed is None or train_samples is None:
            raise ValueError("optimizer_train requires split_seed and train_samples")
        return f"optimizer_train_seed{split_seed}_n{train_samples}.jsonl"
    if kind == "optimizer_val":
        if split_seed is None:
            raise ValueError("optimizer_val requires split_seed")
        if train_samples is not None:
            return f"optimizer_val_seed{split_seed}_n{train_samples}.jsonl"
        return f"optimizer_val_seed{split_seed}.jsonl"
    if kind in {"test", "probe_train", "probe_val", "intervention_val"}:
        return f"{kind}.jsonl"
    if kind == "probe_train_subset":
        if split_seed is None:
            raise ValueError("probe_train_subset requires split_seed")
        return f"probe_train_seed{split_seed}.jsonl"
    raise ValueError(f"Unknown Civil Comments split kind: {kind}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_size: int | None = None,
    allowed_labels: Sequence[str] = CIVIL_COMMENTS_LABELS,
    setup: str = "multilabel",
) -> None:
    setup = validate_setup(setup)
    required = {"id", "text", "labels"}
    if expected_size is not None and len(rows) != expected_size:
        raise ValueError(f"Expected {expected_size} rows, found {len(rows)}")
    ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Row {index} misses required fields: {sorted(missing)}")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError(f"Row {index} has invalid id")
        if row["id"] in ids:
            raise ValueError(f"Duplicate id: {row['id']}")
        ids.add(row["id"])
        if not isinstance(row["text"], str):
            raise TypeError(f"Row {index} has non-string text")
        if not isinstance(row["labels"], list):
            raise TypeError(f"Row {index} has non-list labels")
        canonicalize_labels(row["labels"], allowed_labels=allowed_labels)
        if setup == "binary":
            target = binary_target(row)
            expected = "toxic" if row["labels"] else "safe"
            if target != expected:
                raise ValueError(
                    f"Row {index} binary_label={target!r} conflicts with labels"
                )


def load_manifest(split_root: Path) -> dict[str, Any]:
    path = split_root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_label_contract(payload.get("labels", ()))
    setup = payload.get("setup", payload.get("contract", {}).get("setup", "multilabel"))
    validate_setup(str(setup))
    if payload.get("contract", {}).get("setup", setup) != setup:
        raise ValueError("Top-level and contract setup values differ")
    if setup == "binary":
        validate_binary_output_format(
            str(payload.get("contract", {}).get("binary_output_format", "safe-toxic"))
        )
    if payload.get("contract", {}).get("threshold") != 0.5:
        raise ValueError("manifest threshold must be 0.5")
    return payload


def optimizer_validation_samples(manifest: dict[str, Any], train_samples: int) -> int | None:
    """Choose the largest fixed validation ladder entry not exceeding train N."""
    ladder = manifest.get("contract", {}).get("optimizer_val_ladder")
    if ladder is None:
        return None
    eligible = [int(value) for value in ladder if int(value) <= train_samples]
    if not eligible:
        raise ValueError(
            f"No optimizer validation split is defined for train_samples={train_samples}"
        )
    return max(eligible)


def prepended_virtual_tokens(peft_type: str | None, num_virtual_tokens: int) -> int:
    """Number of virtual positions present in returned hidden-state tensors."""
    if num_virtual_tokens < 0:
        raise ValueError("num_virtual_tokens must be non-negative")
    return num_virtual_tokens if peft_type == "PROMPT_TUNING" else 0


def use_gradient_checkpointing(method: str) -> bool:
    """PEFT Prefix Tuning is incompatible with gradient checkpointing."""
    if method not in {"prompt_tuning", "prefix_tuning"}:
        raise ValueError(f"Unsupported PEFT method: {method}")
    return method == "prompt_tuning"
