from __future__ import annotations

from collections.abc import Sequence

from mrd.civil_multilabel_task import LABELS


PROMPT_CONTRACT_ID = "civil_comments_seed_v2"
# The prompt contract and the v2 splits are those of dense/discrete in this repository.
CONTRACT_SOURCE_REPOSITORY = "dense/discrete"
CONTRACT_SOURCE_COMMIT = "1e4409b526f89fc075dea5be507303015c6852f4"
SEED_INSTRUCTION = (
    "You are a text classifier. Classify the text using the provided labels. "
    "More than one label may apply."
)


def describe_output_contract() -> str:
    canonical = ", ".join(LABELS)
    return (
        "The harness appends a fixed output contract after your instruction and parses the "
        "answer strictly. A valid answer is the applicable labels separated by commas, "
        f"ordered by this exact schema order ({canonical}) - which is not alphabetical - "
        "or the single word NONE when nothing applies. Any other shape is scored zero, "
        "including a differently ordered list, a Python literal, surrounding prose, or an "
        "explanation. The contract is already in the prompt, so do not restate, reformat, "
        "or contradict it; spend the instruction on deciding which labels apply."
    )


def render_user_prompt(instruction: str, text: str) -> str:
    return (
        f"{instruction}\n"
        f"Labels (use exact names): {', '.join(LABELS)}\n"
        f"Text:\n{text}\n"
        "Return every applicable label in the order listed, separated by commas. "
        "If no label applies, return exactly NONE. Return no other text.\n"
        "Answer:"
    )


def format_labels(labels: Sequence[str]) -> str:
    selected = set(labels)
    return ", ".join(label for label in LABELS if label in selected) or "NONE"


def parse_labels(response: str, *, enforce_order: bool = True) -> tuple[str, ...]:
    text = response.strip()
    if "<think>" in text.lower():
        raise ValueError("thinking trace detected")
    if text == "NONE":
        return ()
    decoded = [item.strip() for item in text.split(",")]
    if not decoded or any(not item for item in decoded):
        raise ValueError("expected comma-separated labels or NONE")
    unknown = set(decoded) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    if len(set(decoded)) != len(decoded):
        raise ValueError("duplicate labels are not allowed")
    order = {label: index for index, label in enumerate(LABELS)}
    if enforce_order and [order[label] for label in decoded] != sorted(order[label] for label in decoded):
        raise ValueError("labels are not in canonical order")
    return tuple(sorted(decoded, key=order.__getitem__))


def label_f1(gold: Sequence[str], predicted: Sequence[str]) -> float:
    truth, prediction = set(gold), set(predicted)
    if not truth and not prediction:
        return 1.0
    overlap = len(truth & prediction)
    return 2 * overlap / (len(truth) + len(prediction)) if overlap else 0.0


def score_response(example: dict, response: str) -> tuple[float, tuple[str, ...], str | None]:
    try:
        predicted = parse_labels(response)
    except ValueError as error:
        return 0.0, (), str(error)
    return label_f1(example["labels"], predicted), predicted, None


def reflection_feedback(example: dict, predicted: Sequence[str], error: str | None) -> str:
    gold = set(example["labels"])
    prediction = set(predicted)
    return (
        f"Text: {example['text']}\n"
        f"Expected: {format_labels(gold)}\n"
        f"Predicted: {format_labels(prediction)}\n"
        f"Missed: {format_labels(gold - prediction)}\n"
        f"Extra: {format_labels(prediction - gold)}\n"
        f"Parse error: {error}"
    )
