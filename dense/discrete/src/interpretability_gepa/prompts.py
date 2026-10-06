from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .datasets import CIVIL_LABELS
from .errors import ParseError

# v2: the seed prompt lists five labels.
CIVIL_COMMENTS_PROMPT_CONTRACT_ID = "civil_comments_seed_v2"

SEED_INSTRUCTIONS = {
    "goemotions": (
        "You are a precise emotion classifier. Determine every category that applies to the "
        "text. A text may have multiple categories."
    ),
    "civil_comments": (
        "You are a text classifier. Classify the text using the provided labels. "
        "More than one label may apply."
    ),
    "hallmarks_of_cancer": (
        "You are a biomedical evidence classifier. Determine every Hallmark of Cancer category "
        "supported by the sentence. If none apply, return NONE."
    ),
}


@dataclass(frozen=True)
class RenderedPrompt:
    messages: tuple[dict[str, str], ...]
    text_span_marker: str = "INPUT_TEXT"


def format_labels(labels: Sequence[str], selected: Iterable[str]) -> str:
    """Render a label set exactly as ``parse_labels`` expects to read it back."""
    order = {label: index for index, label in enumerate(labels)}
    chosen = sorted(set(selected), key=lambda label: order[label])
    if not chosen:
        return "NONE"
    if tuple(labels) == CIVIL_LABELS:
        return ", ".join(chosen)
    return json.dumps(chosen)


def describe_output_contract(labels: Sequence[str]) -> str:
    """State the contract the harness appends and the parser enforces."""
    canonical = ", ".join(labels)
    shape = (
        "the applicable labels separated by commas"
        if tuple(labels) == CIVIL_LABELS
        else "a JSON array of the applicable category names"
    )
    return (
        f"The harness appends a fixed output contract after your instruction and parses the "
        f"answer strictly. A valid answer is {shape}, ordered by this exact schema order "
        f"({canonical}) - which is not alphabetical - or the single word NONE when nothing "
        f"applies. Any other shape is scored zero, including a differently ordered list, a "
        f"Python literal, surrounding prose, or an explanation. The contract is already in the "
        f"prompt, so do not restate, reformat, or contradict it; spend the instruction on "
        f"deciding which labels apply."
    )


def render_messages(instruction: str, labels: Sequence[str], text: str) -> RenderedPrompt:
    label_list = ", ".join(labels)
    if tuple(labels) == CIVIL_LABELS:
        user = (
            f"{instruction}\n"
            f"Labels (use exact names): {label_list}\n"
            f"Text:\n{text}\n"
            "Return every applicable label in the order listed, separated by commas. "
            "If no label applies, return exactly NONE. Return no other text.\n"
            "Answer:"
        )
        return RenderedPrompt(({"role": "user", "content": user},))
    user = (
        f"{instruction}\n\nAllowed categories (use exact names): {label_list}\n"
        "Return a JSON array of applicable category names in the listed order. "
        "If no category applies, return exactly NONE.\n\n"
        f"INPUT_TEXT:\n{text}\n\nCategories:"
    )
    return RenderedPrompt(({"role": "user", "content": user},))


def render_condition_messages(
    condition: str, instruction: str, labels: Sequence[str], text: str
) -> RenderedPrompt:
    """Render the shared harness while preserving the preregistered null floor."""
    if condition == "C_null":
        return RenderedPrompt(({"role": "user", "content": text},))
    return render_messages(instruction, labels, text)


def apply_chat_template(tokenizer: Any, rendered: RenderedPrompt, *, non_thinking: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if non_thinking:
        kwargs["enable_thinking"] = False
    return str(tokenizer.apply_chat_template(list(rendered.messages), **kwargs))


def parse_labels(
    response: str, allowed_labels: Sequence[str], *, enforce_order: bool = True
) -> tuple[str, ...]:
    """Parse a response into a canonical label tuple.

    ``enforce_order=True`` is the strict parser (labels must follow schema order);
    ``False`` is the set parser used for the order-lenient score.
    """
    text = response.strip()
    if "<think>" in text.lower():
        raise ParseError("thinking trace detected")
    if tuple(allowed_labels) == CIVIL_LABELS:
        if text == "NONE":
            return ()
        decoded = [item.strip() for item in text.split(",")]
        if not decoded or any(not item for item in decoded):
            raise ParseError("expected comma-separated labels or NONE")
        allowed = set(allowed_labels)
        unknown = set(decoded) - allowed
        if unknown:
            raise ParseError(f"unknown labels: {sorted(unknown)}")
        if len(set(decoded)) != len(decoded):
            raise ParseError("duplicate labels are not allowed")
        order = {label: index for index, label in enumerate(allowed_labels)}
        if enforce_order and [order[label] for label in decoded] != sorted(
            order[label] for label in decoded
        ):
            raise ParseError("labels are not in canonical order")
        return tuple(sorted(decoded, key=order.__getitem__))
    if text.upper() == "NONE":
        return ()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"expected JSON array or NONE: {exc.msg}") from exc
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ParseError("response must be a JSON array of strings")
    allowed = set(allowed_labels)
    unknown = set(decoded) - allowed
    if unknown:
        raise ParseError(f"unknown labels: {sorted(unknown)}")
    order = {label: index for index, label in enumerate(allowed_labels)}
    return tuple(sorted(set(decoded), key=order.__getitem__))


def find_text_token_mask(
    tokenizer: Any, prompt: str, text: str, *, add_special_tokens: bool = False
) -> list[bool]:
    start = prompt.find(text)
    if start < 0:
        raise ValueError("input text is not present verbatim in rendered prompt")
    encoded = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    end = start + len(text)
    return [
        offset_start < end and offset_end > start
        for offset_start, offset_end in encoded["offset_mapping"]
    ]


MULTILABEL_PADDING_FILLER = " Please follow the category list and required output format."


def neutral_length_padding(
    tokenizer: Any, seed: str, adapted: str, *, filler: str = MULTILABEL_PADDING_FILLER
) -> str:
    """Pad ``seed`` to the token length of ``adapted`` with content-free text."""
    target = len(tokenizer.encode(adapted, add_special_tokens=False))
    current = len(tokenizer.encode(seed, add_special_tokens=False))
    if current >= target:
        return seed
    padded = seed
    while len(tokenizer.encode(padded, add_special_tokens=False)) < target:
        padded += filler
    token_ids = tokenizer.encode(padded, add_special_tokens=False)[:target]
    return re.sub(r"\s+", " ", tokenizer.decode(token_ids)).strip()
