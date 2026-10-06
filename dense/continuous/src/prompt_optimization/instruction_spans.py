"""Semantic token spans for instruction-source and fixed-carrier experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch


FIXED_LABELS_PREFIX = "Labels (use exact names):"
FIXED_TEXT_MARKER = "Text:\n"
FIXED_OUTPUT_RULES_PREFIX = "Return every applicable label"
FIXED_ANSWER_MARKER = "Answer:"


@dataclass(frozen=True)
class CarrierTaskSpec:
    common_prefix: str
    text_marker: str
    output_rules_prefix: str


CARRIER_TASK_SPECS = {
    "civil_multilabel": CarrierTaskSpec(
        common_prefix=FIXED_LABELS_PREFIX,
        text_marker=FIXED_TEXT_MARKER,
        output_rules_prefix=FIXED_OUTPUT_RULES_PREFIX,
    ),
    "civil_binary_yesno": CarrierTaskSpec(
        common_prefix="Attribute: toxicity\n",
        text_marker=FIXED_TEXT_MARKER,
        output_rules_prefix="Answer exactly Yes",
    ),
    "amazon_rating": CarrierTaskSpec(
        common_prefix="Review:\n",
        text_marker="Review:\n",
        output_rules_prefix="Answer with exactly one digit",
    ),
}


def normalized_content_start(rendered_prompt: str, content: str) -> int:
    """Locate chat content while allowing template trimming at its right edge."""
    candidates = (content, content.rstrip())
    for candidate in candidates:
        if not candidate:
            continue
        start = rendered_prompt.find(candidate)
        if start < 0:
            continue
        if rendered_prompt.find(candidate, start + 1) >= 0:
            raise ValueError("Prompt content occurs more than once in rendered chat")
        return start
    raise ValueError("Rendered chat prompt does not contain normalized prompt content")


def character_span(content: str, literal: str, *, start: int = 0) -> tuple[int, int]:
    """Return the unique occurrence of ``literal`` at or after ``start``."""
    left = content.find(literal, start)
    if left < 0:
        raise ValueError(f"Prompt content does not contain {literal!r}")
    if content.find(literal, left + 1) >= 0:
        raise ValueError(f"Prompt content contains {literal!r} more than once")
    return left, left + len(literal)


def task_carrier_character_spans(
    content: str,
    sample_text: str,
    *,
    task: str,
) -> dict[str, list[tuple[int, int]]]:
    """Locate shared real-token carrier groups for one task contract.

    The method-editable opening is deliberately excluded. ``all_fixed`` also
    excludes the sample text, while ``all_common_real`` includes the complete
    immutable task suffix and the concrete sample.
    """
    try:
        spec = CARRIER_TASK_SPECS[task]
    except KeyError as error:
        raise ValueError(f"Unknown carrier task: {task}") from error
    common_left, _ = character_span(content, spec.common_prefix)
    text_left, text_right = character_span(
        content, spec.text_marker, start=common_left
    )
    if text_left < common_left:
        raise ValueError("Text marker precedes the common carrier prefix")

    sample_left = text_right
    sample_right = sample_left + len(sample_text)
    if content[sample_left:sample_right] != sample_text:
        raise ValueError("Text marker is not followed by the supplied sample text")

    rules_left, _ = character_span(
        content, spec.output_rules_prefix, start=sample_right
    )
    answer_left, answer_right = character_span(content, FIXED_ANSWER_MARKER, start=rules_left)
    if rules_left >= answer_left:
        raise ValueError("Output rules must precede the Answer marker")
    rules_only = (rules_left, answer_left)
    rules = (rules_left, answer_right)
    answer = (answer_left, answer_right)
    groups = {
        "common_prefix": [(common_left, text_left)] if common_left < text_left else [],
        "text_marker": [(text_left, text_right)],
        "text_only": [(sample_left, sample_right)],
        "output_rules_only": [rules_only],
        "output_rules": [rules],
        "answer": [answer],
    }
    if common_left < text_left:
        groups["labels"] = [(common_left, text_left)]
    groups["all_fixed"] = [(common_left, text_right), rules]
    groups["all_common_real"] = [(common_left, answer_right)]
    return groups


def fixed_carrier_character_spans(
    content: str,
    sample_text: str,
) -> dict[str, list[tuple[int, int]]]:
    """Backward-compatible Civil Comments multilabel carrier spans."""
    return task_carrier_character_spans(
        content,
        sample_text,
        task="civil_multilabel",
    )


def token_positions_for_spans(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    content_start: int,
    spans: Iterable[tuple[int, int]],
    hidden_offset: int = 0,
) -> list[int]:
    """Map content-relative character spans to overlapping tokenizer positions."""
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError("offsets must have shape [sequence, 2]")
    if attention_mask.ndim != 1 or len(attention_mask) != len(offsets):
        raise ValueError("attention_mask must have shape [sequence]")
    if hidden_offset < 0:
        raise ValueError("hidden_offset must be non-negative")
    absolute = []
    for left, right in spans:
        if left < 0 or right <= left:
            raise ValueError("Every character span must be non-empty")
        absolute.append((content_start + int(left), content_start + int(right)))
    positions: list[int] = []
    for index, ((left, right), valid) in enumerate(
        zip(offsets.tolist(), attention_mask.tolist(), strict=True)
    ):
        left, right = int(left), int(right)
        if not bool(valid) or right <= left:
            continue
        if any(left < span_right and right > span_left for span_left, span_right in absolute):
            positions.append(index + hidden_offset)
    positions = sorted(set(positions))
    if not positions:
        raise ValueError("No tokenizer tokens overlap the requested character spans")
    return positions


def token_positions_by_group(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    content_start: int,
    groups: Mapping[str, Iterable[tuple[int, int]]],
    hidden_offset: int = 0,
) -> dict[str, list[int]]:
    """Map every non-empty semantic group to tokenizer positions.

    Some task contracts intentionally have no tokens between the common-prefix
    boundary and the text marker.  Such empty groups are absent from the
    returned mapping; requesting one later still fails explicitly in the
    carrier resolver instead of aborting unrelated carriers during encoding.
    """
    result: dict[str, list[int]] = {}
    for name, raw_spans in groups.items():
        spans = list(raw_spans)
        if not spans:
            continue
        result[str(name)] = token_positions_for_spans(
            offsets,
            attention_mask,
            content_start=content_start,
            spans=spans,
            hidden_offset=hidden_offset,
        )
    return result


def virtual_token_positions(num_virtual_tokens: int) -> list[int]:
    """Return the residual-stream positions occupied by Prompt Tuning tokens."""
    if num_virtual_tokens <= 0:
        raise ValueError("num_virtual_tokens must be positive")
    return list(range(num_virtual_tokens))
