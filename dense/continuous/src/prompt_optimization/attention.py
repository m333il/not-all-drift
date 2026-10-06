"""Attention-span helpers shared by extraction and context masking."""

from __future__ import annotations

import numpy as np

SEGMENTS = ("bos", "virtual", "instruction", "text")


def compute_prompt_spans(
    rendered_prompt: str,
    example_text: str,
    offsets: list[tuple[int, int]] | list[list[int]],
    virtual_tokens: int = 0,
) -> dict[str, list[list[int]]]:
    """Partition real prompt tokens and describe the virtual key interval."""
    start = rendered_prompt.find(example_text)
    if start < 0:
        raise ValueError("Example text was not found in the rendered prompt")
    end = start + len(example_text)
    text_tokens = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not text_tokens:
        raise ValueError("Example text maps to an empty token span")
    first, stop = text_tokens[0], text_tokens[-1] + 1
    if text_tokens != list(range(first, stop)):
        raise ValueError("Example text does not map to one contiguous token span")
    token_count = len(offsets)
    spans = {
        "bos": [[0, 1]],
        "virtual": [[0, virtual_tokens]] if virtual_tokens else [],
        "instruction": [
            interval
            for interval in ([1, first], [stop, token_count])
            if interval[1] > interval[0]
        ],
        "text": [[first, stop]],
    }
    covered = sorted(
        index
        for name in ("bos", "instruction", "text")
        for left, right in spans[name]
        for index in range(left, right)
    )
    if covered != list(range(token_count)):
        raise ValueError("Real-token spans do not partition the prompt")
    return spans


def key_segment_ids(
    spans: dict[str, list[list[int]]],
    *,
    virtual_tokens: int,
    left_padding: int,
    key_count: int,
) -> np.ndarray:
    """Map attention-key coordinates to segment IDs; -1 denotes padding or output."""
    result = np.full(key_count, -1, dtype=np.int8)
    for segment_id, name in enumerate(SEGMENTS):
        for left, right in spans[name]:
            if name == "virtual":
                result[left:right] = segment_id
            else:
                start = virtual_tokens + left_padding + left
                stop = virtual_tokens + left_padding + right
                result[start:stop] = segment_id
    return result


def aggregate_segment_mass(rows: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Sum attention mass for every named segment along the final key axis."""
    return np.stack(
        [rows[..., segment_ids == index].sum(-1) for index in range(len(SEGMENTS))],
        axis=-1,
    )


def unmasked_position_ids(attention_mask):
    """Positions generated from the baseline mask, reused during masking controls."""
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)
