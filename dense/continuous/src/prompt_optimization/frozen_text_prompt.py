"""Helpers for evaluating long, frozen text prompts without silent truncation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


def load_prompt_template(path: Path) -> str:
    """Load a prompt containing exactly one literal ``{text}`` placeholder."""
    template = path.read_text(encoding="utf-8")
    if template.count("{text}") != 1:
        raise ValueError(f"Expected exactly one {{text}} placeholder in {path}")
    return template


def prompt_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def render_prompt(tokenizer: Any, template: str, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": template.format(text=text)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_prompts_strict(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_input_length: int,
    padding_side: str,
    device: torch.device | str,
) -> tuple[Any, list[int]]:
    """Tokenize without truncation and fail if the declared context is exceeded."""
    if max_input_length <= 0:
        raise ValueError("max_input_length must be positive")
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(
            list(prompts),
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_padding_side
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    longest = max((int(length) for length in lengths), default=0)
    if longest > max_input_length:
        raise ValueError(
            f"Frozen prompt input has {longest} tokens, exceeding "
            f"max_input_length={max_input_length}; refusing silent truncation"
        )
    return encoded.to(device), [int(length) for length in lengths]
