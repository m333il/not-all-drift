"""Utilities for aligning hidden states with teacher-forced target tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TeacherForcingExample:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    prediction_positions: tuple[int, ...]


def build_teacher_forcing_example(
    prompt_ids: list[int], target_ids: list[int], max_length: int
) -> TeacherForcingExample:
    """Build inputs whose selected states predict every target token."""
    if not prompt_ids or not target_ids:
        raise ValueError("prompt and target must both contain at least one token")
    if len(target_ids) >= max_length:
        raise ValueError("max_length is too small to hold the target")
    prompt = prompt_ids[-(max_length - len(target_ids)) :]
    inputs = prompt + target_ids[:-1]
    start = len(prompt) - 1
    positions = list(range(start, start + len(target_ids)))
    if positions[-1] >= len(inputs):
        raise RuntimeError("teacher-forcing positions exceed the input sequence")
    return TeacherForcingExample(tuple(inputs), tuple(target_ids), tuple(positions))


def left_pad_teacher_forcing(
    examples: list[TeacherForcingExample], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    if not examples:
        raise ValueError("cannot collate an empty teacher-forcing batch")
    width = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    positions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for example in examples:
        padding = width - len(example.input_ids)
        input_rows.append([pad_token_id] * padding + list(example.input_ids))
        mask_rows.append([0] * padding + [1] * len(example.input_ids))
        positions.append(
            torch.tensor(example.prediction_positions, dtype=torch.long) + padding
        )
        targets.append(torch.tensor(example.target_ids, dtype=torch.long))
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
        positions,
        targets,
    )
