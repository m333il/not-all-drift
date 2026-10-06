"""Learning-rate profiles and checkpoint selection for PEFT scheduler sweeps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SCHEDULER_PROFILES = (
    "constant",
    "linear_decay",
    "linear_warmup_constant",
    "cosine_decay",
    "linear_warmup_cosine",
    "smooth_cosine",
)


def causal_token_loss_sum(logits: Any, labels: Any) -> tuple[float, int]:
    """Return summed next-token CE and the number of supervised target tokens."""
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    token_count = int((shift_labels != -100).sum().item())
    if token_count == 0:
        raise ValueError("Cannot compute causal loss without supervised target tokens")
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss_sum.detach().item()), token_count


def _cosine_decay(progress: float, min_lr_ratio: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def lr_multiplier(
    profile: str,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    """Return the LR multiplier at an optimizer step for a named profile."""
    if profile not in SCHEDULER_PROFILES:
        raise ValueError(f"Unknown scheduler profile: {profile}")
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("Require total_steps > 0 and 0 <= step <= total_steps")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if profile == "constant":
        return 1.0
    if profile == "linear_decay":
        return max(0.0, 1.0 - step / total_steps)
    if profile == "linear_warmup_constant":
        return min(step / max(1, warmup_steps), 1.0)
    if profile == "cosine_decay":
        return _cosine_decay(step / total_steps, min_lr_ratio)
    if step <= warmup_steps:
        progress = step / max(1, warmup_steps)
        if profile == "smooth_cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        return progress
    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return _cosine_decay(decay_progress, min_lr_ratio)


@dataclass
class CheckpointTracker:
    """Track exact validation argmax-F1 and argmin-loss with deterministic ties."""

    best_f1: float = float("-inf")
    loss_at_best_f1: float = float("inf")
    best_f1_epoch: int = 0
    best_loss: float = float("inf")
    f1_at_best_loss: float = float("-inf")
    best_loss_epoch: int = 0

    def observe(self, *, epoch: int, samples_f1: float, val_loss: float) -> tuple[bool, bool]:
        update_f1 = samples_f1 > self.best_f1 or (
            math.isclose(samples_f1, self.best_f1, abs_tol=1e-12)
            and val_loss < self.loss_at_best_f1
        )
        update_loss = val_loss < self.best_loss or (
            math.isclose(val_loss, self.best_loss, abs_tol=1e-12)
            and samples_f1 > self.f1_at_best_loss
        )
        if update_f1:
            self.best_f1 = samples_f1
            self.loss_at_best_f1 = val_loss
            self.best_f1_epoch = epoch
        if update_loss:
            self.best_loss = val_loss
            self.f1_at_best_loss = samples_f1
            self.best_loss_epoch = epoch
        return update_f1, update_loss
