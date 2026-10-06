"""Tuned lens: per-layer affine translators into the final-layer basis.

Each translator is fitted by minimising KL to the model's own final distribution.
The base model stays frozen and one set of translators serves all conditions.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ArtifactError
from .modeling import readout_logits


@dataclass(frozen=True)
class TunedLensConfig:
    """Training geometry for one model's translators."""

    hidden_size: int
    layers: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    positions_per_sequence: int = 64
    max_steps: int = 500
    batch_size: int = 4
    warmup_steps: int = 20
    seed: int = 0

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.layers <= 0:
            raise ArtifactError("tuned lens needs a positive hidden size and layer count")
        if self.positions_per_sequence <= 0:
            raise ArtifactError("positions_per_sequence must be positive")
        if self.max_steps <= 0:
            raise ArtifactError("max_steps must be positive")


def _residual_translator(hidden_size: int) -> Any:
    """``x -> x + W x + b`` with ``W, b`` at zero, so weight decay pulls toward identity."""
    from torch import nn

    class ResidualTranslator(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.delta = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

        def forward(self, states: Any) -> Any:
            return states + self.delta(states)

    return ResidualTranslator()


def build_translators(config: TunedLensConfig, *, device: Any, dtype: Any) -> Any:
    """One affine map per layer, initialised to the identity (= logit lens)."""
    from torch import nn

    config.validate()
    translators = nn.ModuleList(
        [_residual_translator(config.hidden_size) for _ in range(config.layers)]
    )
    return translators.to(device=device, dtype=dtype)


def sample_positions(attention_mask: Any, count: int, generator: Any = None) -> Any:
    """Sample scoring positions per sequence, excluding padding and the final token."""
    import torch

    batch = attention_mask.shape[0]
    picked = torch.zeros((batch, count), dtype=torch.long, device=attention_mask.device)
    for row in range(batch):
        valid = torch.nonzero(attention_mask[row], as_tuple=False).flatten()
        valid = valid[:-1] if len(valid) > 1 else valid
        if len(valid) == 0:
            raise ArtifactError("a sequence has no scoreable position")
        if len(valid) >= count:
            choice = torch.randperm(len(valid), generator=generator, device=valid.device)[:count]
        else:
            choice = torch.randint(len(valid), (count,), generator=generator, device=valid.device)
        picked[row] = valid[choice]
    return picked


def translator_kl(
    *,
    model: Any,
    translator: Any,
    final_norm: Any,
    hidden: Any,
    target_log_probs: Any,
    positions: Any,
) -> Any:
    """KL(final distribution || this layer's translated distribution) at ``positions``."""
    import torch

    gathered = torch.gather(hidden, 1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
    # Translators are float32, the frozen readout keeps the model dtype.
    readout_dtype = model.lm_head.weight.dtype
    translated = translator(gathered).to(readout_dtype)
    states = final_norm(translated) if final_norm is not None else translated
    logits = readout_logits(model, states).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    # Mean KL per scored position.
    vocab = log_probs.shape[-1]
    return torch.nn.functional.kl_div(
        log_probs.reshape(-1, vocab),
        target_log_probs.reshape(-1, vocab),
        log_target=True,
        reduction="batchmean",
    )


def save_translators(path: Path, translators: Any, config: TunedLensConfig) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.to(torch.float32).cpu()
                for key, value in translators.state_dict().items()
            },
            "config": {
                "hidden_size": config.hidden_size,
                "layers": config.layers,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "positions_per_sequence": config.positions_per_sequence,
                "max_steps": config.max_steps,
                "batch_size": config.batch_size,
                "warmup_steps": config.warmup_steps,
                "seed": config.seed,
            },
        },
        path,
    )


def load_translators(path: Path, *, device: Any, dtype: Any) -> tuple[Any, TunedLensConfig]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TunedLensConfig(**payload["config"])
    translators = build_translators(config, device=device, dtype=dtype)
    translators.load_state_dict({k: v.to(dtype) for k, v in payload["state_dict"].items()})
    return translators, config


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def layer_kl_curve(losses: Sequence[float]) -> np.ndarray:
    """Per-layer KL after training, the standard readout of translator quality."""
    return np.asarray(losses, dtype=np.float32)


__all__ = [
    "TunedLensConfig",
    "batched",
    "build_translators",
    "layer_kl_curve",
    "load_translators",
    "sample_positions",
    "save_translators",
    "translator_kl",
]
