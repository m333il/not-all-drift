"""Calibration pass: count how often each expert is selected, per layer.

Counts are taken on prompt positions only (one prefill per batch, no
generation), which is what a pruning decision should be based on: the answer is
six tokens long and its routing is dominated by the label vocabulary, so
including it would let a handful of positions vote on which experts survive.

The counter re-derives the top-k from the gate's logits rather than reading the
block's internals, so it stays valid for any block that takes a plain top-k of
its router logits - and it is the same derivation the mask verifies against.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .frequency import ExpertCounts
from .masking import GateRef, discover_gates

logger = logging.getLogger(__name__)


@dataclass
class _Accumulator:
    gates: Sequence[GateRef]
    top_k: int
    counts: dict[int, np.ndarray] = field(default_factory=dict)
    n_positions: int = 0

    def add(self, layer_idx: int, logits: torch.Tensor, keep: torch.Tensor | None) -> None:
        rows = logits if keep is None else logits[keep]
        if rows.numel() == 0:
            return
        chosen = rows.float().topk(self.top_k, dim=-1).indices.reshape(-1)
        hist = torch.bincount(chosen.cpu(), minlength=logits.shape[-1]).numpy()
        if layer_idx not in self.counts:
            self.counts[layer_idx] = np.zeros(logits.shape[-1], dtype=np.float64)
        self.counts[layer_idx] += hist


class ExpertCounter:
    """Context manager accumulating expert selection counts across forwards.

    ``keep_mask`` is set per batch by the caller to exclude padding positions.
    Forgetting it inflates the counts of whatever experts the pad token likes,
    and in a left-padded batch that is a large, silent bias.
    """

    def __init__(self, model: nn.Module, *, top_k: int) -> None:
        self.gates = discover_gates(model)
        self.top_k = int(top_k)
        self._acc = _Accumulator(self.gates, self.top_k)
        self._keep: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def set_keep_mask(self, keep: torch.Tensor | None) -> None:
        """Flat boolean mask over ``[B*seq]`` rows: True for real tokens."""
        self._keep = None if keep is None else keep.reshape(-1)

    def _make_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
            keep = self._keep
            if keep is not None and keep.shape[0] != output.shape[0]:
                raise RuntimeError(
                    f"keep mask has {keep.shape[0]} rows, gate output has "
                    f"{output.shape[0]}; the mask does not describe this batch"
                )
            self._acc.add(layer_idx, output.detach(), None if keep is None else keep.to(output.device))
            return output

        return hook

    def __enter__(self) -> "ExpertCounter":
        self._handles = [
            g.module.register_forward_hook(self._make_hook(g.layer_idx)) for g in self.gates
        ]
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def result(self, *, source: str, stage: str, n_examples: int) -> ExpertCounts:
        if not self._acc.counts:
            raise RuntimeError("calibration collected no counts - hooks never fired")
        layer_ids = tuple(sorted(self._acc.counts))
        matrix = np.stack([self._acc.counts[i] for i in layer_ids])
        empty = [int(i) for i in layer_ids if matrix[layer_ids.index(i)].sum() == 0]
        if empty:
            raise RuntimeError(f"layers with zero counts after calibration: {empty}")
        return ExpertCounts(
            counts=matrix,
            layer_ids=layer_ids,
            source=source,
            stage=stage,
            n_examples=n_examples,
        )


@torch.no_grad()
def calibrate(
    model: nn.Module,
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    top_k: int,
    source: str,
    n_examples: int,
    stage: str = "prompt",
) -> ExpertCounts:
    """Run prefill over ``batches`` and return per-layer expert counts.

    Each batch is a dict with ``input_ids`` and ``attention_mask`` already on
    the model's device. Nothing is generated: one forward per batch is enough,
    and generation would make the counts depend on the arm's own output length.
    """
    model.eval()
    with ExpertCounter(model, top_k=top_k) as counter:
        for i, batch in enumerate(batches, start=1):
            counter.set_keep_mask(batch["attention_mask"].bool())
            model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if i == 1 or i % 10 == 0:
                logger.info("calibration: %d/%d batches", i, len(batches))
        counter.set_keep_mask(None)
        return counter.result(source=source, stage=stage, n_examples=n_examples)
