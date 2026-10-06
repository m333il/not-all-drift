"""Gate scores for named experts, position by position.

The paper's token-level claim is about routing, not about the profile: Super
Experts receive "exceptionally large" router scores on attention-sink tokens and
ordinary ones elsewhere. Checking that needs the gate's own output, which the
down-projection profiler never sees, so it is a separate probe that runs in the
same forward pass.

``Qwen3MoeTopKRouter.forward`` returns ``(router_logits, router_scores,
router_indices)`` where ``router_scores`` are the renormalised top-k weights.
The score reported here is the full softmax probability, which is what the gate
ranks on and what is comparable across positions and experts; selection is
reported separately because a large probability that still misses top-k routes
no tokens at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterObservation:
    layer: int
    expert: int
    example: int
    probability: list[float]
    """Softmax probability of this expert, one entry per token position."""
    selected: list[bool]
    """Whether the expert was inside top-k at that position."""
    rank: list[int]
    """The expert's rank among all experts at that position, 0 being the largest."""


class RouterScoreProbe:
    """Context manager recording gate scores for ``targets`` = {(layer, expert)}."""

    def __init__(self, model, targets: set[tuple[int, int]]) -> None:
        self.targets = set(targets)
        self.observations: list[RouterObservation] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._example: int | None = None
        self._gates = []
        for name, module in model.named_modules():
            if type(module).__name__.lower().endswith("topkrouter"):
                self._gates.append((int(name.split("layers.")[1].split(".")[0]), module))
        if not self._gates:
            raise RuntimeError("No top-k router modules found")
        layers = {layer for layer, _ in self._gates}
        experts = int(self._gates[0][1].num_experts)
        unknown = [pair for pair in self.targets if pair[0] not in layers or not 0 <= pair[1] < experts]
        if unknown:
            raise ValueError(
                f"Targets outside this model ({len(layers)} MoE layers, {experts} experts each): {sorted(unknown)}")

    def begin_example(self, index: int) -> None:
        self._example = index

    def __enter__(self) -> "RouterScoreProbe":
        layers = {layer for layer, _ in self.targets}
        for layer, gate in self._gates:
            if layer in layers:
                self._handles.append(gate.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer: int):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            if self._example is None:
                raise RuntimeError("Call begin_example(index) before each forward pass")
            logits, _scores, indices = output
            probabilities = torch.softmax(logits.float(), dim=-1)
            order = probabilities.argsort(dim=-1, descending=True).argsort(dim=-1)
            for target_layer, expert in sorted(self.targets):
                if target_layer != layer:
                    continue
                self.observations.append(RouterObservation(
                    layer=layer,
                    expert=expert,
                    example=self._example,
                    probability=probabilities[:, expert].tolist(),
                    selected=(indices == expert).any(dim=-1).tolist(),
                    rank=order[:, expert].tolist(),
                ))
        return hook
