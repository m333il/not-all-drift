"""Router-adapter interface: the seam that lets ``RoutingProbe`` work with any
MoE architecture, not just Ling's ``BailingMoeV2``.

Every architecture scores and selects experts differently (see ``ling.py`` vs
``qwen3_moe.py`` docstrings), but ``routing.py``/``drift.py`` only need five
things from an adapter, and none of them are architecture-specific once named:
which modules are MoE blocks, how to hook them for ``(logits, topk_idx)``, what
the gate actually ranks on (``selection_scores``), whether there is a coarse
group-of-groups selection on top of plain top-k (``selected_groups`` - ``None``
when there is none), and each expert's output-projection norm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint to probe: a repo id, the pretraining stage it represents,
    and an optional pinned revision.

    ``stage`` is metadata, not behavior *except* for one thing: ``routing.py``
    only applies the tokenizer's chat template when the checkpoint actually has
    one, which base checkpoints typically do not. Nothing else in the pipeline
    branches on it - it exists so results are labeled with what was measured.
    """

    repo_id: str
    stage: Literal["base", "sft", "instruct"] = "instruct"
    revision: str | None = None


class RouterAdapter(Protocol):
    """What ``RoutingProbe`` needs from a specific MoE architecture."""

    model_family: str
    num_experts: int
    top_k: int
    n_group: int | None
    topk_group: int | None

    @property
    def layer_ids(self) -> list[int]: ...

    def register_hooks(
        self, collected: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> list[torch.utils.hooks.RemovableHandle]: ...

    def selection_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """The score the gate actually ranks experts on - e.g. softmax
        probabilities, or ``sigmoid(logits) + expert_bias`` for aux-loss-free
        balancing. Shape-preserving: ``[..., num_experts] -> [..., num_experts]``.
        """
        ...

    def selected_groups(self, logits: torch.Tensor) -> torch.Tensor | None:
        """Boolean group-selection mask ``[..., n_group]``, or ``None`` if this
        architecture has no coarse group-of-groups routing (e.g. Qwen3-MoE).
        Callers must treat ``None`` as "this metric does not apply here", not
        as zero drift.
        """
        ...

    def expert_output_norms(self) -> torch.Tensor:
        """Frobenius norm of each expert's output projection, ``[n_moe_layers,
        num_experts]`` - the output-norm axis from *A Closer Look into MoE*
        (2406.18219), used by ``drift.py``'s ``norm_delta_on_flip``.
        """
        ...
