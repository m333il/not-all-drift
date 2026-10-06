"""Unit tests for ``Qwen3MoeAdapter`` against a synthetic stand-in for
``Qwen3MoeTopKRouter`` - no real Qwen3 weights needed.

Two things this adapter does differently from ``LingAdapter`` and that these
tests exist specifically to pin down:
  * it hooks the **gate** module directly, not a parent block, because
    (per the real ``Qwen3MoeSparseMoeBlock.forward``) only the gate's own
    return value carries router logits;
  * ``expert_output_norms`` has two possible expert-storage layouts to
    support (stacked ``nn.Parameter`` vs. per-expert ``nn.ModuleList``) - both
    are exercised here since which one the installed ``transformers`` uses
    was not directly verified against real Qwen3-MoE source in this session
    (see the module docstring in ``mrd/models/qwen3_moe.py``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mrd.models.qwen3_moe import Qwen3MoeAdapter

HIDDEN = 8
MOE_INTER = 4
NUM_EXPERTS = 6
TOP_K = 2


class Qwen3MoeTopKRouter(nn.Module):
    """Name matters: ``Qwen3MoeAdapter`` discovers gates by ``type(m).__name__``."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN))

    def forward(self, hidden_states: torch.Tensor):
        # Real Qwen3Moe/OLMoE-style gates receive already-flattened
        # [batch*seq, hidden] input - with batch_size == 1 this is just
        # [seq, hidden], no separate leading batch dim.
        logits = hidden_states @ self.weight.T
        probs = torch.softmax(logits, dim=-1)
        top_val, top_idx = torch.topk(probs, self.top_k, dim=-1)
        top_val = top_val / top_val.sum(-1, keepdim=True)
        return logits, top_val, top_idx


class _StackedExperts(nn.Module):
    """The batched-parameter expert layout (cf. ``OlmoeExperts``)."""

    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, MOE_INTER))


class _ExpertModule(nn.Module):
    """One entry of the per-expert ``nn.ModuleList`` layout."""

    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(MOE_INTER, HIDDEN, bias=False)


def _fake_qwen_model(num_layers: int, expert_layout: str) -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    for _ in range(num_layers):
        layer = nn.Module()
        mlp = nn.Module()
        mlp.gate = Qwen3MoeTopKRouter()
        mlp.experts = (
            _StackedExperts() if expert_layout == "stacked"
            else nn.ModuleList([_ExpertModule() for _ in range(NUM_EXPERTS)])
        )
        layer.mlp = mlp
        model.model.layers.append(layer)
    return model


def test_discovers_gates_directly_not_the_parent_block():
    model = _fake_qwen_model(num_layers=3, expert_layout="stacked")
    adapter = Qwen3MoeAdapter(model)
    assert adapter.layer_ids == [0, 1, 2]
    assert adapter.num_experts == NUM_EXPERTS
    assert adapter.top_k == TOP_K
    assert adapter.n_group is None
    assert adapter.topk_group is None
    assert adapter.experts_per_group is None


def test_raises_when_no_gates_found():
    empty_model = nn.Module()
    empty_model.model = nn.Module()
    empty_model.model.layers = nn.ModuleList([nn.Module()])  # no .mlp.gate
    try:
        Qwen3MoeAdapter(empty_model)
    except RuntimeError as exc:
        assert "Qwen3MoeTopKRouter" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a model with no gates")


def test_selected_groups_is_always_none():
    model = _fake_qwen_model(num_layers=1, expert_layout="stacked")
    adapter = Qwen3MoeAdapter(model)
    fake_logits = torch.randn(1, 5, NUM_EXPERTS)
    assert adapter.selected_groups(fake_logits) is None


def test_selection_scores_is_plain_softmax():
    model = _fake_qwen_model(num_layers=1, expert_layout="stacked")
    adapter = Qwen3MoeAdapter(model)
    logits = torch.randn(1, 5, NUM_EXPERTS)
    expected = torch.softmax(logits, dim=-1)
    assert torch.allclose(adapter.selection_scores(logits), expected)


def test_register_hooks_captures_logits_and_indices_with_batch_1_no_leading_dim():
    model = _fake_qwen_model(num_layers=2, expert_layout="stacked")
    adapter = Qwen3MoeAdapter(model)
    collected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = adapter.register_hooks(collected)
    try:
        seq = 7
        hidden_flat = torch.randn(seq, HIDDEN)  # batch=1, already flattened
        for layer in model.model.layers:
            layer.mlp.gate(hidden_flat)
    finally:
        for h in handles:
            h.remove()

    assert set(collected) == {0, 1}
    logits, indices = collected[0]
    assert logits.shape == (seq, NUM_EXPERTS)
    assert indices.shape == (seq, TOP_K)
    assert bool((indices >= 0).all() and (indices < NUM_EXPERTS).all())


def test_expert_output_norms_stacked_parameter_layout():
    model = _fake_qwen_model(num_layers=2, expert_layout="stacked")
    adapter = Qwen3MoeAdapter(model)
    norms = adapter.expert_output_norms()
    assert norms.shape == (2, NUM_EXPERTS)
    assert bool((norms >= 0).all())


def test_expert_output_norms_module_list_layout():
    model = _fake_qwen_model(num_layers=2, expert_layout="module_list")
    adapter = Qwen3MoeAdapter(model)
    norms = adapter.expert_output_norms()
    assert norms.shape == (2, NUM_EXPERTS)
    assert bool((norms >= 0).all())
