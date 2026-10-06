"""The batched MoE must produce what the loop produced, or it is not a speedup.

A faster kernel that moves the output is not an optimisation, it is a different
model: greedy decoding turns a last-bit difference in the logits into a
different token whenever two candidates are close, and the pruning table would
then be comparing two harnesses again - the mistake this repo has already paid
for once with `gpt-oss/base`, where a change of ceiling alone was worth 0.075 F1.

So these tests check equality against the stock `Qwen3MoeSparseMoeBlock.forward`
on the same weights and the same input, at float64 where the arithmetic is exact
enough to see a real difference, and they check the guard that keeps a 2000-token
prefill off the dense path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mrd_pruning.grouped_moe import group_qwen3_moe  # noqa: E402


def tiny_block(n_experts=8, hidden=16, inter=12, top_k=3, dtype=torch.float64):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    cfg = Qwen3MoeConfig(
        hidden_size=hidden, intermediate_size=inter, moe_intermediate_size=inter,
        num_experts=n_experts, num_experts_per_tok=top_k, norm_topk_prob=True,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        vocab_size=32,
    )
    torch.manual_seed(0)
    block = Qwen3MoeSparseMoeBlock(cfg).to(dtype)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return block


class Holder(torch.nn.Module):
    """`group_qwen3_moe` walks `model.modules()`, so give it something to walk."""

    def __init__(self, block):
        super().__init__()
        self.block = block


def test_it_converts_the_blocks_it_finds():
    holder = Holder(tiny_block())
    assert group_qwen3_moe(holder) == 1
    assert group_qwen3_moe(holder) == 0, "The second pass must not touch the already assembled block."


def test_batched_output_matches_the_loop_exactly_enough(monkeypatch):
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    ref_block = tiny_block()
    want, want_logits = ref_block(x)

    fast_block = tiny_block()  # same seed, same weights
    group_qwen3_moe(Holder(fast_block), token_budget=64)
    got, got_logits = fast_block(x)

    assert torch.allclose(got_logits, want_logits, atol=0, rtol=0), "The router should not change."
    assert torch.allclose(got, want, atol=1e-10, rtol=1e-10)


def test_the_prefill_path_also_matches():
    # Above the budget the block must fall back and still be correct.
    x = torch.randn(1, 40, 16, dtype=torch.float64)
    want, _ = tiny_block()(x)

    fast_block = tiny_block()
    group_qwen3_moe(Holder(fast_block), token_budget=8)
    got, _ = fast_block(x)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-10, rtol=1e-10)


def test_the_budget_decides_which_path_runs():
    fast_block = tiny_block()
    group_qwen3_moe(Holder(fast_block), token_budget=4)
    assert fast_block._token_budget == 4

    seen = []
    import mrd_pruning.grouped_moe as gm
    original = gm._loop_forward
    gm._loop_forward = lambda *a, **k: (seen.append(1), original(*a, **k))[1]
    try:
        fast_block(torch.randn(1, 3, 16, dtype=torch.float64))   # threshold
        assert seen == [], "Small batch should go the batch way"
        fast_block(torch.randn(1, 9, 16, dtype=torch.float64))   # threshold
        assert seen == [1], "big batch should go into the cycle"
    finally:
        gm._loop_forward = original


def test_the_per_expert_modules_are_not_kept_twice():
    block = tiny_block()
    before = sum(p.numel() for p in block.parameters())
    group_qwen3_moe(Holder(block))
    after = sum(p.numel() for p in block.parameters())
    # Buffers hold the stacked copies; the parameters must not have grown.
    assert after <= before


def test_the_original_experts_are_released():
    block = tiny_block(n_experts=8)
    group_qwen3_moe(Holder(block))
    assert not isinstance(block.experts, torch.nn.ModuleList), (
        "Original Linear must be released, otherwise experts borrow twice as much"
    )
    assert not isinstance(block.experts, torch.nn.Module), "stub"
    names = {n for n, _ in block.named_buffers()}
    assert {"w_gate", "w_up", "w_down"} <= names


def test_the_mask_can_still_find_the_gates_after_conversion():
    """`masking.find_gates` needs both `.gate` and `.experts` on the block.

    Deleting the ModuleList outright made the whole model read as dense: the
    probe on 22-09-2026 loaded for eighteen minutes and died at
    `no MoE gates found`. The placeholder must survive that check and report the
    right count.
    """
    block = tiny_block(n_experts=8)
    group_qwen3_moe(Holder(block))
    assert getattr(block, "gate", None) is not None
    assert getattr(block, "experts", None) is not None
    assert len(block.experts) == 8


def test_the_sweep_keeps_the_grouped_path_opt_in():
    """The 32-level set must not pick up different kernels by accident."""
    body = (ROOT / "scripts" / "run_pruning_sweep.py").read_text()
    assert '"--grouped-moe", action="store_true"' in body, "flaglessly"
    assert "if args.grouped_moe:" in body
    # And it must refuse silently doing nothing on a backbone it cannot convert.
    assert "No Qwen3-MoE block found" in body


def test_the_kernel_travels_with_the_number():
    """A summary that does not say which kernel made it cannot be compared."""
    body = (ROOT / "scripts" / "run_pruning_sweep.py").read_text()
    assert '"moe_kernel": "grouped_bmm" if args.grouped_moe else "per_expert_loop"' in body
