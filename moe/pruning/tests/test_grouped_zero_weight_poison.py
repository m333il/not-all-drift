"""A pruned expert must not be able to poison the token it was pruned from.

The two kernels are only equivalent while every expert output is finite. The
loop computes an expert exactly when the router picked it; the batched form
computes all of them and lets the router's zeros cancel the rest. Those agree
right up to the moment some expert returns a non-finite value, because
`0 * inf` is `NaN`, not `0` - and the sum over experts then turns the whole
token's hidden state into `NaN`.

Pruning is exactly the regime that makes this reachable. A masked expert keeps
its weights and keeps being evaluated by the batched path, on tokens it was
never picked for and, at −50% or −75%, never trained to see. It only has to
overflow once: one `NaN` hidden state propagates through the rest of the
forward, and a model whose state is `NaN` emits the same token until the
ceiling - which is what `qwen/prefix-m500 −50%` did on all 2000 rows.

So this is not a numerical nicety. It is the difference between "the checkpoint
collapsed under pruning" and "our kernel collapsed it".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from mrd_pruning.grouped_moe import _grouped_forward  # noqa: E402


class _Gate:
    """Returns fixed logits, the way the masked gate hook does."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, _flat):
        return self.logits


class _Block:
    """The attributes the two forwards read, and nothing else."""

    def __init__(self, w_gate, w_up, w_down, logits, top_k, budget):
        self.w_gate, self.w_up, self.w_down = w_gate, w_up, w_down
        self.gate = _Gate(logits)
        self.top_k = top_k
        self.norm_topk_prob = True
        self.act_fn = torch.nn.functional.silu
        self._token_budget = budget


def build(dtype, *, huge_expert: int | None, n_experts=6, hidden=8, inter=8,
          tokens=4, top_k=2, budget=4096):
    torch.manual_seed(0)
    w_gate = torch.randn(n_experts, hidden, inter, dtype=dtype) * 0.1
    w_up = torch.randn(n_experts, hidden, inter, dtype=dtype) * 0.1
    w_down = torch.randn(n_experts, inter, hidden, dtype=dtype) * 0.1
    if huge_expert is not None:
        # Large enough that gate @ up overflows the dtype's range.
        big = torch.finfo(dtype).max
        w_gate[huge_expert] = big / 8
        w_up[huge_expert] = big / 8
    logits = torch.randn(tokens, n_experts, dtype=dtype)
    if huge_expert is not None:
        # …and pruned: the router can never pick it. This is what the mask hook
        # writes, and it is why the loop never evaluates this expert.
        logits[:, huge_expert] = torch.finfo(dtype).min
    x = torch.randn(tokens, hidden, dtype=dtype)
    return _Block(w_gate, w_up, w_down, logits, top_k, budget), x


def run(block, x, budget):
    block._token_budget = budget
    out, _ = _grouped_forward(block, x.unsqueeze(0))
    return out.squeeze(0)


def test_healthy_experts_agree_between_the_paths():
    """The baseline: with everything finite the two paths match."""
    block, x = build(torch.float64, huge_expert=None)
    batched = run(block, x, budget=4096)
    looped = run(block, x, budget=1)
    assert torch.allclose(batched, looped, atol=1e-12)


def test_pruned_overflowing_expert_does_not_reach_the_output():
    """The bug: a masked expert that overflows must not poison the token.

    Its router weight is exactly zero, so it contributes nothing - unless the
    implementation multiplies zero by an infinity it computed anyway.
    """
    block, x = build(torch.float32, huge_expert=3)
    batched = run(block, x, budget=4096)
    assert torch.isfinite(batched).all(), (
        "knocked out expert with overflow poisoned the output of the batch way")


def test_the_loop_path_is_unaffected_by_it():
    """Evidence the two kernels really do differ here, not that both break."""
    block, x = build(torch.float32, huge_expert=3)
    looped = run(block, x, budget=1)
    assert torch.isfinite(looped).all()


def test_both_paths_agree_when_an_expert_overflows():
    block, x = build(torch.float32, huge_expert=3)
    batched = run(block, x, budget=4096)
    looped = run(block, x, budget=1)
    assert torch.allclose(batched, looped, atol=1e-4), (
        f"Ways diverged: batch {batched[0, :3]} cycle {looped[0, :3]}")


def test_a_nan_expert_is_also_contained():
    """Not only infinities: a `NaN` weight must stay out of the sum too."""
    block, x = build(torch.float32, huge_expert=None)
    block.w_gate[2] = float("nan")
    block.gate.logits[:, 2] = torch.finfo(torch.float32).min   # pruned
    batched = run(block, x, budget=4096)
    assert torch.isfinite(batched).all(), "NaN knocked out expert hit the amount"


def test_legacy_switch_reproduces_the_defect():
    """The switch has to actually bring the bug back, or the comparison lies.

    The three-way run beside the corrected answers is only evidence if this
    mode is the old code and not a second correct one.
    """
    block, x = build(torch.float32, huge_expert=3)
    block._legacy_poison = True
    poisoned = run(block, x, budget=4096)
    assert not torch.isfinite(poisoned).all()


def test_the_switch_is_off_unless_asked():
    block, x = build(torch.float32, huge_expert=3)
    assert getattr(block, "_legacy_poison", False) is False
    assert torch.isfinite(run(block, x, budget=4096)).all()
