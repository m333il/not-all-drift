"""Two silent assumptions the batched path rests on, pinned.

Everything else about this kernel has been checked against the stock block end
to end. These two are below that level and would fail in a way no end-to-end
test names:

1. The stacked buffers must hold each expert's own weight, transposed. A single
   mis-ordered expert would not crash and would not even look wrong on a short
   prompt - the router would simply be routing to the wrong matrices, and the
   answers would degrade in a way indistinguishable from "pruning hurt".

2. `bmm` is fed a batch dimension created by `expand`, so every expert reads the
   same tokens through a stride of zero. That is the entire memory argument for
   the batched form - materialising `E × tokens × hidden` would be ruinous - and
   it is an assumption about how `bmm` treats a zero stride rather than about
   this code. If it were ever wrong, every number the kernel produced would be
   wrong with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from mrd_pruning.grouped_moe import _stack_transposed  # noqa: E402


class _Expert:
    def __init__(self, linear):
        self.gate_proj = linear


def test_every_expert_lands_in_its_own_slot():
    """Slot i must be expert i, transposed - not sorted, shifted or shared."""
    torch.manual_seed(0)
    experts = [_Expert(torch.nn.Linear(3, 5, bias=False)) for _ in range(7)]
    want = [e.gate_proj.weight.detach().clone() for e in experts]

    out = _stack_transposed(experts, "gate_proj", torch, release=False)

    for i, w in enumerate(want):
        assert torch.equal(out[i], w.t()), f"slot {i} weightless"


def test_distinct_experts_stay_distinct():
    """A stack that accidentally broadcast one expert would still have the
    right shape and the right dtype, and would be caught only here."""
    experts = [_Expert(torch.nn.Linear(2, 2, bias=False)) for _ in range(3)]
    with torch.no_grad():
        for i, e in enumerate(experts):
            e.gate_proj.weight.fill_(float(i + 1))

    out = _stack_transposed(experts, "gate_proj", torch, release=False)

    assert [float(out[i][0, 0]) for i in range(3)] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_bmm_over_an_expanded_batch_matches_a_real_one(dtype):
    """`expand` gives a zero stride; `repeat` gives real memory. Same answer."""
    torch.manual_seed(1)
    n_experts, tokens, hidden, inter = 5, 9, 6, 4
    x = torch.randn(tokens, hidden, dtype=dtype)
    w = torch.randn(n_experts, hidden, inter, dtype=dtype)

    expanded = torch.bmm(x.unsqueeze(0).expand(n_experts, tokens, hidden), w)
    materialised = torch.bmm(x.unsqueeze(0).repeat(n_experts, 1, 1), w)

    assert torch.equal(expanded, materialised), (
        f"bmm on the stretched batch differs from the present in {dtype}")


def test_expanded_bmm_equals_per_expert_matmul():
    """And each slice equals the plain matmul it stands for."""
    torch.manual_seed(2)
    n_experts, tokens, hidden, inter = 4, 7, 5, 3
    x = torch.randn(tokens, hidden, dtype=torch.float64)
    w = torch.randn(n_experts, hidden, inter, dtype=torch.float64)

    got = torch.bmm(x.unsqueeze(0).expand(n_experts, tokens, hidden), w)

    for i in range(n_experts):
        assert torch.allclose(got[i], x @ w[i], atol=1e-12)
