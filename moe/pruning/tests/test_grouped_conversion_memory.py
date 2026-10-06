"""The conversion must not need several copies of a block to build one.

`torch.stack(...).transpose(1, 2).contiguous()` reads well and peaks at three
copies per projection - originals, stack, transposed copy - nine across the
three where three would do. On Qwen3-30B that is roughly 3.6 GB of transient
per block, and on 23-09 it is what killed a run that shared a 139.8 GB card
with a gpt-oss neighbour: the weights fit, the conversion did not, and it died
asking for 384 MiB with 356 MiB left.

So the destination is filled expert by expert and each expert's weight is
dropped as it is consumed. These tests pin both halves: the numbers must be the
transpose they replace, and the originals must actually be released - a version
that copies correctly but keeps every source alive would pass a values-only
test and fail on the card exactly as before.

No transformers here on purpose: the helper takes anything with `.weight`, so
the check runs anywhere torch does.
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
    """The shape the real block presents: `.gate_proj.weight` and friends."""

    def __init__(self, linear):
        self.gate_proj = linear


def make_experts(n: int, out_f: int, in_f: int):
    return [_Expert(torch.nn.Linear(in_f, out_f, bias=False)) for _ in range(n)]


def test_values_match_the_stacked_transpose():
    torch.manual_seed(0)
    experts = make_experts(5, 7, 3)
    want = torch.stack([e.gate_proj.weight.detach().clone()
                        for e in experts]).transpose(1, 2).contiguous()
    got = _stack_transposed(experts, "gate_proj", torch, release=False)
    assert torch.equal(got, want)


def test_shape_is_experts_in_out():
    experts = make_experts(4, 9, 6)
    got = _stack_transposed(experts, "gate_proj", torch, release=False)
    assert got.shape == (4, 6, 9)


def test_dtype_and_device_follow_the_weights():
    experts = make_experts(3, 4, 2)
    for e in experts:
        e.gate_proj.to(torch.float64)
    got = _stack_transposed(experts, "gate_proj", torch, release=False)
    assert got.dtype == torch.float64
    assert got.device == experts[0].gate_proj.weight.device


def test_release_drops_every_source_weight():
    """The point of the whole rewrite: sources go as they are consumed."""
    experts = make_experts(6, 5, 4)
    _stack_transposed(experts, "gate_proj", torch, release=True)
    assert all(e.gate_proj.weight is None for e in experts)


def test_release_false_keeps_them():
    experts = make_experts(3, 5, 4)
    _stack_transposed(experts, "gate_proj", torch, release=False)
    assert all(e.gate_proj.weight is not None for e in experts)


def test_released_sources_are_actually_freed_not_just_unreferenced():
    """A weight still alive somewhere else is not a saving.

    Holding a weakref to one source and dropping every other reference: if the
    release only cleared the attribute while something internal kept the
    tensor, the referent would survive the collection.
    """
    import gc
    import weakref

    experts = make_experts(2, 4, 3)
    ref = weakref.ref(experts[0].gate_proj.weight)
    out = _stack_transposed(experts, "gate_proj", torch, release=True)
    gc.collect()
    assert ref() is None, "base weight survives conversion"
    assert out.shape == (2, 3, 4)


def test_one_expert_is_not_a_special_case():
    experts = make_experts(1, 3, 2)
    got = _stack_transposed(experts, "gate_proj", torch, release=False)
    assert torch.equal(got[0], experts[0].gate_proj.weight.t())
