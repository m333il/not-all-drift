"""The card has to be claimed before the shards load, not after.

The queue picks a card by its free memory and then spends about five minutes
loading Qwen's sixteen shards. On 22-09-2026 two cells died inside that window
eleven minutes apart - `qwen/prefix-m200` and `qwen/prompt-m200` - both in
`load_shard_file`, both with 41 GB left of the 74 the queue had seen. The gate
was not wrong when it fired; the memory left afterwards.

What these tests pin down is the shape of the fix, not the driver behaviour:
that nothing is claimed when nothing was asked for, that a claim that cannot be
met fails immediately and says why, and above all that the reservation is
refused together with `device_map="auto"` - accelerate plans placement from
`mem_get_info`, which cannot see our allocator cache, so a reservation would
read to it as a full card and send the model to the host. That combination
would turn a loud failure into the silent CPU-bound run this repo has already
paid for once.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrd_pruning.arms import ArmSpec, claim_memory, load_arm  # noqa: E402


class FakeOOM(Exception):
    pass


def fake_torch(*, available=True, free_mb=1000, fails=False):
    """Enough of torch for `claim_memory`: allocation, its failure, free memory."""
    calls = []

    def empty(n, dtype=None, device=None):
        calls.append((n, device))
        if fails:
            raise FakeOOM("no memory")
        return object()

    mod = types.SimpleNamespace(
        empty=empty,
        uint8="uint8",
        OutOfMemoryError=FakeOOM,
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            mem_get_info=lambda d=0: (free_mb * 1024 * 1024, 0),
        ),
        calls=calls,
    )
    return mod


@pytest.fixture
def torch_stub(monkeypatch):
    def install(mod):
        monkeypatch.setitem(sys.modules, "torch", mod)
        return mod
    return install


def test_zero_reserves_nothing(torch_stub):
    mod = torch_stub(fake_torch())
    assert claim_memory(0) == 0
    assert mod.calls == []


def test_no_cuda_reserves_nothing(torch_stub):
    mod = torch_stub(fake_torch(available=False))
    assert claim_memory(58000) == 0
    assert mod.calls == []


def test_claim_allocates_exactly_the_megabytes_asked(torch_stub):
    mod = torch_stub(fake_torch())
    assert claim_memory(58000) == 58000
    assert mod.calls == [(58000 * 1024 * 1024, "cuda:0")]


def test_a_claim_that_cannot_be_met_fails_now_and_says_why(torch_stub):
    torch_stub(fake_torch(fails=True, free_mb=41000))
    with pytest.raises(RuntimeError) as exc:
        claim_memory(58000)
    assert "41000" in str(exc.value)
    assert "neighbour" in str(exc.value)


def test_reservation_is_refused_with_device_map_auto():
    spec = ArmSpec(name="base", kind="base", system_prompt_text="x")
    with pytest.raises(ValueError, match="auto"):
        load_arm(spec, model_id="m", device_map="auto", reserve_mb=58000)


def test_device_map_auto_without_a_reservation_is_still_allowed(monkeypatch):
    # The guard must not break the path every earlier run used.
    spec = ArmSpec(name="base", kind="base", system_prompt_text="x")
    with pytest.raises(Exception) as exc:
        load_arm(spec, model_id="knowingly-no-so-model", device_map="auto")
    assert not isinstance(exc.value, ValueError) or "auto" not in str(exc.value)
