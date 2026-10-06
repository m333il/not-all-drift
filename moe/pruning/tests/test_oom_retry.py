"""A shared card takes memory away mid-run; that must not corrupt the counts.

The worker claims a GPU with room to spare, then a neighbour's job takes the
room before the first batch runs. Retrying is right - but a forward that died
part-way has already fired the gate hooks of the layers it reached, so a naive
retry counts those layers twice and the totals still look plausible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "measure_routing_map", ROOT / "scripts" / "measure_routing_map.py"
)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


@pytest.fixture(autouse=True)
def fake_cuda_memory(monkeypatch):
    """Exercise counter rollback on CPU without querying a real CUDA device."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 * 2**30, 16 * 2**30))


class _Counter:
    """The part of GpuCounter the retry touches."""

    def __init__(self) -> None:
        self.counts = torch.zeros(3, dtype=torch.float64)
        self._calls = 0

    snapshot = measure.GpuCounter.snapshot
    restore = measure.GpuCounter.restore

    def fire(self, layer: int) -> None:
        self.counts[layer] += 1
        self._calls += 1


class _Model:
    """Fails the first `n_fail` calls, having counted part of the way through."""

    def __init__(self, counter: _Counter, n_fail: int, n_layers: int = 3) -> None:
        self.counter = counter
        self.n_fail = n_fail
        self.n_layers = n_layers
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask):
        self.calls += 1
        if self.calls <= self.n_fail:
            self.counter.fire(0)          # one layer got counted...
            raise torch.OutOfMemoryError("CUDA out of memory")   # ...then it died
        for layer in range(self.n_layers):
            self.counter.fire(layer)
        return None


def _run(model, counter, **kw):
    ids = torch.ones((1, 4), dtype=torch.long)
    measure.run_forward(model, ids, torch.ones_like(ids), counter, None,
                        attempts=kw.get("attempts", 4), wait=0.0)


def test_successful_forward_counts_once(monkeypatch):
    counter = _Counter()
    _run(_Model(counter, n_fail=0), counter)
    assert counter.counts.tolist() == [1.0, 1.0, 1.0]


def test_partial_forward_is_rolled_back_before_the_retry():
    """Layer 0 fired twice on two failures; after the retry it must read 1."""
    counter = _Counter()
    model = _Model(counter, n_fail=2)
    _run(model, counter)
    assert model.calls == 3
    assert counter.counts.tolist() == [1.0, 1.0, 1.0]


def test_call_count_is_rolled_back_too():
    counter = _Counter()
    _run(_Model(counter, n_fail=1), counter)
    assert counter._calls == 3, "one call per layer of the successful pass"


def test_gives_up_after_the_last_attempt():
    counter = _Counter()
    model = _Model(counter, n_fail=99)
    with pytest.raises(torch.OutOfMemoryError):
        _run(model, counter, attempts=3)
    assert model.calls == 3


def test_counts_are_clean_even_when_it_gives_up():
    """A failed run must not leave half-counted layers behind for the caller."""
    counter = _Counter()
    with pytest.raises(torch.OutOfMemoryError):
        _run(_Model(counter, n_fail=99), counter, attempts=2)
    assert counter.counts.tolist() == [0.0, 0.0, 0.0]


def test_snapshot_is_a_copy_not_a_view():
    counter = _Counter()
    saved = counter.snapshot()
    counter.fire(1)
    counter.restore(saved)
    assert counter.counts.tolist() == [0.0, 0.0, 0.0]


def test_allocator_policy_is_set_before_torch_sees_it():
    import os

    assert "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]
