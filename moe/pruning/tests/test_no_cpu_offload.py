"""A model spilled onto the host must stop the run, not slow it down.

`device_map="auto"` treats a card that cannot hold the weights as a layout
problem, not an error: it puts the overflow on CPU and carries on. Generation
still produces correct answers, so nothing in the output says anything is wrong
 - the only symptom is speed, and on a queue where a level legitimately takes
hours, "slow" is indistinguishable from "stuck".

`gpt-oss/base` once ran for more than four hours without finishing its first
sixty-four rows, with all weights in host RAM and no GPU allocation at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mrd_pruning.arms import ArmSpec, require_on_gpu  # noqa: E402


class FakeModel:
    def __init__(self, placement) -> None:
        if placement is not None:
            self.hf_device_map = placement


@pytest.fixture
def spec():
    return ArmSpec(name="base", kind="base")


def test_a_fully_placed_model_passes(spec):
    require_on_gpu(FakeModel({"model.layers.0": 0, "lm_head": 0}), spec)


def test_a_single_module_on_cpu_is_enough_to_stop_the_run(spec):
    """One layer on the host is the case that makes a level take a day."""
    with pytest.raises(RuntimeError, match="not only on the GPU"):
        require_on_gpu(FakeModel({"model.layers.0": 0, "model.layers.1": "cpu"}), spec)


def test_disk_offload_is_refused_too(spec):
    with pytest.raises(RuntimeError, match="disk"):
        require_on_gpu(FakeModel({"model.layers.0": "disk"}), spec)


def test_the_message_names_a_module_so_the_log_is_actionable(spec):
    with pytest.raises(RuntimeError) as err:
        require_on_gpu(FakeModel({"a": 0, "b": "cpu", "c": "cpu"}), spec)
    assert "2 modules" in str(err.value)
    assert "base" in str(err.value)


def test_a_model_without_a_device_map_is_left_alone(spec):
    """A single-device load sets no `hf_device_map`; that is not an offload."""
    require_on_gpu(FakeModel(None), spec)
    require_on_gpu(FakeModel({}), spec)


def test_a_device_index_named_as_a_string_is_not_mistaken_for_a_host(spec):
    """`torch.device` objects and ints both stringify; only cpu/disk are hosts.

    A naive `"cpu" in str(dev)` would also fire on a hypothetical device string
    containing it, and treating `0` as suspicious would fail every healthy run.
    """
    require_on_gpu(FakeModel({"a": 0, "b": "cuda:1"}), spec)
