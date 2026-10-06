from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("selection", Path(__file__).parents[1] / "scripts/select_checkpoints.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def example():
    manifest = {"keys": ["a", "b"], "arms": [
        {"name": str(step), "adapter": f"/run/step_{step:06d}/adapter", "adapter_hashes": {}}
        for step in [0, 2, 1]]}
    scores = {"arms": {str(step): {"score": value, "per_example": [
        {"key": key, "score": value} for key in ["a", "b"]]}
        for step, value in [(0, 1.0), (2, 0.5), (1, 0.5)]}}
    return manifest, scores


def test_initialization_is_control_and_ties_choose_earliest_trained_step():
    manifest, scores = example()
    result = runner.select_checkpoints(manifest, scores)
    assert result["selected"][0]["best"]["step"] == 1
    assert result["controls"][0]["step"] == 0


def test_duplicate_id_cannot_hide_a_missing_validation_example():
    manifest, scores = example()
    scores["arms"]["1"]["per_example"][1]["key"] = "a"
    with pytest.raises(ValueError, match="exactly one"):
        runner.select_checkpoints(manifest, scores)


def test_missing_arm_and_inconsistent_primary_score_are_rejected():
    manifest, scores = example()
    missing = deepcopy(scores)
    del missing["arms"]["2"]
    with pytest.raises(ValueError, match="Scored arms"):
        runner.select_checkpoints(manifest, missing)
    scores["arms"]["1"]["score"] = 0.9
    with pytest.raises(ValueError, match="Aggregate primary"):
        runner.select_checkpoints(manifest, scores)
