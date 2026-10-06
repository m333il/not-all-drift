"""Arm specs: checkpoint pinning and the system-prompt policy."""
from __future__ import annotations

from pathlib import Path

import pytest

from mrd_pruning.arms import ArmSpec, provenance, sha256_of_dir
from mrd_pruning.task import SEED_SYSTEM_PROMPT


def make_adapter(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts)
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text('{"peft_type": "PROMPT_TUNING"}')
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path


def test_epoch_pinned_adapter_is_accepted(tmp_path) -> None:
    """The published layout nests the real checkpoint as
    checkpoints/epoch_002/adapter - that directory is pinned, not ambiguous."""
    path = make_adapter(tmp_path, "arms", "pt", "checkpoints", "epoch_002", "adapter")
    ArmSpec(name="prompt_tuning", kind="prompt_tuning", adapter_path=path).validate()


def test_top_level_adapter_is_refused(tmp_path) -> None:
    path = make_adapter(tmp_path, "arms", "pt", "adapter")
    spec = ArmSpec(name="prompt_tuning", kind="prompt_tuning", adapter_path=path)
    with pytest.raises(ValueError, match="top-level adapter"):
        spec.validate()


def test_top_level_adapter_can_be_opted_into(tmp_path) -> None:
    path = make_adapter(tmp_path, "arms", "pt", "adapter")
    ArmSpec(name="pt", kind="prompt_tuning", adapter_path=path,
            allow_unpinned_checkpoint=True).validate()


def test_missing_adapter_path_fails_early(tmp_path) -> None:
    spec = ArmSpec(name="pt", kind="prompt_tuning", adapter_path=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        spec.validate()
    with pytest.raises(ValueError, match="needs adapter_path"):
        ArmSpec(name="pt", kind="prompt_tuning").validate()


def test_gepa_needs_its_prompt() -> None:
    with pytest.raises(ValueError, match="optimised prompt"):
        ArmSpec(name="gepa", kind="gepa").validate()


def test_system_policies() -> None:
    base = ArmSpec(name="base", kind="base", system_prompt_text=SEED_SYSTEM_PROMPT)
    pt = ArmSpec(name="pt", kind="prompt_tuning", system_prompt_text=None)
    assert base.system_prompt("as_trained") == SEED_SYSTEM_PROMPT
    assert pt.system_prompt("as_trained") is None      # the trained asymmetry
    assert pt.system_prompt("seed_all") == SEED_SYSTEM_PROMPT
    assert base.system_prompt("none_all") is None
    with pytest.raises(ValueError, match="policy"):
        base.system_prompt("whatever")  # type: ignore[arg-type]


def test_provenance_carries_digests_not_just_names(tmp_path) -> None:
    path = make_adapter(tmp_path, "arms", "pt", "checkpoints", "epoch_002", "adapter")
    record = provenance(ArmSpec(name="pt", kind="prompt_tuning", adapter_path=path))
    assert set(record["adapter_sha256"]) == {"adapter_config.json", "adapter_model.safetensors"}
    assert len(record["adapter_sha256"]["adapter_model.safetensors"]) == 64


def test_sha256_refuses_an_empty_directory(tmp_path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        sha256_of_dir(tmp_path / "empty")


def test_gepa_arm_may_carry_its_prompt_in_the_user_turn() -> None:
    """The v25 contract has no system message at all.

    A GEPA arm there is configured correctly with no system text - its optimised
    instruction leads the user turn instead. Rejecting that configuration failed
    every GEPA cell of the grid before a single one had run (12-09-2026).
    """
    ArmSpec(name="gepa", kind="gepa", prompt_in_user_turn=True).validate()

    with pytest.raises(ValueError, match="optimised prompt text"):
        ArmSpec(name="gepa", kind="gepa").validate()


def test_gepa_arm_with_a_system_prompt_still_validates() -> None:
    ArmSpec(name="gepa", kind="gepa", system_prompt_text="instructions").validate()
