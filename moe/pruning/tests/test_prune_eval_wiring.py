"""The wiring between a measured map and the sweep that prunes from it.

Every map in `routing_maps_s42_native` holds exactly one arm, under the name
`measure_routing_map.py` was given: `prefix_tuning`, not `prefix`. The sweep's
name table said `prefix`, so `--counts-arm own` on a prefix arm raised KeyError
after loading a 30B checkpoint. These tests pin the resolution and the levels
that the 50%/75% run depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mrd_pruning.frequency import resolve_level, resolve_levels  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "run_pruning_sweep", ROOT / "scripts" / "run_pruning_sweep.py"
)
sweep = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(sweep)
except Exception as exc:  # pragma: no cover - torch missing on a laptop
    pytest.skip(f"sweep script needs its deps: {exc}", allow_module_level=True)


def _write_npz(tmp_path: Path, arms: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "expert_counts.npz"
    payload = {f"{arm}|__all__": m for arm, m in arms.items()}
    payload["_meta"] = np.array('{"n_examples": 500}')
    np.savez(path, **payload)
    return path


def test_single_arm_file_answers_for_itself(tmp_path):
    """The real per-cell maps: one arm, whatever it is called."""
    path = _write_npz(tmp_path, {"prefix_tuning": np.ones((4, 8))})
    assert sweep.npz_arm_key(path, "prefix_tuning") == "prefix_tuning"


def test_single_arm_file_ignores_the_name_table(tmp_path):
    """Even a name the table would have mapped elsewhere."""
    path = _write_npz(tmp_path, {"prefix_tuning": np.ones((4, 8))})
    # The old table said "prefix"; the file says otherwise and the file wins.
    assert sweep.npz_arm_key(path, "prefix_tuning") != "prefix"


def test_multi_arm_file_falls_back_to_the_table(tmp_path):
    path = _write_npz(tmp_path, {"base": np.ones((4, 8)), "gepa": np.ones((4, 8))})
    assert sweep.npz_arm_key(path, "gepa") == "gepa"


def test_multi_arm_file_without_a_match_fails_loudly(tmp_path):
    path = _write_npz(tmp_path, {"base": np.ones((4, 8)), "gepa": np.ones((4, 8))})
    with pytest.raises(SystemExit, match="--counts-arm"):
        sweep.npz_arm_key(path, "prompt_tuning")


@pytest.mark.parametrize("n_experts,expected", [
    (32, [0, 16, 24]),     # gpt-oss
    (128, [0, 64, 96]),    # Qwen
])
def test_the_run_levels_resolve_per_backbone(n_experts, expected):
    """0 / 50% / 75% must mean the same fraction on both backbones."""
    assert resolve_levels(["0", "50%", "75%"], n_experts) == expected


def test_a_level_never_exceeds_the_layer_width():
    with pytest.raises(ValueError, match="exceeds"):
        resolve_level("129", 128)


def test_percent_and_fraction_agree():
    assert resolve_level("50%", 128) == resolve_level(0.5, 128) == 64


def test_zero_is_kept_as_the_control():
    """Level 0 is the unpruned reference the other two are read against."""
    assert resolve_levels(["0", "50%", "75%"], 128)[0] == 0


def _map_with_provenance(tmp_path: Path, adapter_dir: Path, digest: str) -> Path:
    """A counts file that records which checkpoint it was measured from."""
    import json as _json
    meta = {
        "n_examples": 500,
        "arm_provenance": {
            "adapter_path": str(adapter_dir),
            "adapter_sha256": {"adapter_model.safetensors": digest},
        },
    }
    path = tmp_path / "expert_counts.npz"
    np.savez(path, **{"prompt_tuning|__all__": np.ones((4, 8)),
                      "_meta": np.array(_json.dumps(meta))})
    return path


def _adapter(tmp_path: Path, name: str, payload: bytes) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(payload)
    (d / "adapter_config.json").write_text("{}")
    return d


def _spec(adapter: Path):
    from mrd_pruning.arms import ArmSpec
    return ArmSpec(name="prompt_tuning", kind="prompt_tuning", adapter_path=adapter,
                   system_prompt_text=None, prompt_in_user_turn=True)


def test_map_from_another_checkpoint_stops_the_run(tmp_path):
    """The mistake this guard exists for: two adapter trees, same cell names.

    `adapters/` and `adapters-v25/` hold the same eighteen names with different
    weights. Pruning by a map measured from one while scoring the other is
    invisible in every log line - the mask is simply built from the wrong
    routing.
    """
    from mrd_pruning.arms import provenance
    measured = _adapter(tmp_path, "v25", b"the checkpoint the map saw")
    other = _adapter(tmp_path, "other", b"a different checkpoint entirely")
    digest = provenance(_spec(measured))["adapter_sha256"]["adapter_model.safetensors"]
    counts = _map_with_provenance(tmp_path, measured, digest)

    with pytest.raises(SystemExit, match="mask would come from one checkpoint"):
        sweep.check_map_matches_arm(_spec(other), counts, allow_mismatch=False)


def test_matching_checkpoint_passes_and_reports(tmp_path):
    from mrd_pruning.arms import provenance
    measured = _adapter(tmp_path, "v25", b"the checkpoint the map saw")
    digest = provenance(_spec(measured))["adapter_sha256"]["adapter_model.safetensors"]
    counts = _map_with_provenance(tmp_path, measured, digest)

    report = sweep.check_map_matches_arm(_spec(measured), counts, allow_mismatch=False)
    assert report["checked"] and report["matches"]
    assert report["map_sha256"] == report["arm_sha256"]


def test_mismatch_can_be_allowed_on_purpose(tmp_path):
    """Pruning one arm by another's map is a real experiment, just not a default."""
    from mrd_pruning.arms import provenance
    measured = _adapter(tmp_path, "v25", b"the checkpoint the map saw")
    other = _adapter(tmp_path, "other", b"a different checkpoint entirely")
    digest = provenance(_spec(measured))["adapter_sha256"]["adapter_model.safetensors"]
    counts = _map_with_provenance(tmp_path, measured, digest)

    report = sweep.check_map_matches_arm(_spec(other), counts, allow_mismatch=True)
    assert report["checked"] and not report["matches"]


def test_base_arm_has_no_adapter_to_check(tmp_path):
    from mrd_pruning.arms import ArmSpec
    counts = _map_with_provenance(tmp_path, tmp_path / "x", "deadbeef")
    base = ArmSpec(name="base", kind="base", adapter_path=None,
                   system_prompt_text=None, prompt_in_user_turn=True)
    report = sweep.check_map_matches_arm(base, counts, allow_mismatch=False)
    assert report["checked"] is False


def test_answer_on_the_line_after_the_prefix_is_read():
    """`Answer:` alone, the answer below it - GEPA's optimised layout.

    458 of 2000 responses on gepa-n1000 looked like this and were all thrown
    away as unreadable, which is most of why GEPA scored below the base.
    """
    from mrd_pruning.task import parse_response
    r = parse_response("Answer:\nNONE")
    assert r.saw_none and not r.unparsable

    r = parse_response("Answer:\ntoxicity, insult")
    assert r.labels == ("toxicity", "insult") and not r.unparsable


def test_answer_on_the_same_line_still_works():
    from mrd_pruning.task import parse_response
    assert parse_response("Answer: NONE").saw_none
    assert parse_response("Answer: insult").labels == ("insult",)


def test_blank_lines_between_prefix_and_answer_are_skipped():
    from mrd_pruning.task import parse_response
    assert parse_response("Answer:\n\n  toxicity").labels == ("toxicity",)


def test_a_second_line_is_read_only_when_the_first_is_empty():
    """Verbosity after a real answer must still be ignored."""
    from mrd_pruning.task import parse_response
    r = parse_response("Answer: NONE\ntoxicity, insult")
    assert r.saw_none and r.labels == (), "the second line must not be harvested"


def test_still_unparsable_when_nothing_is_there():
    from mrd_pruning.task import parse_response
    assert parse_response("Answer:\nText: some echoed template").unparsable
    assert parse_response("").unparsable
