"""Composite stage slices for the frequency map.

``__all__`` is not a like-for-like axis across arms: a prompt-tuning arm routes
its own virtual tokens and they dominate its counts, while a prefix-projected
arm has no virtual positions at all. ``__text__`` is the slice that compares
the same thing on both.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mrd_pruning.frequency import load_counts_npz, resolve_stage_parts, select_pruned

STAGES_PRESENT = ["__all__", "answer", "comment", "template", "virtual"]


def write_npz(tmp_path, arm="prompt_tuning", *, with_virtual=True):
    """A two-layer, four-expert map whose stages disagree on the least-used expert."""
    comment = np.array([[10.0, 1.0, 5.0, 4.0], [8.0, 2.0, 6.0, 4.0]])
    template = np.array([[6.0, 2.0, 3.0, 9.0], [5.0, 3.0, 2.0, 10.0]])
    answer = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    payload = {
        f"{arm}|comment": comment,
        f"{arm}|template": template,
        f"{arm}|answer": answer,
    }
    stages = ["__all__", "answer", "comment", "template"]
    total = comment + template + answer
    if with_virtual:
        # Virtual tokens pile onto expert 1, the one text uses least.
        virtual = np.array([[0.0, 900.0, 0.0, 0.0], [0.0, 900.0, 0.0, 0.0]])
        payload[f"{arm}|virtual"] = virtual
        stages.append("virtual")
        total = total + virtual
    payload[f"{arm}|__all__"] = total
    meta = {"n_examples": 500, "stages_present": sorted(stages), "entries": {}}
    path = tmp_path / "expert_counts.npz"
    np.savez(path, _meta=json.dumps(meta), **payload)
    return path


def test_single_stage_is_unchanged(tmp_path):
    path = write_npz(tmp_path)
    counts = load_counts_npz(path, "prompt_tuning", "comment")
    assert counts.counts[0].tolist() == [10.0, 1.0, 5.0, 4.0]


def test_sum_of_stages(tmp_path):
    path = write_npz(tmp_path)
    counts = load_counts_npz(path, "prompt_tuning", "comment+template")
    assert counts.counts[0].tolist() == [16.0, 3.0, 8.0, 13.0]


def test_text_slice_drops_virtual(tmp_path):
    path = write_npz(tmp_path)
    text = load_counts_npz(path, "prompt_tuning", "__text__")
    every = load_counts_npz(path, "prompt_tuning", "__all__")
    assert text.counts[0].tolist() == [17.0, 4.0, 9.0, 14.0]
    assert every.counts[0][1] == 904.0  # virtual dominates the same expert


def test_virtual_flips_which_expert_looks_least_used(tmp_path):
    """The whole reason the slice matters: the victim changes."""
    path = write_npz(tmp_path)
    by_all = select_pruned(
        load_counts_npz(path, "prompt_tuning", "__all__"), 1, top_k=2
    )
    by_text = select_pruned(
        load_counts_npz(path, "prompt_tuning", "__text__"), 1, top_k=2
    )
    assert by_text[0] == [1]  # text uses expert 1 least
    assert by_all[0] != by_text[0]  # virtual tokens protect it


def test_text_slice_works_without_virtual(tmp_path):
    path = write_npz(tmp_path, arm="prefix_tuning", with_virtual=False)
    text = load_counts_npz(path, "prefix_tuning", "__text__")
    every = load_counts_npz(path, "prefix_tuning", "__all__")
    assert text.counts.tolist() == every.counts.tolist()


def test_unknown_stage_is_rejected(tmp_path):
    path = write_npz(tmp_path)
    with pytest.raises(ValueError, match="unknown stage"):
        load_counts_npz(path, "prompt_tuning", "bogus")


def test_missing_stage_names_what_is_available(tmp_path):
    path = write_npz(tmp_path, with_virtual=False)
    with pytest.raises(KeyError, match="virtual"):
        load_counts_npz(path, "prompt_tuning", "comment+virtual")


def test_unknown_arm_is_rejected(tmp_path):
    path = write_npz(tmp_path)
    with pytest.raises(KeyError, match="no stage of arm"):
        load_counts_npz(path, "nope", "__all__")


def test_resolve_parts_lists_stages():
    assert resolve_stage_parts("__text__", STAGES_PRESENT) == [
        "answer", "comment", "template",
    ]
    assert resolve_stage_parts("comment", STAGES_PRESENT) == ["comment"]
