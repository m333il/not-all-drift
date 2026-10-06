"""Guards for the thirty-two pruning runs of the final set.

Every failure this file is written against produced a plausible number last
time. A level scored against the wrong checkpoint's mask, answers cut off by a
ceiling and read as if the model had nothing to say, a mask that removed the
experts a layer uses most instead of least - each of those filled the quality
table with something that looked like a result.

Tests over levels that do not exist yet skip, so the file can be run on every
pass while the queue fills.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.artifacts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / os.environ.get("FINAL_PRUNE_OUT", "results/prune_final_5075")
MAPS = ROOT / os.environ.get("FINAL_MAPS", "routing_maps_final")
TEST = ROOT / "data" / "test_canonical_n2000.jsonl"
CALIB = ROOT / "data" / "calib_val_n500.jsonl"

BASES = (("gpt-oss", "base"), ("qwen", "base"))
GEPA = (("gpt-oss", "gepa-gpt-oss-20b-n1000-s42"),
        ("qwen", "gepa-qwen3-2507-n1000-s42"))
ARMS = tuple(
    (model, f"{kind}-m{m}-s42")
    for model in ("gpt-oss", "qwen")
    for kind in ("prompt", "prefix-projected")
    for m in (100, 200, 500)
)
CELLS = BASES + ARMS + GEPA

# How many experts each level removes per layer, by backbone.
PRUNED = {"gpt-oss": {"50%": 16, "75%": 24}, "qwen": {"50%": 64, "75%": 96}}
N_EXPERTS = {"gpt-oss": 32, "qwen": 128}
CEILING = {"gpt-oss": 4096, "qwen": 512}

LEVELS = ("50%", "75%")
END_MARKERS = ("<|im_end|>", "<|return|>", "<|endoftext|>", "<|end|>")


def _kind(cell: str) -> str:
    if cell == "base":
        return "base"
    if cell.startswith("gepa-"):
        return "gepa"
    return "prompt_tuning" if cell.startswith("prompt-") else "prefix_tuning"


def _level_dir(model: str, cell: str, level: str) -> Path:
    return OUT / model / cell / f"{_kind(cell)}_prune{PRUNED[model][level]:03d}"


def _summary(model: str, cell: str, level: str) -> dict:
    d = _level_dir(model, cell, level)
    path = d / "summary.json"
    if not path.is_file():
        pytest.skip(f"level not measured yet: {model}/{cell}/{level}")
    return json.loads(path.read_text())


def _config(model: str, cell: str) -> dict:
    """The sweep's own record of what it was asked to measure.

    `summary.json` carries the metrics; the data file, the example count and the
    ceiling live one level up in `sweep_config.json`, and that is where they
    have to be checked.
    """
    path = OUT / model / cell / "sweep_config.json"
    if not path.is_file():
        pytest.skip(f"cell not started yet: {model}/{cell}")
    return json.loads(path.read_text())


def _rescored(model: str, cell: str, level: str) -> dict | None:
    path = _level_dir(model, cell, level) / "summary_rescored.json"
    return json.loads(path.read_text()) if path.is_file() else None


def test_the_queue_is_thirty_two_levels():
    assert len(CELLS) * len(LEVELS) == 32


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_mask_was_built_from_this_cells_own_map(model, cell, level):
    """The guard that stops a mask from one checkpoint scoring another.

    The sweep writes `map_matches_arm` by comparing the adapter digest recorded
    in the map against the adapter this run loaded. That comparison only exists
    where there is an adapter: `base` loads none and GEPA's arm is a prompt, so
    for those two the sweep reports `checked: false` and this test must not read
    that as a pass *or* as a failure. What it demands instead is that the
    inapplicable case is the only reason it went unchecked - a `checked: false`
    for any other reason is a real hole - and that the map file the run names is
    this cell's own.
    """
    s = _summary(model, cell, level)
    match = s.get("map_matches_arm")
    assert match is not None, f"{model}/{cell}/{level}: no check of the card"

    if cell == "base" or cell.startswith("gepa-"):
        assert match.get("checked") is False and "no adapter" in str(match.get("reason")), (
            f"{model}/{cell}/{level}: an adapterless cell has an unexpected verdict. {match}")
        named = (s.get("counts_npz") or {}).get("path")
        if named is not None:
            assert f"/{model}/{cell}/" in str(named), (
                f"{model}/{cell}/{level}The mask was built on someone else's map {named}")
        # The identity of the mask itself is checked against this cell's map by
        # `test_the_mask_took_the_least_used_experts`, which is what actually
        # catches a map from the wrong checkpoint here.
        return

    assert match.get("matches") is True, (
        f"{model}/{cell}/{level}The card is not from this checkpoint. {match}")


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_named_map_file_is_the_one_on_disk(model, cell, level):
    """The map's sha256 in the summary must match the file it names.

    Written for the two cells whose provenance rests on nothing else, but it
    costs nothing to demand of every cell: a map replaced under the same path
    after the run would otherwise leave no trace.
    """
    s = _summary(model, cell, level)
    named = s.get("counts_npz")
    if not named or not named.get("sha256"):
        pytest.skip("level measured before the sweep started writing counts_npz")
    path = Path(named["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        pytest.skip(f"map {path} is not available on this machine")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == named["sha256"], (
        f"{model}/{cell}/{level}map {path} changed after running.")


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_level_scored_the_whole_test_split(model, cell, level):
    s = _summary(model, cell, level)
    assert s["n"] == 2000, f"{model}/{cell}/{level}: n={s['n']}It should be 2000."
    cfg = _config(model, cell)
    # The queue passes it through the environment, so it lands as a string.
    assert int(cfg["n_examples"]) == 2000
    assert str(cfg["data"]).endswith("test_canonical_n2000.jsonl"), (
        f"{model}/{cell}: {cfg['data']}")


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_ceiling_is_the_wide_one(model, cell, level):
    """Native ceilings threw away nineteen levels last time.

    gpt-oss reasons again once pruned and needs 4096; Qwen answers in tens of
    tokens and gets 512, which is twelve times its p99.
    """
    s = _summary(model, cell, level)
    got = s["generation"]["max_new_tokens"]
    assert got == CEILING[model], (
        f"{model}/{cell}/{level}ceiling {got}expected {CEILING[model]}")


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_answers_were_kept(model, cell, level):
    """Two thousand rows of what the model actually said, beside the score.

    The truncation bug survived an entire run because the answers were not read;
    a number without them cannot be audited.
    """
    d = _level_dir(model, cell, level)
    if not (d / "summary.json").is_file():
        pytest.skip(f"level not measured yet: {model}/{cell}/{level}")
    rows = d / "results.jsonl"
    assert rows.is_file(), f"{model}/{cell}/{level}: no results.jsonl"
    assert sum(1 for _ in rows.open()) == 2000


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_mask_removed_the_right_number_of_experts_per_layer(model, cell, level):
    """Per-layer, not global: every layer loses exactly its share.

    A global mask takes the experts that are weak on average, and the seed-42
    maps showed that costs five to seventy times more traffic. If the count per
    layer drifts, the run is measuring the other policy.
    """
    d = _level_dir(model, cell, level)
    path = d / "pruned_experts.json"
    if not (d / "summary.json").is_file():
        pytest.skip(f"level not measured yet: {model}/{cell}/{level}")
    assert path.is_file(), f"{model}/{cell}/{level}: no pruned_experts.json"
    per_layer = json.loads(path.read_text())
    want = PRUNED[model][level]
    sizes = {len(v) for v in per_layer.values()}
    assert sizes == {want}, (
        f"{model}/{cell}/{level}knocked out {sorted(sizes)} experts on the layer, "
        f"expected {want}")
    # Every layer, not just the ones that happened to be written out.
    s = _summary(model, cell, level)
    assert len(per_layer) == s["mask_audit"]["layers_touched"]
    assert s["n_pruned_total"] == want * len(per_layer)


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_the_mask_took_the_least_used_experts(model, cell, level):
    """What was removed must be the quiet end of this cell's own map.

    Inverting the comparison keeps every shape and every count intact and
    removes the busiest experts instead. The quality then collapses for a
    reason that has nothing to do with the hypothesis, and nothing in the
    output says so.
    """
    d = _level_dir(model, cell, level)
    path = d / "pruned_experts.json"
    npz = MAPS / model / cell / "expert_counts.npz"
    if not (d / "summary.json").is_file() or not npz.is_file():
        pytest.skip(f"No level or map: {model}/{cell}/{level}")
    per_layer = json.loads(path.read_text())
    layers = [per_layer[str(i)] for i in range(len(per_layer))]

    with np.load(npz, allow_pickle=True) as bundle:
        key = next(k for k in bundle.files if k.endswith("|__all__"))
        counts = np.asarray(bundle[key], float)

    want = PRUNED[model][level]
    for layer, removed in enumerate(layers):
        order = np.argsort(counts[layer], kind="stable")
        quiet = set(order[:want].tolist())
        overlap = len(quiet & set(removed))
        # Ties at zero make the exact set ambiguous; the load carried by what
        # was removed is the honest comparison and must not exceed the load of
        # the quietest set of the same size.
        assert counts[layer][list(removed)].sum() <= counts[layer][order[:want]].sum(), (
            f"{model}/{cell}/{level} layer {layer}knocked out {overlap}/{want} from "
            "quiet, total loading of knocked out above the minimum")


@pytest.mark.parametrize("model,cell", CELLS)
@pytest.mark.parametrize("level", LEVELS)
def test_a_score_on_the_floor_is_explained(model, cell, level):
    """F1 0.330 is what silence scores, not what a model scores.

    A third of the test rows carry no labels, so an arm that answers nothing
    still reads 0.330. Such a level is only meaningful next to the share of
    empty and truncated answers, and this test makes the pairing mandatory.
    """
    s = _summary(model, cell, level)
    f1 = s.get("f1_mean")
    if f1 is None or f1 > 0.34:
        return
    empty = s.get("empty_pred_rate")
    assert empty is not None, (
        f"{model}/{cell}/{level}: F1 on the floor {f1}But there are no empty answers.")
    rescored = _rescored(model, cell, level)
    trunc = (rescored or {}).get("truncated_rate")
    assert empty > 0.5 or (trunc or 0) > 0.5, (
        f"{model}/{cell}/{level}: F1 {f1:.3f} on the floor, but empty. "
        f"{empty:.2f} circumcised {trunc} It's not silence, you know.")


def test_the_mask_never_saw_the_test_split():
    """The map is calibration-only; the score is test-only.

    Shared rows would leak the test set into the mask and the gain would be an
    artefact of the leak.
    """
    if not (CALIB.is_file() and TEST.is_file()):
        pytest.skip("no split")
    calib = {json.loads(l)["id"] for l in CALIB.open()}
    test = {json.loads(l)["id"] for l in TEST.open()}
    assert not (calib & test)
