"""Guards for the final sixteen-cell map run.

The failures this file is written against are the ones that stay silent. A map
measured from the wrong checkpoint, an adapter that never attached, a stage
split computed against the wrong number of virtual tokens - all three produce a
well-formed ``expert_counts.npz`` whose numbers describe a different experiment.
Nothing downstream notices: the pruning mask is built, the sweep runs, the
quality table fills, and the result is wrong in a way no assertion catches.

So the checks here are about identity, not shape. Does this adapter come from
the run the provenance table names? Does this map differ from the base map, as
it must if the adapter was applied at all? Is the calibration set still disjoint
from the test set the F1 numbers come from?

Tests over artefacts that do not exist yet skip rather than fail: the file is
meant to be run repeatedly while the queue fills, and a missing map is a cell
not yet measured, not a defect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.artifacts

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / os.environ.get("FINAL_ADAPTERS", "adapters-final")
# Overridable so the same guards can be pointed at the frozen seed-42 run: a
# check that passes on known-good maps and fails on a broken one is worth more
# than a check that has only ever seen the run it was written for.
MAPS = ROOT / os.environ.get("FINAL_MAPS", "routing_maps_final")
PROVENANCE = ROOT / "arm-provenance.json"
CALIB = ROOT / "data" / "calib_val_n500.jsonl"
TEST = ROOT / "data" / "test_canonical_n2000.jsonl"

# The sixteen cells of the final set, as (model, cell). Two bases carry no
# adapter and two GEPA arms are prompts, so only twelve are checkpoints.
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

# Shape and routing width are fixed by the backbone, not by the arm.
GEOMETRY = {"gpt-oss": (24, 32, 4), "qwen": (48, 128, 8)}


def _provenance() -> dict[tuple[str, str], dict]:
    if not PROVENANCE.is_file():
        pytest.skip(f"no {PROVENANCE}")
    table = json.loads(PROVENANCE.read_text())["arms"]
    out = {}
    for row in table:
        model = "gpt-oss" if row["model"].startswith("gpt-oss") else "qwen"
        kind, m = row["arm"].split("-m")
        cell = f"{'prefix-projected' if kind == 'prefix' else kind}-m{m}-s42"
        out[(model, cell)] = row
    return out


def _map(model: str, cell: str) -> tuple[dict[str, np.ndarray], dict]:
    path = MAPS / model / cell / "expert_counts.npz"
    if not path.is_file():
        pytest.skip(f"map not measured yet: {model}/{cell}")
    with np.load(path, allow_pickle=True) as bundle:
        meta = json.loads(str(bundle["_meta"]))
        stages = {k.split("|", 1)[1]: np.asarray(bundle[k], float)
                  for k in bundle.files if k != "_meta"}
    return stages, meta


def test_the_final_set_is_sixteen_cells():
    assert len(CELLS) == 16
    assert len(ARMS) == 12
    # A GEPA arm that is not n1000 does not belong to the final set: the earlier
    # grid measured n200 and n500 too, and quietly carrying them forward would
    # make the note claim sixteen sections while rendering twenty.
    assert all("n1000" in cell for _, cell in GEPA)


# One arm's two sources disagree about which step was selected, and neither can
# be checked against the other from here. `arm-provenance.json` says 750; the
# archive's own `run/selection.json` says 1125, scoring it 0.8317 on validation
# against 0.8307 for 750. The published rule is that the archive names its own
# step, so that is what was extracted - but the test F1 in the table may belong
# to the other one. Recorded here rather than silently accepted.
STEP_DISPUTED = {
    ("qwen", "prefix-projected-m500-s42"): (
        "Archive selects step_001125 (val 0.8317), table names 750 "
        "(val 0.8307); which of these gave test F1 0.8070 is unknown "
        "artifactually"),
}


@pytest.mark.parametrize("model,cell", ARMS)
def test_an_adapter_comes_from_the_run_the_table_names(model, cell):
    """Family, cell and step must match the published provenance.

    Picking a neighbouring step is the quiet failure: the weights load, the map
    is measured, and the F1 in the table belongs to a checkpoint that was never
    profiled.
    """
    rows = _provenance()
    expected = rows[(model, cell)]
    local = ADAPTERS / model / cell / "provenance.json"
    if not local.is_file():
        pytest.skip(f"adapter not available: {model}/{cell}")
    got = json.loads(local.read_text())

    assert got["run_path"] == f"{expected['family']}/{expected['cell']}", (
        f"{model}/{cell}: {got['run_path']}, "
        f"table {expected['family']}/{expected['cell']}")

    want = f"step_{expected['selected_step']:06d}"
    if got["step"] != want and (model, cell) in STEP_DISPUTED:
        pytest.xfail(f"{model}/{cell}: {STEP_DISPUTED[(model, cell)]}")
    assert got["step"] == want, (
        f"{model}/{cell}step {got['step']}The table says {want}")


@pytest.mark.parametrize("model,cell", ARMS)
def test_the_adapter_has_weights_and_a_config(model, cell):
    d = ADAPTERS / model / cell
    if not d.is_dir():
        pytest.skip(f"adapter not available: {model}/{cell}")
    weights = d / "adapter_model.safetensors"
    config = d / "adapter_config.json"
    assert weights.is_file() and weights.stat().st_size > 0
    assert config.is_file()


@pytest.mark.parametrize("model,cell", ARMS)
def test_the_config_declares_the_virtual_tokens_its_name_promises(model, cell):
    """`m200` must mean two hundred virtual tokens.

    The stage split slices the prompt by position, so a config that disagrees
    with the arm's name silently attributes virtual positions to the comment and
    comment positions to the template.
    """
    config = ADAPTERS / model / cell / "adapter_config.json"
    if not config.is_file():
        pytest.skip(f"adapter not available: {model}/{cell}")
    declared = json.loads(config.read_text()).get("num_virtual_tokens")
    promised = int(cell.split("-m")[1].split("-")[0])
    assert declared == promised, (
        f"{model}/{cell}in configuration {declared} virtual tokens, "
        f"name {promised}")


def test_calibration_never_touches_the_test_split():
    """The map is measured on val500, the F1 on test2000, and they must not meet.

    Building the pruning mask from rows the arm is scored on would leak the test
    set into the mask, and the leak would show up as an unexplained gain.
    """
    if not (CALIB.is_file() and TEST.is_file()):
        pytest.skip("no split")
    calib = {json.loads(l)["id"] for l in CALIB.open()}
    test = {json.loads(l)["id"] for l in TEST.open()}
    assert len(calib) == 500
    assert not (calib & test), (
        f"{len(calib & test)} calibration lines are also in the test")


@pytest.mark.parametrize("model,cell", CELLS)
def test_all_equals_the_sum_of_its_stages(model, cell):
    """`__all__` is not a separate measurement, it is the total.

    Any drift between the two means a token was counted into a stage it does not
    belong to, or dropped entirely.
    """
    stages, _ = _map(model, cell)
    parts = [v for k, v in stages.items() if k != "__all__"]
    assert parts, f"{model}/{cell}There is no stage other than __all__"
    np.testing.assert_array_equal(
        stages["__all__"], np.sum(parts, axis=0),
        err_msg=f"{model}/{cell}__all___ is not the sum of the steps")


@pytest.mark.parametrize("model,cell", CELLS)
def test_the_map_has_the_geometry_of_its_backbone(model, cell):
    stages, meta = _map(model, cell)
    layers, experts, top_k = GEOMETRY[model]
    assert stages["__all__"].shape == (layers, experts)
    assert meta["top_k"] == top_k
    assert meta["num_experts"] == experts


@pytest.mark.parametrize("model,cell", CELLS)
def test_the_map_was_measured_on_the_calibration_split(model, cell):
    """Five hundred rows of val, contract v25 - the protocol of the frozen run.

    Measuring on a different split or contract makes the new maps
    incomparable with the seed-42 ones, which is the whole point of repeating
    the protocol.
    """
    _, meta = _map(model, cell)
    assert meta["n_examples"] == 500
    assert meta["data"].endswith("calib_val_n500.jsonl")
    assert meta["prompt_contract"] == "v25"


@pytest.mark.parametrize("model,cell", CELLS)
def test_every_layer_routed_something(model, cell):
    """A layer with no assignments means the hook missed it.

    A silently unhooked layer keeps its row at zero, and the pruning mask then
    treats every expert there as dead and removes all of them.
    """
    stages, _ = _map(model, cell)
    totals = stages["__all__"].sum(axis=1)
    assert (totals > 0).all(), (
        f"{model}/{cell}: layers without a single purpose: "
        f"{np.flatnonzero(totals == 0).tolist()}")


@pytest.mark.parametrize("model,cell", CELLS)
def test_counts_are_non_negative_whole_numbers(model, cell):
    stages, _ = _map(model, cell)
    for name, counts in stages.items():
        assert np.isfinite(counts).all(), f"{model}/{cell}/{name}not a number"
        assert (counts >= 0).all(), f"{model}/{cell}/{name}: negative counters"
        assert np.allclose(counts, np.round(counts)), (
            f"{model}/{cell}/{name}fractional counters")


@pytest.mark.parametrize("model,cell", ARMS)
def test_an_arm_routes_differently_from_its_base(model, cell):
    """If the adapter never attached, the arm's map equals the base's.

    PEFT loading is forgiving: a config it cannot apply leaves the model intact
    and raises nothing. The map is then the base map under an arm's name, and
    every drift number computed from it is zero by construction.
    """
    arm, _ = _map(model, cell)
    base, _ = _map(model, "base")
    assert not np.array_equal(arm["__all__"], base["__all__"]), (
        f"{model}/{cell}: the map matched the base by bit. "
        "The adapter was probably not used.")


@pytest.mark.parametrize("model,cell", ARMS)
def test_a_prompt_tuning_arm_reports_its_virtual_stage(model, cell):
    """Virtual tokens route too, and they must land in their own stage.

    An earlier run put them into the padding stage and the padding into theirs;
    the numbers looked plausible and were wrong. The guard here is cheap: a
    prompt-tuning arm has a virtual stage, and it is not empty.
    """
    stages, meta = _map(model, cell)
    if not cell.startswith("prompt-"):
        pytest.skip("prefix tuning adds no positions to the input")
    assert meta["n_virtual"] == int(cell.split("-m")[1].split("-")[0])
    virtual = [v for k, v in stages.items() if "virtual" in k]
    assert virtual, f"{model}/{cell}There is no stage of virtual tokens"
    assert virtual[0].sum() > 0, f"{model}/{cell}: The virtual stage is empty"


@pytest.mark.parametrize("model,cell", CELLS)
def test_the_calibration_answers_sit_beside_the_map(model, cell):
    """Five hundred answers, one per calibration row.

    Without them a surprising count cannot be traced back to what the model
    actually said, and that is how the truncation bug survived a whole run.
    """
    path = MAPS / model / cell / "expert_counts.npz"
    if not path.is_file():
        pytest.skip(f"map not measured yet: {model}/{cell}")
    answers = MAPS / model / f"{cell}.answers.jsonl"
    assert answers.is_file(), f"{model}/{cell}: no answer file"
    assert sum(1 for _ in answers.open()) == 500
