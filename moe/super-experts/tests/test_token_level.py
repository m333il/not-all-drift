"""The token-level pass has to report the position an expert actually fired at
and the gate's own score at that position, for the same forward pass.

Run on CPU against a tiny random model: this checks the wiring between the
profiler's traces and the router probe, not any claim about real Super Experts.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.criterion import SuperExpert
from se_gepa.profiler import FusedExpertProfiler
from se_gepa.router import RouterScoreProbe
from test_profiler import tiny_model


def load_script():
    spec = importlib.util.spec_from_file_location(
        "profile_super_experts", ROOT / "scripts" / "profile_super_experts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_probe_reports_a_probability_per_position():
    model, config = tiny_model("grouped_mm", seed=11)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    with RouterScoreProbe(model, {(0, 2), (1, 5)}) as probe:
        probe.begin_example(0)
        with torch.no_grad():
            model(input_ids=input_ids)
    assert {(row.layer, row.expert) for row in probe.observations} == {(0, 2), (1, 5)}
    for row in probe.observations:
        assert len(row.probability) == input_ids.shape[1]
        assert len(row.selected) == len(row.rank) == input_ids.shape[1]
        assert all(0.0 <= value <= 1.0 for value in row.probability)
        assert all(0 <= value < config.num_experts for value in row.rank)
        # Rank 0 is the largest probability, so selection must hold wherever the
        # expert ranks inside top-k.
        for rank, selected in zip(row.rank, row.selected):
            assert selected == (rank < config.num_experts_per_tok)


def test_token_level_pass_agrees_with_the_trace_it_summarises():
    module = load_script()
    model, config = tiny_model("grouped_mm", seed=12)
    generator = torch.Generator().manual_seed(12)
    segments = torch.randint(0, config.vocab_size, (3, 10), generator=generator)
    with FusedExpertProfiler(model) as profiler:
        for index in range(segments.shape[0]):
            profiler.begin_example(index, segments[index].tolist())
            with torch.no_grad():
                model(input_ids=segments[index : index + 1])
    records = profiler.records
    busiest = max(records, key=lambda key: records[key].output_max)
    super_experts = [SuperExpert(layer=busiest[0], expert=busiest[1],
                                 output_max=records[busiest].output_max, rank=1)]

    rows = module.token_level_pass(model, segments, super_experts, count=3)
    assert len(rows) == 1
    row = rows[0]
    assert (row["layer"], row["expert"]) == busiest
    assert row["segments"], "the expert was routed to at least once"
    for entry in row["segments"]:
        assert entry["activation_at_max"] >= (entry["activation_max_elsewhere"] or 0.0)
        assert 0.0 <= entry["router_probability_first"] <= 1.0
        assert entry["positions_routed"] >= 1
    # The corpus maximum must show up as one segment's own maximum.
    assert max(entry["activation_at_max"] for entry in row["segments"]) == pytest.approx(
        records[busiest].output_max, rel=1e-6)
    assert 0.0 <= row["argmax_at_first_position_rate"] <= 1.0


def test_token_level_pass_is_skipped_without_super_experts():
    module = load_script()
    model, _config = tiny_model("grouped_mm", seed=13)
    assert module.token_level_pass(model, torch.zeros(1, 4, dtype=torch.long), [], count=4) is None
