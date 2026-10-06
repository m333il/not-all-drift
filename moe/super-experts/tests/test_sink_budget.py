"""The GPT-OSS sink-budget probe: its arithmetic, and its wiring on a tiny model.

The budget groups must add up to one for every query, the learned sink must be the
returned weights' deficit (GPT-OSS drops the sink column before returning them),
and both interventions must change what later layers see and then leave the model
exactly as they found it. CPU, tiny random model.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_sink_budget import (
    CONDITIONS, GROUPS, attention_budget, parse_experts, probe, residual_summary,
)
from test_two_prunings import LAYER, busiest


def tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(0)
    config = GptOssConfig(
        vocab_size=97, hidden_size=32, intermediate_size=16, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=8,
        num_experts_per_tok=2, max_position_embeddings=2048, sliding_window=4,
        experts_implementation="grouped_mm",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    return GptOssForCausalLM(config).to(torch.float32).eval()


def test_groups_sum_to_one_and_the_sink_is_the_deficit():
    weights = torch.tril(torch.rand(3, 6, 6) + 0.1)
    weights = weights / weights.sum(dim=-1, keepdim=True) * 0.7
    means, heads = attention_budget(weights, first_query=3)
    assert sum(means.values()) == pytest.approx(1.0, abs=1e-6)
    assert means["learned_sink"] == pytest.approx(0.3, abs=1e-6)
    assert len(heads["learned_sink"]) == 3 and len(heads["key_0"]) == 3


def test_queries_before_first_query_are_excluded():
    weights = torch.zeros(1, 5, 5)
    weights[0, :3, 0] = 1.0
    weights[0, 3:, 4] = 1.0
    means, _ = attention_budget(weights, first_query=3)
    assert means["key_0"] == 0.0
    assert means["rest"] == pytest.approx(1.0)
    assert means["learned_sink"] == 0.0


def test_residual_summary_separates_position_zero():
    summary = residual_summary([900.0, 2.0, 3.0, 50.0])
    assert summary["position_0"] == 900.0
    assert summary["max_elsewhere"] == 50.0 and summary["argmax_elsewhere"] == 3


def test_parse_experts():
    assert parse_experts("17:5,6:5") == {(17, 5), (6, 5)}


def test_probe_measures_three_conditions_and_restores_the_model():
    model = tiny_gpt_oss()
    ids = torch.arange(1, 13).unsqueeze(0)
    experts = {(LAYER, busiest(model, ids))}
    sequences = [list(range(1, 13)), list(range(20, 34))]
    result = probe(model, sequences, experts, first_query=3)
    rows = result["rows"]
    layers = model.config.num_hidden_layers
    assert len(rows) == len(CONDITIONS) * len(sequences) * layers
    for row in rows:
        assert sum(row[f"mass_{name}"] for name in GROUPS) == pytest.approx(1.0, abs=1e-4)
    assert any(row["mass_learned_sink"] > 1e-3 for row in rows), "the sink logit takes some mass"
    assert set(result["layer_types"]) == {"sliding_attention", "full_attention"}
    assert result["restoration"]["max_abs_difference"] == 0.0

    def seen(condition):
        return [row["residual_position_0"] for row in rows
                if row["condition"] == condition and row["layer"] == layers - 1]
    assert seen("zeroed") != seen("intact")
    assert seen("router_masked") != seen("intact")
    assert seen("zeroed") != seen("router_masked")
    summary = result["summary"]
    assert set(summary) == set(CONDITIONS)
    assert all(len(per_layer) == layers for per_layer in summary.values())
