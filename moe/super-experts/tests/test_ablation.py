"""Pruning has to be exactly "this expert contributes nothing", and reversible.

The claim these tests protect is a causal one - that an arm still needs its Super
Experts, or no longer does - so the intervention must be the same operation
upstream performs (zero the down projection, do not renormalise routing) and must
leave the model bit-identical afterwards, or a second arm measured on the same
loaded weights would inherit the first one's damage.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.ablation import ExpertAblation
from se_gepa.profiler import FusedExpertProfiler
from se_gepa.residual import ResidualNormProbe, attention_sink_summary
from test_profiler import tiny_model


def busiest_expert(model, ids):
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, ids[0].tolist())
        with torch.no_grad():
            model(input_ids=ids)
    records = profiler.records
    return max(records, key=lambda key: records[key].output_max), records


def test_ablation_silences_the_expert_and_nothing_else():
    model, config = tiny_model("grouped_mm", seed=40)
    ids = torch.randint(0, config.vocab_size, (1, 12))
    target, before = busiest_expert(model, ids)
    with ExpertAblation(model, {target}):
        _, during = busiest_expert(model, ids)
    assert during[target].output_max == 0.0
    # Other experts in other layers keep their own maxima; the ablated layer's
    # neighbours may shift because the residual stream changed downstream, so the
    # invariant checked here is restricted to layers before the ablated one.
    earlier = [key for key in before if key[0] < target[0]]
    assert earlier
    for key in earlier:
        assert during[key].output_max == pytest.approx(before[key].output_max, rel=1e-6)


def test_ablation_restores_the_weights_exactly():
    model, config = tiny_model("grouped_mm", seed=41)
    ids = torch.randint(0, config.vocab_size, (1, 12))
    with torch.no_grad():
        baseline = model(input_ids=ids).logits.clone()
    target, _ = busiest_expert(model, ids)
    with ExpertAblation(model, {target}):
        with torch.no_grad():
            ablated = model(input_ids=ids).logits.clone()
    with torch.no_grad():
        restored = model(input_ids=ids).logits
    assert not torch.equal(baseline, ablated), "ablating the busiest expert changed nothing"
    assert torch.equal(baseline, restored)


def test_ablation_refuses_an_expert_outside_the_model():
    model, _config = tiny_model("grouped_mm", seed=42)
    with pytest.raises(ValueError, match="outside this model"):
        ExpertAblation(model, {(1, 9999)})


def test_residual_probe_reports_one_row_per_layer_and_position():
    model, config = tiny_model("grouped_mm", seed=43)
    ids = torch.randint(0, config.vocab_size, (1, 9))
    with ResidualNormProbe(model, layers=[0, 1]) as probe:
        probe.begin_example(0)
        with torch.no_grad():
            model(input_ids=ids)
    assert [row.layer for row in probe.records] == [0, 1]
    for row in probe.records:
        assert len(row.per_position) == ids.shape[1]
        assert all(value >= 0 for value in row.per_position)


def test_residual_probe_sees_an_ablation_downstream():
    model, config = tiny_model("grouped_mm", seed=44)
    ids = torch.randint(0, config.vocab_size, (1, 9))

    def norms():
        with ResidualNormProbe(model) as probe:
            probe.begin_example(0)
            with torch.no_grad():
                model(input_ids=ids)
        return {row.layer: row.per_position for row in probe.records}

    target, _ = busiest_expert(model, ids)
    before = norms()
    with ExpertAblation(model, {target}):
        during = norms()
    assert before[target[0]] != during[target[0]]


def test_attention_sink_summary_splits_prefix_from_real_keys():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(45)
    config = Qwen3MoeConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        max_position_embeddings=64, experts_implementation="grouped_mm",
        attn_implementation="eager",
    )
    model = Qwen3MoeForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 10))
    rows = attention_sink_summary(model, ids, layers=[0, 1], prefix_length=0)
    assert [row["layer"] for row in rows] == [0, 1]
    for row in rows:
        assert row["prefix_mass_mean"] == 0.0
        assert row["learned_sink_mass_mean"] == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= row["real_key_0_mass_mean"] <= 1.0
        assert 0 <= row["argmax_real_key_mode"] < ids.shape[1]
        assert 0.0 < row["argmax_real_key_share"] <= 1.0


def test_attention_sink_summary_recovers_gpt_oss_learned_sink_mass():
    from transformers import GptOssConfig, GptOssForCausalLM

    config = GptOssConfig(
        vocab_size=97, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_local_experts=8,
        num_experts_per_tok=2, max_position_embeddings=2048,
        experts_implementation="grouped_mm", attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    model = GptOssForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 10))
    rows = attention_sink_summary(model, ids, layers=[0, 1], prefix_length=0)
    for row in rows:
        assert 0.0 < row["learned_sink_mass_mean"] < 1.0
        assert row["real_mass_mean"] + row["learned_sink_mass_mean"] == pytest.approx(1.0)
        assert 0.0 <= row["learned_sink_is_largest_share"] <= 1.0


def test_attention_split_from_third_position_sums_to_one():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(46)
    config = Qwen3MoeConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        max_position_embeddings=64, experts_implementation="grouped_mm",
        attn_implementation="eager",
    )
    model = Qwen3MoeForCausalLM(config).to(torch.float32).eval()
    ids = torch.randint(0, config.vocab_size, (1, 12))
    for offset in (0, 4):
        for row in attention_sink_summary(model, ids, layers=[0, 1], prefix_length=0, offset=offset):
            total = (row["from3_virtual"] + row["from3_key0"] + row["from3_key1"] + row["from3_key2"]
                     + row["from3_rest"] + row["from3_learned_sink"])
            assert total == pytest.approx(1.0, abs=1e-5)
            assert row["from3_queries"] == 12 - offset - 3
            assert (row["from3_virtual"] == 0.0) == (offset == 0)
