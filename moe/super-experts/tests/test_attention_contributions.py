"""Attention contribution wiring on tiny random Qwen3-MoE models."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.attention_contributions import AttentionContributionProbe, _conditional_virtual_stats

pytest.importorskip("peft")

VIRTUAL = 4


def tiny_model(seed=0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        max_position_embeddings=64, experts_implementation="eager", attn_implementation="eager",
    )
    return Qwen3MoeForCausalLM(config).to(torch.float32).eval(), config


def wrapped(kind, seed=0):
    from peft import PrefixTuningConfig, PromptTuningConfig, get_peft_model

    model, config = tiny_model(seed)
    if kind == "prompt":
        peft_config = PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=VIRTUAL)
    else:
        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM", num_virtual_tokens=VIRTUAL,
            prefix_projection=True, encoder_hidden_size=config.hidden_size,
        )
    return get_peft_model(model, peft_config).eval(), config


def measure(model, ids, *, layers=(1,), virtual_queries=0, prefix_keys=0, audit_vector_dims=0,
            reconstruction_rtol=1e-5):
    with torch.no_grad():
        baseline = model(input_ids=ids, use_cache=False).logits
    with AttentionContributionProbe(
        model, list(layers), virtual_query_tokens=virtual_queries, prefix_key_tokens=prefix_keys,
        audit_vector_dims=audit_vector_dims, reconstruction_rtol=reconstruction_rtol,
    ) as probe:
        probe.begin_example(7, ids.shape[1], "sequence")
        with torch.no_grad():
            instrumented = model(input_ids=ids, use_cache=False).logits
    return baseline, instrumented, probe.records


def assert_record(record, config, real_tokens, virtual_queries, prefix_keys):
    assert record["example"] == 7
    assert record["real_queries"] == real_tokens
    assert record["attention_heads"] == config.num_attention_heads
    assert record["key_value_heads"] == config.num_key_value_heads
    assert record["virtual_query_tokens"] == virtual_queries
    assert record["virtual_key_tokens"] == virtual_queries + prefix_keys
    for head in range(config.num_attention_heads):
        total = sum(record["per_head_attention_mass"][group][head]
                    for group in ("virtual", "real_0", "real_1", "real_2", "real_remaining"))
        assert total == pytest.approx(1.0, abs=1e-6)
    assert record["reconstruction"]["before_o_proj"]["relative_l2"] < 1e-6
    assert record["reconstruction"]["after_o_proj"]["relative_l2"] < 1e-6


def test_selected_layer_scope_gqa_and_noop_logits():
    model, config = tiny_model(seed=10)
    ids = torch.randint(0, config.vocab_size, (1, 9))
    baseline, instrumented, records = measure(model, ids, layers=(1,))
    assert torch.equal(baseline, instrumented)
    assert [row["layer"] for row in records] == [1]
    assert_record(records[0], config, ids.shape[1], 0, 0)
    assert records[0]["within_virtual"] is None


@pytest.mark.parametrize("kind", ["prompt", "prefix"])
def test_peft_virtual_geometry_and_real_query_alignment(kind):
    model, config = wrapped(kind, seed=11)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    virtual_queries = VIRTUAL if kind == "prompt" else 0
    prefix_keys = VIRTUAL if kind == "prefix" else 0
    baseline, instrumented, records = measure(
        model, ids, virtual_queries=virtual_queries, prefix_keys=prefix_keys,
    )
    assert torch.equal(baseline, instrumented)
    record = records[0]
    assert_record(record, config, ids.shape[1], virtual_queries, prefix_keys)
    assert record["within_virtual"] is not None
    assert len(record["within_virtual"]["conditional_entropy"]) == config.num_attention_heads

    with torch.no_grad():
        weights = model(input_ids=ids, use_cache=False, output_attentions=True).attentions[1][0]
    expected = weights[:, virtual_queries:, :VIRTUAL].float().sum(dim=-1).mean(dim=-1)
    assert record["per_head_attention_mass"]["virtual"] == pytest.approx(expected.tolist(), abs=1e-6)
    spans = {
        "real_0": (VIRTUAL, VIRTUAL + 1),
        "real_1": (VIRTUAL + 1, VIRTUAL + 2),
        "real_2": (VIRTUAL + 2, VIRTUAL + 3),
        "real_remaining": (VIRTUAL + 3, VIRTUAL + ids.shape[1]),
    }
    for group, (start, stop) in spans.items():
        expected = weights[:, virtual_queries:, start:stop].float().sum(dim=-1).mean(dim=-1)
        assert record["per_head_attention_mass"][group] == pytest.approx(expected.tolist(), abs=1e-6)


@pytest.mark.parametrize("real_tokens", [1, 2])
def test_short_real_sequences_partition_without_overlap(real_tokens):
    model, config = tiny_model(seed=15 + real_tokens)
    ids = torch.randint(0, config.vocab_size, (1, real_tokens))
    _baseline, _instrumented, records = measure(model, ids)
    record = records[0]
    assert_record(record, config, real_tokens, 0, 0)
    assert all(value == 0 for value in record["per_head_attention_mass"]["real_2"])
    assert all(value == 0 for value in record["per_head_attention_mass"]["real_remaining"])
    if real_tokens == 1:
        assert all(value == 0 for value in record["per_head_attention_mass"]["real_1"])


def test_full_group_mean_vectors_are_opt_in_and_bounded():
    model, config = tiny_model(seed=13)
    ids = torch.randint(0, config.vocab_size, (1, 5))
    captured = {}

    def capture(_module, _inputs, output):
        captured["attention"] = output[0].detach().clone()

    handle = model.model.layers[1].self_attn.register_forward_hook(capture)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    handle.remove()
    _baseline, _instrumented, records = measure(
        model, ids, audit_vector_dims=config.hidden_size,
    )
    summed = torch.stack([
        torch.tensor(summary["mean_vector"])
        for summary in records[0]["projected_contribution"].values()
    ]).sum(dim=0)
    assert torch.allclose(summed, captured["attention"][0].float().mean(dim=0), atol=1e-6, rtol=1e-6)
    for summary in records[0]["projected_contribution"].values():
        assert len(summary["mean_vector"]) == config.hidden_size
        assert summary["mean_squared_l2_about_mean"] >= 0
        assert len(summary["per_head_mean_vector_l2"]) == config.num_attention_heads


def test_zero_virtual_mass_has_null_conditional_statistics():
    stats = _conditional_virtual_stats(torch.zeros(3, 4, VIRTUAL))
    assert stats["eligible_real_queries"] == [0, 0, 0]
    for field in ("conditional_entropy", "conditional_effective_count",
                  "conditional_top1_mass", "conditional_top5_mass"):
        assert stats[field] == [None, None, None]


def test_bfloat16_reconstruction_roundtrip_on_cpu():
    model, config = tiny_model(seed=14)
    model = model.to(torch.bfloat16)
    ids = torch.randint(0, config.vocab_size, (1, 7))
    baseline, instrumented, records = measure(model, ids, reconstruction_rtol=0.02)
    assert torch.equal(baseline, instrumented)
    assert records[0]["reconstruction"]["after_o_proj"]["relative_l2"] < 0.02


def test_cached_decode_is_rejected_instead_of_misaligned():
    model, config = tiny_model(seed=12)
    ids = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        prefill = model(input_ids=ids, use_cache=True)
    with AttentionContributionProbe(model, [0]) as probe:
        probe.begin_example(0, 1)
        with pytest.raises(RuntimeError, match="Cached decoding is unsupported"):
            with torch.no_grad():
                model(input_ids=ids[:, :1], past_key_values=prefill.past_key_values, use_cache=True)


def test_non_qwen_is_rejected():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attn_implementation="eager",
    ))
    with pytest.raises(TypeError, match="Only Qwen3"):
        AttentionContributionProbe(model, [0])
