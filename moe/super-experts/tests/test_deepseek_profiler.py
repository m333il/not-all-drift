"""The Super-Expert profiler on DeepSeek-V2's MoE block, and its shared-expert record.

DeepSeek-V2 keeps a dense first layer, routes over fused experts like Qwen3-MoE
and adds an ungated shared expert to every token. The profiler must skip the dense
layer, leave the forward unchanged under both experts backends, find an injected
outlier expert, and record the shared expert separately. CPU, tiny random model.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from se_gepa.criterion import identify, output_max_map
from se_gepa.profiler import FusedExpertProfiler, SharedExpertRecorder
from test_profiler import tiny_model as tiny_qwen


def tiny_deepseek(backend, seed=0):
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    torch.manual_seed(seed)
    config = DeepseekV2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=2, first_k_dense_replace=1,
        kv_lora_rank=16, q_lora_rank=None, qk_rope_head_dim=4, qk_nope_head_dim=8, v_head_dim=8,
        max_position_embeddings=64, experts_implementation=backend,
    )
    return DeepseekV2ForCausalLM(config).to(torch.float32).eval()


def run(model, ids, profiler=None):
    with torch.no_grad():
        if profiler is not None:
            profiler.begin_example(0, ids[0].tolist())
        return model(input_ids=ids).logits


@pytest.mark.parametrize("backend", ["eager", "grouped_mm"])
def test_profiler_skips_the_dense_layer_and_leaves_the_forward_unchanged(backend):
    model = tiny_deepseek(backend)
    ids = torch.arange(1, 13).unsqueeze(0)
    plain = run(model, ids)
    with FusedExpertProfiler(model, backend=backend) as profiler:
        instrumented = run(model, ids, profiler)
    assert profiler.layer_ids == [1, 2, 3]
    assert torch.allclose(plain, instrumented, atol=1e-5)
    assert profiler.records


def test_an_injected_outlier_expert_is_identified():
    model = tiny_deepseek("grouped_mm")
    ids = torch.arange(1, 13).unsqueeze(0)
    with FusedExpertProfiler(model, backend="grouped_mm") as profiler:
        run(model, ids, profiler)
    layer, expert = max(profiler.records, key=lambda key: profiler.records[key].hits)
    with torch.no_grad():
        model.model.layers[layer].mlp.experts.down_proj[expert] *= 1000.0
    with FusedExpertProfiler(model, backend="grouped_mm") as profiler:
        run(model, ids, profiler)
    found = identify(output_max_map(profiler.records), total_layers=4, include_fraction=1.0)
    assert (layer, expert) in {(row.layer, row.expert) for row in found}


def test_shared_expert_is_recorded_per_moe_layer():
    model = tiny_deepseek("grouped_mm")
    ids = torch.arange(1, 13).unsqueeze(0)
    with SharedExpertRecorder(model) as shared:
        run(model, ids)
    assert sorted(shared.records) == [1, 2, 3]
    assert all(record["max"] > 0 and 0 <= record["position"] < 12 for record in shared.records.values())


def test_models_without_a_shared_expert_record_nothing():
    model, _config = tiny_qwen("grouped_mm")
    ids = torch.arange(1, 13).unsqueeze(0)
    with SharedExpertRecorder(model) as shared:
        run(model, ids)
    assert shared.records == {}


def test_gate_token_level_pass_reads_deepseek_router_scores():
    sys.path.insert(0, str(ROOT / "scripts"))
    from profile_super_experts import token_level_pass

    model = tiny_deepseek("grouped_mm")
    segments = torch.randint(1, 97, (3, 12), generator=torch.Generator().manual_seed(0))
    with FusedExpertProfiler(model, backend="grouped_mm") as profiler:
        for index in range(3):
            profiler.begin_example(index, segments[index].tolist())
            with torch.no_grad():
                model(input_ids=segments[index:index + 1])
    found = identify(output_max_map(profiler.records), total_layers=4, include_fraction=1.0)
    rows = token_level_pass(model, segments, found, 2)
    assert rows and all("argmax_at_first_position_rate" in row for row in rows)
