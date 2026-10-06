"""The ported profiler must (a) not change the model's output and (b) record the
same numbers an independent recomputation gets, under both experts backends.

Everything here runs on CPU against a randomly initialised tiny Qwen3-MoE, so it
is a contract test for the port, not evidence about any real checkpoint. The
``grouped_mm`` path runs through transformers' CPU fallback kernel, which is the
same Python implementation the GPU takes with a different matmul underneath.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_gepa.criterion import SuperExpert, identify
from se_gepa.profiler import FusedExpertProfiler

BACKENDS = ["grouped_mm", "eager"]


def tiny_model(backend, seed=0):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        max_position_embeddings=64,
        experts_implementation=backend,
    )
    return Qwen3MoeForCausalLM(config).to(torch.float32).eval(), config


def moe_blocks(model):
    return {int(name.split("layers.")[1].split(".")[0]): module
            for name, module in model.named_modules()
            if getattr(module, "experts", None) is not None and hasattr(module, "gate")}


def reference_maxima(model, input_ids):
    """Independent recomputation: capture each MoE block's expert input and the
    router's choices, then evaluate every routed (expert, token) pair directly.
    """
    captured = {}

    def make_hook(layer):
        def hook(module, inputs, _output):
            hidden = inputs[0].reshape(-1, inputs[0].shape[-1])
            _logits, _weights, indices = module.gate(hidden)
            captured[layer] = (hidden.detach().clone(), indices.detach().clone())
        return hook

    blocks = moe_blocks(model)
    handles = [module.register_forward_hook(make_hook(layer)) for layer, module in blocks.items()]
    with torch.no_grad():
        model(input_ids=input_ids)
    for handle in handles:
        handle.remove()

    expected = {}
    for layer, (hidden, indices) in captured.items():
        experts = blocks[layer].experts
        for expert in range(experts.num_experts):
            rows = (indices == expert).any(dim=-1).nonzero().flatten()
            if rows.numel() == 0:
                continue
            with torch.no_grad():
                gate, up = torch.nn.functional.linear(
                    hidden[rows], experts.gate_up_proj[expert]).chunk(2, dim=-1)
                entry = experts.act_fn(gate) * up
                output = torch.nn.functional.linear(entry, experts.down_proj[expert])
            best = int(output.abs().amax(dim=-1).argmax())
            expected[(layer, expert)] = {
                "output_max": float(output.abs().max()),
                "input_max": float(entry.abs().max()),
                "hits": rows.numel(),
                "position": int(rows[best]),
                "channel": int(output[best].abs().argmax()),
            }
    return expected


@pytest.fixture(scope="module", params=BACKENDS)
def profiled(request):
    model, config = tiny_model(request.param)
    input_ids = torch.randint(0, config.vocab_size, (1, 24))
    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with torch.no_grad():
            instrumented = model(input_ids=input_ids).logits
    return model, input_ids, baseline, instrumented, profiler


def test_instrumented_forward_matches_the_model(profiled):
    _model, _ids, baseline, instrumented, profiler = profiled
    assert profiler.backend in BACKENDS
    assert torch.equal(baseline, instrumented)


def test_records_match_independent_recomputation(profiled):
    model, input_ids, _baseline, _instrumented, profiler = profiled
    expected = reference_maxima(model, input_ids)
    records = profiler.records
    assert set(records) == set(expected)
    ids = input_ids[0].tolist()
    for key, record in records.items():
        row = expected[key]
        assert record.output_max == pytest.approx(row["output_max"], rel=1e-6)
        assert record.input_max == pytest.approx(row["input_max"], rel=1e-6)
        assert record.hits == row["hits"]
        assert record.position == row["position"]
        assert record.channel == row["channel"]
        assert record.token_id == ids[row["position"]]


def test_both_backends_record_the_same_profile():
    profiles = {}
    for backend in BACKENDS:
        model, config = tiny_model(backend, seed=7)
        input_ids = torch.randint(0, config.vocab_size, (1, 20), generator=torch.Generator().manual_seed(7))
        with FusedExpertProfiler(model) as profiler:
            profiler.begin_example(0, input_ids[0].tolist())
            with torch.no_grad():
                model(input_ids=input_ids)
        profiles[backend] = profiler.records
    assert set(profiles["eager"]) == set(profiles["grouped_mm"])
    for key, record in profiles["eager"].items():
        other = profiles["grouped_mm"][key]
        assert record.output_max == pytest.approx(other.output_max, rel=1e-5)
        assert record.position == other.position
        assert record.hits == other.hits


def test_grouped_profiler_supports_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.manual_seed(11)
    config = GptOssConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        experts_implementation="grouped_mm",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    model = GptOssForCausalLM(config).to(torch.float32).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        baseline = model(input_ids=input_ids).logits
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with torch.no_grad():
            instrumented = model(input_ids=input_ids).logits

    assert torch.equal(baseline, instrumented)
    assert profiler.layer_ids == [0, 1, 2, 3]
    assert profiler.num_experts == 8
    assert profiler.records
    assert all(record.hits > 0 for record in profiler.records.values())


def test_maxima_accumulate_across_examples_and_keep_their_token():
    model, config = tiny_model("grouped_mm", seed=3)
    generator = torch.Generator().manual_seed(3)
    batches = [torch.randint(0, config.vocab_size, (1, 16), generator=generator) for _ in range(4)]
    per_example = []
    for ids in batches:
        with FusedExpertProfiler(model) as profiler:
            profiler.begin_example(0, ids[0].tolist())
            with torch.no_grad():
                model(input_ids=ids)
        per_example.append(profiler.records)
    with FusedExpertProfiler(model) as profiler:
        for index, ids in enumerate(batches):
            profiler.begin_example(index, ids[0].tolist())
            with torch.no_grad():
                model(input_ids=ids)
    corpus = profiler.records
    for key, record in corpus.items():
        seen = [index for index in range(len(batches)) if key in per_example[index]]
        best = max(seen, key=lambda index: per_example[index][key].output_max)
        assert record.output_max == pytest.approx(per_example[best][key].output_max, rel=1e-6)
        assert record.example == best
        assert record.position == per_example[best][key].position
        assert record.token_id == batches[best][0].tolist()[record.position]


@pytest.mark.parametrize("backend", BACKENDS)
def test_tracked_experts_get_a_per_token_trace(backend):
    model, config = tiny_model(backend, seed=1)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with FusedExpertProfiler(model) as discover:
        discover.begin_example(0, input_ids[0].tolist())
        with torch.no_grad():
            model(input_ids=input_ids)
    baseline = discover.records
    busiest = max(baseline, key=lambda key: baseline[key].hits)
    with FusedExpertProfiler(model, track={busiest}) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with torch.no_grad():
            model(input_ids=input_ids)
    trace = profiler.traces[0].values[busiest]
    assert len(trace) == baseline[busiest].hits
    assert trace[baseline[busiest].position] == pytest.approx(baseline[busiest].output_max, rel=1e-6)
    assert max(trace.values()) == pytest.approx(baseline[busiest].output_max, rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_profiler_restores_the_original_forward(backend):
    model, config = tiny_model(backend, seed=2)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    before = [module.experts.forward for module in moe_blocks(model).values()]
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with torch.no_grad():
            model(input_ids=input_ids)
    after = [module.experts.forward for module in moe_blocks(model).values()]
    assert [f.__func__ for f in before] == [f.__func__ for f in after]


@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_forward_is_refused(backend):
    model, config = tiny_model(backend, seed=3)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with pytest.raises(RuntimeError, match="batch size 1"):
            with torch.no_grad():
                model(input_ids=input_ids)


@pytest.mark.parametrize("backend", BACKENDS)
def test_generation_is_refused(backend):
    model, config = tiny_model(backend, seed=5)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with FusedExpertProfiler(model) as profiler:
        profiler.begin_example(0, input_ids[0].tolist())
        with pytest.raises(RuntimeError, match="one full sequence"):
            model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=False)


def test_forward_without_begin_example_is_refused():
    model, config = tiny_model("grouped_mm", seed=4)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with FusedExpertProfiler(model):
        with pytest.raises(RuntimeError, match="begin_example"):
            with torch.no_grad():
                model(input_ids=input_ids)


def test_profiling_a_different_backend_than_the_model_runs_is_refused():
    model, _config = tiny_model("grouped_mm", seed=6)
    with pytest.raises(ValueError, match="numbers the model did not produce"):
        FusedExpertProfiler(model, backend="eager")


def test_published_criterion_selects_the_extreme_tail():
    output_max = {(layer, expert): 1.0 + 0.01 * expert for layer in range(48) for expert in range(8)}
    output_max[(1, 3)] = 900.0
    output_max[(2, 5)] = 500.0
    output_max[(40, 0)] = 800.0  # outside the massive-activation prefix
    selected = identify(output_max, total_layers=48)
    assert selected == [
        SuperExpert(layer=1, expert=3, output_max=900.0, rank=1),
        SuperExpert(layer=2, expert=5, output_max=500.0, rank=2),
    ]


def test_ratio_test_uses_floor_division_like_upstream():
    # Upstream writes the ratio test as ``value > max // times``. With a maximum
    # of 105 that floor is 10, not 10.5, so 10.2 is a Super Expert under the
    # published criterion and would not be under a tidied-up ``max / times``.
    output_max = {(0, expert): 1.0 for expert in range(401)}
    output_max[(0, 0)] = 105.0
    output_max[(0, 1)] = 10.2
    assert [row.expert for row in identify(output_max, total_layers=4)] == [0, 1]
