import json

import pytest
import torch
import torch.nn.functional as F

from test_attention_contributions import tiny_model, wrapped
from se_gepa.expert_channel import EarlyExpertContribution, ExpertChannelIntervention


IDS = torch.tensor([[2, 3, 4, 5, 6]])


def experts(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.layers[1].mlp.experts


def inputs(dtype):
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(7, 32, generator=generator).to(dtype)
    indices = torch.tensor([[1, 2], [2, 3], [1, 3], [1, 2], [2, 3], [1, 3], [2, 3]])
    weights = torch.tensor([[.75, .25]] * 7, dtype=dtype)
    return hidden, indices, weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_replay_exact_locality_and_restoration(dtype):
    model, _ = tiny_model()
    model.to(dtype)
    module = experts(model)
    args = inputs(dtype)
    with torch.no_grad():
        native = module(*args)
        saved = module.down_proj.clone()
        module.down_proj[1, 7].zero_()
        masked = module(*args)
        module.down_proj.copy_(saved)
    with ExpertChannelIntervention(model, 1, 1, 7, length=5) as hook:
        result = module(*args)
        later = module(*args)
    expected = native.clone()
    expected[:3, 7] = masked[:3, 7]
    assert torch.equal(result, expected)
    assert torch.equal(later, native)
    assert torch.equal(module.down_proj, saved)
    assert hook.calls == 2 and hook.patched_positions == 3 and hook.weight_restored
    assert hook.audit["routing"]["hit"] == [True, False, True]
    assert hook.audit["other_channels_exact"] and hook.audit["unrouted_rows_exact"]
    assert hook.audit["later_positions_exact"]
    assert "forward" not in module.__dict__
    json.dumps(hook.audit, allow_nan=False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_control_actual_rounded_direction_and_seed(dtype):
    model, _ = tiny_model()
    model.to(dtype)
    module, args = experts(model), inputs(dtype)
    with torch.no_grad():
        native = module(*args)
    with ExpertChannelIntervention(model, 1, 1, 7, "control", length=5) as hook:
        result = module(*args)
    scalar = torch.tensor(hook.audit["signed_contribution"])
    expected = native.clone()
    expected[:3] = (native[:3].float() - scalar[:, None] * hook.direction).to(dtype)
    assert torch.equal(expected, result)
    assert hook.direction[7] == 0
    assert hook.direction.norm().item() == pytest.approx(1, abs=1e-7)
    assert torch.equal(result[:, 7], native[:, 7])
    assert torch.equal(hook.direction, ExpertChannelIntervention(model, 1, 1, 7, length=5).direction)
    actual_norms = (result[:3].float() - native[:3].float()).norm(dim=-1)
    torch.testing.assert_close(torch.tensor(hook.audit["realized_norm"]), actual_norms)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("mode", ["observe", "rescue"])
def test_identity_and_observer_noop_including_prefix(prefix, mode):
    model, _ = wrapped("prefix") if prefix else tiny_model()
    with torch.no_grad():
        plain = model(input_ids=IDS, use_cache=False).logits
        with ExpertChannelIntervention(model, 1, 1, 7, mode, length=5) as hook:
            actual = model(input_ids=IDS, use_cache=False).logits
        with EarlyExpertContribution(model, 1, 1, 5) as observer:
            observed = model(input_ids=IDS, use_cache=False).logits
    assert torch.equal(plain, actual) and torch.equal(plain, observed)
    assert hook.calls == observer.calls == 1
    assert observer.contribution.shape == (3, 32)
    assert observer.contribution.dtype == torch.float32
    assert observer.contribution.device.type == "cpu"
    assert observer.audit["finite"]
    assert not experts(model)._forward_hooks
    assert not model.get_base_model().model.layers[1].mlp._forward_pre_hooks if prefix else not model.model.layers[1].mlp._forward_pre_hooks


def test_observer_matches_selected_expert_manual_output_and_nonrouted_zero():
    model, _ = tiny_model()
    module, args = experts(model), inputs(torch.float32)
    with torch.no_grad(), EarlyExpertContribution(model, 1, 1, 5) as observer:
        module(*args)
        gate, up = F.linear(args[0][:3], module.gate_up_proj[1]).chunk(2, -1)
        projected = F.linear(module.act_fn(gate) * up, module.down_proj[1])
        expected = projected * torch.tensor([.75, 0, .75])[:, None]
        torch.testing.assert_close(observer.contribution, expected, rtol=0, atol=0)
        original_capture = observer.contribution.clone()
        module(*args)
    assert torch.equal(observer.contribution, original_capture)
    assert observer.calls == 2


@pytest.mark.parametrize("cache", [False, True])
def test_only_first_forward_is_patched_during_generate(cache):
    model, _ = tiny_model()
    module = experts(model)
    outputs = []
    handle = module.register_forward_hook(lambda m, a, out: outputs.append(out.detach().clone()))
    with torch.no_grad(), ExpertChannelIntervention(model, 1, 1, 7, length=5) as hook:
        result = model.generate(input_ids=IDS, attention_mask=torch.ones_like(IDS),
                                max_new_tokens=3, do_sample=False, eos_token_id=None,
                                pad_token_id=0, use_cache=cache)
    handle.remove()
    assert hook.calls == 3 and hook.patched_positions == 3
    assert [len(x) for x in outputs] == ([5, 1, 1] if cache else [5, 6, 7])
    # Replay each later call with the intervention absent; cached history may carry the first change.
    with torch.no_grad():
        for i in range(1, 3):
            replay = []
            handle = module.register_forward_hook(lambda m, a, out: replay.append(out.clone()))
            if cache:
                with ExpertChannelIntervention(model, 1, 1, 7, length=5):
                    previous = model(input_ids=IDS, use_cache=True)
                for j in range(1, i + 1):
                    previous = model(input_ids=result[:, 5+j-1:5+j],
                                     past_key_values=previous.past_key_values, use_cache=True)
            else:
                model(input_ids=result[:, :5+i], use_cache=False)
            handle.remove()
            torch.testing.assert_close(outputs[i], replay[-1], rtol=0, atol=0)


def test_transposed_physical_layout_targets_output_channel():
    model, _ = tiny_model()
    module = experts(model)
    args = inputs(torch.float32)
    with torch.no_grad():
        native = module(*args)
        module.gate_up_proj = torch.nn.Parameter(module.gate_up_proj.transpose(-1, -2).contiguous())
        module.down_proj = torch.nn.Parameter(module.down_proj.transpose(-1, -2).contiguous())
    module.is_transposed = True

    def transposed_backend(hidden, indices, weights):
        result = torch.zeros_like(hidden)
        for expert in range(module.num_experts):
            token, slot = torch.where(indices == expert)
            gate, up = F.linear(hidden[token], module.gate_up_proj[expert].T).chunk(2, -1)
            output = F.linear(module.act_fn(gate) * up, module.down_proj[expert].T)
            result.index_add_(0, token, output * weights[token, slot, None])
        return result

    module.forward = transposed_backend
    with torch.no_grad():
        transposed_native = module(*args)
    torch.testing.assert_close(transposed_native, native)
    saved = module.down_proj.detach().clone()
    with ExpertChannelIntervention(model, 1, 1, 7, length=5) as hook:
        result = module(*args)
    with EarlyExpertContribution(model, 1, 1, 5) as observer, torch.no_grad():
        unchanged = module(*args)
    assert torch.equal(unchanged, transposed_native)
    assert torch.equal(saved, module.down_proj)
    assert hook.audit["is_transposed"] and observer.audit["is_transposed"]
    assert torch.equal(result[:, :7], transposed_native[:, :7])
    assert not torch.equal(result[:3, 7], transposed_native[:3, 7])
    assert module.forward is transposed_backend


def test_exception_restores_weight_and_original_forward():
    model, _ = tiny_model()
    module, args = experts(model), inputs(torch.float32)
    original = module.forward
    saved = module.down_proj.detach().clone()
    calls = 0

    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("masked replay failed")
        return original(*args)

    module.forward = failing
    with pytest.raises(RuntimeError, match="masked replay failed"):
        with ExpertChannelIntervention(model, 1, 1, 7, length=5) as hook:
            module(*args)
    assert hook.weight_restored and torch.equal(saved, module.down_proj)
    assert module.forward is failing
    assert not model.model.layers[1].mlp._forward_pre_hooks


def test_batch_and_original_length_guards():
    model, _ = tiny_model()
    with pytest.raises(ValueError, match="batch size one"):
        with ExpertChannelIntervention(model, 1, 1, 7, length=5):
            model(input_ids=IDS.expand(2, -1))
    with pytest.raises(ValueError, match="original prefill"):
        with EarlyExpertContribution(model, 1, 1, 6):
            model(input_ids=IDS)


@pytest.mark.parametrize("bad_call,match", [(2, "unrelated"), (3, "not exact")])
def test_rejects_nonlocal_masked_replay_or_unstable_native(bad_call, match):
    model, _ = tiny_model()
    module, args = experts(model), inputs(torch.float32)
    original = module.forward
    saved = module.down_proj.detach().clone()
    calls = 0

    def corrupt(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == bad_call:
            result[0, 0] += 1
        return result

    module.forward = corrupt
    with pytest.raises(RuntimeError, match=match):
        with ExpertChannelIntervention(model, 1, 1, 7, length=5) as hook:
            module(*args)
    assert hook.weight_restored and torch.equal(saved, module.down_proj)
    assert module.forward is corrupt
