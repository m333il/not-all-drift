import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.prefix_intervention import PrefixValueIntervention
from test_attention_contributions import wrapped, VIRTUAL


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_noop_and_restore_keep_greedy_cached_generation_exact(dtype):
    model, config = wrapped("prefix", seed=41)
    model.to(dtype)
    ids = torch.tensor([[3, 8, 2, 7]])
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        baseline = model.generate(**options)
        for mode in ("observe", "restore"):
            with PrefixValueIntervention(model, 1, mode=mode) as hook:
                actual = model.generate(**options)
            assert hook.calls == 4 and hook.decode_calls == 3
            assert torch.equal(baseline.sequences, actual.sequences)
            assert all(torch.equal(a, b) for a, b in zip(baseline.scores, actual.scores))
    assert not model.get_base_model().model.layers[1].self_attn._forward_hooks


@pytest.mark.parametrize("scope", ["all", "last"])
def test_zero_matches_independent_value_zeroing_and_keeps_attention(scope):
    model, config = wrapped("prefix", seed=42)
    ids = torch.tensor([[5, 3, 9, 7, 2]])
    attention = model.get_base_model().model.layers[1].self_attn
    with torch.no_grad():
        reference = model(input_ids=ids, use_cache=True, output_attentions=True)
        with PrefixValueIntervention(model, 1, mode="zero", scope=scope) as hook:
            intervened = model(input_ids=ids, use_cache=True, output_attentions=True)
            intervened_next = model.get_base_model()(input_ids=ids[:, :1],
                past_key_values=intervened.past_key_values, use_cache=True)
    assert torch.equal(reference.attentions[1], intervened.attentions[1])
    assert torch.equal(reference.past_key_values.layers[1].values,
                       intervened.past_key_values.layers[1].values[:, :, :-1])
    assert not torch.equal(reference.logits[:, -1], intervened.logits[:, -1])

    def independent(_module, _args, kwargs, output):
        weights = output[1]
        values = kwargs["past_key_values"].layers[1].values.clone()
        values[:, :, :VIRTUAL] = 0
        values = values.repeat_interleave(attention.num_key_value_groups, dim=1)
        pre = (weights @ values).transpose(1, 2).reshape(1, weights.shape[-2], -1)
        actual = attention.o_proj(pre)
        if scope == "last":
            actual[:, :-1] = output[0][:, :-1]
        return (actual, output[1])

    handle = attention.register_forward_hook(independent, with_kwargs=True)
    try:
        with torch.no_grad():
            expected = model(input_ids=ids, use_cache=True)
            expected_next = model.get_base_model()(input_ids=ids[:, :1],
                past_key_values=expected.past_key_values, use_cache=True)
    finally:
        handle.remove()
    assert torch.allclose(intervened.logits, expected.logits, atol=2e-6, rtol=2e-5)
    assert torch.allclose(intervened_next.logits, expected_next.logits, atol=2e-6, rtol=2e-5)
    if scope == "last":
        assert torch.equal(reference.logits[:, :-1], intervened.logits[:, :-1])


def test_constant_replaces_selected_output_in_prefill_and_decode():
    model, config = wrapped("prefix", seed=43)
    ids = torch.tensor([[8]])
    vector = torch.linspace(-0.1, 0.1, config.hidden_size)
    seen = []
    attention = model.get_base_model().model.layers[1].self_attn
    before = attention.register_forward_hook(lambda _m, _a, out: seen.append(out[0].clone()))
    try:
        with torch.no_grad(), PrefixValueIntervention(model, 1, mode="constant", vector=vector) as hook:
            first = model(input_ids=ids, use_cache=True)
            c1 = hook.last_contribution.clone()
            cached = model.get_base_model()(input_ids=ids, past_key_values=first.past_key_values, use_cache=True)
            c2 = hook.last_contribution.clone()
            assert hook.calls == 2 and hook.decode_calls == 1
        assert c1.shape == c2.shape == (1, config.hidden_size)
        assert torch.isfinite(cached.logits).all()
    finally:
        before.remove()
    captured = []
    with torch.no_grad(), PrefixValueIntervention(model, 1, mode="constant", vector=vector):
        after = attention.register_forward_hook(lambda _m, _a, out: captured.append(out[0].clone()))
        try:
            model(input_ids=ids, use_cache=True)
        finally:
            after.remove()
    assert torch.allclose(captured[0], seen[0] - c1 + vector, atol=1e-7)


def test_prompt_adapter_is_rejected():
    model, _ = wrapped("prompt", seed=44)
    with pytest.raises(ValueError, match="PREFIX_TUNING"):
        PrefixValueIntervention(model, 1)
