from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.prefill_patch import PrefillStatePatch
from test_attention_contributions import tiny_model, wrapped


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scope", ["all", "after3", "last"])
def test_cached_generation_matches_fixed_length_uncached_replay(dtype, scope):
    model, _ = tiny_model(seed=103)
    model.to(dtype)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    length = ids.shape[1]
    donor = torch.randn(length, 32) * .1
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        native = model.generate(**options)
        with PrefillStatePatch(model, 1, length, scope, donor) as patch:
            cached = model.generate(**options)
        assert patch.calls == 4
        assert patch.patched_positions == {"all": 5, "after3": 2, "last": 1}[scope]
        assert not torch.equal(native.scores[0], cached.scores[0])
        replay = ids.clone()
        for scores in cached.scores:
            with PrefillStatePatch(model, 1, length, scope, donor) as full:
                logits = model(input_ids=replay, use_cache=False).logits[:, -1]
            assert full.before.shape == (length, 32)
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(logits.float(), scores.float(), atol=tolerance, rtol=tolerance)
            replay = torch.cat((replay, scores.argmax(-1, keepdim=True)), dim=1)
        assert torch.equal(replay, cached.sequences)
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_observation_zero_alpha_and_exact_interpolation(dtype):
    model, _ = wrapped("prefix", seed=104)
    model.to(dtype)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    donor = torch.randn(5, 32)
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        native = model.generate(**options)
        with PrefillStatePatch(model, 1, 5) as observer:
            observed = model.generate(**options)
        with PrefillStatePatch(model, 1, 5, donor=donor, alpha=0) as identity:
            zero = model.generate(**options)
        for result in (observed, zero):
            assert torch.equal(native.sequences, result.sequences)
            assert all(torch.equal(a, b) for a, b in zip(native.scores, result.scores))
        assert observer.calls == identity.calls == 3
        assert observer.patched_positions == 0 and observer.after is None
        assert identity.patched_positions == 5
        assert torch.equal(observer.before, identity.before)
        assert torch.equal(identity.before, identity.after)
        assert observer.before.dtype == torch.float32 and observer.before.device.type == "cpu"
        for alpha in (.25, 1):
            with PrefillStatePatch(model, 1, 5, "after3", donor, alpha) as patch:
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
            expected = ((1 - alpha) * patch.before[3:] + alpha * donor[3:]).to(dtype).float()
            assert torch.equal(patch.after[:3], patch.before[:3])
            assert torch.equal(patch.after[3:], expected)
    assert not model.get_base_model().model.layers[1]._forward_hooks


def test_donor_exact_and_current_layer_cache_boundary():
    teacher, _ = wrapped("prefix", seed=105)
    base = teacher.get_base_model()
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    with torch.no_grad():
        with PrefillStatePatch(teacher, 1, 5) as capture:
            teacher(input_ids=ids, use_cache=False, logits_to_keep=1)
        native = base(input_ids=ids, use_cache=True)
        with PrefillStatePatch(base, 1, 5, donor=capture.before) as patch:
            changed = base(input_ids=ids, use_cache=True)
        assert torch.equal(patch.after, capture.before)
        assert not torch.equal(native.logits, changed.logits)
        for layer in (0, 1):
            for field in ("keys", "values"):
                assert torch.equal(getattr(native.past_key_values.layers[layer], field),
                                   getattr(changed.past_key_values.layers[layer], field))
        for field in ("keys", "values"):
            assert not torch.equal(getattr(native.past_key_values.layers[2], field),
                                   getattr(changed.past_key_values.layers[2], field))
    assert not base.model.layers[1]._forward_hooks


def test_input_validation_and_hook_cleanup_on_failure():
    model, _ = tiny_model(seed=106)
    with pytest.raises(ValueError, match="positive integer"):
        PrefillStatePatch(model, 1, 0)
    with pytest.raises(ValueError, match="scope"):
        PrefillStatePatch(model, 1, 3, "unknown")
    with pytest.raises(ValueError, match="donor shape"):
        PrefillStatePatch(model, 1, 3, donor=torch.zeros(2, 32))
    for ids, message in ((torch.tensor([[2, 3]]), "original prefill"),
                         (torch.tensor([[2, 3, 4], [2, 3, 4]]), "batch size one")):
        with pytest.raises(ValueError, match=message):
            with PrefillStatePatch(model, 1, 3):
                model(input_ids=ids, use_cache=False)
        assert not model.model.layers[1]._forward_hooks


def test_after3_short_prefill_is_identity():
    model, _ = tiny_model(seed=107)
    ids = torch.tensor([[2, 3]])
    with torch.no_grad():
        native = model(input_ids=ids, use_cache=False).logits
        with PrefillStatePatch(model, 1, 2, "after3", torch.randn(2, 32)) as patch:
            actual = model(input_ids=ids, use_cache=False).logits
    assert patch.patched_positions == 0
    assert torch.equal(native, actual)
    assert torch.equal(patch.before, patch.after)
