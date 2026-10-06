from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.prefill_shift import PrefillShift
from test_attention_contributions import tiny_model, wrapped


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_identity_and_cached_generation_matches_fixed_prefill_replay(dtype):
    model, _ = tiny_model(seed=133)
    model.to(dtype)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    vector = torch.randn(32) * .1
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        native = model.generate(**options)
        with PrefillShift(model, 1, 5, torch.zeros(32)) as zero:
            identity = model.generate(**options)
        assert torch.equal(native.sequences, identity.sequences)
        assert all(torch.equal(a, b) for a, b in zip(native.scores, identity.scores))
        assert zero.calls == 4 and zero.patched_positions == 2
        assert torch.equal(zero.before, zero.after)
        with PrefillShift(model, 1, 5, vector) as shift:
            cached = model.generate(**options)
        assert shift.calls == 4 and shift.patched_positions == 2
        assert not torch.equal(native.scores[0], cached.scores[0])
        assert torch.equal(shift.before[:3], shift.after[:3])
        assert torch.equal(shift.after[3:], (shift.before[3:] + vector).to(dtype).float())
        assert shift.before.dtype == shift.after.dtype == torch.float32
        assert shift.before.device.type == shift.after.device.type == "cpu"
        replay = ids.clone()
        for scores in cached.scores:
            with PrefillShift(model, 1, 5, vector) as full:
                actual = model(input_ids=replay, use_cache=False).logits[:, -1]
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(actual.float(), scores.float(), atol=tolerance, rtol=tolerance)
            assert full.patched_positions == 2 and full.before.shape == (5, 32)
            replay = torch.cat((replay, scores.argmax(-1, keepdim=True)), dim=1)
        assert torch.equal(replay, cached.sequences)
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_only_original_positions_change_and_later_forwards_are_unpatched(dtype):
    model, _ = wrapped("prefix", seed=134)
    model.to(dtype)
    layer = model.get_base_model().model.layers[1]
    before, after = [], []

    def capture(target):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            target.append(hidden.detach().clone())
        return hook

    first = layer.register_forward_hook(capture(before))
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    vector = torch.linspace(-.11, .17, 32)
    try:
        with torch.no_grad(), PrefillShift(model, 1, 5, vector) as shift:
            last = layer.register_forward_hook(capture(after))
            try:
                output = model(input_ids=ids, use_cache=True)
                model.get_base_model()(input_ids=torch.tensor([[9]]), past_key_values=output.past_key_values, use_cache=True)
                model(input_ids=ids, use_cache=False)
            finally:
                last.remove()
        assert shift.calls == 3 and shift.patched_positions == 2
        assert torch.equal(before[0][:, :3], after[0][:, :3])
        assert torch.equal(before[0][:, 5:], after[0][:, 5:])
        expected = (before[0][:, 3:5].float() + vector).to(dtype)
        assert torch.equal(after[0][:, 3:5], expected)
        assert not torch.equal(before[0][:, 3:5], after[0][:, 3:5])
        assert torch.equal(before[1], after[1])
        assert torch.equal(before[2], after[2])
    finally:
        first.remove()
    assert not layer._forward_hooks


@pytest.mark.parametrize("length", [1, 3])
def test_short_prefill_is_identity(length):
    model, _ = tiny_model(seed=135)
    ids = torch.tensor([[2, 3, 4]])[:, :length]
    with torch.no_grad():
        native = model(input_ids=ids, use_cache=False).logits
        with PrefillShift(model, 1, length, torch.ones(32)) as shift:
            actual = model(input_ids=ids, use_cache=False).logits
    assert torch.equal(native, actual)
    assert torch.equal(shift.before, shift.after) and shift.patched_positions == 0


def test_shape_validation_and_cleanup_on_failure():
    model, _ = tiny_model(seed=136)
    with pytest.raises(ValueError, match="positive integer"):
        PrefillShift(model, 1, 0, torch.ones(32))
    with pytest.raises(ValueError, match="one vector"):
        PrefillShift(model, 1, 5, torch.ones(5, 32))
    for ids, message in ((torch.tensor([[2, 3]]), "original prefill"),
                         (torch.tensor([[2, 3, 4], [2, 3, 4]]), "batch size one")):
        with pytest.raises(ValueError, match=message):
            with PrefillShift(model, 1, 3, torch.ones(32)):
                model(input_ids=ids, use_cache=False)
        assert not model.model.layers[1]._forward_hooks
