from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from se_gepa.prefill_template import PrefillTemplateShift, materialize_table
from test_attention_contributions import tiny_model, wrapped


@pytest.mark.parametrize("length", [8, 9, 12])
def test_materialize_prefix_middle_and_end_alignment(length):
    table = torch.arange(10).reshape(5, 2).float()
    actual = materialize_table(table, length, prefix_length=5, suffix_length=2)
    indices = [0, 1] + [2] * (length - 7) + [3, 4]
    assert torch.equal(actual, table[indices])
    assert actual.shape == (length - 3, 2)
    assert torch.equal(actual[-2:], table[-2:])


def test_materialize_default_geometry_and_empty_static_segments():
    table = torch.arange(76).float()[:, None]
    actual = materialize_table(table, 80)
    assert torch.equal(actual[:41], table[:41])
    assert torch.equal(actual[41:43], table[41:42].expand(2, 1))
    assert torch.equal(actual[-34:], table[-34:])
    assert torch.equal(materialize_table(torch.tensor([[7.]]), 6, 3, 0), torch.full((3, 1), 7.))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_identity_and_cached_generation_matches_fixed_length_replay(dtype):
    model, _ = tiny_model(seed=143)
    model.to(dtype)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    table = torch.randn(4, 32) * .1
    options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4,
                   do_sample=False, eos_token_id=None, pad_token_id=0, use_cache=True,
                   return_dict_in_generate=True, output_scores=True)
    with torch.no_grad():
        native = model.generate(**options)
        with PrefillTemplateShift(model, 1, 8, torch.zeros_like(table), 4, 2) as zero:
            identity = model.generate(**options)
        assert torch.equal(native.sequences, identity.sequences)
        assert all(torch.equal(a, b) for a, b in zip(native.scores, identity.scores))
        assert torch.equal(zero.before, zero.after)
        assert zero.calls == 4 and zero.patched_positions == 5
        with PrefillTemplateShift(model, 1, 8, table, 4, 2) as shift:
            cached = model.generate(**options)
        assert shift.calls == 4 and shift.patched_positions == 5
        assert not torch.equal(native.scores[0], cached.scores[0])
        assert torch.equal(shift.before[:3], shift.after[:3])
        expected = (shift.before[3:] + table[[0, 1, 1, 2, 3]]).to(dtype).float()
        assert torch.equal(shift.after[3:], expected)
        assert shift.before.dtype == shift.after.dtype == torch.float32
        assert shift.before.device.type == shift.after.device.type == "cpu"
        replay = ids.clone()
        for scores in cached.scores:
            with PrefillTemplateShift(model, 1, 8, table, 4, 2) as full:
                actual = model(input_ids=replay, use_cache=False).logits[:, -1]
            tolerance = .004 if dtype == torch.bfloat16 else 1e-6
            assert torch.allclose(actual.float(), scores.float(), atol=tolerance, rtol=tolerance)
            assert full.patched_positions == 5 and full.before.shape == (8, 32)
            replay = torch.cat((replay, scores.argmax(-1, keepdim=True)), dim=1)
        assert torch.equal(replay, cached.sequences)
    assert not model.model.layers[1]._forward_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_suffix_does_not_move_to_generated_tail_and_later_calls_are_untouched(dtype):
    model, _ = wrapped("prefix", seed=144)
    model.to(dtype)
    layer = model.get_base_model().model.layers[1]
    before, after = [], []

    def capture(target):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            target.append(hidden.detach().clone())
        return hook

    first = layer.register_forward_hook(capture(before))
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    table = torch.randn(4, 32) * .1
    try:
        with torch.no_grad(), PrefillTemplateShift(model, 1, 8, table, 4, 2) as shift:
            last = layer.register_forward_hook(capture(after))
            try:
                output = model(input_ids=ids, use_cache=True)
                model.get_base_model()(input_ids=torch.tensor([[12]]), past_key_values=output.past_key_values, use_cache=True)
                model(input_ids=ids, use_cache=False)
            finally:
                last.remove()
        assert shift.calls == 3 and shift.patched_positions == 5
        assert torch.equal(before[0][:, :3], after[0][:, :3])
        assert torch.equal(before[0][:, 8:], after[0][:, 8:])
        expected = (before[0][:, 3:8].float() + table[[0, 1, 1, 2, 3]]).to(dtype)
        assert torch.equal(after[0][:, 3:8], expected)
        assert not torch.equal(before[0][:, 6:8], after[0][:, 6:8])
        assert torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
    finally:
        first.remove()
    assert not layer._forward_hooks


def test_validation_and_cleanup_on_failure():
    model, _ = tiny_model(seed=145)
    table = torch.ones(4, 32)
    for length, prefix, suffix in [(6, 4, 2), (8, 2, 2), (8, 4, -1)]:
        with pytest.raises(ValueError, match="at least one middle"):
            materialize_table(table, length, prefix, suffix)
    with pytest.raises(ValueError, match="integers"):
        materialize_table(table, 8.0, 4, 2)
    with pytest.raises(ValueError, match="table shape"):
        materialize_table(torch.ones(3, 32), 8, 4, 2)
    with pytest.raises(ValueError, match="hidden size"):
        PrefillTemplateShift(model, 1, 8, torch.ones(4, 31), 4, 2)
    for ids, message in ((torch.tensor([[2, 3]]), "original prefill"),
                         (torch.tensor([[2, 3, 4], [2, 3, 4]]), "batch size one")):
        with pytest.raises(ValueError, match=message):
            with PrefillTemplateShift(model, 1, 8, table, 4, 2):
                model(input_ids=ids, use_cache=False)
        assert not model.model.layers[1]._forward_hooks
