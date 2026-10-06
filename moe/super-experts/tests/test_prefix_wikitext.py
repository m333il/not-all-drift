import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_prefix_wikitext import measure
from se_gepa.prefix_intervention import PrefixKeyMask
from test_attention_contributions import tiny_model, wrapped, VIRTUAL


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_key_mask_renormalizes_real_attention_and_preserves_cache_and_positions(dtype):
    model, _ = wrapped("prefix", seed=71)
    model.to(dtype)
    ids = torch.tensor([[5, 3, 9, 7, 2]])
    attention = model.get_base_model().model.layers[1].self_attn
    positions = []
    handle = attention.register_forward_pre_hook(
        lambda _m, _a, kw: positions.append(tuple(x.clone() for x in kw['position_embeddings'])), with_kwargs=True)
    try:
        with torch.no_grad():
            ref = model(input_ids=ids, use_cache=True, output_attentions=True)
            with PrefixKeyMask(model, 1) as hook:
                actual = model(input_ids=ids, use_cache=True, output_attentions=True)
        assert hook.calls == 1
    finally:
        handle.remove()
    assert all(torch.equal(a, b) for a, b in zip(*positions))
    assert torch.equal(ref.past_key_values.layers[1].keys, actual.past_key_values.layers[1].keys)
    assert torch.equal(ref.past_key_values.layers[1].values, actual.past_key_values.layers[1].values)
    weights = actual.attentions[1].float()
    assert torch.count_nonzero(weights[..., :VIRTUAL]) == 0
    reference = ref.attentions[1][..., VIRTUAL:].float()
    reference /= reference.sum(-1, keepdim=True)
    tolerance = .005 if dtype == torch.bfloat16 else 2e-7
    assert torch.allclose(weights[..., VIRTUAL:], reference, atol=tolerance, rtol=tolerance)
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=tolerance)
    assert torch.count_nonzero(weights[..., VIRTUAL:].triu(1)) == 0
    assert not torch.equal(ref.logits, actual.logits)
    assert not attention._forward_pre_hooks


@pytest.mark.parametrize("prefix", [False, True])
def test_paired_nll_and_mechanisms_runner(tmp_path, prefix):
    model, _ = wrapped("prefix", seed=72) if prefix else tiny_model(seed=72)
    tokens = torch.tensor([[3, 5, 9, 1, 4], [2, 8, 5, 6, 7]])
    summaries, gates = measure(model, tokens, [0, 1, 2], [(1, 2), (2, 3)], tmp_path)
    expected = ["intact", "zero_values", "mask_keys"] if prefix else ["intact"]
    assert [r['condition'] for r in summaries] == expected
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 2 * len(expected)
    for row in rows:
        assert row['targets'] == 4
        assert len(row['attention']) == len(row['residual']) == 3
        assert len(row['experts']) == 2
        assert row['sequence_sha256'] == rows[row['window']]['sequence_sha256']
        assert row['delta_nll'] == pytest.approx(row['nll'] - rows[row['window']]['nll'])
        if row['condition'] == 'mask_keys':
            assert all(sum(layer['prefix_mass_per_head']) == 0 for layer in row['attention'])
    for condition in expected:
        assert gates[condition]['observation_logits_exact']
    if prefix:
        assert gates['restore']['logits_exact']
    with torch.no_grad():
        output = model(input_ids=tokens[:1], use_cache=False)
        independently_scored = torch.nn.functional.cross_entropy(
            output.logits[0, :-1].float(), tokens[0, 1:])
    assert rows[0]['nll'] == pytest.approx(float(independently_scored), abs=1e-6)
