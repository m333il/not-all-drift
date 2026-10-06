import gzip
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_expert_channel import calibrate, diagnostics, evaluate, measure, table_context
from se_gepa.arms import load_contract
from se_gepa.expert_channel import ExpertChannelIntervention
from test_attention_contributions import tiny_model, wrapped, VIRTUAL


class Tokenizer:
    pad_token_id = 0
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=False):
        return 'NONE'


@pytest.mark.parametrize('task', ['civil', 'wiki'])
@pytest.mark.parametrize('arm', ['base', 'prefix', 'table'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_runner_calibration_generation_and_late_attention_are_paired(tmp_path, task, arm, dtype):
    if task == 'civil':
        pytest.importorskip('sklearn')
    model, config = wrapped('prefix', 71) if arm == 'prefix' else tiny_model(71)
    model.to(dtype)
    base = model.get_base_model() if arm == 'prefix' else model
    ids = [[2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 9, 10, 11, 12]]
    sources = [{'id': 'a', 'labels': []}, {'id': 'b', 'labels': []}]
    routing = []
    handle = base.model.layers[1].mlp.experts.register_forward_pre_hook(
        lambda module, args: routing.append(args[1].clone()))
    with torch.no_grad():
        base(input_ids=torch.tensor([ids[0]]), use_cache=False)
    handle.remove()
    expert = int(routing[0][0, 0])
    selection = calibrate(base, ids, sources, tmp_path, layer=1, expert=expert)
    assert json.loads((tmp_path / 'selection.json').read_text()) == selection
    assert selection['distinct_first3_ids'] == 1
    assert selection['channel'] == max(range(config.hidden_size), key=selection['energy'].__getitem__)
    assert selection['calibration_n'] == 2 and selection['observation_logits_exact']
    table = (torch.randn(4, config.hidden_size, generator=torch.Generator().manual_seed(83)) * .1
             if arm == 'table' else None)
    original_down = base.model.layers[1].mlp.experts.down_proj.detach().clone()
    prefix_keys = VIRTUAL if arm == 'prefix' else 0
    summary = evaluate(model, ids, sources, task, Tokenizer(), load_contract(), tmp_path,
        selection['channel'], table, prefix_keys, layer=1, expert=expert, readout_layers=[1, 2],
        table_layer=2, prefix_length=4, suffix_length=2, max_new_tokens=3)
    assert [r['condition'] for r in summary] == ['intact', 'remove', 'control']
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    with gzip.open(tmp_path / 'diagnostics.jsonl.gz', 'rt') as stream:
        saved_diags = [json.loads(line) for line in stream]
    assert [(r['key'], r['condition']) for r in rows] == [
        (s['id'], c) for s in sources for c in ('intact', 'remove', 'control')]
    assert len(saved_diags) == 6
    assert all(g['unhooked_exact'] and g['rescue_exact'] for g in json.loads((tmp_path / 'gates.json').read_text()))
    assert not base.model.layers[1].mlp._forward_pre_hooks
    assert not base.model.layers[2]._forward_hooks
    for row, diag in zip(rows, saved_diags):
        sequence = ids[0 if row['key'] == 'a' else 1]
        mode = 'observe' if row['condition'] == 'intact' else row['condition']
        assert row['l1_attention_exact']
        assert row['intervention_calls'] == (3 if task == 'civil' else 1)
        assert row['intervention']['positions'] == [0, 1, 2]
        assert row['intervention']['later_positions_exact'] and row['intervention']['weight_restored']
        assert all(a['real_queries'] == len(sequence) - 3 for a in diag['attention'])
        with torch.no_grad(), table_context(model, sequence, table, 2, 4, 2), \
                ExpertChannelIntervention(model, 1, expert, selection['channel'], mode=mode,
                                          length=len(sequence)):
            native = model(input_ids=torch.tensor([sequence]), use_cache=False, output_attentions=True)
        expected = native.attentions[2][0, :, 3:, prefix_keys:prefix_keys + 3].float().sum(-1).mean()
        assert row['early_key_mass_late_macro'] == pytest.approx(float(expected), abs=1e-6)
        if task == 'civil':
            assert row['truncated'] and row['score'] == 0 and row['parsed_score'] == 1
        else:
            assert row['targets'] == len(sequence) - 1
    for i, sequence in enumerate(ids):
        with table_context(model, sequence, table, 2, 4, 2):
            unhooked, _ = measure(model, sequence, task, Tokenizer(), [], load_contract(), 3)
        intact = rows[3 * i]
        if task == 'civil':
            assert unhooked['generation_scores_sha256'] == intact['generation_scores_sha256']
        else:
            assert unhooked['logits_sha256'] == intact['logits_sha256']
            assert unhooked['nll'] == intact['nll']
    assert torch.equal(original_down, base.model.layers[1].mlp.experts.down_proj)


def test_diagnostic_logits_remain_observational_in_bfloat16():
    model, _ = wrapped('prefix', 89)
    model.to(torch.bfloat16)
    sequence = [2, 3, 4, 5, 6, 7]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([sequence]), use_cache=False, logits_to_keep=1).logits
    diag, measured = diagnostics(model, sequence, 3, 'observe', None, VIRTUAL, 0,
                                 layer=1, expert=0, readout_layers=[1, 2])
    assert torch.equal(logits[0, -1].float(), measured)
    assert diag['attention'][0]['real_query_start'] == 3
    assert diag['intervention']['later_positions_exact']
