import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_prefill_confirmation import (BOOTSTRAP_DRAWS, CIVIL_SLICE, WIKI_SLICE, CONDITIONS,
    ROLE_MASKS, civil_summary, evaluate, measure, role_table)
from se_gepa.arms import load_contract
from se_gepa.prefill_template import PrefillTemplateShift
from test_attention_contributions import tiny_model


def test_civil_summary_matches_independent_ten_thousand_draw_bootstrap():
    scores = [.1, .7, 1., .3, .5, .9, .2, .6, .8, .4, .15, .85, .05, .95, .25, .75, .35]
    baselines = [.4, .2, .8, .4, .6, .1, .5, .2, .9, .3, .6, .2, .1, .4, .7, .8, .2]
    rows = [dict(key=i, score=value, valid=i % 3 != 0, finished=i % 4 != 0, truncated=i % 4 == 0,
                 completion_tokens=i + 1, first_token_kl_from_intact=i / 100) for i, value in enumerate(scores)]
    reference = {i: {'row': {'score': score}} for i, score in enumerate(baselines)}
    actual = civil_summary(rows, reference)
    deltas = torch.tensor([x - y for x, y in zip(scores, baselines)], dtype=torch.float64)
    indices = torch.randint(17, (10000, 17), generator=torch.Generator().manual_seed(42))
    expected = torch.quantile(deltas[indices].mean(1), torch.tensor([.025, .975], dtype=torch.float64))
    assert BOOTSTRAP_DRAWS == 10000
    assert actual['paired_bootstrap_ci95'] == expected.tolist()
    assert actual['paired_score_delta'] == float(deltas.mean())
    assert actual['improved'] == int((deltas > 0).sum()) and actual['worsened'] == int((deltas < 0).sum())
    assert actual['valid'] == sum(r['valid'] for r in rows) / 17
    assert actual['truncated'] == sum(r['truncated'] for r in rows) / 17


@pytest.mark.parametrize('task', ['civil', 'wiki'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_five_conditions_use_correct_init_table_and_preserve_inactive_middle(tmp_path, task, dtype):
    pytest.importorskip('sklearn')
    model, _ = tiny_model(seed=254)
    model.to(dtype)
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None
        def decode(self, _ids, skip_special_tokens=False):
            return 'NONE'
    tokenizer = Tokenizer()
    contract = load_contract()
    ids = [[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 10, 8, 9]]
    tables = {'trained': torch.randn(5, 32) * .1, 'init': torch.randn(5, 32) * .1}
    saved = {kind: table.clone() for kind, table in tables.items()}
    sources = [{'id': 'a', 'labels': []}, {'id': 'b', 'labels': []}]
    summary = evaluate(model, ids, sources, task, tokenizer, contract, tmp_path, tables,
                       layer=1, max_new_tokens=3, prefix_length=5, suffix_length=2)
    assert CONDITIONS == ['base', 'trained_full', 'init_full', 'trained_prefix_suffix', 'init_prefix_suffix']
    assert list(ROLE_MASKS.values()) == [0, 7, 7, 5, 5]
    assert [s['condition'] for s in summary] == CONDITIONS
    assert all(s['bootstrap_draws'] == 10000 for s in summary)
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    groups = {name: [row for row in rows if row['condition'] == name] for name in CONDITIONS}
    for row in rows:
        assert row['role_mask'] == ROLE_MASKS[row['condition']]
        if row['condition'] != 'base':
            middle_omitted = row['condition'].endswith('_prefix_suffix')
            assert row['inactive_positions'] == (row['prompt_tokens'] - 7 if middle_omitted else 0)
            assert row['active_positions'] == (4 if middle_omitted else row['prompt_tokens'] - 3)
            assert row['inactive_positions_exact'] and row['target_max_abs_error'] == 0
            assert row['first3_before_sha256'] == row['first3_after_sha256']
            assert row['intervention_calls'] == (3 if task == 'civil' else 1)
        if task == 'civil':
            assert row['score'] == 0 and row['parsed_score'] == 1 and row['truncated']
    # Distinct donor tables make using trained instead of init in the PS arm observable.
    with torch.no_grad(), PrefillTemplateShift(model, 1, len(ids[0]), role_table(tables['init'], 5, 5, 2), 5, 2):
        expected, _ = measure(model, ids[0], task, tokenizer, [], contract, 3)
    field = 'generation_scores_sha256' if task == 'civil' else 'logits_sha256'
    assert expected[field] == groups['init_prefix_suffix'][0][field]
    assert groups['init_prefix_suffix'][0][field] != groups['trained_prefix_suffix'][0][field]
    if task == 'wiki':
        for s in summary:
            delta = torch.tensor([r['nll'] - b['nll'] for r, b in zip(groups[s['condition']], groups['base'])], dtype=torch.float64)
            indices = torch.randint(2, (10000, 2), generator=torch.Generator().manual_seed(42))
            ci = torch.quantile(delta[indices].mean(1), torch.tensor([.025, .975], dtype=torch.float64))
            assert s['paired_bootstrap_ci95'] == ci.tolist()
            assert s['targets'] == sum(len(x) - 1 for x in ids)
    gates = json.loads((tmp_path / 'gates.json').read_text())
    assert sum(bool(g.get('zero_table_exact')) for g in gates) == (2 if task == 'civil' else 1)
    assert sum(bool(g.get('observation_logits_exact')) for g in gates) == 1
    assert all(torch.equal(tables[kind], saved[kind]) for kind in tables)
    assert not model.model.layers[1]._forward_hooks


def test_final_reserved_data_counts_and_prior_exclusions():
    root = Path(__file__).resolve().parents[1]
    validation, test = [[json.loads(line) for line in (root / 'data' / name).read_text().splitlines()]
        for name in ('civil_v2_val_seed42_n200.jsonl', 'civil_v2_test_head2000.jsonl')]
    assert CIVIL_SLICE == (200, 1000) and WIKI_SLICE == (96, 128)
    selected = test[CIVIL_SLICE[0]:CIVIL_SLICE[1]]
    assert len(selected) * len(CONDITIONS) == 4000
    assert len(range(*WIKI_SLICE)) * len(CONDITIONS) == 160
    for field in ('id', 'text'):
        values = {r[field] for r in selected}
        assert len(values) == 800
        assert not values & {r[field] for r in validation + test[:200] + test[1000:2000]}
