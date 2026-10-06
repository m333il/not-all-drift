import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_prefill_roles import CONDITIONS, ROLE_MASKS, REVISION, TABLE_HASHES, evaluate, load_fitted, role_table
from se_gepa.prefill_template import PrefillTemplateShift, materialize_table
from se_gepa.arms import load_contract
from test_attention_contributions import tiny_model


@pytest.mark.parametrize('length', [8, 11])
def test_role_bits_expand_without_rescaling_or_suffix_motion(length):
    table = torch.arange(1, 11).reshape(5, 2).float()
    for mask in range(8):
        selected = role_table(table, mask, 5, 2)
        for bit, section in ((1, slice(0, 2)), (2, slice(2, 3)), (4, slice(3, 5))):
            assert torch.equal(selected[section], table[section] if mask & bit else torch.zeros_like(table[section]))
        full = materialize_table(selected, length, 5, 2)
        assert torch.equal(full[:2], selected[:2])
        assert torch.equal(full[2:-2], selected[2:3].expand(length - 7, 2))
        assert torch.equal(full[-2:], selected[-2:])
    assert torch.equal(role_table(table, 7, 5, 2), table)
    with pytest.raises(ValueError, match='Role mask'):
        role_table(table, 8, 5, 2)


def test_omitted_roles_leave_actual_bfloat16_states_exact():
    model, _ = tiny_model(seed=244)
    model.to(torch.bfloat16)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    table = torch.randn(5, 32) * .1
    with torch.no_grad():
        for mask in (1, 2, 4):
            selected = role_table(table, mask, 5, 2)
            with PrefillTemplateShift(model, 1, 8, selected, 5, 2) as hook:
                model(input_ids=ids, use_cache=False)
            addition = materialize_table(selected, 8, 5, 2)
            inactive = (addition == 0).all(-1)
            assert torch.equal(hook.before[:3], hook.after[:3])
            assert torch.equal(hook.before[3:][inactive], hook.after[3:][inactive])
            assert torch.equal(hook.after[3:], (hook.before[3:] + addition).bfloat16().float())


@pytest.mark.parametrize('task', ['civil', 'wiki'])
def test_native_runner_all_role_conditions_and_cap(tmp_path, task):
    pytest.importorskip('sklearn')
    model, _ = tiny_model(seed=245)
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None
        def decode(self, _ids, skip_special_tokens=False):
            return 'NONE'
    ids = [[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 10, 8, 9]]
    tables = {'trained': torch.randn(5, 32) * .1, 'init': torch.randn(5, 32) * .1}
    original = {k: v.clone() for k, v in tables.items()}
    summary = evaluate(model, ids, [{'id': 'a', 'labels': []}, {'id': 'b', 'labels': []}], task,
        Tokenizer(), load_contract(), tmp_path, tables, layer=1, max_new_tokens=3, prefix_length=5, suffix_length=2)
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 18 and [row['condition'] for row in summary] == CONDITIONS
    for row in rows:
        mask = ROLE_MASKS[row['condition']]
        assert row['role_mask'] == mask
        if mask:
            expected_active = (2 if mask & 1 else 0) + (row['prompt_tokens'] - 7 if mask & 2 else 0) + (2 if mask & 4 else 0)
            assert row['active_positions'] == expected_active
            assert row['inactive_positions'] == row['prompt_tokens'] - 3 - expected_active
            assert row['inactive_positions_exact'] and row['target_max_abs_error'] == 0
            assert row['first3_before_sha256'] == row['first3_after_sha256']
        if task == 'civil':
            assert row['score'] == 0 and row['parsed_score'] == 1 and row['truncated']
    gates = json.loads((tmp_path / 'gates.json').read_text())
    assert sum(bool(g.get('zero_table_exact')) for g in gates) == (2 if task == 'civil' else 1)
    assert sum(bool(g.get('observation_logits_exact')) for g in gates) == 1
    assert all(torch.equal(tables[k], original[k]) for k in tables)
    assert not model.model.layers[1]._forward_hooks


def test_frozen_objects_authenticate_and_modified_file_is_rejected(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / 'data/prefill_template_frozen_20260924'
    tables, provenance, template, calibration, hashes = load_fitted(source, REVISION)
    assert all(table.shape == (76, 2048) for table in tables.values())
    assert provenance['origin_status'] == 'VERIFIED' and len(calibration) == 64
    assert template['prefix_length'] == 44 and template['suffix_length'] == 34
    assert set(TABLE_HASHES) == set(tables) and len(hashes) == 6
    copied = tmp_path / 'fitted'
    shutil.copytree(source, copied)
    path = copied / 'template.json'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='file hash mismatch'):
        load_fitted(copied, REVISION)
    with pytest.raises(ValueError, match='provenance mismatch'):
        load_fitted(source, 'wrong-revision')


def test_fresh_role_confirmation_rows_are_disjoint():
    root = Path(__file__).resolve().parents[1]
    validation, test = [[json.loads(line) for line in (root / 'data' / name).read_text().splitlines()]
        for name in ('civil_v2_val_seed42_n200.jsonl', 'civil_v2_test_head2000.jsonl')]
    for field in ('id', 'text'):
        values = {r[field] for r in test[1000:1200]}
        assert len(values) == 200
        assert not values & {r[field] for r in validation + test[:200] + test[1200:2000]}
