import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_wording_transfer import (ADAPTERS, BOOTSTRAP_DRAWS, CONDITIONS, REVISION, WEIGHT_HASHES,
    PrefillStatePatch, capture, check_teachers, evaluate, measure)
from se_gepa.arms import load_contract
from test_attention_contributions import wrapped


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_intact_teachers_and_exact_input_matched_donors_are_separate_conditions(tmp_path, dtype):
    pytest.importorskip('sklearn')
    teacher, _ = wrapped('prefix', seed=274)
    for name in ADAPTERS.values():
        teacher.add_adapter(name, teacher.peft_config['default'])
    teacher.to(dtype).eval()
    base = teacher.get_base_model()
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None
        def decode(self, _ids, skip_special_tokens=False):
            return 'NONE'
    tokenizer = Tokenizer(); contract = load_contract()
    ids = [[2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 9, 10, 11, 12]]
    sources = [{'id': 'a', 'labels': []}, {'id': 'b', 'labels': []}]
    summary = evaluate(teacher, ids, sources, tokenizer, contract, tmp_path, layer=1, max_new_tokens=3)
    assert [s['condition'] for s in summary] == CONDITIONS
    assert CONDITIONS == ['base', 'trained_teacher', 'init_teacher', 'trained_donor', 'init_donor']
    assert all(s['bootstrap_draws'] == BOOTSTRAP_DRAWS == 10000 for s in summary)
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    assert [(r['key'], r['condition']) for r in rows] == [(source['id'], name) for source in sources for name in CONDITIONS]
    groups = {name: [r for r in rows if r['condition'] == name] for name in CONDITIONS}
    with torch.no_grad():
        for kind, adapter in ADAPTERS.items():
            teacher.set_adapter(adapter); teacher.eval()
            actual, _ = measure(teacher, ids[0], 'civil', tokenizer, [], contract, 3)
            assert actual['generation_scores_sha256'] == groups[f'{kind}_teacher'][0]['generation_scores_sha256']
            donor = capture(teacher, ids[0], 1)
            with PrefillStatePatch(base, 1, len(ids[0]), 'after3', donor, 1) as hook:
                actual, _ = measure(base, ids[0], 'civil', tokenizer, [], contract, 3)
            assert actual['generation_scores_sha256'] == groups[f'{kind}_donor'][0]['generation_scores_sha256']
            assert torch.equal(hook.after[3:], donor[3:]) and torch.equal(hook.before[:3], hook.after[:3])
            native, _ = measure(base, ids[0], 'civil', tokenizer, [], contract, 3)
            assert native['generation_scores_sha256'] == groups['base'][0]['generation_scores_sha256']
    assert groups['trained_teacher'][0]['generation_scores_sha256'] != groups['init_teacher'][0]['generation_scores_sha256']
    for r in rows:
        assert r['score'] == 0 and r['parsed_score'] == 1 and r['truncated']
        if r['condition'].endswith('_donor'):
            assert r['scope'] == 'after3' and r['alpha'] == 1 and r['target_max_abs_error'] == 0
            assert r['patched_positions'] == r['prompt_tokens'] - 3 and r['intervention_calls'] == 3
            assert r['first3_exact'] and r['first3_before_sha256'] == r['first3_after_sha256']
    gates = json.loads((tmp_path / 'gates.json').read_text())
    assert sum(bool(g.get('zero_alpha_exact')) for g in gates) == 2
    assert {g['base_after_teacher_exact'] for g in gates if 'base_after_teacher_exact' in g} == {'trained', 'init'}
    assert gates[0]['observation_logits_exact'] and gates[0]['models'] == ['base', 'trained', 'init']
    diagnostics = [json.loads(line) for line in (tmp_path / 'state-diagnostics.jsonl').read_text().splitlines()]
    for diag, sequence in zip(diagnostics, ids):
        assert len(diag['base_norm_per_position']) == len(sequence)
        for kind in ADAPTERS:
            assert len(diag[kind]['donor_norm_per_position']) == len(sequence)
            assert len(diag[kind]['delta_norm_per_position']) == len(sequence)
    assert not base.model.layers[1]._forward_hooks


def test_audited_teacher_provenance_rejects_swapped_steps_weights_or_cell():
    metadata = []
    for kind, step in [('trained', 1125), ('init', 0)]:
        metadata.append({'name': ADAPTERS[kind], 'adapter_config': {'num_virtual_tokens': 500},
            'receipt': {'repo': 'archive', 'commit': 'same', 'prefix': 'same-cell', 'archive_sha256': 'same-hash',
                        'peft_type': 'PREFIX_TUNING', 'base_revision': REVISION, 'num_virtual_tokens': 500,
                        'selected_step': step, 'adapter_files': {'adapter_model.safetensors': WEIGHT_HASHES[kind]}}})
    check_teachers(metadata, REVISION)
    for field, wrong in [('selected_step', 1125), ('prefix', 'another-cell'), ('base_revision', 'wrong')]:
        broken = copy.deepcopy(metadata); broken[1]['receipt'][field] = wrong
        with pytest.raises(ValueError):
            check_teachers(broken, REVISION)
    broken = copy.deepcopy(metadata)
    broken[1]['receipt']['adapter_files']['adapter_model.safetensors'] = WEIGHT_HASHES['trained']
    with pytest.raises(ValueError, match='audited checkpoint'):
        check_teachers(broken, REVISION)
    broken = copy.deepcopy(metadata); broken[1]['adapter_config']['num_virtual_tokens'] = 200
    with pytest.raises(ValueError, match='configurations'):
        check_teachers(broken, REVISION)


def test_existing_wording_render_gate_is_reused_and_no_input_slice_changes():
    from probe_wording_transfer import render_pair
    from probe_prefill_wording import render_pair as frozen_render_pair
    assert render_pair is frozen_render_pair
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'data/prefill_wording_20260924.json').read_text())
    assert config['civil_slice'] == [200, 400] and config['instruction_span'] == [3, 25]
    assert list(config['variants']) == ['original', 'paraphrase_a', 'paraphrase_b']
