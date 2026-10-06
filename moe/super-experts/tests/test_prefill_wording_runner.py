import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_prefill_wording import (BOOTSTRAP_DRAWS, CONDITIONS, ROLE_MASKS, evaluate,
    load_contract, render_pair)
from test_attention_contributions import tiny_model


class Tokenizer:
    def __call__(self, rendered, add_special_tokens=False):
        instruction, text = json.loads(rendered)
        ids = list(range(100, 144)) + [500 + ord(char) for char in text] + list(range(200, 234))
        if instruction == 'paraphrase':
            ids[10] = 900
            ids[24] = 901
        elif instruction == 'label_change':
            ids[30] = 902
        elif instruction == 'content_change':
            ids[45] = 903
        elif instruction == 'suffix_change':
            ids[-1] = 904
        elif instruction == 'first3_change':
            ids[1] = 905
        elif instruction == 'length_change':
            ids.insert(10, 906)
        return {'input_ids': ids}


def mock_contract():
    return ([], {'civil_comments': 'original'},
            lambda instruction, _labels, text: SimpleNamespace(instruction=instruction, text=text),
            lambda _tokenizer, result, non_thinking: json.dumps([result.instruction, result.text]))


def test_position_matched_instruction_changes_are_accepted():
    template = {'prefix_input_ids': list(range(100, 144)), 'suffix_input_ids': list(range(200, 234))}
    original, variant, changed = render_pair(Tokenizer(), 'paraphrase', 'abc', mock_contract(), template)
    assert changed == [10, 24]
    assert len(original) == len(variant) == 81
    assert original[:3] == variant[:3] and original[25:] == variant[25:]
    same, replay, positions = render_pair(Tokenizer(), 'original', 'abc', mock_contract(), template)
    assert same == replay and positions == []


@pytest.mark.parametrize('instruction', ['label_change', 'content_change', 'suffix_change'])
def test_label_content_or_suffix_changes_are_rejected(instruction):
    template = {'prefix_input_ids': list(range(100, 144)), 'suffix_input_ids': list(range(200, 234))}
    with pytest.raises(ValueError, match='positions >=25'):
        render_pair(Tokenizer(), instruction, 'abc', mock_contract(), template)


@pytest.mark.parametrize('instruction,message', [('first3_change', 'first three'), ('length_change', 'sequence length')])
def test_first_three_or_length_changes_are_rejected(instruction, message):
    template = {'prefix_input_ids': list(range(100, 144)), 'suffix_input_ids': list(range(200, 234))}
    with pytest.raises(ValueError, match=message):
        render_pair(Tokenizer(), instruction, 'abc', mock_contract(), template)


def test_original_frozen_contract_mismatch_is_rejected():
    template = {'prefix_input_ids': list(range(99, 143)), 'suffix_input_ids': list(range(200, 234))}
    with pytest.raises(ValueError, match='Original rendering'):
        render_pair(Tokenizer(), 'paraphrase', 'abc', mock_contract(), template)


def test_reused_evaluator_preserves_all_five_conditions_and_cap(tmp_path):
    pytest.importorskip('sklearn')
    from probe_prefill_confirmation import evaluate as frozen_evaluate
    assert evaluate is frozen_evaluate
    assert CONDITIONS == ['base', 'trained_full', 'init_full', 'trained_prefix_suffix', 'init_prefix_suffix']
    assert list(ROLE_MASKS.values()) == [0, 7, 7, 5, 5] and BOOTSTRAP_DRAWS == 10000
    model, _ = tiny_model(seed=264)
    class OutputTokenizer:
        pad_token_id = 0
        eos_token_id = None
        def decode(self, _ids, skip_special_tokens=False):
            return 'NONE'
    ids = [[2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 10, 8, 9]]
    tables = {'trained': torch.randn(5, 32) * .1, 'init': torch.randn(5, 32) * .1}
    summary = evaluate(model, ids, [{'id': 'a', 'labels': []}, {'id': 'b', 'labels': []}], 'civil',
        OutputTokenizer(), load_contract(), tmp_path, tables, layer=1, max_new_tokens=3, prefix_length=5, suffix_length=2)
    rows = [json.loads(line) for line in (tmp_path / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 10 and [s['condition'] for s in summary] == CONDITIONS
    assert all(s['bootstrap_draws'] == 10000 for s in summary)
    assert all(r['truncated'] and r['score'] == 0 and r['parsed_score'] == 1 for r in rows)
    assert all(r['target_max_abs_error'] == 0 and r['inactive_positions_exact'] for r in rows if r['condition'] != 'base')


def test_wording_rows_are_disclosed_subset_of_confirmation():
    root = Path(__file__).resolve().parents[1]
    data = [json.loads(line) for line in (root / 'data/civil_v2_test_head2000.jsonl').read_text().splitlines()]
    selected = data[200:400]
    assert selected == data[200:1000][:200]
    assert len({r['id'] for r in selected}) == len({r['text'] for r in selected}) == 200
    assert len(selected) * len(CONDITIONS) == 1000
