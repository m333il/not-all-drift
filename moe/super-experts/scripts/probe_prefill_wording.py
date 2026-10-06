#!/usr/bin/env python3
"""Test frozen residual tables under position-matched instruction paraphrases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch

from probe_prefill_confirmation import (BOOTSTRAP_DRAWS, CONDITIONS, MODEL, REVISION, ROLE_MASKS, TABLE_HASHES,
    build_base, evaluate, load_contract, load_fitted, render, SEED_KEY, write_json)
from probe_prefill_template import common_boundaries


def render_pair(tokenizer, instruction, text, contract, template):
    original = render(tokenizer, contract[1][SEED_KEY], text, contract)
    variant = render(tokenizer, instruction, text, contract)
    if (len(original) <= 78 or original[:44] != template['prefix_input_ids']
            or original[-34:] != template['suffix_input_ids']):
        raise ValueError('Original rendering differs from frozen template')
    if len(variant) != len(original):
        raise ValueError('Instruction wording changed sequence length')
    if variant[:3] != original[:3]:
        raise ValueError('Instruction wording changed the first three tokens')
    if variant[25:] != original[25:]:
        raise ValueError('Instruction wording changed label, content or suffix tokens at positions >=25')
    changed = [i for i, (a, b) in enumerate(zip(original, variant)) if a != b]
    return original, variant, changed


def main():
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True); parser.add_argument('--model-revision', required=True)
    parser.add_argument('--variant', choices=['original', 'paraphrase_a', 'paraphrase_b'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--fitted-dir', type=Path, default=root / 'data/prefill_template_frozen_20260924')
    args = parser.parse_args()
    if transformers.__version__ != '5.16.1':
        raise RuntimeError('Unexpected runtime')
    config_path = root / 'data/prefill_wording_20260924.json'
    config = json.loads(config_path.read_text())
    if (config['schema'] != 'prefill_wording_v1' or config['model_revision'] != REVISION
            or config['civil_slice'] != [200, 400] or config['instruction_span'] != [3, 25]
            or config['prefix_length'] != 44 or config['suffix_length'] != 34
            or set(config['variants']) != {'original', 'paraphrase_a', 'paraphrase_b'}):
        raise ValueError('Frozen wording configuration mismatch')
    tables, provenance, template, calibration, fitted_hashes = load_fitted(args.fitted_dir, args.model_revision)
    paths = [root / 'data' / name for name in ('civil_v2_val_seed42_n200.jsonl', 'civil_v2_test_head2000.jsonl')]
    validation, test = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    if [{k: v for k, v in row.items() if k != 'input_ids'} for row in calibration] != validation[:64]:
        raise ValueError('Frozen calibration witness differs from validation[:64]')
    rows = test[200:400]
    if len(rows) != 200:
        raise ValueError('Incomplete wording evaluation rows')
    for field in ('id', 'text'):
        if len({r[field] for r in rows}) != 200 or {r[field] for r in rows} & {r[field] for r in validation}:
            raise ValueError('Duplicate wording rows or overlap with validation')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    contract = load_contract()
    if config['variants']['original'] != contract[1][SEED_KEY]:
        raise ValueError('Original instruction differs from frozen rendering contract')
    instruction = config['variants'][args.variant]
    witnesses = {}
    for split, sources in [('calibration', calibration), ('evaluation', rows)]:
        entries = []
        for source in sources:
            original, variant, changed = render_pair(tokenizer, instruction, source['text'], contract, template)
            if split == 'calibration' and original != source['input_ids']:
                raise ValueError('Native rendering differs from frozen calibration witness')
            if bool(changed) != (args.variant != 'original'):
                raise ValueError('Original or paraphrased instruction identity mismatch')
            entries.append({**source, 'original_input_ids': original, 'input_ids': variant,
                            'changed_instruction_positions': changed, 'changed_instruction_count': len(changed)})
        if common_boundaries([r['input_ids'] for r in entries]) != (44, 34):
            raise ValueError('Wording changed the common prefix/suffix boundaries')
        witnesses[split] = entries
    variant_prefix = witnesses['calibration'][0]['input_ids'][:44]
    if any(row['input_ids'][:44] != variant_prefix for row in witnesses['evaluation']):
        raise ValueError('Evaluation wording prefix differs from calibration rendering')
    args.out.mkdir(parents=True, exist_ok=False)
    fitted_out = args.out / 'fitted'; fitted_out.mkdir()
    for name in fitted_hashes:
        shutil.copyfile(args.fitted_dir / name, fitted_out / name)
    shutil.copyfile(config_path, args.out / 'wording-config.json')
    write_json(args.out / 'inputs.json', witnesses['evaluation'])
    write_json(args.out / 'calibration-witness.json', witnesses['calibration'])
    write_json(args.out / 'rendering-gates.json', {
        'variant': args.variant, 'calibration_examples': 64, 'evaluation_examples': 200,
        'original_calibration_ids_exact': True, 'equal_sequence_lengths': True,
        'first3_exact': True, 'positions_25_onward_exact': True, 'instruction_span': [3, 25],
        'common_prefix_length': 44, 'common_suffix_length': 34,
        'original_prefix_input_ids': template['prefix_input_ids'], 'variant_prefix_input_ids': variant_prefix,
        'suffix_input_ids': template['suffix_input_ids']})
    write_json(args.out / 'used-slices.json', {
        'civil_evaluation': [200, 400], 'previous_confirmation_slice': [200, 1000],
        'reused_from_confirmation': True, 'calibration_witness': 'Civil validation[0:64], rendering only; no refitting',
        'civil_validation_excluded': True, 'unique_evaluation_ids_and_text': True,
        'history': 'Same 200 previously evaluated confirmation rows; this measures wording robustness, not a fresh held-out task improvement.'})
    base = build_base(args.model_dir, 'cuda')
    if base.config.model_type != 'qwen3_moe' or base.config.hidden_size != 2048:
        raise ValueError('Wrong base model architecture')
    sequences = [row['input_ids'] for row in witnesses['evaluation']]
    summaries = evaluate(base, sequences, rows, 'civil', tokenizer, contract, args.out, tables)
    count = len((args.out / 'rows.jsonl').read_text().splitlines())
    if count != 1000 or len(summaries) != 5:
        raise RuntimeError('Incomplete wording results')
    write_json(args.out / 'manifest.json', {
        'status': 'PASS', 'probe': 'prefill_wording_v1', 'task': 'civil', 'variant': args.variant,
        'instruction': instruction, 'instruction_span': [3, 25],
        'wording_config_path': str(config_path.relative_to(root)),
        'wording_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'model': MODEL, 'model_revision': args.model_revision, 'layer': 3, 'alpha': 1,
        'scope': 'after3 original prefill only', 'prefix_length': 44, 'suffix_length': 34, 'table_slots': 76,
        'conditions': CONDITIONS, 'role_masks': ROLE_MASKS,
        'recipient': 'native base without adapters; generated queries unpatched',
        'fitting': 'none; unchanged prior tables under a position-matched instruction variant',
        'fitted_provenance': provenance, 'fitted_files_sha256': fitted_hashes, 'table_tensor_sha256': TABLE_HASHES,
        'contract_examples_verified': 64, 'rendering_examples_verified': 264, 'observation_logits_exact': True,
        'selection': 'Civil test[200:400], reused from final confirmation', 'evaluation_examples': 200,
        'metric_rows': count, 'max_new_tokens': 64, 'seed': 42, 'bootstrap_draws': BOOTSTRAP_DRAWS,
        'source_rows_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'files_sha256': {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in args.out.rglob('*') if path.is_file()}})
    print(f'PREFILL_WORDING_ROWS={count}', flush=True)


if __name__ == '__main__':
    main()
