#!/usr/bin/env python3
"""Confirm frozen full and prefix/suffix tables on the final reserved inputs."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from se_gepa.arms import SEED_KEY, build_base, device_of, load_contract, render
from se_gepa.prefill_template import PrefillTemplateShift, materialize_table
from probe_attention_contributions import sequence_hash
from probe_prefill_transfer import capture, state_hash
from probe_residual_transfer import measure, equal_outputs, write_json
from eval_ppl_arms import windows, WIKITEXT_REVISION

from probe_prefill_roles import MODEL, REVISION, TABLE_HASHES, load_fitted, role_table

CONDITIONS = ['base', 'trained_full', 'init_full', 'trained_prefix_suffix', 'init_prefix_suffix']
ROLE_MASKS = dict(zip(CONDITIONS, [0, 7, 7, 5, 5]))
CIVIL_SLICE = (200, 1000)
WIKI_SLICE = (96, 128)
BOOTSTRAP_DRAWS = 10000


def civil_summary(rows, reference):
    delta = torch.tensor([r['score'] - reference[r['key']]['row']['score'] for r in rows], dtype=torch.float64)
    indices = torch.randint(len(rows), (BOOTSTRAP_DRAWS, len(rows)), generator=torch.Generator().manual_seed(42))
    interval = torch.quantile(delta[indices].mean(1), torch.tensor([.025, .975], dtype=torch.float64))
    return {'n': len(rows), 'score': sum(r['score'] for r in rows) / len(rows),
            'valid': sum(r['valid'] for r in rows) / len(rows),
            'finished': sum(r['finished'] for r in rows) / len(rows),
            'truncated': sum(r['truncated'] for r in rows) / len(rows),
            'mean_completion_tokens': sum(r['completion_tokens'] for r in rows) / len(rows),
            'paired_score_delta': float(delta.mean()), 'paired_bootstrap_ci95': interval.tolist(),
            'improved': int((delta > 0).sum()), 'worsened': int((delta < 0).sum()),
            'first_token_kl_from_intact': sum(r['first_token_kl_from_intact'] for r in rows) / len(rows)}


@torch.no_grad()
def evaluate(base, sequences, sources, task, tokenizer, contract, out, tables,
             layer=3, max_new_tokens=64, prefix_length=44, suffix_length=34):
    selected = {name: role_table(tables['init' if name.startswith('init_') else 'trained'], ROLE_MASKS[name],
                                prefix_length, suffix_length) for name in CONDITIONS if name != 'base'}
    reference = {}; groups = {name: [] for name in CONDITIONS}; gates = []
    with (out / 'rows.jsonl').open('w') as stream:
        for i, (ids, source) in enumerate(zip(sequences, sources)):
            original = capture(base, ids, layer, i == 0)
            if i == 0:
                gates.append({'key': source['id'], 'observation_logits_exact': True})
            for name in CONDITIONS:
                mask = ROLE_MASKS[name]
                context = (nullcontext() if name == 'base' else
                           PrefillTemplateShift(base, layer, len(ids), selected[name], prefix_length, suffix_length))
                with context as hook:
                    row, first = measure(base, ids, task, tokenizer, source.get('labels'), contract, max_new_tokens)
                row.update(key=source['id'], condition=name, role_mask=mask,
                           sequence_sha256=sequence_hash(ids), prompt_tokens=len(ids))
                if hook is not None:
                    if hook.calls != (row['completion_tokens'] if task == 'civil' else 1) or hook.patched_positions != len(ids) - 3:
                        raise RuntimeError('Incorrect original-prefill coverage')
                    if not torch.equal(hook.before[:3], hook.after[:3]):
                        raise RuntimeError('Early states changed')
                    if not torch.allclose(hook.before, original, rtol=.005, atol=.005):
                        raise RuntimeError('Base capture differs from recipient')
                    addition = materialize_table(selected[name], len(ids), prefix_length, suffix_length)
                    expected = (hook.before[3:].to(device_of(base)) + addition.to(device_of(base))).to(base.dtype).float().cpu()
                    error = float((hook.after[3:] - expected).abs().max())
                    if error != 0:
                        raise RuntimeError('Rounded role-table addition mismatch')
                    active_slots = role_table(torch.ones(selected[name].shape[0], 1), mask, prefix_length, suffix_length)
                    inactive = materialize_table(active_slots, len(ids), prefix_length, suffix_length)[:, 0] == 0
                    if not torch.equal(hook.before[3:][inactive], hook.after[3:][inactive]):
                        raise RuntimeError('Omitted role states changed')
                    row.update(patched_positions=hook.patched_positions, intervention_calls=hook.calls,
                        active_positions=int((~inactive).sum()), inactive_positions=int(inactive.sum()),
                        inactive_positions_exact=True, target_max_abs_error=error,
                        capture_max_abs_difference=float((hook.before - original).abs().max()),
                        first3_before_sha256=state_hash(hook.before[:3]), first3_after_sha256=state_hash(hook.after[:3]))
                if name == 'base':
                    reference[source['id']] = {'row': dict(row), 'first': first}
                    if i < (5 if task == 'civil' else 1):
                        with PrefillTemplateShift(base, layer, len(ids), torch.zeros_like(tables['trained']), prefix_length, suffix_length):
                            identity, _ = measure(base, ids, task, tokenizer, source.get('labels'), contract, max_new_tokens)
                        if not equal_outputs(row, identity, task):
                            raise RuntimeError('Zero-table identity failed')
                        gates.append({'key': source['id'], 'zero_table_exact': True})
                ref = reference[source['id']]
                if task == 'civil':
                    p, q = ref['first'].log_softmax(-1), first.log_softmax(-1)
                    row['first_token_kl_from_intact'] = float((p.exp() * (p - q)).sum())
                else:
                    row['delta_nll'] = row['nll'] - ref['row']['nll']
                groups[name].append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps({'example': i, 'key': source['id'], 'conditions_completed': len(CONDITIONS)}), flush=True)
    summaries = []
    for name, rows in groups.items():
        if task == 'civil':
            summary = civil_summary(rows, reference)
        else:
            delta = torch.tensor([r['delta_nll'] for r in rows], dtype=torch.float64)
            draws = torch.randint(len(rows), (BOOTSTRAP_DRAWS, len(rows)), generator=torch.Generator().manual_seed(42))
            ci = torch.quantile(delta[draws].mean(-1), torch.tensor([.025, .975], dtype=torch.float64))
            nll = sum(r['nll'] for r in rows) / len(rows)
            summary = {'n': len(rows), 'nll': nll, 'ppl': math.exp(nll), 'paired_delta_nll': float(delta.mean()),
                       'paired_bootstrap_ci95': ci.tolist(), 'targets': sum(r['targets'] for r in rows)}
        summaries.append({'condition': name, 'role_mask': ROLE_MASKS[name], 'bootstrap_draws': BOOTSTRAP_DRAWS, **summary})
    write_json(out / 'summary.json', summaries); write_json(out / 'gates.json', gates)
    return summaries


def main():
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True); parser.add_argument('--model-revision', required=True)
    parser.add_argument('--task', choices=['civil', 'wiki'], required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fitted-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'data/prefill_template_frozen_20260924')
    args = parser.parse_args()
    if transformers.__version__ != '5.16.1':
        raise RuntimeError('Unexpected runtime')
    tables, provenance, template, calibration, fitted_hashes = load_fitted(args.fitted_dir, args.model_revision)
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'data' / name for name in ('civil_v2_val_seed42_n200.jsonl', 'civil_v2_test_head2000.jsonl')]
    validation, test = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    if [{k: v for k, v in r.items() if k != 'input_ids'} for r in calibration] != validation[:64]:
        raise ValueError('Frozen calibration witness differs from validation[:64]')
    rows = test[CIVIL_SLICE[0]:CIVIL_SLICE[1]]
    if len(rows) != 800:
        raise ValueError('Incomplete confirmation rows')
    for field in ('id', 'text'):
        if len({r[field] for r in rows}) != 800 or {r[field] for r in rows} & {r[field] for r in validation + test[:200] + test[1000:2000]}:
            raise ValueError('Duplicate or overlapping confirmation rows')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    contract = load_contract()
    for witness in calibration:
        ids = render(tokenizer, contract[1][SEED_KEY], witness['text'], contract)
        if (ids != witness['input_ids'] or ids[:44] != template['prefix_input_ids']
                or ids[-34:] != template['suffix_input_ids'] or len(ids) <= 78):
            raise ValueError('Native rendering differs from frozen calibration witness')
    if args.task == 'civil':
        sequences = [render(tokenizer, contract[1][SEED_KEY], row['text'], contract) for row in rows]
        if any(ids[:44] != template['prefix_input_ids'] or ids[-34:] != template['suffix_input_ids'] for ids in sequences):
            raise ValueError('Civil confirmation template differs from frozen template')
    else:
        tokens = windows(tokenizer, 2048, WIKI_SLICE[1])
        if tokens.shape != (WIKI_SLICE[1], 2048):
            raise RuntimeError('Missing WikiText confirmation windows')
        sequences = tokens[WIKI_SLICE[0]:WIKI_SLICE[1]].tolist(); rows = [{'id': f'wiki-test-window-{i}'} for i in range(*WIKI_SLICE)]
    if any(len(ids) <= 78 for ids in sequences):
        raise ValueError('No middle position in confirmation input')
    args.out.mkdir(parents=True, exist_ok=False)
    fitted_out = args.out / 'fitted'; fitted_out.mkdir()
    for name in fitted_hashes:
        shutil.copyfile(args.fitted_dir / name, fitted_out / name)
    write_json(args.out / 'inputs.json', [{**row, 'input_ids': ids} for row, ids in zip(rows, sequences)])
    write_json(args.out / 'used-slices.json', {'calibration_witness': 'Civil validation[0:64], rendering only; no refitting',
        'civil_evaluation': list(CIVIL_SLICE), 'civil_excluded_test_slices': [[0, 200], [1000, 2000]],
        'civil_validation_excluded': True, 'civil_id_and_text_disjoint': True, 'wiki_evaluation': list(WIKI_SLICE),
        'wiki_previous_intervention_windows': [0, 96],
        'history': 'Civil head2000 previously evaluated by base; fresh confirmation of the unchanged fitted table.'})
    base = build_base(args.model_dir, 'cuda')
    if base.config.model_type != 'qwen3_moe' or base.config.hidden_size != 2048:
        raise ValueError('Wrong base model architecture')
    summaries = evaluate(base, sequences, rows, args.task, tokenizer, contract, args.out, tables)
    count = len((args.out / 'rows.jsonl').read_text().splitlines())
    if count != len(summaries) * len(rows):
        raise RuntimeError('Missing final confirmation rows')
    write_json(args.out / 'manifest.json', {'status': 'PASS', 'probe': 'prefill_frozen_confirmation_v1', 'task': args.task,
        'model': MODEL, 'model_revision': args.model_revision, 'layer': 3, 'alpha': 1,
        'scope': 'after3 original prefill only', 'prefix_length': 44, 'suffix_length': 34, 'table_slots': 76,
        'conditions': CONDITIONS, 'role_masks': ROLE_MASKS, 'recipient': 'native base without adapters; generated queries unpatched',
        'fitting': 'none; reused immutable prior tables', 'fitted_provenance': provenance,
        'fitted_files_sha256': fitted_hashes, 'table_tensor_sha256': TABLE_HASHES,
        'contract_examples_verified': len(calibration), 'observation_logits_exact': True,
        'selection': 'Civil test[200:1000] or WikiText test windows[96:128],2048 tokens',
        'wikitext_revision': WIKITEXT_REVISION, 'metric_rows': count, 'max_new_tokens': 64, 'seed': 42,
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'noninferiority_margin_f1': .02,
        'source_rows_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'files_sha256': {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in args.out.rglob('*') if path.is_file()}})
    print(f'PREFILL_CONFIRMATION_ROWS={count}', flush=True)


if __name__ == '__main__':
    main()
