#!/usr/bin/env python3
"""Confirm a frozen prefill table and measure its three role contributions."""
import argparse
import base64
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zlib

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from se_gepa.arms import SEED_KEY, build_base, device_of, load_contract, render
from se_gepa.prefill_template import PrefillTemplateShift, materialize_table
from probe_attention_contributions import sequence_hash
from probe_prefix_causal import paired_summary
from probe_prefill_transfer import capture, state_hash
from probe_residual_transfer import measure, equal_outputs, write_json
from eval_ppl_arms import windows, WIKITEXT_REVISION

MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
REVISION = '0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
CONDITIONS = ['base', 'trained_prefix', 'trained_middle', 'trained_prefix_middle',
              'trained_suffix', 'trained_prefix_suffix', 'trained_middle_suffix', 'trained_full', 'init_full']
ROLE_MASKS = dict(zip(CONDITIONS, [0, 1, 2, 3, 4, 5, 6, 7, 7]))
TABLE_HASHES = {'trained': 'd7e7e866c1f6135f352968651055aeb8a89be26db9870e68a549cebe29881d9f',
                'init': '27e23908e56820d4e469ea2460be4f8800d961c2023fe281e4d66aa7c4275cba'}
FITTED_FILES = {'table-trained.json', 'table-init.json', 'template.json', 'calibration-inputs.json', 'manifest.json'}


def role_table(table, mask, prefix_length=44, suffix_length=34):
    if mask not in range(8):
        raise ValueError('Role mask must be an integer from 0 through 7')
    prefix_slots = prefix_length - 3
    if table.ndim != 2 or table.shape[0] != prefix_slots + 1 + suffix_length:
        raise ValueError('Incorrect template table shape')
    selected = table.clone()
    for bit, section in ((1, slice(0, prefix_slots)), (2, slice(prefix_slots, prefix_slots + 1)),
                         (4, slice(prefix_slots + 1, None))):
        if not mask & bit:
            selected[section] = 0
    return selected


def load_fitted(folder, revision):
    provenance = json.loads((folder / 'provenance.json').read_text())
    if (revision != REVISION or provenance['model_revision'] != REVISION or provenance['model'] != MODEL
            or provenance['layer'] != 3 or provenance['prefix_length'] != 44 or provenance['suffix_length'] != 34
            or provenance['table_shape'] != [76, 2048] or provenance['calibration_n'] != 64
            or provenance['origin_status'] != 'VERIFIED' or not provenance['origin_cross_job_calibration_exact']):
        raise ValueError('Frozen fitted-object provenance mismatch')
    if set(provenance['files_sha256']) != FITTED_FILES:
        raise ValueError('Incomplete frozen fitted-object file list')
    hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FITTED_FILES | {'provenance.json'}}
    if any(hashes[name] != expected for name, expected in provenance['files_sha256'].items()):
        raise ValueError('Frozen fitted-object file hash mismatch')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if (manifest['status'] != 'PASS' or manifest['probe'] != 'prefill_static_template_v1'
            or manifest['model_revision'] != REVISION or manifest['layer'] != 3
            or manifest['prefix_length'] != 44 or manifest['suffix_length'] != 34 or manifest['table_slots'] != 76):
        raise ValueError('Wrong original fitted-object manifest')
    for name in FITTED_FILES - {'manifest.json'}:
        if manifest['files_sha256'][name] != hashes[name]:
            raise ValueError('Frozen file differs from original manifest')
    tables = {}
    for kind in TABLE_HASHES:
        entry = json.loads((folder / f'table-{kind}.json').read_text())
        record = entry['table']
        raw = zlib.decompress(base64.b64decode(record['zlib_base64']))
        if (record['dtype'] != 'float32-little-endian' or record['shape'] != [76, 2048]
                or entry['calibration_n'] != 64 or entry['prefix_length'] != 44 or entry['suffix_length'] != 34
                or hashlib.sha256(raw).hexdigest() != record['sha256'] or record['sha256'] != TABLE_HASHES[kind]):
            raise ValueError('Frozen table tensor hash or shape mismatch')
        tables[kind] = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(76, 2048).clone()
        if not torch.isfinite(tables[kind]).all():
            raise ValueError('Non-finite frozen table')
    template = json.loads((folder / 'template.json').read_text())
    calibration = json.loads((folder / 'calibration-inputs.json').read_text())
    if (template['source'] != 'Civil calibration only' or template['prefix_length'] != 44
            or template['suffix_length'] != 34 or len(template['prefix_input_ids']) != 44
            or len(template['suffix_input_ids']) != 34 or len(calibration) != 64):
        raise ValueError('Frozen rendering witness geometry mismatch')
    return tables, provenance, template, calibration, hashes


@torch.no_grad()
def evaluate(base, sequences, sources, task, tokenizer, contract, out, tables,
             layer=3, max_new_tokens=64, prefix_length=44, suffix_length=34):
    selected = {name: role_table(tables['init' if name == 'init_full' else 'trained'], ROLE_MASKS[name],
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
            summary = paired_summary(rows, reference, 42)
        else:
            delta = torch.tensor([r['delta_nll'] for r in rows], dtype=torch.float64)
            draws = torch.randint(len(rows), (10000, len(rows)), generator=torch.Generator().manual_seed(42))
            ci = torch.quantile(delta[draws].mean(-1), torch.tensor([.025, .975], dtype=torch.float64))
            nll = sum(r['nll'] for r in rows) / len(rows)
            summary = {'n': len(rows), 'nll': nll, 'ppl': math.exp(nll), 'paired_delta_nll': float(delta.mean()),
                       'paired_bootstrap_ci95': ci.tolist(), 'targets': sum(r['targets'] for r in rows)}
        summaries.append({'condition': name, 'role_mask': ROLE_MASKS[name], **summary})
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
    rows = test[1000:1200]
    if len(rows) != 200:
        raise ValueError('Incomplete confirmation rows')
    for field in ('id', 'text'):
        if len({r[field] for r in rows}) != 200 or {r[field] for r in rows} & {r[field] for r in validation + test[:200] + test[1200:2000]}:
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
        tokens = windows(tokenizer, 2048, 96)
        if tokens.shape != (96, 2048):
            raise RuntimeError('Missing WikiText confirmation windows')
        sequences = tokens[80:96].tolist(); rows = [{'id': f'wiki-test-window-{i}'} for i in range(80, 96)]
    if any(len(ids) <= 78 for ids in sequences):
        raise ValueError('No middle position in confirmation input')
    args.out.mkdir(parents=True, exist_ok=False)
    fitted_out = args.out / 'fitted'; fitted_out.mkdir()
    for name in fitted_hashes:
        shutil.copyfile(args.fitted_dir / name, fitted_out / name)
    write_json(args.out / 'inputs.json', [{**row, 'input_ids': ids} for row, ids in zip(rows, sequences)])
    write_json(args.out / 'used-slices.json', {'calibration_witness': 'Civil validation[0:64], rendering only; no refitting',
        'civil_evaluation': [1000, 1200], 'civil_excluded_test_slices': [[0, 200], [1200, 2000]],
        'civil_validation_excluded': True, 'civil_id_and_text_disjoint': True, 'wiki_evaluation': [80, 96],
        'wiki_previous_intervention_windows': [0, 80],
        'history': 'Civil head2000 previously evaluated by base; fresh confirmation of the unchanged fitted table.'})
    base = build_base(args.model_dir, 'cuda')
    if base.config.model_type != 'qwen3_moe' or base.config.hidden_size != 2048:
        raise ValueError('Wrong base model architecture')
    summaries = evaluate(base, sequences, rows, args.task, tokenizer, contract, args.out, tables)
    count = len((args.out / 'rows.jsonl').read_text().splitlines())
    if count != len(summaries) * len(rows):
        raise RuntimeError('Missing role factorial rows')
    write_json(args.out / 'manifest.json', {'status': 'PASS', 'probe': 'prefill_frozen_roles_v1', 'task': args.task,
        'model': MODEL, 'model_revision': args.model_revision, 'layer': 3, 'alpha': 1,
        'scope': 'after3 original prefill only', 'prefix_length': 44, 'suffix_length': 34, 'table_slots': 76,
        'conditions': CONDITIONS, 'role_masks': ROLE_MASKS, 'recipient': 'native base without adapters; generated queries unpatched',
        'fitting': 'none; reused immutable prior tables', 'fitted_provenance': provenance,
        'fitted_files_sha256': fitted_hashes, 'table_tensor_sha256': TABLE_HASHES,
        'contract_examples_verified': len(calibration), 'observation_logits_exact': True,
        'selection': 'Civil test[1000:1200] or WikiText test windows[80:96],2048 tokens',
        'wikitext_revision': WIKITEXT_REVISION, 'metric_rows': count, 'max_new_tokens': 64, 'seed': 42,
        'source_rows_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'files_sha256': {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in args.out.rglob('*') if path.is_file()}})
    print(f'PREFILL_ROLES_ROWS={count}', flush=True)


if __name__ == '__main__':
    main()
