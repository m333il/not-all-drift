#!/usr/bin/env python3
"""Fixed early-expert output-channel intervention and downstream readouts."""
import argparse
from contextlib import ExitStack, nullcontext
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from probe_attention_contributions import arm_metadata, sequence_hash
from probe_prefill_roles import MODEL, REVISION, TABLE_HASHES, load_fitted
from probe_prefill_transfer import state_hash
from probe_residual_transfer import equal_outputs, measure, write_json
from probe_wording_transfer import ADAPTERS, check_teachers
from eval_ppl_arms import windows, WIKITEXT_REVISION
from se_gepa.arms import build_base, check_contract, device_of, load_contract, render, SEED_KEY, wrap
from se_gepa.attention_contributions import AttentionContributionProbe
from se_gepa.expert_channel import EarlyExpertContribution, ExpertChannelIntervention
from se_gepa.prefill_template import PrefillTemplateShift

LAYER, EXPERT = 1, 68
READOUT_LAYERS = [1, 2, 3, 4, 5]
CONDITIONS = ['intact', 'remove', 'control']
ARMS = ['base', 'trained', 'init', 'table']
CIVIL_SLICE, WIKI_SLICE = (200, 400), (128, 144)


@torch.no_grad()
def calibrate(base, sequences, sources, out, layer=LAYER, expert=EXPERT):
    records, energy = [], None
    for i, (ids, source) in enumerate(zip(sequences, sources)):
        tensor = torch.tensor([ids], device=device_of(base))
        options = dict(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1)
        reference = base(**options).logits if i == 0 else None
        with EarlyExpertContribution(base, layer, expert, len(ids)) as observer:
            observed = base(**options).logits
        if reference is not None and not torch.equal(reference, observed):
            raise RuntimeError('Calibration observation changes base logits')
        contribution = observer.contribution
        if contribution.shape != (3, base.config.hidden_size) or not torch.isfinite(contribution).all():
            raise RuntimeError('Invalid calibration contribution')
        squared = contribution.double().square().sum(0)
        energy = squared if energy is None else energy + squared
        records.append({'key': source['id'], 'sequence_sha256': sequence_hash(ids),
                        'first3_ids': ids[:3], 'contribution_sha256': state_hash(contribution),
                        'audit': observer.audit})
    if float(energy.max()) <= 0:
        raise RuntimeError('Selected expert has no early calibration contribution')
    channel = int(energy.argmax())
    selection = {'layer': layer, 'expert': expert, 'channel': channel, 'source': 'native base only',
        'criterion': 'argmax sum of squared routed expert output over calibration first3; first index on tie',
        'calibration_n': len(sequences), 'energy': energy.tolist(),
        'selected_energy_fraction': float(energy[channel] / energy.sum()),
        'distinct_first3_ids': len({tuple(r['first3_ids']) for r in records}),
        'distinct_contribution_hashes': len({r['contribution_sha256'] for r in records}),
        'examples': records, 'observation_logits_exact': True,
        'selection_precedes_evaluation': True, 'cross_job_energy_rtol': .005, 'cross_job_energy_atol': 1e-8}
    write_json(out / 'selection.json', selection)
    return selection


def residual_hook(layer, channel, rows):
    def hook(_module, _args, output):
        h = (output[0] if isinstance(output, tuple) else output)[0].detach().float()
        rows.append({'layer': layer, 'first3_max_abs': h[:3].abs().amax(-1).tolist(),
                     'remaining_max_abs': float(h[3:].abs().max()),
                     'first3_channel': h[:3, channel].tolist(),
                     'first3_other_max_abs': h[:3, torch.arange(h.shape[-1], device=h.device) != channel].abs().amax(-1).tolist(),
                     'first3_l2': h[:3].norm(dim=-1).tolist()})
    return hook


def table_context(model, ids, table, layer=3, prefix_length=44, suffix_length=34):
    return (nullcontext() if table is None else
            PrefillTemplateShift(model, layer, len(ids), table, prefix_length, suffix_length))


@torch.no_grad()
def diagnostics(model, ids, channel, mode, table, prefix_keys, index, layer=LAYER, expert=EXPERT,
                readout_layers=READOUT_LAYERS, table_layer=3, prefix_length=44, suffix_length=34):
    residuals = []
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with ExitStack() as stack:
        stack.enter_context(table_context(model, ids, table, table_layer, prefix_length, suffix_length))
        edit = stack.enter_context(ExpertChannelIntervention(model, layer, expert, channel,
                                     mode=mode, length=len(ids), seed=42))
        probe = stack.enter_context(AttentionContributionProbe(model, readout_layers,
            prefix_key_tokens=prefix_keys, reconstruction_rtol=.02 if base.dtype == torch.bfloat16 else 1e-5,
            real_query_start=3))
        probe.begin_example(index, len(ids), sequence_hash(ids))
        for selected in readout_layers:
            handle = base.model.layers[selected].register_forward_hook(residual_hook(selected, channel, residuals))
            stack.callback(handle.remove)
        tensor = torch.tensor([ids], device=device_of(model))
        logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False, logits_to_keep=1).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError('Non-finite instrumented prefill')
    if edit.calls != 1 or len(probe.records) != len(readout_layers) or len(residuals) != len(readout_layers):
        raise RuntimeError('Incomplete prefill diagnostic coverage')
    late = [r for r in probe.records if r['layer'] > layer]
    mass = sum(sum(sum(r['per_head_attention_mass'][f'real_{i}'][h] for i in range(3))
                         for h in range(r['attention_heads'])) / r['attention_heads'] for r in late) / len(late)
    return {'attention': probe.records, 'residuals': residuals, 'intervention': edit.audit,
            'early_key_mass_late_macro': mass}, logits[0, -1].float().cpu()


@torch.no_grad()
def evaluate(model, sequences, sources, task, tokenizer, contract, out, channel, table=None,
             prefix_keys=0, layer=LAYER, expert=EXPERT, readout_layers=READOUT_LAYERS,
             table_layer=3, prefix_length=44, suffix_length=34, max_new_tokens=64):
    groups = {c: [] for c in CONDITIONS}; gates = []
    with (out / 'rows.jsonl').open('w') as stream, gzip.open(out / 'diagnostics.jsonl.gz', 'wt') as ds:
        for index, (ids, source) in enumerate(zip(sequences, sources)):
            reference, reference_diag = None, None
            for condition in CONDITIONS:
                mode = 'observe' if condition == 'intact' else condition
                with table_context(model, ids, table, table_layer, prefix_length, suffix_length):
                    with ExpertChannelIntervention(model, layer, expert, channel, mode=mode,
                                                   length=len(ids), seed=42) as edit:
                        row, first = measure(model, ids, task, tokenizer, source.get('labels'), contract, max_new_tokens)
                if edit.calls != (row['completion_tokens'] if task == 'civil' else 1):
                    raise RuntimeError('Intervention generation coverage mismatch')
                for gate in ('native_repeat_exact', 'weight_restored', 'other_channels_exact',
                             'unrouted_rows_exact', 'later_positions_exact', 'finite'):
                    if edit.audit[gate] is not True:
                        raise RuntimeError(f'Intervention gate failed: {gate}')
                diag, diag_first = diagnostics(model, ids, channel, mode, table, prefix_keys, index,
                    layer, expert, readout_layers, table_layer, prefix_length, suffix_length)
                if diag['intervention'] != edit.audit:
                    raise RuntimeError('Diagnostic intervention differs from evaluation intervention')
                if first is not None and not torch.allclose(first, diag_first, atol=.005, rtol=.005):
                    raise RuntimeError('Diagnostic prefill differs from generation first logits')
                if condition == 'intact':
                    reference, reference_diag = dict(row), diag
                    if index < (5 if task == 'civil' else 1):
                        for identity_mode in ('unhooked', 'rescue'):
                            with table_context(model, ids, table, table_layer, prefix_length, suffix_length):
                                context = (nullcontext() if identity_mode == 'unhooked' else
                                    ExpertChannelIntervention(model, layer, expert, channel,
                                        mode='rescue', length=len(ids), seed=42))
                                with context:
                                    check, _ = measure(model, ids, task, tokenizer, source.get('labels'), contract, max_new_tokens)
                            if not equal_outputs(row, check, task):
                                raise RuntimeError(f'{identity_mode} identity gate failed')
                        gates.append({'key': source['id'], 'unhooked_exact': True, 'rescue_exact': True})
                first_layer = next(r for r in diag['attention'] if r['layer'] == layer)
                reference_layer = next(r for r in reference_diag['attention'] if r['layer'] == layer)
                if first_layer != reference_layer:
                    raise RuntimeError('Intervention changed earlier attention at its own layer')
                row.update(key=source['id'], condition=condition, sequence_sha256=sequence_hash(ids),
                    prompt_tokens=len(ids), early_key_mass_late_macro=diag['early_key_mass_late_macro'],
                    delta_early_key_mass=diag['early_key_mass_late_macro'] - reference_diag['early_key_mass_late_macro'],
                    intervention=edit.audit, intervention_calls=edit.calls, l1_attention_exact=True)
                if task == 'wiki':
                    row['delta_nll'] = row['nll'] - reference['nll']
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                ds.write(json.dumps({'key': source['id'], 'condition': condition,
                    'sequence_sha256': sequence_hash(ids), **diag}, allow_nan=False) + '\n'); ds.flush()
                groups[condition].append(row)
            print(f'EXPERT_CHANNEL_ROW={index + 1}/{len(sequences)}', flush=True)
    summaries = []
    for condition, rows in groups.items():
        if task == 'civil':
            summary = {'n': len(rows),
                **{k: sum(r[k] for r in rows) / len(rows) for k in ('score', 'valid', 'finished', 'truncated')},
                'mean_completion_tokens': sum(r['completion_tokens'] for r in rows) / len(rows)}
        else:
            nll = sum(r['nll'] for r in rows) / len(rows)
            summary = {'n': len(rows), 'nll': nll, 'ppl': math.exp(nll), 'targets': sum(r['targets'] for r in rows)}
        summaries.append({'condition': condition, **summary})
    write_json(out / 'summary.json', summaries); write_json(out / 'gates.json', gates)
    return summaries


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True); p.add_argument('--model-revision', required=True)
    p.add_argument('--arms-spec', type=Path, required=True); p.add_argument('--contract-sample', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--task', choices=['civil', 'wiki'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if transformers.__version__ != '5.16.1' or peft.__version__ != '0.20.0':
        raise RuntimeError('Unpinned runtime')
    root = Path(__file__).resolve().parents[1]
    arms = json.loads(args.arms_spec.read_text()); metadata = [arm_metadata(a) for a in arms]
    check_teachers(metadata, args.model_revision)
    tables, provenance, template, calibration, fitted_hashes = load_fitted(
        root / 'data/prefill_template_frozen_20260924', args.model_revision)
    paths = [root / 'data' / name for name in ('civil_v2_val_seed42_n200.jsonl', 'civil_v2_test_head2000.jsonl')]
    validation, test = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    if [{k: v for k, v in r.items() if k != 'input_ids'} for r in calibration] != validation[:64]:
        raise ValueError('Calibration witness source mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    contract = load_contract()
    checked = check_contract(tokenizer, contract, validation, json.loads(args.contract_sample.read_text()))
    calibration_ids = [render(tokenizer, contract[1][SEED_KEY], r['text'], contract) for r in calibration]
    if any(ids != r['input_ids'] for ids, r in zip(calibration_ids, calibration)):
        raise ValueError('Calibration witness token mismatch')
    if args.task == 'civil':
        rows = test[CIVIL_SLICE[0]:CIVIL_SLICE[1]]
        if len(rows) != 200 or len({r['id'] for r in rows}) != 200 or len({r['text'] for r in rows}) != 200:
            raise ValueError('Civil evaluation slice invalid')
        if any({r[k] for r in rows} & {r[k] for r in validation} for k in ('id', 'text')):
            raise ValueError('Calibration and evaluation overlap')
        sequences = [render(tokenizer, contract[1][SEED_KEY], r['text'], contract) for r in rows]
        if any(ids[:44] != template['prefix_input_ids'] or ids[-34:] != template['suffix_input_ids'] for ids in sequences):
            raise ValueError('Civil evaluation template mismatch')
    else:
        tokens = windows(tokenizer, 2048, WIKI_SLICE[1])
        if tokens.shape != (WIKI_SLICE[1], 2048):
            raise ValueError('Missing Wiki windows')
        sequences = tokens[WIKI_SLICE[0]:WIKI_SLICE[1]].tolist()
        rows = [{'id': f'wiki-test-window-{i}'} for i in range(*WIKI_SLICE)]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'inputs.json', [{**r, 'input_ids': ids} for r, ids in zip(rows, sequences)])
    write_json(args.out / 'calibration-inputs.json', calibration)
    base = build_base(args.model_dir, 'cuda')
    if base.config.model_type != 'qwen3_moe' or base.config.hidden_size != 2048:
        raise ValueError('Unexpected architecture')
    selection = calibrate(base, calibration_ids, calibration, args.out)
    model = base
    if args.arm in ADAPTERS:
        tensor = torch.tensor([calibration_ids[0]], device=device_of(base))
        with torch.no_grad():
            original = base(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
            model = wrap(base, arms); model.set_adapter(ADAPTERS[args.arm]); model.eval()
            after = model.get_base_model()(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
        if not torch.equal(original, after):
            raise RuntimeError('Wrapping changed base logits')
    summaries = evaluate(model, sequences, rows, args.task, tokenizer, contract, args.out,
        selection['channel'], tables['trained'] if args.arm == 'table' else None,
        500 if args.arm in ADAPTERS else 0)
    count = len((args.out / 'rows.jsonl').read_text().splitlines())
    if count != len(rows) * 3 or len(summaries) != 3:
        raise RuntimeError('Incomplete experiment')
    write_json(args.out / 'manifest.json', {'status': 'PASS', 'probe': 'expert_channel_causal_v1',
        'task': args.task, 'arm': args.arm, 'model': MODEL, 'model_revision': REVISION,
        'layer': LAYER, 'expert': EXPERT, 'channel': selection['channel'], 'scope': 'first3 original prefill only',
        'selection_sha256': hashlib.sha256((args.out / 'selection.json').read_bytes()).hexdigest(),
        'conditions': CONDITIONS, 'readout_layers': READOUT_LAYERS, 'readout_query_start': 3,
        'primary_attention_readout': 'equal-head/layer mean real0:3 mass, layers2:6, real queries>=3',
        'civil_slice': list(CIVIL_SLICE), 'wiki_slice': list(WIKI_SLICE), 'civil_rows_reused': True,
        'wikitext_revision': WIKITEXT_REVISION, 'arms': metadata, 'fitted_files_sha256': fitted_hashes,
        'table_tensor_sha256': TABLE_HASHES['trained'] if args.arm == 'table' else None,
        'contract_examples_verified': checked, 'evaluation_examples': len(rows), 'metric_rows': count,
        'max_new_tokens': 64, 'seed': 42, 'bootstrap_draws': 10000,
        'source_rows_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
        'files_sha256': {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in args.out.rglob('*') if path.is_file()}})
    print(f'EXPERT_CHANNEL_ROWS={count}', flush=True)


if __name__ == '__main__':
    main()
