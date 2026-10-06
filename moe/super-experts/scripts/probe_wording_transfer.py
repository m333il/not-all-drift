#!/usr/bin/env python3
"""Separate intact-prefix and donor-state robustness under fixed wordings."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil

import torch

from probe_prefill_confirmation import BOOTSTRAP_DRAWS, MODEL, REVISION, civil_summary
from probe_prefill_wording import render_pair
from probe_prefill_template import common_boundaries
from probe_prefill_transfer import capture, state_hash
from probe_residual_transfer import equal_outputs, measure, write_json
from probe_attention_contributions import arm_metadata, sequence_hash
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, wrap
from se_gepa.prefill_patch import PrefillStatePatch

ADAPTERS = {'trained': 'prefix-m500-best', 'init': 'prefix-m500-init'}
CONDITIONS = ['base', 'trained_teacher', 'init_teacher', 'trained_donor', 'init_donor']
WEIGHT_HASHES = {'trained': '9b3da8cb8177383fe9d8b290541424bb64ac21a8c9d14787339787b03dff44ea',
                 'init': 'ed0ec06abe15f4dd9a8323aad1ebb37d0e7255c060e382487a348cc12b06b965'}


def check_teachers(metadata, revision):
    if revision != REVISION or len(metadata) != 2 or {m['name'] for m in metadata} != set(ADAPTERS.values()):
        raise ValueError('Requires matched trained/init teachers on the pinned base')
    by_name = {m['name']: m for m in metadata}
    trained, init = [by_name[ADAPTERS[k]] for k in ('trained', 'init')]
    for field in ('repo', 'commit', 'prefix', 'archive_sha256'):
        if trained['receipt'][field] != init['receipt'][field]:
            raise ValueError('Teachers originate from different training cells')
    for kind, step in [('trained', 1125), ('init', 0)]:
        entry = by_name[ADAPTERS[kind]]; receipt = entry['receipt']
        if (receipt['peft_type'] != 'PREFIX_TUNING' or receipt['base_revision'] != REVISION
                or receipt['num_virtual_tokens'] != 500 or receipt['selected_step'] != step
                or receipt['adapter_files']['adapter_model.safetensors'] != WEIGHT_HASHES[kind]):
            raise ValueError('Teacher differs from the audited checkpoint')
    configs = [{k: v for k, v in entry['adapter_config'].items() if k != 'base_model_name_or_path'} for entry in (trained, init)]
    if configs[0] != configs[1]:
        raise ValueError('Teacher configurations differ')


@torch.no_grad()
def evaluate(teacher, sequences, sources, tokenizer, contract, out, layer=3, max_new_tokens=64):
    base = teacher.get_base_model()
    references = {}; groups = {name: [] for name in CONDITIONS}; gates = []
    with (out / 'rows.jsonl').open('w') as stream, (out / 'state-diagnostics.jsonl').open('w') as diagnostic:
        for i, (ids, source) in enumerate(zip(sequences, sources)):
            original = capture(base, ids, layer, i == 0)
            donors = {}
            diag = {'key': source['id'], 'sequence_sha256': sequence_hash(ids), 'prompt_tokens': len(ids),
                    'base_sha256': state_hash(original), 'base_norm_per_position': original.norm(dim=-1).tolist(),
                    'base_max_abs_per_position': original.abs().amax(-1).tolist()}
            for kind, adapter in ADAPTERS.items():
                teacher.set_adapter(adapter); teacher.eval()
                donor = capture(teacher, ids, layer, i == 0); donors[kind] = donor
                diag[kind] = {'donor_sha256': state_hash(donor), 'donor_norm_per_position': donor.norm(dim=-1).tolist(),
                    'donor_max_abs_per_position': donor.abs().amax(-1).tolist(),
                    'delta_norm_per_position': (donor - original).norm(dim=-1).tolist(),
                    'base_donor_cosine_per_position': torch.nn.functional.cosine_similarity(original, donor, dim=-1).tolist()}
            diagnostic.write(json.dumps(diag, allow_nan=False) + '\n'); diagnostic.flush()
            if i == 0:
                gates.append({'key': source['id'], 'observation_logits_exact': True, 'models': ['base', 'trained', 'init']})
            for name in CONDITIONS:
                kind = name.split('_')[0]
                is_teacher, is_donor = name.endswith('_teacher'), name.endswith('_donor')
                if is_teacher:
                    teacher.set_adapter(ADAPTERS[kind]); teacher.eval()
                context = PrefillStatePatch(base, layer, len(ids), 'after3', donors[kind], 1) if is_donor else nullcontext()
                with context as hook:
                    row, first = measure(teacher if is_teacher else base, ids, 'civil', tokenizer,
                                         source['labels'], contract, max_new_tokens)
                row.update(key=source['id'], condition=name, sequence_sha256=sequence_hash(ids), prompt_tokens=len(ids))
                if hook is not None:
                    if hook.calls != row['completion_tokens'] or hook.patched_positions != len(ids) - 3:
                        raise RuntimeError('Incorrect original-prefill donation coverage')
                    if not torch.equal(hook.before[:3], hook.after[:3]):
                        raise RuntimeError('Early native states changed')
                    if not torch.allclose(hook.before, original, rtol=.005, atol=.005):
                        raise RuntimeError('Base capture differs from native recipient')
                    error = float((hook.after[3:] - donors[kind][3:]).abs().max())
                    if error != 0 or not torch.isfinite(hook.after).all():
                        raise RuntimeError('Donor-state assignment is not finite and exact')
                    row.update(scope='after3', alpha=1, patched_positions=hook.patched_positions,
                        intervention_calls=hook.calls, target_max_abs_error=error, first3_exact=True,
                        capture_max_abs_difference=float((hook.before - original).abs().max()),
                        first3_before_sha256=state_hash(hook.before[:3]), first3_after_sha256=state_hash(hook.after[:3]))
                if name == 'base':
                    references[source['id']] = {'row': dict(row), 'first': first}
                    if i < 5:
                        with PrefillStatePatch(base, layer, len(ids), 'after3', donors['trained'], 0) as identity_hook:
                            identity, _ = measure(base, ids, 'civil', tokenizer, source['labels'], contract, max_new_tokens)
                        if (not equal_outputs(row, identity, 'civil') or identity_hook.calls != row['completion_tokens']
                                or identity_hook.patched_positions != len(ids) - 3):
                            raise RuntimeError('Zero-alpha donation identity failed')
                        gates.append({'key': source['id'], 'zero_alpha_exact': True})
                if is_teacher and i == 0:
                    restored, _ = measure(base, ids, 'civil', tokenizer, source['labels'], contract, max_new_tokens)
                    if not equal_outputs(references[source['id']]['row'], restored, 'civil'):
                        raise RuntimeError('Teacher generation or adapter switch contaminated native base')
                    gates.append({'key': source['id'], 'base_after_teacher_exact': kind,
                                  'active_adapter': ADAPTERS[kind]})
                ref = references[source['id']]['first']
                p, q = ref.log_softmax(-1), first.log_softmax(-1)
                row['first_token_kl_from_intact'] = float((p.exp() * (p - q)).sum())
                groups[name].append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps({'example': i, 'key': source['id'], 'conditions_completed': len(CONDITIONS)}), flush=True)
    summaries = [{'condition': name, 'bootstrap_draws': BOOTSTRAP_DRAWS, **civil_summary(rows, references)}
                 for name, rows in groups.items()]
    write_json(out / 'summary.json', summaries); write_json(out / 'gates.json', gates)
    return summaries


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True); parser.add_argument('--model-revision', required=True)
    parser.add_argument('--arms-spec', type=Path, required=True); parser.add_argument('--contract-sample', type=Path, required=True)
    parser.add_argument('--variant', choices=['original', 'paraphrase_a', 'paraphrase_b'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != '5.16.1' or peft.__version__ != '0.20.0':
        raise RuntimeError('Unexpected runtime')
    arms = json.loads(args.arms_spec.read_text())
    metadata = [arm_metadata(arm) for arm in arms]
    check_teachers(metadata, args.model_revision)
    root = Path(__file__).resolve().parents[1]
    config_path = root / 'data/prefill_wording_20260924.json'
    config = json.loads(config_path.read_text())
    if (config['schema'] != 'prefill_wording_v1' or config['model_revision'] != REVISION
            or config['civil_slice'] != [200, 400] or config['instruction_span'] != [3, 25]
            or config['prefix_length'] != 44 or config['suffix_length'] != 34
            or set(config['variants']) != {'original', 'paraphrase_a', 'paraphrase_b'}):
        raise ValueError('Frozen wording configuration mismatch')
    witness_root = root / 'data/prefill_template_frozen_20260924'
    witness_provenance = json.loads((witness_root / 'provenance.json').read_text())
    witness_hashes = {name: hashlib.sha256((witness_root / name).read_bytes()).hexdigest()
                      for name in ('provenance.json', 'template.json', 'calibration-inputs.json')}
    if (witness_provenance['model_revision'] != REVISION or witness_provenance['origin_status'] != 'VERIFIED'
            or any(witness_hashes[name] != witness_provenance['files_sha256'][name]
                   for name in ('template.json', 'calibration-inputs.json'))):
        raise ValueError('Frozen rendering witnesses fail provenance verification')
    template = json.loads((witness_root / 'template.json').read_text())
    calibration = json.loads((witness_root / 'calibration-inputs.json').read_text())
    if (len(calibration) != 64 or template['prefix_length'] != 44 or template['suffix_length'] != 34
            or len(template['prefix_input_ids']) != 44 or len(template['suffix_input_ids']) != 34):
        raise ValueError('Frozen rendering witness geometry mismatch')
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
    checked = check_contract(tokenizer, contract, validation, json.loads(args.contract_sample.read_text()))
    if config['variants']['original'] != contract[1][SEED_KEY]:
        raise ValueError('Original instruction differs from archived contract')
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
        if common_boundaries([row['input_ids'] for row in entries]) != (44, 34):
            raise ValueError('Wording changed common prefix/suffix boundaries')
        witnesses[split] = entries
    variant_prefix = witnesses['calibration'][0]['input_ids'][:44]
    if any(row['input_ids'][:44] != variant_prefix for row in witnesses['evaluation']):
        raise ValueError('Evaluation prefix differs from calibration rendering')
    args.out.mkdir(parents=True, exist_ok=False)
    original_out = args.out / 'rendering-source'; original_out.mkdir()
    for name in witness_hashes:
        shutil.copyfile(witness_root / name, original_out / name)
    shutil.copyfile(config_path, args.out / 'wording-config.json')
    write_json(args.out / 'inputs.json', witnesses['evaluation'])
    write_json(args.out / 'calibration-witness.json', witnesses['calibration'])
    write_json(args.out / 'rendering-gates.json', {'variant': args.variant, 'calibration_examples': 64,
        'evaluation_examples': 200, 'original_calibration_ids_exact': True, 'equal_sequence_lengths': True,
        'first3_exact': True, 'positions_25_onward_exact': True, 'instruction_span': [3, 25],
        'common_prefix_length': 44, 'common_suffix_length': 34,
        'original_prefix_input_ids': template['prefix_input_ids'], 'variant_prefix_input_ids': variant_prefix,
        'suffix_input_ids': template['suffix_input_ids']})
    write_json(args.out / 'used-slices.json', {'civil_evaluation': [200, 400], 'previous_confirmation_slice': [200, 1000],
        'reused_from_confirmation': True, 'reused_from_wording': True,
        'calibration_witness': 'Civil validation[0:64], rendering only; no fitting',
        'civil_validation_excluded': True, 'unique_evaluation_ids_and_text': True,
        'history': 'Same 200 wording/confirmation examples; no fresh-sample claim.'})
    base = build_base(args.model_dir, 'cuda')
    if base.config.model_type != 'qwen3_moe' or base.config.hidden_size != 2048:
        raise ValueError('Wrong base model architecture')
    sequences = [row['input_ids'] for row in witnesses['evaluation']]
    tensor = torch.tensor([sequences[0]], device=device_of(base))
    with torch.no_grad():
        native = base(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
        teacher = wrap(base, arms)
        for adapter in ADAPTERS.values():
            teacher.set_adapter(adapter); teacher.eval()
            after = teacher.get_base_model()(input_ids=tensor, use_cache=False, logits_to_keep=1).logits
            if not torch.equal(native, after):
                raise RuntimeError('Wrapping or adapter switch changed native base')
    del native, after, tensor
    summaries = evaluate(teacher, sequences, rows, tokenizer, contract, args.out)
    count = len((args.out / 'rows.jsonl').read_text().splitlines())
    if count != 1000 or len(summaries) != 5:
        raise RuntimeError('Incomplete wording transfer results')
    write_json(args.out / 'manifest.json', {'status': 'PASS', 'probe': 'wording_transfer_v1', 'task': 'civil',
        'variant': args.variant, 'instruction': instruction, 'instruction_span': [3, 25], 'arms': metadata,
        'wording_config_path': str(config_path.relative_to(root)), 'wording_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'model': MODEL, 'model_revision': args.model_revision, 'layer': 3, 'alpha': 1,
        'scope': 'after3 original prefill only', 'prefix_length': 44, 'suffix_length': 34,
        'conditions': CONDITIONS, 'recipient': 'native base without prefix; generated queries unpatched',
        'donor': 'input-matched intact trained/init prefix under the same wording', 'fitting': 'none',
        'rendering_source_files_sha256': witness_hashes,
        'contract_examples_verified': 64, 'archived_contract_examples_verified': checked,
        'contract_sample_sha256': hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
        'rendering_examples_verified': 264, 'observation_logits_exact': True,
        'base_wrap_logits_exact': True, 'base_adapter_switch_logits_exact': True,
        'selection': 'Civil test[200:400], reused from wording and final confirmation', 'evaluation_examples': 200,
        'metric_rows': count, 'max_new_tokens': 64, 'seed': 42, 'bootstrap_draws': BOOTSTRAP_DRAWS,
        'source_rows_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__},
        'files_sha256': {str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in args.out.rglob('*') if path.is_file()}})
    print(f'WORDING_TRANSFER_ROWS={count}', flush=True)


if __name__ == '__main__':
    main()
