#!/usr/bin/env python3
"""Test a fixed positional mean table before fitting an input-dependent shift."""
import argparse
import base64
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import zlib

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, render, wrap
from se_gepa.prefill_patch import PrefillStatePatch
from se_gepa.prefill_shift import PrefillShift
from se_gepa.prefill_template import PrefillTemplateShift, materialize_table
from probe_attention_contributions import arm_metadata, sequence_hash
from probe_prefix_causal import paired_summary
from probe_prefill_transfer import capture, state_hash
from probe_residual_transfer import measure, equal_outputs, packed_tensor, write_json
from eval_ppl_arms import windows, WIKITEXT_REVISION

ADAPTERS = {'trained': 'prefix-m500-best', 'init': 'prefix-m500-init'}
CONDITIONS = ['base', 'trained_teacher', 'trained_donor', 'init_donor',
              'trained_global_constant', 'init_global_constant', 'trained_table',
              'init_table', 'trained_table_permuted']


def common_boundaries(sequences):
    def common(items):
        count = 0
        for tokens in zip(*items):
            if len(set(tokens)) != 1:
                break
            count += 1
        return count
    prefix = common(sequences)
    suffix = common([ids[::-1] for ids in sequences])
    if prefix < 3 or suffix < 1 or any(len(ids) <= prefix + suffix for ids in sequences):
        raise ValueError('Template must have prefix, nonempty variable middle and suffix')
    return prefix, suffix


def permutation_indices(prefix_length, suffix_length):
    generator = torch.Generator().manual_seed(42)
    prefix_slots = prefix_length - 3
    return torch.cat((torch.randperm(prefix_slots, generator=generator),
                      torch.tensor([prefix_slots]),
                      prefix_slots + 1 + torch.randperm(suffix_length, generator=generator)))


def role_parts(delta, prefix_length, suffix_length):
    return (delta[3:prefix_length], delta[prefix_length:-suffix_length], delta[-suffix_length:])


def slots_and_energy(delta, prefix_length, suffix_length):
    prefix, middle, suffix = role_parts(delta, prefix_length, suffix_length)
    slots = torch.cat((prefix, middle.mean(0)[None], suffix))
    energies = torch.cat((prefix.double().square().sum(-1),
                          middle.double().square().sum(-1).mean()[None],
                          suffix.double().square().sum(-1)))
    return slots, energies


def packed_double(tensor):
    raw = tensor.contiguous().numpy().astype('<f8').tobytes()
    return {'dtype': 'float64-little-endian', 'shape': list(tensor.shape),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'zlib_base64': base64.b64encode(zlib.compress(raw)).decode()}


def new_moments(slot_count, hidden_size):
    return {'n': 0, 'slot_sum': torch.zeros(slot_count, hidden_size, dtype=torch.float64),
            'slot_mean_squared_norm_sum': torch.zeros(slot_count, dtype=torch.float64),
            'slot_token_energy_sum': torch.zeros(slot_count, dtype=torch.float64),
            'global_mean_sum': torch.zeros(hidden_size, dtype=torch.float64),
            'global_token_energy_sum': 0., 'prefix_first': None, 'prefix_max_abs_difference': 0.}


def add_moments(moments, slots, energies, global_mean, global_energy, prefix_length):
    moments['n'] += 1
    moments['slot_sum'] += slots.double()
    moments['slot_mean_squared_norm_sum'] += slots.double().square().sum(-1)
    moments['slot_token_energy_sum'] += energies
    moments['global_mean_sum'] += global_mean.double()
    moments['global_token_energy_sum'] += global_energy
    prefix = slots[:prefix_length-3]
    if moments['prefix_first'] is None:
        moments['prefix_first'] = prefix.clone()
    difference = float((prefix-moments['prefix_first']).abs().max())
    moments['prefix_max_abs_difference'] = max(moments['prefix_max_abs_difference'], difference)


def write_moments(path, moments):
    n = moments['n']
    variance = (moments['slot_mean_squared_norm_sum']/n -
                (moments['slot_sum']/n).square().sum(-1)).clamp_min(0)
    write_json(path, {'n': n, 'slot_sum': packed_double(moments['slot_sum']),
        'slot_mean_squared_norm_sum': moments['slot_mean_squared_norm_sum'].tolist(),
        'slot_token_energy_sum': moments['slot_token_energy_sum'].tolist(),
        'global_mean_sum': packed_double(moments['global_mean_sum']),
        'global_token_energy_sum': moments['global_token_energy_sum'],
        'slot_between_example_variance': variance.tolist(),
        'prefix_slots_exact': moments['prefix_max_abs_difference'] == 0,
        'prefix_max_abs_difference': moments['prefix_max_abs_difference'],
        'prefix_invariance_policy': 'Exactness and maximum drift reported; no numeric invariance threshold used.'})


def role_geometry(slots, energies, vector, table, length, prefix_length, suffix_length):
    boundaries = {'prefix': slice(0, prefix_length-3),
                  'middle': slice(prefix_length-3, prefix_length-2),
                  'suffix': slice(prefix_length-2, None)}
    sizes = {'prefix': prefix_length-3, 'middle': length-prefix_length-suffix_length,
             'suffix': suffix_length}
    result = {}
    for name, section in boundaries.items():
        mean, target, energy = slots[section].double(), table[section].double(), float(energies[section].mean())
        global_error = max(0., energy - 2*float((mean*vector.double()).sum(-1).mean()) + float(vector.double().square().sum()))
        table_error = max(0., energy - 2*float((mean*target).sum(-1).mean()) + float(target.square().sum(-1).mean()))
        result[name] = {'positions': sizes[name], 'delta_energy': energy,
                        'global_mse': global_error, 'table_mse': table_error}
    return result


@torch.no_grad()
def fit_tables(teacher, sequences, sources, out, layer=3, prefix_length=44, suffix_length=34):
    base = teacher.get_base_model()
    stored = {kind: [] for kind in ADAPTERS}
    for i, (ids, source) in enumerate(zip(sequences, sources)):
        original = capture(base, ids, layer, i == 0)
        for kind, adapter in ADAPTERS.items():
            teacher.set_adapter(adapter); teacher.eval()
            donor = capture(teacher, ids, layer, i == 0)
            delta = donor-original
            slots, energies = slots_and_energy(delta, prefix_length, suffix_length)
            stored[kind].append({'slots': slots, 'energies': energies, 'mean': delta[3:].mean(0),
                'energy': float(delta[3:].double().square().sum(-1).mean()),
                'key': source['id'], 'sequence_sha256': sequence_hash(ids), 'length': len(ids),
                'base_sha256': state_hash(original), 'donor_sha256': state_hash(donor)})
        print(json.dumps({'calibration_example': i, 'key': source['id']}), flush=True)
    vectors = {kind: torch.stack([r['mean'] for r in records]).mean(0) for kind, records in stored.items()}
    tables = {kind: torch.stack([r['slots'] for r in records]).mean(0) for kind, records in stored.items()}
    permutation = permutation_indices(prefix_length, suffix_length)
    tables['permuted'] = tables['trained'][permutation].clone()
    for kind in ADAPTERS:
        moments = new_moments(*tables[kind].shape)
        with (out/f'calibration-{kind}.jsonl').open('w') as stream:
            for record in stored[kind]:
                slots, energies, mean, energy = [record[k] for k in ('slots', 'energies', 'mean', 'energy')]
                add_moments(moments, slots, energies, mean, energy, prefix_length)
                row = {k: v for k,v in record.items() if k not in ('slots','energies','mean','energy')}
                row.update(global_mean=packed_tensor(mean), middle_mean=packed_tensor(slots[prefix_length-3]),
                    global_delta_energy=energy, slot_token_energy=energies.tolist(),
                    roles=role_geometry(slots,energies,vectors[kind],tables[kind],record['length'],prefix_length,suffix_length))
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        write_moments(out/f'calibration-moments-{kind}.json', moments)
        write_json(out/f'table-{kind}.json', {'table': packed_tensor(tables[kind]),
            'definition': 'equal-example prefix slots, middle token mean, end-aligned suffix slots',
            'calibration_n': len(sequences), 'prefix_length': prefix_length, 'suffix_length': suffix_length})
    write_json(out/'vectors.json', {'calibration_n': len(sequences),
        'definition': 'mean over positions >=3 within each example, then equal mean across calibration examples',
        'global_vectors': {kind: packed_tensor(v) for kind,v in vectors.items()},
        'global_norms': {kind: float(v.norm()) for kind,v in vectors.items()},
        'permutation_seed': 42, 'permutation': permutation.tolist(),
        'prefix_length': prefix_length, 'suffix_length': suffix_length})
    return vectors, tables


@torch.no_grad()
def evaluate(teacher, sequences, sources, task, tokenizer, contract, out, vectors, tables,
             layer=3, max_new_tokens=64, prefix_length=44, suffix_length=34):
    base = teacher.get_base_model(); reference = {}; groups = {name: [] for name in CONDITIONS}; gates = []
    moments = {kind: new_moments(*tables[kind].shape) for kind in ADAPTERS}
    with (out/'rows.jsonl').open('w') as stream, (out/'state-diagnostics.jsonl').open('w') as diagnostic:
        for i, (ids, source) in enumerate(zip(sequences,sources)):
            original = capture(base,ids,layer,i == 0); donors = {}
            diag = {'key': source['id'], 'sequence_sha256': sequence_hash(ids), 'length': len(ids),
                    'base_sha256': state_hash(original)}
            for kind, adapter in ADAPTERS.items():
                teacher.set_adapter(adapter); teacher.eval()
                donors[kind] = capture(teacher,ids,layer,i == 0)
                delta = donors[kind]-original
                slots, energies = slots_and_energy(delta,prefix_length,suffix_length)
                energy = float(delta[3:].double().square().sum(-1).mean())
                add_moments(moments[kind],slots,energies,delta[3:].mean(0),energy,prefix_length)
                diag[kind] = {'donor_sha256': state_hash(donors[kind]), 'global_delta_energy': energy,
                    'slot_token_energy': energies.tolist(),
                    'roles': role_geometry(slots,energies,vectors[kind],tables[kind],len(ids),prefix_length,suffix_length)}
            diagnostic.write(json.dumps(diag,allow_nan=False)+'\n'); diagnostic.flush()
            for name in CONDITIONS:
                kind = name.split('_')[0]; context = nullcontext(); addition = None
                if name.endswith('_donor'):
                    context = PrefillStatePatch(base,layer,len(ids),'after3',donors[kind],1)
                elif name.endswith('_global_constant'):
                    addition = vectors[kind].expand(len(ids)-3,-1)
                    context = PrefillShift(base,layer,len(ids),vectors[kind])
                elif '_table' in name:
                    table = tables['permuted' if name.endswith('_permuted') else kind]
                    addition = materialize_table(table,len(ids),prefix_length,suffix_length)
                    context = PrefillTemplateShift(base,layer,len(ids),table,prefix_length,suffix_length)
                if name == 'trained_teacher':
                    teacher.set_adapter(ADAPTERS['trained']); teacher.eval()
                with context as hook:
                    row, first = measure(teacher if name == 'trained_teacher' else base,ids,task,
                                         tokenizer,source.get('labels'),contract,max_new_tokens)
                row.update(key=source['id'],condition=name,sequence_sha256=sequence_hash(ids),prompt_tokens=len(ids))
                if hook is not None:
                    if hook.calls != (row['completion_tokens'] if task == 'civil' else 1) or hook.patched_positions != len(ids)-3:
                        raise RuntimeError('Incorrect original-prefill coverage')
                    if not torch.equal(hook.before[:3],hook.after[:3]): raise RuntimeError('Early states changed')
                    if not torch.allclose(hook.before,original,rtol=.005,atol=.005): raise RuntimeError('Base capture differs from recipient')
                    expected = donors[kind][3:] if addition is None else (
                        hook.before[3:].to(device_of(base))+addition.to(device_of(base))).to(base.dtype).float().cpu()
                    error = float((hook.after[3:]-expected).abs().max())
                    if error != 0: raise RuntimeError('Residual target mismatch')
                    row.update(patched_positions=hook.patched_positions,intervention_calls=hook.calls,
                        target_max_abs_error=error,capture_max_abs_difference=float((hook.before-original).abs().max()),
                        first3_before_sha256=state_hash(hook.before[:3]),first3_after_sha256=state_hash(hook.after[:3]))
                if name == 'base':
                    reference[source['id']] = {'row': dict(row),'first':first}
                    if i < (5 if task == 'civil' else 1):
                        with PrefillTemplateShift(base,layer,len(ids),torch.zeros_like(tables['trained']),prefix_length,suffix_length):
                            identity,_ = measure(base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                        if not equal_outputs(row,identity,task): raise RuntimeError('Zero-table identity failed')
                        gates.append({'key':source['id'],'zero_table_exact':True})
                if name == 'trained_teacher' and i == 0:
                    for adapter in ADAPTERS.values():
                        teacher.set_adapter(adapter); teacher.eval()
                        again,_ = measure(base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                        if not equal_outputs(reference[source['id']]['row'],again,task): raise RuntimeError('Adapter contaminated base')
                    gates.append({'key':source['id'],'base_after_each_adapter_exact':True})
                if name.endswith('_global_constant') and i == 0:
                    repeated = vectors[kind].expand_as(tables[kind]).clone()
                    with PrefillTemplateShift(base,layer,len(ids),repeated,prefix_length,suffix_length):
                        parity,_ = measure(base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                    if not equal_outputs(row,parity,task): raise RuntimeError('Global-vector path differs from repeated table')
                    gates.append({'key':source['id'],'global_table_path_exact':kind})
                ref = reference[source['id']]
                if task == 'civil':
                    p,q = ref['first'].log_softmax(-1),first.log_softmax(-1)
                    row['first_token_kl_from_intact'] = float((p.exp()*(p-q)).sum())
                else: row['delta_nll'] = row['nll']-ref['row']['nll']
                groups[name].append(row); stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
            print(json.dumps({'example':i,'key':source['id'],'conditions_completed':len(CONDITIONS)}),flush=True)
    summaries = []
    for name, rows in groups.items():
        if task == 'civil': summary = paired_summary(rows,reference,42)
        else:
            delta = torch.tensor([r['delta_nll'] for r in rows],dtype=torch.float64)
            draws = torch.randint(len(rows),(10000,len(rows)),generator=torch.Generator().manual_seed(42))
            ci = torch.quantile(delta[draws].mean(-1),torch.tensor([.025,.975],dtype=torch.float64))
            nll = sum(r['nll'] for r in rows)/len(rows)
            summary = {'n':len(rows),'nll':nll,'ppl':math.exp(nll),'paired_delta_nll':float(delta.mean()),
                'paired_bootstrap_ci95':ci.tolist(),'targets':sum(r['targets'] for r in rows)}
        summaries.append({'condition':name,**summary})
    for kind in ADAPTERS: write_moments(out/f'evaluation-moments-{kind}.json',moments[kind])
    write_json(out/'summary.json',summaries); write_json(out/'gates.json',gates)
    return summaries


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True); parser.add_argument('--model-revision', required=True)
    parser.add_argument('--arms-spec', type=Path, required=True); parser.add_argument('--contract-sample', type=Path, required=True)
    parser.add_argument('--task', choices=['civil','wiki'], required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if transformers.__version__ != '5.16.1' or peft.__version__ != '0.20.0': raise RuntimeError('Unexpected runtime')
    arms = json.loads(args.arms_spec.read_text())
    if len(arms) != 2 or {a['name'] for a in arms} != set(ADAPTERS.values()): raise ValueError('Requires matched trained and init donors')
    metadata = [arm_metadata(a) for a in arms]
    if any(m['receipt']['peft_type'] != 'PREFIX_TUNING' or m['receipt']['base_revision'] != args.model_revision for m in metadata):
        raise ValueError('Wrong donor architecture or base revision')
    by_name = {m['name']:m for m in metadata}
    trained, init = [by_name[ADAPTERS[k]] for k in ('trained','init')]
    for field in ('repo','commit','prefix','archive_sha256'):
        if trained['receipt'][field] != init['receipt'][field]: raise ValueError('Donors are from different training cells')
    if [trained['receipt']['selected_step'],init['receipt']['selected_step']] != [1125,0]:
        raise ValueError('Wrong checkpoint steps')
    if any(m['receipt']['num_virtual_tokens'] != 500 for m in metadata): raise ValueError('Wrong prefix length')
    configs = [{k:v for k,v in m['adapter_config'].items() if k != 'base_model_name_or_path'} for m in (trained,init)]
    if configs[0] != configs[1]: raise ValueError('Donor configurations differ')
    expected_weights = ['9b3da8cb8177383fe9d8b290541424bb64ac21a8c9d14787339787b03dff44ea',
                        'ed0ec06abe15f4dd9a8323aad1ebb37d0e7255c060e382487a348cc12b06b965']
    if [m['receipt']['adapter_files']['adapter_model.safetensors'] for m in (trained,init)] != expected_weights:
        raise ValueError('Donor weights differ from audited checkpoints')
    root = Path(__file__).resolve().parents[1]
    paths = [root/'data'/n for n in ['civil_v2_val_seed42_n200.jsonl','civil_v2_test_head2000.jsonl']]
    validation,test = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    calibration,rows = validation[:64],test[1200:1400]
    if len(calibration) != 64 or len(rows) != 200: raise ValueError('Incomplete frozen rows')
    for field in ('id','text'):
        if len({r[field] for r in calibration}) != 64 or len({r[field] for r in rows}) != 200:
            raise ValueError('Duplicate frozen rows')
        if {r[field] for r in rows} & {r[field] for r in validation+test[:200]+test[1400:2000]}:
            raise ValueError('Calibration or previous evaluation overlap')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True); contract = load_contract()
    checked = check_contract(tokenizer,contract,validation,json.loads(args.contract_sample.read_text()))
    calibration_ids = [render(tokenizer,contract[1][SEED_KEY],r['text'],contract) for r in calibration]
    prefix_length,suffix_length = common_boundaries(calibration_ids)
    if (prefix_length,suffix_length) != (44,34): raise RuntimeError('Frozen template boundaries changed')
    if args.task == 'civil':
        sequences = [render(tokenizer,contract[1][SEED_KEY],r['text'],contract) for r in rows]
        if any(ids[:prefix_length] != calibration_ids[0][:prefix_length] or
               ids[-suffix_length:] != calibration_ids[0][-suffix_length:] for ids in sequences):
            raise ValueError('Evaluation rendering differs from calibration template')
    else:
        tokens = windows(tokenizer,2048,80)
        if tokens.shape != (80,2048): raise RuntimeError('Missing Wiki windows')
        sequences = tokens[64:80].tolist(); rows = [{'id':f'wiki-test-window-{i}'} for i in range(64,80)]
    if any(len(ids) <= prefix_length+suffix_length for ids in calibration_ids+sequences):
        raise ValueError('No variable middle in input')
    args.out.mkdir(parents=True,exist_ok=False)
    for name,sources,ids_list in [('inputs.json',rows,sequences),('calibration-inputs.json',calibration,calibration_ids)]:
        write_json(args.out/name,[{**r,'input_ids':ids} for r,ids in zip(sources,ids_list)])
    write_json(args.out/'used-slices.json',{'calibration':'Civil validation[0:64]',
        'civil_evaluation':[1200,1400],'civil_excluded_test_slices':[[0,200],[1400,2000]],
        'civil_validation_excluded':True,'civil_id_and_text_disjoint':True,
        'wiki_evaluation':[64,80],'wiki_previous_intervention_windows':[0,64],
        'history':'Civil head2000 previously evaluated by base; new slices are fresh to this intervention series.'})
    write_json(args.out/'template.json',{'source':'Civil calibration only','prefix_length':prefix_length,
        'suffix_length':suffix_length,'prefix_input_ids':calibration_ids[0][:prefix_length],
        'suffix_input_ids':calibration_ids[0][-suffix_length:],
        'evaluation_template_ids_verified':args.task == 'civil',
        'wiki_rule':'same numerical positions; no template-match gate'})
    base = build_base(args.model_dir,'cuda'); tensor = torch.tensor([sequences[0]],device=device_of(base))
    with torch.no_grad():
        native = base(input_ids=tensor,use_cache=False,logits_to_keep=1).logits
        teacher = wrap(base,arms)
        for adapter in ADAPTERS.values():
            teacher.set_adapter(adapter); teacher.eval()
            wrapped = teacher.get_base_model()(input_ids=tensor,use_cache=False,logits_to_keep=1).logits
            if not torch.equal(native,wrapped): raise RuntimeError('Wrapping or adapter switch changed base')
    del native,wrapped,tensor
    vectors,tables = fit_tables(teacher,calibration_ids,calibration,args.out)
    summaries = evaluate(teacher,sequences,rows,args.task,tokenizer,contract,args.out,vectors,tables)
    count = len((args.out/'rows.jsonl').read_text().splitlines())
    if count != len(summaries)*len(rows): raise RuntimeError('Missing rows')
    write_json(args.out/'manifest.json',{
        'status':'PASS','probe':'prefill_static_template_v1','task':args.task,'arms':metadata,
        'model_revision':args.model_revision,'layer':3,'conditions':CONDITIONS,'alpha':1,
        'scope':'after3 original prefill only','prefix_length':44,'suffix_length':34,'table_slots':76,
        'recipient':'base without prefix; generated queries unpatched',
        'calibration':'Civil validation[:64]; each slot example-balanced; middle first averaged within each example',
        'selection':'Civil test[1200:1400] or WikiText test windows[64:80],2048 tokens',
        'wikitext_revision':WIKITEXT_REVISION,'data_history':'new intervention rows; Civil previously scored by base',
        'source_rows_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'contract_examples_verified':checked,'contract_sample_sha256':hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
        'base_wrap_logits_exact':True,'observation_logits_exact':True,'metric_rows':count,
        'max_new_tokens':64,'permutation_seed':42,
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}})
    print(f'PREFILL_TEMPLATE_ROWS={count}',flush=True)


if __name__ == '__main__': main()
