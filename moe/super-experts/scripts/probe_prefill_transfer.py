#!/usr/bin/env python3
"""Test donor-state scope while preserving the base's early residual states."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from se_gepa.arms import SEED_KEY, build_base, check_contract, device_of, load_contract, render, wrap
from se_gepa.prefill_patch import PrefillStatePatch
from probe_attention_contributions import arm_metadata, sequence_hash
from probe_prefix_causal import paired_summary
from probe_residual_transfer import measure, equal_outputs, write_json
from eval_ppl_arms import windows, WIKITEXT_REVISION


def designs(task):
    result = [('base',None,0),('teacher',None,0),('all_prefill','all',1),('after3','after3',1),('after3_0.1','after3',.1)]
    return result + [('last_query','last',1)] if task=='civil' else result


def state_hash(states):
    return hashlib.sha256(states.contiguous().numpy().astype('<f4').tobytes()).hexdigest()


@torch.no_grad()
def capture(model, ids, layer, check=False):
    tensor=torch.tensor([ids],device=device_of(model))
    options=dict(input_ids=tensor,attention_mask=torch.ones_like(tensor),use_cache=False,logits_to_keep=1)
    native=model(**options).logits if check else None
    with PrefillStatePatch(model,layer,len(ids)) as hook:
        observed=model(**options).logits
    if check and not torch.equal(native,observed):
        raise RuntimeError('Observation changed logits')
    if hook.calls!=1 or not torch.isfinite(hook.before).all():
        raise RuntimeError('Invalid donor capture')
    return hook.before


@torch.no_grad()
def evaluate(teacher,sequences,sources,task,tokenizer,contract,out,layer=3,max_new_tokens=64):
    base=teacher.get_base_model()
    panel=designs(task)
    reference={};groups={name:[] for name,_,_ in panel};gates=[]
    with (out/'rows.jsonl').open('w') as stream,(out/'state-diagnostics.jsonl').open('w') as diagnostic:
        for i,(ids,source) in enumerate(zip(sequences,sources)):
            original=capture(base,ids,layer,i==0)
            donor=capture(teacher,ids,layer,i==0)
            delta=donor-original
            diagnostic.write(json.dumps({'key':source['id'],'sequence_sha256':sequence_hash(ids),
                'base_sha256':state_hash(original),'donor_sha256':state_hash(donor),
                'base_norm_per_position':original.norm(dim=-1).tolist(),'donor_norm_per_position':donor.norm(dim=-1).tolist(),
                'delta_norm_per_position':delta.norm(dim=-1).tolist(),
                'base_donor_cosine_per_position':torch.nn.functional.cosine_similarity(original,donor,dim=-1).tolist(),
                'base_max_abs_per_position':original.abs().amax(-1).tolist(),'donor_max_abs_per_position':donor.abs().amax(-1).tolist()},allow_nan=False)+'\n');diagnostic.flush()
            for name,scope,alpha in panel:
                context=PrefillStatePatch(base,layer,len(ids),scope,donor,alpha) if scope else nullcontext()
                with context as hook:
                    row,first=measure(teacher if name=='teacher' else base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                row.update(key=source['id'],condition=name,sequence_sha256=sequence_hash(ids),prompt_tokens=len(ids))
                if hook is not None:
                    start=0 if scope=='all' else 3 if scope=='after3' else len(ids)-1
                    if hook.calls!=(row['completion_tokens'] if task=='civil' else 1) or hook.patched_positions!=len(ids)-start:
                        raise RuntimeError('Incorrect prefill intervention coverage')
                    if not torch.equal(hook.after[:start],hook.before[:start]):
                        raise RuntimeError('Unselected residual states changed')
                    discrepancy=float((hook.before-original).abs().max())
                    if not torch.allclose(hook.before,original,rtol=.005,atol=.005):
                        raise RuntimeError('Captured base differs from recipient prefill')
                    if alpha==1 and not torch.equal(hook.after[start:],donor[start:]):
                        raise RuntimeError('Donor assignment not exact')
                    interpolation_error=0.0
                    if alpha!=1:
                        expected=((1-alpha)*hook.before[start:].to(device_of(base))+alpha*donor[start:].to(device_of(base))).to(base.dtype).float().cpu()
                        interpolation_error=float((hook.after[start:]-expected).abs().max())
                        if interpolation_error!=0:
                            raise RuntimeError('Interpolated residual does not match target')
                    row.update(scope=scope,alpha=alpha,interpolation_max_abs_error=interpolation_error,patched_positions=hook.patched_positions,intervention_calls=hook.calls,
                        unselected_states_exact=True,capture_max_abs_difference=discrepancy,
                        donor_target_max_abs_error=float((hook.after[start:]-donor[start:]).abs().max()),
                        first3_after_sha256=state_hash(hook.after[:3]),first3_before_sha256=state_hash(hook.before[:3]))
                if name=='base':
                    reference[source['id']]={'row':dict(row),'first':first}
                    if i<(5 if task=='civil' else 1):
                        with PrefillStatePatch(base,layer,len(ids),'all',donor,0):
                            identity,_=measure(base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                        if not equal_outputs(row,identity,task):
                            raise RuntimeError('Zero-alpha identity failed')
                        gates.append({'key':source['id'],'zero_alpha_exact':True})
                if name=='teacher' and i==0:
                    base_again,_=measure(base,ids,task,tokenizer,source.get('labels'),contract,max_new_tokens)
                    if not equal_outputs(reference[source['id']]['row'],base_again,task):
                        raise RuntimeError('Teacher contaminated base')
                    gates.append({'key':source['id'],'base_after_teacher_exact':True})
                ref=reference[source['id']]
                if task=='civil':
                    p,q=ref['first'].log_softmax(-1),first.log_softmax(-1)
                    row['first_token_kl_from_intact']=float((p.exp()*(p-q)).sum())
                else:
                    row['delta_nll']=row['nll']-ref['row']['nll']
                groups[name].append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            print(json.dumps({'example':i,'key':source['id'],'conditions_completed':len(panel)}),flush=True)
    summaries=[]
    for name,_,_ in panel:
        rows=groups[name]
        if task=='civil':
            summary=paired_summary(rows,reference,42)
        else:
            delta=torch.tensor([r['delta_nll'] for r in rows],dtype=torch.float64)
            draws=torch.randint(len(rows),(10000,len(rows)),generator=torch.Generator().manual_seed(42))
            ci=torch.quantile(delta[draws].mean(-1),torch.tensor([.025,.975],dtype=torch.float64))
            nll=sum(r['nll'] for r in rows)/len(rows)
            summary={'n':len(rows),'nll':nll,'ppl':math.exp(nll),'paired_delta_nll':float(delta.mean()),'paired_bootstrap_ci95':ci.tolist(),'targets':sum(r['targets'] for r in rows)}
        summaries.append({'condition':name,**summary})
    write_json(out/'summary.json',summaries);write_json(out/'gates.json',gates)
    return summaries


def main():
    import peft
    import transformers
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',required=True);parser.add_argument('--model-revision',required=True)
    parser.add_argument('--arms-spec',type=Path,required=True);parser.add_argument('--contract-sample',type=Path,required=True)
    parser.add_argument('--task',choices=['civil','wiki'],required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if transformers.__version__!='5.16.1' or peft.__version__!='0.20.0':raise RuntimeError('Unexpected runtime')
    arms=json.loads(args.arms_spec.read_text())
    if len(arms)!=1 or arms[0]['name']!='prefix-m500-best':raise ValueError('Frozen low-LR oracle protocol requires prefix-m500-best')
    metadata=arm_metadata(arms[0])
    if metadata['receipt']['peft_type']!='PREFIX_TUNING' or metadata['receipt']['base_revision']!=args.model_revision:raise ValueError('Wrong donor')
    root=Path(__file__).resolve().parents[1]
    paths=[root/'data'/name for name in ['civil_v2_val_seed42_n200.jsonl','civil_v2_test_head2000.jsonl']]
    validation,test=[[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    rows=test[1400:1600]
    excluded=validation+test[1800:2000]
    if len(rows)!=200 or {r['id'] for r in rows}&{r['id'] for r in excluded} or {r['text'] for r in rows}&{r['text'] for r in excluded}:raise ValueError('Overlapping or unavailable confirmation rows')
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True);contract=load_contract()
    checked=check_contract(tokenizer,contract,validation,json.loads(args.contract_sample.read_text()))
    if args.task=='civil':
        sequences=[render(tokenizer,contract[1][SEED_KEY],r['text'],contract) for r in rows]
    else:
        tokens=windows(tokenizer,2048,48)
        if tokens.shape!=(48,2048):raise RuntimeError('Missing WikiText windows')
        sequences=tokens[32:48].tolist();rows=[{'id':f'wiki-test-window-{i}'} for i in range(32,48)]
    if any(len(ids)<=3 for ids in sequences):raise ValueError('Input too short for after3')
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'inputs.json',[{**r,'input_ids':ids} for r,ids in zip(rows,sequences)])
    base=build_base(args.model_dir,'cuda');tensor=torch.tensor([sequences[0]],device=device_of(base))
    with torch.no_grad():
        native=base(input_ids=tensor,use_cache=False,logits_to_keep=1).logits
        teacher=wrap(base,arms)
        wrapped=teacher.get_base_model()(input_ids=tensor,use_cache=False,logits_to_keep=1).logits
    if not torch.equal(native,wrapped):raise RuntimeError('Wrapping changed base')
    del native,wrapped,tensor
    summaries=evaluate(teacher,sequences,rows,args.task,tokenizer,contract,args.out)
    count=len((args.out/'rows.jsonl').read_text().splitlines())
    if count!=len(summaries)*len(rows):raise RuntimeError('Missing rows')
    write_json(args.out/'manifest.json',{'status':'PASS','probe':'prefill_scope_transfer_v1','task':args.task,'arm':metadata,
        'model_revision':args.model_revision,'layer':3,'designs':[{'condition':n,'scope':s,'alpha':a} for n,s,a in designs(args.task)],
        'recipient':'base without prefix; generated queries unpatched','donor':'intact low-LR prefix on same input',
        'selection':'Civil test[1400:1600] or WikiText test windows[32:48],2048tokens','wikitext_revision':WIKITEXT_REVISION,
        'data_history':'new to intervention grid; Civil previously scored by base','source_rows_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'contract_examples_verified':checked,'contract_sample_sha256':hashlib.sha256(args.contract_sample.read_bytes()).hexdigest(),
        'base_wrap_logits_exact':True,'observation_logits_exact':True,'metric_rows':count,'max_new_tokens':64,'seed':42,
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__},
        'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}})
    print(f'PREFILL_TRANSFER_ROWS={count}',flush=True)


if __name__=='__main__':main()
