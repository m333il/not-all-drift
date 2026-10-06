import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_prefill_transfer import evaluate, designs
from se_gepa.arms import load_contract
from test_attention_contributions import wrapped


@pytest.mark.parametrize('task',['civil','wiki'])
def test_runner_scope_scoring_identity_and_small_artifacts(tmp_path,task):
    pytest.importorskip('sklearn')
    model,_=wrapped('prefix',seed=120)
    class Tokenizer:
        pad_token_id=0
        eos_token_id=None
        def decode(self,_ids,skip_special_tokens=False):return 'NONE'
    summary=evaluate(model,[[2,3,4,5],[6,7,8,9,10]],
        [{'id':'a','labels':[]},{'id':'b','labels':[]}],task,Tokenizer(),load_contract(),tmp_path,layer=1,max_new_tokens=3)
    rows=[json.loads(line) for line in (tmp_path/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==2*len(designs(task))
    assert [r['condition'] for r in summary]==[n for n,_,_ in designs(task)]
    assert ('last_query' in {r['condition'] for r in rows})==(task=='civil')
    for r in rows:
        if r['condition'] in ('after3','after3_0.1'):
            assert r['first3_after_sha256']==r['first3_before_sha256']
            assert r['patched_positions']==r['prompt_tokens']-3
        if r.get('alpha')==1:assert r['donor_target_max_abs_error']==0
        if r.get('alpha')==.1:assert r['interpolation_max_abs_error']==0
        if task=='civil':assert r['score']==0 and r['parsed_score']==1 and r['truncated']
    assert len(json.loads((tmp_path/'gates.json').read_text()))==(3 if task=='civil' else 2)
    assert len((tmp_path/'state-diagnostics.jsonl').read_text().splitlines())==2


def test_frozen_rows_exclude_prior_eval_and_validation():
    root=Path(__file__).resolve().parents[1]
    validation,test=[[json.loads(l) for l in (root/'data'/name).read_text().splitlines()]
        for name in ['civil_v2_val_seed42_n200.jsonl','civil_v2_test_head2000.jsonl']]
    rows=test[1400:1600];old=validation+test[1800:2000]
    assert len(rows)==200
    for field in ('id','text'):
        assert not {r[field] for r in rows}&{r[field] for r in old}
