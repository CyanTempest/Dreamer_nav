"""Audit full evaluation coverage, runtime, model selection, and video identity."""
import hashlib
import itertools
import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
selection=read('metrics/lwy_runtime2_model_selection.json')
methods=['dreamer_lwy','ppo_lwy','dreamer_s0','ppo_s0','planner']
hashes={m:digest(selection['selected'][m]['checkpoint'] if m in selection['selected'] else f'checkpoints/{m}/latest.pkl') for m in methods if m!='planner'}
for name,run in selection['training'].items():
    best=max(run['rounds'],key=lambda r:(r['success_rate'],r['mean_gates'],-float(r['label'])))
    assert run['best']==best==selection['selected'][name]
    assert all(read(r['evaluation'])['dataset']=='val' for r in run['rounds'])
key_numbers={};summaries=[]
for method,dataset,ob,sf in itertools.product(methods,['original','blind','heldout','mini'],[0,1],[0,1]):
    p=f'metrics/lwy_runtime2_final_{method}_{dataset}_ob{ob}_sf{sf}_rt1.json';d=read(p);n=30 if dataset=='mini' else 100
    assert d['method']==method and d['dataset']==dataset and d['episodes']==len(d['rows'])==n
    assert d['random_obstacles']==bool(ob) and d['safety_filter']==bool(sf)
    assert d['runtime']['xla_flags']=='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
    assert d['runtime']['tf_cudnn_deterministic']=='1' and d['runtime']['policy_batch_width']==100
    assert d['runtime']['noise_seeds']==list(range(940000,940000+n))
    assert not d['runtime']['dummy_policy_streams_enter_environment']
    assert d['successes']==sum(r['success'] for r in d['rows'])
    if method!='planner':assert d['checkpoint_sha256']==hashes[method]
    if sf==0 and dataset in ['original','blind']:key_numbers[f'{method}/{dataset}/ob{ob}']=dict(successes=d['successes'],episodes=n,success_rate=d['success_rate'])
    bins={}
    for r in d['rows']:
        if r['success']:continue
        b=str(r['section']['index']) if r.get('section') else str(min(9,int(r['progress_m']//100)))
        bins[b]=bins.get(b,0)+1
    summaries.append(dict(artifact=p,method=method,dataset=dataset,ob=ob,sf=sf,successes=d['successes'],failures=d['failures'],failure_s_sections=bins))
video=read('metrics/lwy_runtime2_dreamer_vs_ppo_rt1.json')
assert Path(video['path']).is_file() and Path(video['path']).stat().st_size>1000
for r in video['rollouts']:
    d=read(f"metrics/lwy_runtime2_final_{r['method']}_original_ob0_sf0_rt1.json")
    assert r['action_pose_sha256']==d['rows'][0]['action_pose_sha256']
result=dict(passed=True,final_conditions=len(summaries),episodes=sum(30 if s['dataset']=='mini' else 100 for s in summaries),key_numbers=key_numbers,selected=selection['selected'],video=video['path'],conditions=summaries)
with Path('metrics/lwy_runtime2_acceptance_final.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='conditions'}))
