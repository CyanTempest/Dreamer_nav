"""Verify the requested two-method budget and all 24 paired evaluation cells."""
import itertools
import json
import math
from pathlib import Path


root=Path(__file__).resolve().parents[1]
metrics=root/'metrics'
fingerprint='a3350106d8849a7fff38c58ca5d2ebaa759b52e4a02443022b9f75109bfc896d'
methods=('planner','dreamer_s0','ppo_s0')
expected=set(itertools.product(methods,('heldout','test'),(False,True),(False,True)))
evaluation=json.loads((metrics/'evaluation_summary.json').read_text())
selected=[r for r in evaluation if r['method'] in methods]
keys=[(r['method'],r['split'],r['obstacles'],r['filter']) for r in selected]
checks=dict(all_24_conditions=set(keys)==expected and len(keys)==len(expected))
progress={name:json.loads((root/'checkpoints'/name/'progress.json').read_text())
          for name in methods if name!='planner'}
for name,p in progress.items():
    checks[name+'_at_least_5h']=p['wall_seconds']>=18000
    checks[name+'_positive_updates']=p['updates']>0
checks['dreamer_continued_original_checkpoint']=progress['dreamer_s0']['steps']>1299872
signatures={}
for row,key in zip(selected,keys):
    raw=json.loads((metrics/row['source']).read_text())
    episodes=raw['rows'];tag='_'.join(map(str,key))
    checks[tag+'_episodes']=len(episodes)==row['episodes'] and len(episodes)>=(100 if row['split']=='heldout' else 30)
    checks[tag+'_fingerprint']=raw['environment_fingerprint']==fingerprint
    checks[tag+'_failures']=sum(raw['failures'].values())==len(episodes)
    checks[tag+'_numeric_summary']=all(math.isfinite(row[k]) for k in ('mean_gates','success_rate','fail_per_100m','takeover_rate','mean_speed')) and math.isclose(row['mean_gates'],sum(e['gates'] for e in episodes)/len(episodes),abs_tol=1e-12) and math.isclose(row['success_rate'],sum(e['success'] for e in episodes)/len(episodes),abs_tol=1e-12)
    signatures[key]=sorted((e['map'],e['map_seed'],e['episode'],e['scenario_seed']) for e in episodes)
for split,ob,sf in itertools.product(('heldout','test'),(False,True),(False,True)):
    checks[f'paired_{split}_{ob}_{sf}']=all(signatures.get((m,split,ob,sf))==signatures.get(('planner',split,ob,sf)) for m in methods)
for name in ('g1','g2','audit'):
    checks[name+'_passed']=json.loads((metrics/f'{name}.json').read_text())['passed']
recovery=json.loads((metrics/'oom_recovery_summary.json').read_text())
checks['host_memory_validation']=recovery['validation_passed'] and recovery['validation_seconds']>=600 and recovery['rss_max_gb']<45
jobs=json.loads((metrics/'jobs.json').read_text())
for name in ('formal_dreamer_s0','formal_ppo_s0','eval_planner','eval_dreamer_s0','eval_ppo_s0'):
    matching=[j for j in jobs if j['name']==name]
    checks[name+'_completed']=bool(matching) and matching[-1]['status']=='complete' and matching[-1]['exit_code']==0
result=dict(passed=all(checks.values()),checks=checks,progress=progress,
            evaluation_conditions=len(selected),evaluation_episodes=sum(r['episodes'] for r in selected),
            environment_fingerprint=fingerprint)
(metrics/'final_queue_acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=result['passed'],progress=progress,evaluation_conditions=result['evaluation_conditions'],evaluation_episodes=result['evaluation_episodes'],failed_checks=[k for k,v in checks.items() if not v]),indent=2))
assert result['passed'],result
