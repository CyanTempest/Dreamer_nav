"""Census, gate recording and isolated mixed training manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def census():
    from .diag_geometry import feature_row
    result={}
    for root,split,key in [('maps','train','old_train'),('maps_v2','train','new_train'),('maps_v2','blind','blind')]:
        manifest=json.loads((Path(root)/'manifest.json').read_text())
        records=[r for r in manifest['maps'] if r['split']==split]
        rows=[feature_row(r['name'],root) for r in records]
        counts=dict(right_rock=sum(r['opposing_rock_3_5m_any_lateral'] for r in rows),
            either_opposing_rock=sum(any(abs(q['ds'])<3.5 and q['lateral']*r['lateral']<0 and abs(q['lateral'])<1.6 for q in r['rocks']) for r in rows),
            gate_within_1m=sum(r['gate_distance']<1 for r in rows),
            gate_within_3m=sum(r['gate_distance']<3 for r in rows),
            triple=sum(r['gate_distance']<3 and any(abs(q['ds'])<3.5 and q['lateral']*r['lateral']<0 and abs(q['lateral'])<1.6 for q in r['rocks']) for r in rows),
            left_log=sum(r['lateral']>0 for r in rows),right_log=sum(r['lateral']<0 for r in rows))
        result[key]=dict(n=len(rows),counts=counts,rows=rows)
    with Path('metrics/fix_census.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:dict(n=v['n'],counts=v['counts']) for k,v in result.items()}),flush=True)


def gate():
    result=json.loads(Path('metrics/fix_eval_v1_planner_blind_ob0_sf0.json').read_text())
    record=dict(passed=result['success_rate']>=.95,threshold=.95,episodes=result['episodes'],
        success_rate=result['success_rate'],mean_gates=result['mean_gates'],
        environment_fingerprint=result['environment_fingerprint'],
        manifest_sha256=hashlib.sha256(Path('maps_v2/manifest.json').read_bytes()).hexdigest())
    with Path('metrics/fix_g1.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record),flush=True)
    assert record['passed'],'G1 failed: training is forbidden'


def prepare():
    from .fix_training import require_fix_g1
    require_fix_g1()
    root=Path('maps_mix_v2');root.mkdir(exist_ok=True)
    records=[]
    for source,label in [('maps','old'),('maps_v2','new')]:
        manifest=json.loads((Path(source)/'manifest.json').read_text())
        for record in manifest['maps']:
            if record['split']!='train':continue
            name=f'{label}_{record["seed"]}'
            link=root/name
            if not link.exists():link.symlink_to((Path(source)/record['name']).resolve(),target_is_directory=True)
            records.append({**record,'name':name,'source_root':source,'source_name':record['name']})
    assert len(records)==400 and len({r['seed'] for r in records})==400
    if not (root/'manifest.json').exists():
        (root/'manifest.json').write_text(json.dumps(dict(maps=records,old_new_ratio=[.5,.5]),indent=2))
    copy_records={}
    for name in ['dreamer_s0','ppo_s0']:
        source=Path('checkpoints')/name;dest=Path('checkpoints')/(name+'_ft')
        if not dest.exists():shutil.copytree(source,dest)
        sha=lambda p:hashlib.sha256((p/'latest.pkl').read_bytes()).hexdigest()
        assert sha(source)==sha(dest),'Refuse to restart preparation over a trained checkpoint'
        progress=json.loads((source/'progress.json').read_text())
        copy_records[name]=dict(source_sha256=sha(source),copy_sha256=sha(dest),source_progress=progress,
            replay='fresh mixed-map replay; old rewards are not carried into the changed reward objective')
    with Path('metrics/fix_initial_checkpoints.json').open('x') as f:json.dump(copy_records,f,indent=2)
    print(json.dumps(copy_records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['census','gate','prepare']);a=p.parse_args()
    globals()[a.action]()
