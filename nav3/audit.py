"""Audit actual saved manifests, used training seeds, upstream branch and read-only references."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--local',action='store_true');a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'metrics/map_manifest.json').read_text())
    train=[r for r in manifest['maps'] if r['split']=='train']
    held=[r for r in manifest['maps'] if r['split']=='heldout']
    ts={r['seed'] for r in train};hs={r['seed'] for r in held}
    checks=dict(training_maps_at_least_200=len(train)>=200,heldout_maps_at_least_20=len(held)>=20,
                seeds_disjoint=not ts&hs,mini_not_in_train=all(not r['mini'] and r['name']!='mini_trail' for r in train))
    used={}
    for path in (root/'metrics').glob('config_*.json'):
        cfg=json.loads(path.read_text())
        if 'map_seeds' not in cfg:continue
        seeds=set(cfg['map_seeds']);used[path.stem]=sorted(seeds)
        checks[f'{path.stem}_training_only']=seeds<=ts and not seeds&hs
        checks[f'{path.stem}_no_mini']='mini_trail' not in cfg['map_names']
    actual=[]
    reset_records={}
    for path in (root/'metrics').glob('trainmaps_*.jsonl'):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line]
        checks[f'{path.stem}_actual_resets_training_only']=all(r['seed'] in ts and r['seed'] not in hs and r['split']=='train' and not r['mini'] and r['map']!='mini_trail' for r in rows)
        reset_records[path.stem]=dict(episodes=len(rows),distinct_seeds=sorted({r['seed'] for r in rows}))
    for path in (root/'metrics').glob('audit_actor_*.json'):
        data=json.loads(path.read_text())
        if data:
            checks[f'{path.stem}_no_priv_leak']=data['privileged_actor_carry_max_error']==0 and all(not k.startswith('priv_') for k in data['encoder_keys'])
            actual.append(path.name)
    upstream=root.parent/'dreamerv3' if a.local else root.parents[1]/'dreamerv3'
    try:
        head=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
        branch=subprocess.check_output(['git','-C',str(upstream),'branch','--show-current'],text=True).strip()
        checks['upstream_head_unchanged']=head=='e01491fad6434b2245a3b8ca201dd7faedcc458c'
        checks['upstream_still_main']=branch=='main'
    except subprocess.CalledProcessError:
        head=branch='not a git checkout';checks['upstream_git_verifiable']=False
    before=root/'metrics/reference_stat_before.txt';after=root/'metrics/reference_stat_after.txt'
    checks['read_only_snapshots_present']=before.exists() and after.exists()
    if before.exists() and after.exists():
        checks['read_only_reference_stats_unchanged']=before.read_bytes()==after.read_bytes()
    gate=json.loads((root/'metrics/g1.json').read_text())
    fingerprints={json.loads(p.read_text()).get('environment_fingerprint') for p in (root/'metrics').glob('config_*.json') if 'map_seeds' in json.loads(p.read_text())}
    checks['training_environment_matches_g1']=fingerprints=={gate['environment_fingerprint']}
    checks['privileged_noninterference_was_tested']=bool(actual)
    result=dict(passed=all(checks.values()),checks=checks,train_maps=len(train),heldout_maps=len(held),
                used_training_seeds=used,actor_leakage_files=actual,upstream_head=head,upstream_branch=branch,
                actual_reset_records=reset_records,
                training_runs_audited=len(used),training_not_run=(len(used)==0))
    (root/'metrics'/('audit_local.json' if a.local else 'audit.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert result['passed'],checks


if __name__=='__main__':main()
