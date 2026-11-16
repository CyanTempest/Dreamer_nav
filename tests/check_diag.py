"""Regression for undefined IoU; data-integrity checks for diagnostic results."""
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
from nav3.report import normalized_iou
from nav3.provenance import fingerprint

old=json.loads(Path('metrics/eval_dreamer_s0_test_ob0_sf0.json').read_text())
values=normalized_iou(old)
assert values[:2]==old['trav_iou'][:2] and values[2] is None
empty=dict(trav_iou=[0.,0.,0.],random_obstacles=True,rows=[dict(trav_union=[0,0,0])])
assert normalized_iou(empty)==[None,None,None]
assert normalized_iou(dict(trav_iou=None)) is None
assert fingerprint()==old['environment_fingerprint']
new=json.loads(Path('metrics/diag_full_dreamer.json').read_text())
assert new['failures']==dict(segment_pass=1,collision=29)
assert new['trace_endpoint_max_error_m']==0
for a,b in zip(old['rows'],new['rows']):
    if a['failure']=='collision':assert a['progress_m']==b['progress_m']
for split in ['heldout','test']:
    for ob in [0,1]:
        for sf in [0,1]:
            d=json.loads(Path(f'metrics/diag_eval_margin_{split}_ob{ob}_sf{sf}.json').read_text())
            assert d['episodes']==(100 if split=='heldout' else 30)
            assert d['environment_fingerprint']==fingerprint()
            if not ob:assert d['trav_iou'][2] is None
            if not sf:
                original=json.loads(Path(f'metrics/eval_dreamer_s0_{split}_ob{ob}_sf0.json').read_text())
                assert d['success_rate']==original['success_rate']
                assert d['mean_gates']==original['mean_gates']
original_map=np.load('maps/mini_trail/map.npz')['layers']
variant=np.load('maps/diag_log_center_occupancy_v2/mini_trail/map.npz')['layers']
changed=(original_map[...,5]<0)!=(variant[...,5]<0)
assert not np.any(changed&(original_map[...,6]!=2))
assert np.array_equal(original_map[...,:5],variant[...,:5])
manifest=json.loads(Path('maps/manifest.json').read_text())
train={r['seed'] for r in manifest['maps'] if r['split']=='train'}
heldout={r['seed'] for r in manifest['maps'] if r['split']=='heldout'}
assert not train&heldout and 871002 not in train
assert not train&set(range(30000,30004))
protected=json.loads(Path('metrics/diag_original_hashes.json').read_text())
changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
result=dict(passed=True,undefined_iou_checked=True,protected_original_files=len(protected),protected_changed=changed,
    environment_fingerprint=fingerprint(),isolated_variant_nonlog_occupancy_changes=0,mini_in_train=False,synthetic_seeds_in_train=False,
    original_checkpoint_hashes={name:hashlib.sha256(Path(f'checkpoints/{name}/latest.pkl').read_bytes()).hexdigest() for name in ['dreamer_s0','ppo_s0']})
for name,tag in [('dreamer_s0','full_dreamer'),('ppo_s0','full_ppo')]:
    assert result['original_checkpoint_hashes'][name]==json.loads(Path(f'videos/{name}_rollout.json').read_text())['checkpoint_sha256']
    assert result['original_checkpoint_hashes'][name]==json.loads(Path(f'metrics/diag_{tag}.json').read_text())['checkpoint_sha256']
p=argparse.ArgumentParser();p.add_argument('--output',default='metrics/diag_checks.json');args=p.parse_args()
if not Path(args.output).exists():Path(args.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result))
