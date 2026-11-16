"""Small read-only status snapshot for remote polling."""
import datetime
import json
from pathlib import Path
import subprocess


def read(path,default=None):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def last_event(path):
    p=Path(path)
    if not p.exists():return {}
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-50000))
        lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row=json.loads(line)
            if 'wall_seconds' in row:return row
        except ValueError:pass
    return {}


initial=read('metrics/fix_initial_checkpoints.json',{})
sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
jobs=read('metrics/fix_jobs.json',[]);gate=read('metrics/fix_g1.json')
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    generated=sum(len(list(Path('maps_v2',s).glob('*/metadata.json'))) for s in ['train','blind']),
    g1=None if gate is None else dict(passed=gate['passed'],rate=gate['success_rate']),sessions=sessions,
    job=None if not jobs else {k:jobs[-1].get(k) for k in ['label','status','start_utc','exit_code']},training={})
for name in ['dreamer_s0','ppo_s0']:
    row=last_event(f'metrics/train_{name}_ft.jsonl')
    if row and name in initial:
        result['training'][name]=dict(event=row.get('event'),steps=row.get('steps'),
            fine_tuning_hours=round((row['wall_seconds']-initial[name]['source_progress']['wall_seconds'])/3600,3),
            mean_gates=row.get('mean_gates'))
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
result['gpu_mib_util']=gpu.stdout.strip()
print(json.dumps(result),flush=True)
