"""Compact queue status with an audit trail of each SSH supervision check."""
import json
from pathlib import Path
import subprocess
import time


def latest(path,events=None):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-131072))
        lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if events and row.get('event') not in events:continue
        return {k:v for k,v in row.items() if k!='metrics'}


jobs=json.loads(Path('metrics/jobs.json').read_text())
result=dict(time=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip())
result['jobs']=[{k:v for k,v in job.items() if k!='command'} for job in jobs[-4:]]
for name in ['dreamer_s0','ppo_s0']:
    progress=Path(f'checkpoints/{name}/progress.json')
    result[name]=dict(checkpoint=json.loads(progress.read_text()) if progress.exists() else None,
        latest=latest(Path(f'metrics/train_{name}.jsonl'),{'train','end','resume'}),
        memory=latest(Path(f'metrics/memory_{name}.jsonl')))
result['evaluation_files']=len([p for name in ['planner','dreamer_s0','ppo_s0'] for p in Path('metrics').glob(f'eval_{name}_*.json') if '_at_' not in p.name])
result['telemetry']=latest(Path('metrics/queue_telemetry.jsonl'))
result['process_status']=[]
for process in (result['telemetry'] or {}).get('processes',[]):
    proc=Path('/proc')/str(process['pid'])
    try:
        if (proc/'cwd').resolve()!=Path.cwd():continue
        values={line.split(':',1)[0]:line.split(':',1)[1].strip()
                for line in (proc/'status').read_text().splitlines() if ':' in line}
        result['process_status'].append(dict(pid=process['pid'],
            rss_bytes=int(values['VmRSS'].split()[0])*1024,
            rss_hwm_bytes=int(values['VmHWM'].split()[0])*1024,
            swap_bytes=int(values['VmSwap'].split()[0])*1024))
    except (FileNotFoundError,ProcessLookupError,PermissionError,KeyError):
        continue
with Path('metrics/ssh_supervision.jsonl').open('a') as f:
    f.write(json.dumps(result)+'\n')
print(json.dumps(result,indent=2))
