"""Resumable 2 x 3-hour fine-tuning, in bounded 30-minute jobs."""
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .memory import atomic_json


def main():
    from .fix_training import require_fix_g1
    require_fix_g1()
    initial=json.loads(Path('metrics/fix_initial_checkpoints.json').read_text())
    ledger=Path('metrics/fix_jobs.json')
    jobs=json.loads(ledger.read_text()) if ledger.exists() else []
    for record in jobs:
        if record['status']=='running':record['status']='interrupted'
    def run(label,args,timeout):
        elapsed=sum(r.get('wall_seconds',0) for r in jobs)
        if elapsed+timeout>8*3600:raise RuntimeError('Round-five queue budget exhausted')
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        assert free>18000,f'Insufficient free GPU memory: {free} MiB; no other process will be stopped'
        rec=dict(label=label,command=[sys.executable,*map(str,args)],status='running',
            start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout=timeout,gpu_free_mib_before=free)
        jobs.append(rec);atomic_json(ledger,jobs);start=time.monotonic()
        with Path(f'logs/fix_{label}.log').open('a') as f:
            proc=subprocess.Popen(rec['command'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            rec['pid']=proc.pid;atomic_json(ledger,jobs)
            try:code=proc.wait(timeout=timeout-120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:code=proc.wait(timeout=110)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
        rec.update(exit_code=code,wall_seconds=time.monotonic()-start,status='complete' if code==0 else 'failed')
        atomic_json(ledger,jobs);print(json.dumps(rec),flush=True)
        return code
    for method,envs in [('dreamer',4608),('ppo',640)]:
        source=f'{method}_s0';name=source+'_ft';base=initial[source]['source_progress']['wall_seconds']
        failures=0
        while True:
            progress=json.loads(Path(f'checkpoints/{name}/progress.json').read_text())
            remaining=10800-(progress['wall_seconds']-base)
            if remaining<=0:break
            seconds=min(1800,math.ceil(remaining));fresh=progress['steps']==initial[source]['source_progress']['steps']
            flags=['--reset-environment'] if fresh else []
            if fresh and method=='dreamer':flags+=['--fresh-replay']
            command=['-m',f'nav3.train_{method}','--name',name,'--seed',0,'--envs',envs,'--steps',2000000000,
                '--seconds',seconds,'--resume','--map-root','maps_mix_v2','--fix-round','--margin-weight',.25,'--margin-distance',.20,*flags]
            code=run(f'train_{name}_{progress["steps"]}',command,seconds+300)
            if code not in [0,75]:
                failures+=1
                if failures>=3:raise RuntimeError(f'{name} failed three times; inspect raw logs')
                continue
            progress=json.loads(Path(f'checkpoints/{name}/progress.json').read_text())
            tag=f'curve_{name}_{progress["steps"]}'
            if not Path(f'metrics/fix_{tag}.json').exists():
                code=run(tag,['-m','nav3.fix_eval','--name',name,'--split','blind','--root','maps_v2','--episodes',1,'--tag',tag],240)
                if code:raise RuntimeError(f'Blind learning curve evaluation failed: {name}')
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
