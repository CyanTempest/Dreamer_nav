"""Sequential remote tmux workload, bounded jobs, resumable checkpoints, gate stops."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .provenance import require_g1
from .memory import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true')
    ap.add_argument('--budget-seconds',type=int,default=86400)
    ap.add_argument('--train-seconds',type=int,default=18000)
    a=ap.parse_args();require_g1()
    ledgerpath=Path('metrics/jobs.json')
    jobs=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
    # Old failed/killed runners must not reserve time from the next method.
    charged_seconds=0.
    failures={}
    for job in jobs:
        if job.get('status')=='running':
            job['status']='interrupted_before_recovery'
    def save():atomic_json(ledgerpath,jobs)
    def attempt(name,module,args,timeout):
        nonlocal charged_seconds
        runtime_path=Path('configs/runtime.json')
        runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        if module=='nav3.train_dreamer' and 'dreamer_envs' in runtime:
            args=[*args,'--envs',int(runtime['dreamer_envs'])]
            args += ['--chunksize',int(runtime.get('dreamer_chunksize',16))]
        if module=='nav3.train_ppo' and 'ppo_envs' in runtime:
            args=[*args,'--envs',int(runtime['ppo_envs'])]
        if charged_seconds+timeout>a.budget_seconds:
            print(f'SKIP budget: {name}; successful_seconds={charged_seconds:.0f}, reserve={timeout}',flush=True);return False
        query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(query.strip().splitlines()[0])<18000:
            print(f'STOP insufficient free GPU memory before {name}: {query.strip()} MiB',flush=True);return False
        command=[sys.executable,'-m',module,*map(str,args)]
        rec=dict(name=name,command=command,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='running',timeout_seconds=timeout)
        jobs.append(rec);save();start=time.monotonic()
        with (Path('logs')/f'{name}.log').open('a') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            rec['pid']=proc.pid;save()
            try:code=proc.wait(timeout=timeout-180)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGTERM)
                try:code=proc.wait(timeout=170)
                except subprocess.TimeoutExpired:
                    proc.kill();code=proc.wait();rec['forced_stop']=True
        rec.update(exit_code=code,wall_seconds=time.monotonic()-start,status='complete' if code==0 else 'failed');save()
        if code==0:charged_seconds+=rec['wall_seconds']
        subprocess.run([sys.executable,'-m','nav3.report'],check=False)
        print(json.dumps(rec),flush=True)
        return code==0
    def run(name,module,args,timeout):
        while failures.get(name,0)<3:
            if attempt(name,module,args,timeout):return True
            failures[name]=failures.get(name,0)+1
            print(f'RETRY {name}: failure {failures[name]}/3; resume latest checkpoint',flush=True)
        print(f'SKIP {name}: three failures; continue next configuration',flush=True)
        return False
    def progress(name):
        p=Path('checkpoints')/name/'progress.json'
        return json.loads(p.read_text()) if p.exists() else dict(steps=0,wall_seconds=0)
    for method,limit in [('dreamer',7200),('ppo',3600)]:
        name=f'{method}_s0';p=progress(name)
        if p['steps']<100000:
            args=['--name',name,'--seed',0,'--steps',100000,'--seconds',limit,'--resume']
            if not run(f'smoke_{name}',f'nav3.train_{method}',args,limit+180):continue
    # Preserve the already accepted smoke gate; formal training must not replace
    # its evidence with a later transient evaluation or a memory-probe run.
    if not Path('metrics/g2.json').exists():
        subprocess.run([sys.executable,'-m','nav3.gates'],check=True)
    gate=json.loads(Path('metrics/g2.json').read_text())
    if not gate['passed']:
        print('STOP: G2 did not pass. Formal jobs are not authorized by the gate.',flush=True);return
    if a.smoke_only:return
    target=a.train_seconds
    assert target>0
    # Current user scope: one seed per main method; defer extra comparisons.
    schedule=[('dreamer','dreamer_s0',0,[]),('ppo','ppo_s0',0,[])]
    for method,name,seed,flags in schedule:
        while (remaining:=math.ceil(target-progress(name)['wall_seconds']))>0:
            seconds=min(remaining,10000)
            if not run(f'formal_{name}',f'nav3.train_{method}',
                ['--name',name,'--seed',seed,'--steps',1000000000,'--seconds',seconds,'--resume',*flags],seconds+180):break
    names=['planner']+[n for _,n,_,_ in schedule]
    for name in names:
        run(f'eval_{name}','nav3.eval_model',['--name',name,'--all'],7200)
    subprocess.run([sys.executable,'-m','nav3.report'],check=True)


if __name__=='__main__':main()
