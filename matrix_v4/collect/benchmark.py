"""Run one fixed wall-time tier on the provisioned Colab VM, including cold resets."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

try:
    import pynvml
except ImportError:
    pynvml=None


def process_utilization():
    if pynvml is None:
        return {}
    try:
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        samples=pynvml.nvmlDeviceGetProcessUtilization(handle,int((time.time()-6)*1e6))
        util={}
        for row in samples:
            if row.smUtil<=100:
                util.setdefault(row.pid,[]).append(row.smUtil)
        return {pid:sum(values)/len(values) for pid,values in util.items()}
    except pynvml.NVMLError:
        return {}
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def slot_loop(c, batch, slot, n, gpu, start, end):
    count=0
    while time.time()<end-5:
        run=batch/f'{batch.name}-s{slot}-e{count:03d}'
        run.mkdir()
        seed=20261119+slot*10000+count
        cmd=['bash',str(c/'collect/instance.sh'),str(run),str(slot),str(seed),str(end),str(n),gpu]
        with (run/'launch.log').open('w') as log:
            status=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
        if not (run/'episode.json').exists():
            error='benchmark_end' if time.time()>=end-1 else 'instance_crash'
            spawn=run/'spawn.json'
            meta=dict(reason=error,status=status,steps=0,qualified=0,stand_accepted=False,
                      closed=time.time(),window=f'window_{slot:02d}',seed=seed,gpu=gpu,instances=n,
                      driver='pure_pursuit',config=json.loads((c/'collect/config.json').read_text()))
            if spawn.is_file():
                meta['spawn']=json.loads(spawn.read_text())
            # A run that never entered control still has an auditable empty shard.
            # Keep a failed writer's incomplete bytes separate from sealed data.
            from shard import Episode
            partial=c/'data'/(run.name+'.partial')
            if partial.exists():
                partial.replace(run/'interrupted.partial')
            if not (c/'data'/(run.name+'.npz')).exists():
                Episode(c/'data',run.name,meta).finish(error)
            (run/'episode.json').write_text(json.dumps(meta))
        count+=1
        if status:
            time.sleep(min(3,max(0,end-time.time())))


def sample(c):
    gpu=subprocess.check_output(['nvidia-smi'],text=True)
    utilization=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
    per_pid={int(p):int(m) for p,m in re.findall(r'\s(\d+)\s+(?:C\+G|C|G)\s+[^\n]*?\s(\d+)MiB',gpu)}
    sm=process_utilization()
    procs={}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            exe=(proc/'exe').readlink()
            if exe.name!='UeSim' or not exe.is_relative_to(c/'instances'):
                continue
            slot=exe.relative_to(c/'instances').parts[0]
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            procs[slot]=dict(pid=int(proc.name),cpu_s=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'),
                             gpu_mib=per_pid.get(int(proc.name)),sm_pct=sm.get(int(proc.name)))
        except (OSError,IndexError):
            continue
    return dict(epoch=time.time(),card=utilization,slots=procs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('n',type=int,choices=range(1,15))
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--tag',default='bench')
    a=p.parse_args()
    c=Path('/content/col')
    assert Path('/content/matrix-v4/graphics.env').exists()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    batch=c/'runs'/f'{a.tag}-{a.n}-{uuid.uuid4().hex[:8]}'
    batch.mkdir(parents=True)
    (c/'data').mkdir(exist_ok=True)
    start=time.time()
    end=math.ceil(start+a.seconds)
    spec=dict(n=a.n,gpu=gpu,start=start,end=end)
    (batch/'tier.json').write_text(json.dumps(spec))
    print(batch,flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.n) as pool:
        tasks=[pool.submit(slot_loop,c,batch,i,a.n,gpu,start,end) for i in range(a.n)]
        with (batch/'resources.jsonl').open('w',buffering=1) as log:
            while time.time()<end:
                print(json.dumps(sample(c)),file=log)
                time.sleep(min(4,max(0,end-time.time())))
        for task in tasks:
            task.result()
    episodes=[json.loads(f.read_text()) for f in batch.glob('*/episode.json')]
    spec.update(episodes=len(episodes),steps=sum(r['steps'] for r in episodes),
                qualified=sum(r['qualified'] for r in episodes),finished=time.time())
    (batch/'done.json').write_text(json.dumps(spec))
    print(spec,flush=True)


if __name__=='__main__':
    main()
