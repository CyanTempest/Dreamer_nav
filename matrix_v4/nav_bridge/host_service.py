"""CPU host service for pose fitting, incremental pulls and the 27 CU cutoff."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

base=Path(os.environ.get('MXNAV_RUN_ROOT',Path.cwd())).resolve()
os.chdir(base)
os.environ.update(MV4_SESSION='mxnav',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',
                  PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(base/'frozen'))
code=base.parent/'exp_nav3_matrix_v4/matrix_v4/nav_bridge'
mux=['bash','ops/mux.sh']
cli=str(Path.home()/'.local/bin/colab')
results=base/'results'
budget_t=0
sync_t=0
sync_process=None
def vm(command,**kw):
    return subprocess.run([*mux,command],check=True,timeout=120,**kw)
def budget():
    s=subprocess.check_output([cli,'usage'],text=True,timeout=30)
    start=float(re.search(r'balance: ([\d.]+)',Path('logs/start_usage').read_text()).group(1))
    spent=start-float(re.search(r'balance: ([\d.]+)',s).group(1))
    if spent>=27 or time.time()-Path('logs/new.log').stat().st_mtime>=27/8.88*3600:
        vm('mkdir -p /content/eval && touch /content/eval/wrap')
    return spent

def service():
    global budget_t,sync_t,sync_process
    if time.monotonic()-budget_t>40:
        budget()
        budget_t=time.monotonic()
    msg=vm('cd /content/eval 2>/dev/null && cat request.json 2>/dev/null || true',capture_output=True,text=True).stdout.strip()
    if msg:
        req=json.loads(msg)
        name=f"pose-{req['leg']:04}.json"
        local=results/name
        flag=results/(name+'.sent')
        if not flag.exists():
            if not local.exists():
                proc=subprocess.run([str(base.parent/'envs/mjmap/bin/python'),str(code/'fit.py'),str(req['station'])],
                                    capture_output=True,text=True,timeout=240,check=True)
                json.loads(proc.stdout)
                local.write_text(proc.stdout)
            with local.open('rb') as f: vm(f'cd /content/eval && cat >{name}.tmp && mv {name}.tmp {name}',stdin=f)
            flag.touch()
    if time.monotonic()-sync_t>15 and (sync_process is None or sync_process.poll() is not None):
        includes=['/leg-*/','/calibration/','steps-*.npz','clip.mp4','end.json',
                  'calibration.json','cal_trace.npz','frame_stats.json','config.json',
                  'outcome.json','events.json','status.json']
        sync_process=subprocess.Popen(['rsync','-rt','--timeout=90',
            '-e',f'ssh -T -S {base}/logs/mxnav.sock -o ProxyCommand=false',
            *['--include='+k for k in includes],'--exclude=*',
            'root@colab-runtime:/content/eval/',str(results)+'/'])
        sync_t=time.monotonic()

if __name__=='__main__':
    while not Path('logs/host.done').exists():
        try: service()
        except Exception as e: print(time.ctime(),repr(e),flush=True)
        time.sleep(2)
