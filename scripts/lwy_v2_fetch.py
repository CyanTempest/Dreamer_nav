"""Bounded client-side result synchronization; no remote job depends on this."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
HOST='hpf@100.108.114.64'
REMOTE='wang-sm/ATEC_NAV_WSM/research/exp_nav3/'


def final_cleanup():
    script="""
import json,os,pathlib,subprocess,time
from datetime import datetime,timezone
root=pathlib.Path.home()/'wang-sm/ATEC_NAV_WSM/research/exp_nav3';os.chdir(root)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(root/'cache/matplotlib'),
 XDG_CACHE_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'))
state=json.loads(pathlib.Path('metrics/lwy_v2_queue.json').read_text())
pid=state['pid']
for _ in range(30):
 if not pathlib.Path(f'/proc/{pid}').exists():break
 time.sleep(1)
owned=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  if (path/'cwd').resolve()!=root:continue
  args=(path/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  if ' -m nav3.lwy_' in args or (' -m nav3.train_ppo' in args and '--lwy-round' in args) or ' tests/check_lwy.py' in args or ' tests/check_diag.py' in args:
   owned.append(dict(pid=int(path.name),command=args))
 except (OSError,UnicodeError):pass
sessions=subprocess.run(['tmux','list-sessions','-F','#S'],capture_output=True,text=True).stdout.splitlines()
result=dict(utc=datetime.now(timezone.utc).isoformat(),queue_pid=pid,queue_running=pathlib.Path(f'/proc/{pid}').exists(),
 nav3_sessions=[s for s in sessions if s.startswith('nav3_')],owned_background_processes=owned)
pathlib.Path('metrics/lwy_v2_cleanup.json').write_text(json.dumps(result,indent=2))
subprocess.run(['.venv_lwy/bin/python','-m','nav3.lwy_report_v2'],check=True,timeout=180)
"""
    subprocess.run(['ssh','-o','ConnectTimeout=10',HOST,'python3 -'],input=script,text=True,check=True,timeout=240)


def fetch():
    subprocess.run(['rsync','-az',f'{HOST}:{REMOTE}LWY_REPORT.md',str(ROOT/'LWY_REPORT.md')],check=True,timeout=120)
    rules={
        'metrics':['lwy_v2_*.json','lwy_v2_*.jsonl','train_*_lwy.jsonl','trainmaps_*_lwy.jsonl','memory_*_lwy.jsonl','config_*_lwy.json'],
        'figures':['lwy_v2_*'],
        'logs':['lwy_v2_*'],
        'videos':['lwy_v2_*.mp4','lwy_v2_*.json'],
    }
    for directory,patterns in rules.items():
        (ROOT/directory).mkdir(exist_ok=True)
        subprocess.run(['rsync','-az',*[f'--include={p}' for p in patterns],'--exclude=*',
            f'{HOST}:{REMOTE}{directory}/',str(ROOT/directory)+'/'],check=True,timeout=300)


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    deadline=time.monotonic()+14*3600
    while True:
        try:
            raw=subprocess.check_output(['ssh','-o','ConnectTimeout=10',HOST,
                f'cat {REMOTE}metrics/lwy_v2_queue.json'],text=True,timeout=30)
            state=json.loads(raw)
            finished=state['status'] in ['complete','failed','interrupted','awaiting_protocol_decision']
            if finished:final_cleanup()
            fetch()
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),status=state['status'],
                active_job=state.get('active_job'),synced=True)),flush=True)
            if finished or not a.watch:break
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as exc:
            print(json.dumps(dict(error=repr(exc))),flush=True)
            if not a.watch:raise
        if time.monotonic()>=deadline:break
        time.sleep(300)


if __name__=='__main__':main()
