"""Resumable evaluation matrix, diagnostics and curve time zero."""
import argparse
import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',action='store_true');a=p.parse_args()
    def run(tag,args):
        if Path(f'metrics/fix_{tag}.json').exists():return
        started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
        with Path(f'logs/fix_{tag}.log').open('a') as f:
            result=subprocess.run([sys.executable,'-m','nav3.fix_eval','--tag',tag,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,timeout=600)
        with Path('metrics/fix_evaluation_jobs.jsonl').open('a') as f:
            f.write(json.dumps(dict(tag=tag,start_utc=started,wall_seconds=time.monotonic()-clock,exit_code=result.returncode))+'\n')
        result.check_returncode()
        print(tag,flush=True)
    names=['dreamer_s0','ppo_s0','planner'] if a.initial else ['dreamer_s0','dreamer_s0_ft','ppo_s0','ppo_s0_ft','planner']
    for name in names:
        for split,root,episodes in [('heldout','maps',5),('blind','maps_v2',5),('test','maps',100)]:
            for ob,sf in itertools.product([0,1],repeat=2):
                flags=(['--ob'] if ob else [])+(['--sf'] if sf else [])
                run(f'eval_v1_{name}_{split}_ob{ob}_sf{sf}', ['--name',name,'--split',split,'--root',root,'--episodes',episodes,*flags])
        if a.initial and name!='planner':
            run(f'curve_initial_{name}',['--name',name,'--split','blind','--root','maps_v2','--episodes',1])
        if not a.initial and name!='planner':
            run(f'repeat100_{name}',['--name',name,'--split','test','--episodes',100,
                '--compare',f'metrics/fix_eval_v1_{name}_test_ob0_sf0.json'])
    if not a.initial:
        for name in names:
            for case in ['short_log','start150','start150_seed9141','synthetic_base','synthetic_rocks']:
                if case.startswith('synthetic'):
                    args=['--split','heldout','--root',f'maps/diag_{case}','--episodes',15,'--around-log']
                else:
                    args=['--split','test','--episodes',30,'--start',158 if case=='short_log' else 150,'--stop',171]
                    if case=='start150_seed9141':args+=['--seed',9141]
                for ob,sf in itertools.product([0,1],repeat=2):
                    flags=(['--ob'] if ob else [])+(['--sf'] if sf else [])
                    run(f'diagnostic_{name}_{case}_ob{ob}_sf{sf}',['--name',name,*args,*flags])


if __name__=='__main__':main()
