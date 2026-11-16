"""Explicit fresh-process evaluation protocol for round five.

Run this module before importing JAX. Batch size and map order are part of the
protocol: the legacy environment uses batch-shaped random draws.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform


def setup():
    os.environ['XLA_FLAGS']='--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    os.environ['NAV3_REPRO']='1'


def run(name,split='heldout',episodes=5,root='maps',ob=False,sf=False,seed=9137,tag=None,start_s=None,stop_s=None):
    import jax
    from .diag_rollout import load_policy
    from .evaluate import evaluate
    from .memory import release_host_memory
    path=Path('metrics')/f'fix_{tag or f"eval_v1_{name}_{split}_ob{int(ob)}_sf{int(sf)}"}.json'
    if path.exists():raise FileExistsError(path)
    policy=load_policy(name)
    release_host_memory()
    result=evaluate(policy,split,episodes,ob,sf,seed,name,save=False,root=root,
        reset_policy=True,plots=False,trace_hash=True,start_s=start_s,stop_s=stop_s)
    result['runtime']=dict(python=platform.python_version(),jax=jax.__version__,device=str(jax.devices()[0]),
        xla_flags=os.environ.get('XLA_FLAGS'),cudnn_deterministic=os.environ.get('TF_CUDNN_DETERMINISTIC'),
        protocol_version=1,batch_random_stream='legacy fixed-shape; batch size and map order are part of configuration')
    if name!='planner':
        result['checkpoint_sha256']=hashlib.sha256(Path(f'checkpoints/{name}/latest.pkl').read_bytes()).hexdigest()
    with path.open('x') as f:json.dump(result,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='planner');p.add_argument('--split',default='heldout')
    p.add_argument('--root',default='maps');p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--ob',action='store_true');p.add_argument('--sf',action='store_true')
    p.add_argument('--seed',type=int,default=9137);p.add_argument('--tag');p.add_argument('--compare')
    p.add_argument('--start',type=float);p.add_argument('--stop',type=float)
    p.add_argument('--around-log',action='store_true')
    a=p.parse_args();setup()
    if a.around_log:
        records=[r for r in json.loads((Path(a.root)/'manifest.json').read_text())['maps'] if r['split']==a.split]
        stations=[r['features']['log'] for r in records for _ in range(a.episodes)]
        a.start=[s-6 for s in stations];a.stop=[s+7 for s in stations]
    result=run(a.name,a.split,a.episodes,a.root,a.ob,a.sf,a.seed,a.tag,a.start,a.stop)
    if a.compare:
        old=json.loads(Path(a.compare).read_text())
        changed=[i for i,(x,y) in enumerate(zip(old['rows'],result['rows'])) if x!=y]
        assert len(old['rows'])==len(result['rows']) and not changed,changed
        print('EXACT_REPEAT: all episode fields and every-step action/pose hashes match',flush=True)


if __name__=='__main__':main()
