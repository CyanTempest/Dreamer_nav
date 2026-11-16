"""Evaluate a generic observed-obstacle margin guard, without changing policies.

This is a diagnostic safety-filter candidate, not a fix to unfiltered Dreamer.
All evaluations keep the original map/seed/episode protocol and save new JSONs.
"""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from .control import assess,decode,env_x,env_y,planner
from .diag_rollout import load_policy
from .evaluate import evaluate


@jax.jit
def margin_filter(obs,action):
    radius=.5+(jnp.clip(action[:,0],-1,1)+1)*1.25
    angle=jnp.clip(action[:,1],-1,1)*jnp.pi/3
    target=jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1)
    bev=decode(obs['bev']);risk,unknown,corridor=assess(bev,target[:,None])
    # Keep the planner in charge while a visible obstacle is alongside the
    # chassis, not just when sparse points on the target segment hit it.
    near=jnp.any((bev[...,1]>.21)&(bev[...,3]<.5)&(env_x()>-.6)&(env_x()<3.5)&(jnp.abs(env_y())<1.4),axis=(1,2))
    intervene=near|(risk[:,0]>.01)|(corridor[:,0]>.15)|(unknown[:,0]>.65)
    return jnp.where(intervene[:,None],planner(obs),action),intervene


def main():
    p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');args=p.parse_args()
    policy=load_policy('dreamer_s0');conditions=[(s,o,f) for s in ['heldout','test'] for o in [False,True] for f in [False,True]] if args.all else [('test',False,True)]
    for split,ob,filtered in conditions:
        path=Path(f'metrics/diag_eval_margin_{split}_ob{int(ob)}_sf{int(filtered)}.json')
        if path.exists():continue
        policy.carry=None;policy.key=jax.random.PRNGKey(9137)
        start=time.monotonic()
        result=evaluate(policy,split,30 if split=='test' else 5,ob,filtered,method='dreamer_s0_diag_margin',save=False,filter_fn=margin_filter)
        result['diagnostic_candidate']=dict(kind='observation-only margin safety filter',learned_policy_changed=False,training=False,
            description='Observed height >.21 m, known, x in (-.6,3.5), |y|<1.4; use existing planner while near, plus old triggers')
        result['total_wall_seconds']=time.monotonic()-start
        path.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
