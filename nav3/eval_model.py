import argparse
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
from .evaluate import evaluate
from .control import planner


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='planner');ap.add_argument('--all',action='store_true')
    ap.add_argument('--split',default='heldout');ap.add_argument('--random-obstacles',action='store_true');ap.add_argument('--filter',action='store_true')
    a=ap.parse_args()
    if a.name=='planner':policy=planner
    else:
        config=json.loads((Path('metrics')/f'config_{a.name}.json').read_text())
        with (Path('checkpoints')/a.name/'latest.pkl').open('rb') as f:data=pickle.load(f)
        if a.name.startswith('ppo'):
            from .ppo import PPOPolicy
            policy=PPOPolicy(config['seed']);policy.params=jax.tree.map(jnp.asarray,data['policy'][0])
        else:
            from .dreamer_adapter import DreamerPolicy
            policy=DreamerPolicy(config['seed'],f'checkpoints/{a.name}',not config['no_head'],config['amnesic'])
            policy.agent.load(data['agent'])
        del data
        from .memory import release_host_memory
        release_host_memory()
    conditions=[(s,o,f) for s in ('heldout','test') for o in (False,True) for f in (False,True)] if a.all else [(a.split,a.random_obstacles,a.filter)]
    for split,obstacles,filtered in conditions:
        if hasattr(policy,'carry'):policy.carry=None;policy.key=jax.random.PRNGKey(9137)
        direct=False
        if a.name!='planner':direct=config.get('direct',False)
        evaluate(policy,split,30 if split=='test' else 5,obstacles,filtered,method=a.name,direct=direct)


if __name__=='__main__':main()
