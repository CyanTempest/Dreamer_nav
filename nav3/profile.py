import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from .bank import MapBank
from .env import FastEnv


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--envs',type=int,default=1024)
    ap.add_argument('--steps',type=int,default=100)
    ap.add_argument('--split',default='train')
    a=ap.parse_args()
    bank=MapBank(a.split)
    env=FastEnv(bank,a.envs,True)
    start=time.monotonic()
    state,obs,_=env.reset(jax.random.PRNGKey(42))
    action=jnp.tile(jnp.array([0.,0.,.2]),(a.envs,1))
    for _ in range(3):
        state,obs,_=env.step(state,action)
    jax.block_until_ready(obs)
    compile_seconds=time.monotonic()-start
    trials=[]
    for _ in range(3):
        start=time.monotonic()
        for step in range(a.steps):
            state,obs,_=env.step(state,action)
        jax.block_until_ready(obs)
        elapsed=time.monotonic()-start
        trials.append(dict(seconds=elapsed,env_steps_per_second=a.envs*a.steps/elapsed))
    result=dict(envs=a.envs,steps_per_trial=a.steps,trials=trials,compile_seconds=compile_seconds,
                includes=['dynamics 20 Hz','noisy sensing','accumulation','reset','privileged erosion'],backend=jax.default_backend(),map_count=bank.count,map_split=a.split)
    Path('metrics/profile.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
