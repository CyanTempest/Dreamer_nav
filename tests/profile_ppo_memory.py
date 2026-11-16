"""Two real PPO rollouts and optimizer passes per candidate, no allocation padding."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from nav3.bank import MapBank
from nav3.env import FastEnv
from nav3.ppo import PPOPolicy, act, advantages, update
from nav3.memory import rss_bytes, release_host_memory

ap=argparse.ArgumentParser();ap.add_argument('--envs',type=int,required=True)
a=ap.parse_args()
bank=MapBank('train');env=FastEnv(bank,a.envs,True);policy=PPOPolicy(0)
with Path('checkpoints/ppo_s0/latest.pkl').open('rb') as f:checkpoint=pickle.load(f)
policy.params,policy.optimizer=jax.tree.map(jnp.asarray,checkpoint['policy'][:2])
del checkpoint
release_host_memory()
state,obs,_=env.reset(jax.random.PRNGKey(0));rng=np.random.default_rng(42)
samples=[];times=[]
def sample(stage):
    used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    samples.append(dict(stage=stage,used_mib=used,total_mib=total,fraction=used/total,rss_gb=rss_bytes()/1e9))
    print(json.dumps(samples[-1]),flush=True)
for trial in range(2):
    start=time.monotonic();rollout=[]
    for t in range(128):
        action,info=policy.policy(obs)
        nstate,nobs,_=env.step(state,action)
        _,_,_,_,nv=act(policy.params,policy.carry,nobs,jax.random.PRNGKey(0))
        rollout.append(dict(bev=obs['bev'],state=obs['state'],first=obs['is_first'],valid=~obs['is_last'],
            reward=nobs['reward'],last=nobs['is_last'],terminal=nobs['is_terminal'],next_value=nv,**info))
        state,obs=nstate,nobs
    data=jax.tree.map(lambda *x:jnp.stack(x),*rollout)
    jax.block_until_ready(data);del rollout
    adv,returns=advantages(data['reward'],data['value'],data['next_value'],data['last'],data['terminal'],data['valid'])
    data.update(advantage=adv,**{'return':returns})
    sequences=jax.tree.map(lambda x:x.reshape((4,32,a.envs,*x.shape[2:])).swapaxes(1,2).reshape((-1,32,*x.shape[2:])),data)
    jax.block_until_ready(sequences);del data,adv,returns
    sample(f'rollout_{trial}')
    for epoch in range(4):
        order=rng.permutation(4*a.envs)
        for i in range(0,len(order),16):
            batch=jax.tree.map(lambda x:x[order[i:i+16]].swapaxes(0,1),sequences)
            policy.params,policy.optimizer,metrics=update(policy.params,policy.optimizer,batch)
    jax.block_until_ready(policy.params)
    assert all(np.isfinite(np.asarray(x)).all() for x in metrics.values())
    times.append(time.monotonic()-start)
    sample(f'optimizer_{trial}')
    del sequences,batch
result=dict(envs=a.envs,samples=samples,trial_seconds=times,
            env_steps_per_second=128*a.envs/times[-1],finite_loss=True,
            peak_gpu_fraction=max(x['fraction'] for x in samples),peak_rss_gb=max(x['rss_gb'] for x in samples))
Path(f'metrics/ppo_memory_probe_{a.envs}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
