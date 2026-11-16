"""Measure real environment, policy and optimizer allocations; no preallocation padding."""
import argparse
import json
import time
import subprocess
from pathlib import Path
import jax
import numpy as np
from nav3.bank import MapBank
from nav3.env import FastEnv
from nav3.dreamer_adapter import DreamerPolicy, embodied

p=argparse.ArgumentParser()
p.add_argument('--envs',type=int,required=True)
a=p.parse_args()
start=time.monotonic()
bank=MapBank('train')
policy=DreamerPolicy(logdir=f'checkpoints/memory_probe_{a.envs}')
env=FastEnv(bank,a.envs,True)
state,obs,_=env.reset(jax.random.PRNGKey(743))
replay=embodied.Replay(length=65,capacity=4096,chunksize=256,online=True,seed=0)
samples=[]
def sample(stage):
    used,total=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
    row=dict(stage=stage,used_mib=used,total_mib=total,fraction=used/total)
    samples.append(row);print(json.dumps(row),flush=True)
for t in range(66):
    action,outs=policy.policy(obs)
    tran=jax.device_get({**obs,'action':action,**outs})
    for i in range(16):replay.add({k:v[i] for k,v in tran.items()},i)
    state,obs,_=env.step(state,action)
jax.block_until_ready(obs)
sample('environment_and_actor')
def batches():
    while True:
        batch=replay.sample(16)
        batch['consec']=np.zeros(batch['is_first'].shape,np.int32)
        yield batch
stream=iter(policy.agent.stream(batches))
carry=policy.agent.init_train(16)
losses=[]
def check_metrics(metrics):
    for key,value in metrics.items():
        if key.startswith('loss/') or key=='opt/loss':
            value=np.asarray(value)
            assert np.isfinite(value).all(),key
            losses.append(float(value))
for i in range(8):
    carry,outs,metrics=policy.agent.train(carry,next(stream))
    check_metrics(metrics)
sample('optimizer_warm')
tick=time.monotonic()
for _ in range(5):
    action,_=policy.policy(obs)
    state,obs,_=env.step(state,action)
    # Same online update ratio as the actual job.
    for i in range(max(1,a.envs//32)):
        carry,outs,metrics=policy.agent.train(carry,next(stream))
        check_metrics(metrics)
    jax.block_until_ready(obs)
sample('mixed_training')
assert losses, 'No optimizer metrics received'
result=dict(envs=a.envs,samples=samples,env_steps_per_second=5*a.envs/(time.monotonic()-tick),wall_seconds=time.monotonic()-start,finite_loss=True)
Path(f'metrics/memory_probe_{a.envs}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
