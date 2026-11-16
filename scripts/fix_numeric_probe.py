"""Isolate runtime arithmetic: identical input, weights, RNG; record compiled HLO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--level',type=int);p.add_argument('--warm',action='store_true');p.add_argument('--name',default='ppo_s0');a=p.parse_args()
if a.level is not None:
    os.environ['XLA_FLAGS']=f'--xla_gpu_autotune_level={a.level} --xla_gpu_deterministic_ops=true'
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
import jax
import jax.numpy as jnp
import numpy as np
startup_flags=os.environ.get('XLA_FLAGS')
if a.name.startswith('dreamer') and a.level is not None:jax.devices()
from nav3.diag_rollout import load_policy
from nav3.bank import MapBank
from nav3.env import FastEnv
from nav3.ppo import act
policy=load_policy(a.name)
bank=MapBank('test');env=FastEnv(bank,30,False)
state,obs,_=env.reset(jax.random.PRNGKey(9137),jnp.zeros(30,jnp.int32))
if a.warm:
    warm={k:jnp.repeat(v[:1],100,axis=0) for k,v in obs.items()}
    policy(warm);jax.block_until_ready(policy.carry)
policy.carry=None;policy.key=jax.random.PRNGKey(9137)
action=policy(obs)
prediction=policy.decoder() if policy.head else jnp.zeros(1)
def sha(tree):
    h=hashlib.sha256()
    for x in jax.tree.leaves(tree):h.update(np.asarray(x).tobytes())
    return h.hexdigest()
np.savez_compressed(f'metrics/fix_numeric_{a.tag}.npz',action=np.asarray(action),prediction=np.asarray(prediction,dtype=np.float32))
result=dict(name=a.name,flags=os.environ.get('XLA_FLAGS'),startup_flags=startup_flags,obs_sha256=sha(obs),state_sha256=sha(state),
    params_sha256=sha(policy.agent.params if policy.head else policy.params),key=policy.key.tolist(),
    action_sha256=sha(action),prediction_sha256=sha(prediction),
    prediction_min=float(jnp.min(prediction)),prediction_max=float(jnp.max(prediction)),
    finite_prediction=bool(jnp.isfinite(prediction).all()))
if not policy.head:
    compiled=act.lower(policy.params,jnp.zeros((30,256)),obs,jax.random.PRNGKey(0)).compile()
    hlo=compiled.as_text();Path(f'logs/fix_numeric_{a.tag}.hlo').write_text(hlo)
    result['hlo_sha256']=hashlib.sha256(hlo.encode()).hexdigest()
with Path(f'metrics/fix_numeric_{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
