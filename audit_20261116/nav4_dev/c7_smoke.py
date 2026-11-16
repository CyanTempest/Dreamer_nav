"""C7 smoke: load data/trail_a (forward only), reset + 200 planner steps at 64 envs; shapes, memory, steps/s."""
import json
import os
import resource
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.planner import Planner

t0 = time.time()
bank = TrailBank('data/trail_a', ['forward'])
load_s = time.time()-t0
env = TrailEnv(bank, 64)
planner = Planner(env.cfg)
t = time.time()
state, obs, extra = env.reset(jax.random.PRNGKey(20261116))            # training-style random stations
action = planner(obs)
state, obs, extra = env.step(state, action)
jax.block_until_ready(obs['bev'])
compile_s = time.time()-t
t = time.time()
done = 0
for _ in range(200):
    state, obs, extra = env.step(state, planner(obs))
    done += int(np.asarray(obs['is_last']).sum())
jax.block_until_ready(obs['bev'])
dt = time.time()-t
stats = jax.devices()[0].memory_stats() or {}
print('C7_SMOKE', json.dumps(dict(
    maps=bank.names, bank_shapes={k: list(getattr(bank, k).shape) for k in bank.KEYS},
    bank_dtypes={k: str(getattr(bank, k).dtype) for k in bank.KEYS},
    bank_bytes=int(sum(getattr(bank, k).nbytes for k in bank.KEYS)), load_s=load_s, compile_s=compile_s,
    obs_shapes={k: [list(v.shape), str(v.dtype)] for k, v in obs.items()},
    extra_shapes={k: list(v.shape) for k, v in extra.items()},
    steps=200, envs=64, seconds=dt, env_steps_per_s_with_planner=64*200/dt, episodes_ended=done,
    failures=np.bincount(np.asarray(state['failure']), minlength=6).tolist(),
    start_stations=np.asarray(state['s']).round(1).tolist()[:16],
    device=str(jax.devices()[0]), jax_peak_bytes_in_use=stats.get('peak_bytes_in_use'),
    host_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)), flush=True)
