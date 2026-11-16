"""Failure-code tally for 200 planner steps at 64 random stations on trail_a forward, with failure stations."""
import json
import jax
import numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.planner import Planner
bank = TrailBank('data/trail_a', ['forward'])
env = TrailEnv(bank, 64)
planner = Planner(env.cfg)
state, obs, _ = env.reset(jax.random.PRNGKey(20261116))
valid = np.asarray(env.feasible(state, state['pos'], state['yaw']))
print('feasible random starts', int(valid.sum()), '/ 64')
ends = []
for t in range(200):
    starts = np.asarray(state['s']).copy()
    state, obs, _ = env.step(state, planner(obs))
    h = jax.device_get({k: state[k] for k in ('done', 'failure', 's', 'steps', 'arrived')})
    for i in np.flatnonzero(np.asarray(obs['is_last'])):
        ends.append((int(h['failure'][i]), round(float(h['s'][i]), 1), int(h['steps'][i])))
codes = {}
for c, s, n in ends:
    codes.setdefault(c, []).append((s, n))
print(json.dumps({str(k): dict(count=len(v), first=v[:12]) for k, v in codes.items()}))
