import numpy as np, jax, jax.numpy as jnp
from nav4.common import decode, bev_xy
from nav4.planner import Planner
from nav4.tests.helpers import env as make_env, reset
env = make_env(('wall',), n=1)
pl = Planner(env.cfg)
state, obs, _ = reset(env, 8.)
for t in range(80):
    a, info = pl.plan(obs)
    state, obs, _ = env.step(state, a)
    print(t, 'pos', np.asarray(state['pos'][0]).round(2), 'yaw %.2f'%float(state['yaw'][0]), 'vel', np.asarray(state['vel'][0]).round(2), 'a', np.asarray(a[0]).round(2), 'off %.2f risk %.3f border %.3f minrisk %.3f'%(float(info['offset'][0]), float(info['selected_risk'][0]), float(info['selected_border'][0]), float(info['min_risk'][0])), 'fail', int(state['failure'][0]))
    if state['done'][0]: break
