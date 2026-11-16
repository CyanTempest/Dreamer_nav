import numpy as np, jax, jax.numpy as jnp
from nav4.common import decode, bev_xy
from nav4.planner import Planner, Hybrid
from nav4.tests.helpers import env as make_env, reset
env = make_env(('wall',), n=1)
pl = Planner(env.cfg)
state, obs, _ = reset(env, 9.5)
b = np.asarray(decode(obs['bev'], env.cfg))[0]
xy = bev_xy(env.cfg)
sel = (np.abs(xy[...,0]-2.7) < .3) & (xy[...,1] > -.5) & (xy[...,1] < 1.8)
print('wall cells unknown', b[...,3][sel].round(2))
print('wall cells obstacle', b[...,1][sel].round(2))
print('truth layer5 at wall', np.asarray(env.bank.sample(jnp.zeros(1,jnp.int32), jnp.array([[12.2, 0.]])) ))
g, low, high, cov, known = env.bank.atlas(jnp.zeros(1,jnp.int32), jnp.array([[12.2, 0.]]), 3)
print('atlas', g, low, high, cov, known)
risk, unk, cor = pl.assess(jnp.asarray(b[None]), jnp.array([[[3.,0.]]]))
print('assess', risk, unk, cor)
a, info = pl.plan(obs); print('plan', a, {k: np.asarray(v) for k,v in info.items()})
print('support risk', pl.support_risk(obs, jnp.array([[1.,0.,1.]])))
