import sys, time, tempfile
import jax, jax.numpy as jnp, numpy as np
from nav4.tests import synthetic
from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.planner import Planner
d = tempfile.mkdtemp(dir='audit_20261116/tmp')
m = synthetic.write(d, {'flat': synthetic.build('flat'), 'ramp': synthetic.build('ramp_down', dtype=np.float16)})
bank = TrailBank(m)
env = TrailEnv(bank, 4)
t=time.time(); state, obs, extra = env.reset(jax.random.PRNGKey(0), start_s=2.0); jax.block_until_ready(obs['bev']); print('reset', time.time()-t)
print({k:(v.shape, v.dtype) for k,v in obs.items()})
print({k:(v.shape, v.dtype) for k,v in extra.items()})
pl = Planner(env.cfg)
t=time.time()
for i in range(40):
    a = pl(obs)
    state, obs, extra = env.step(state, a)
jax.block_until_ready(obs['bev']); print('40 steps', time.time()-t)
print('s', state['s'], 'fail', state['failure'], 'vel', state['vel'][:, :2], 'gates', state['gates'])
print('state vec', obs['state'][0])
print('extra', extra['progress'], extra['lateral'], extra['heading_error'])
t=time.time()
for i in range(40):
    a = pl(obs); state, obs, extra = env.step(state, a)
jax.block_until_ready(obs['bev']); print('40 more', time.time()-t, state['s'], state['failure'])
