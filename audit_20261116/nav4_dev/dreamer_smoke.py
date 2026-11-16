import time, jax, jax.numpy as jnp, numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.policy import DreamerPolicy
bank = TrailBank('r9_official_runner/maps_scan_holdout')
env = TrailEnv(bank, bank.count*8)
state, obs, _ = env.reset(jax.random.PRNGKey(914901), start_s=2.)
t = time.time()
d = DreamerPolicy(0, 'audit_20261116/nav4_dev/policy_runtime', jax_overrides=dict(autotune=1, deterministic=True), deterministic=True).load('checkpoints/dreamer_lwy_r8_pilot_20261113/latest.pkl')
print('load', time.time()-t, jax.devices())
print('leak', d.leakage_check(obs))
t = time.time()
for i in range(20):
    a = d(obs); d.executed(a); state, obs, _ = env.step(state, a)
jax.block_until_ready(a); print('20 steps', time.time()-t, np.asarray(a[:3]))
t = time.time()
for i in range(20):
    a = d(obs); d.executed(a); state, obs, _ = env.step(state, a)
jax.block_until_ready(a); print('20 more steps', time.time()-t)
