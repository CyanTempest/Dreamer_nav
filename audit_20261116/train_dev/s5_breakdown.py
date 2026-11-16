"""S5 follow-up: per-component wall time of the trainer's driver step (env step, policy step, host copy), each
synchronised, with policy actions, and env-only with random actions, on CPU.

  JAX_PLATFORMS=cpu $NAV3_PYTHON audit_20261116/train_dev/s5_breakdown.py 64 30
"""
import json
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.policy import DreamerPolicy

cfg = json.load(open('nav4/config/train.json'))
envs, steps = int(sys.argv[1]), int(sys.argv[2])
bank = TrailBank(cfg['manifest'], cfg['maps'])
env = TrailEnv(bank, envs, cfg['env'])
key = jax.random.PRNGKey(0)
state, obs, _ = env.reset(key)
policy = DreamerPolicy(0, tempfile.mkdtemp(prefix='s5b_', dir='audit_20261116/train_dev'), obs=obs,
                       jax_overrides=dict(platform='cpu', autotune=1, deterministic=True))
keys = list(policy.obs_space)


def act(carry, obs, k):
    _, (carry, a, outs) = policy._policy(policy.agent.params, carry, {x: obs[x] for x in keys}, mode='train', seed=k)
    action = jnp.where(obs['is_last'][:, None], 0., a['action'])
    return (*carry[:3], {'action': action}), action, outs


random = jax.jit(lambda k: jax.random.uniform(k, (envs, 3), minval=-1, maxval=1))
carry = policy.initial(envs)
carry, action, outs = act(carry, obs, key)
state, obs, _ = env.step(state, action)
jax.block_until_ready((carry, obs))
parts = dict(env=[], policy=[], host=[], resets=[])
for i in range(steps):
    t = time.time()
    state, obs, extra = env.step(state, action)
    jax.block_until_ready(obs)
    parts['env'].append(time.time()-t)
    t = time.time()
    carry, action, outs = act(carry, obs, jax.random.fold_in(key, i))
    jax.block_until_ready(action)
    parts['policy'].append(time.time()-t)
    t = time.time()
    tran = jax.device_get(({k: obs[k] for k in keys}, outs, action))
    parts['host'].append(time.time()-t)
    parts['resets'].append(int(tran[0]['is_first'].sum()))
rand = []
for i in range(steps):
    t = time.time()
    state, obs, _ = env.step(state, random(jax.random.fold_in(key, 1000+i)))
    jax.block_until_ready(obs)
    rand.append(time.time()-t)
print(json.dumps(dict(envs=envs, steps=steps, mean_s={k: float(np.mean(v)) for k, v in parts.items() if k != 'resets'},
                      median_s={k: float(np.median(v)) for k, v in parts.items() if k != 'resets'},
                      env_random_actions_mean_s=float(np.mean(rand)), env_random_actions_median_s=float(np.median(rand)),
                      mean_resets_per_step=float(np.mean(parts['resets'])),
                      env_steps_per_s=envs/sum(float(np.mean(v)) for k, v in parts.items() if k != 'resets'))))
