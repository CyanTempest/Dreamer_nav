"""Env-only CPU step time (random actions, 64 envs, trail_a forward) per input variant, split by whether any slot
reset in that step (lax.cond branch)."""
import json
import time

import jax
import numpy as np

from nav4.bank import TrailBank
from nav4.env import TrailEnv

bank = TrailBank("data/trail_a", ["forward"])
n, steps, out = 64, 40, {}
for name, config in (("default", {}), ("route_off", {"route_inputs": False}),
                     ("train", {"route_inputs": False, "odometry_inputs": True, "priv_route": True})):
    env = TrailEnv(bank, n, config)
    key = jax.random.PRNGKey(0)
    state, obs, _ = env.reset(key)
    act = jax.jit(lambda k: jax.random.uniform(k, (n, 3), minval=-1, maxval=1))
    for i in range(3):
        state, obs, _ = env.step(state, act(jax.random.fold_in(key, i)))
    jax.block_until_ready(obs)
    times, resets = [], []
    for i in range(steps):
        resetting = bool(np.asarray(state["done"]).any())
        t = time.time()
        state, obs, _ = env.step(state, act(jax.random.fold_in(key, 10+i)))
        jax.block_until_ready(obs)
        times.append(time.time()-t)
        resets.append(resetting)
    times, resets = np.array(times), np.array(resets)
    out[name] = dict(mean_s=float(times.mean()), median_s=float(np.median(times)),
                     with_reset_mean_s=float(times[resets].mean()) if resets.any() else None,
                     without_reset_mean_s=float(times[~resets].mean()) if (~resets).any() else None,
                     steps_with_reset=int(resets.sum()), steps=steps)
print(json.dumps(out))
