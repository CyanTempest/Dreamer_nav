"""Fraction of training start poses (env.start_station, start_lateral offset, start yaw jitter) whose envelope is
infeasible in truth on data/trail_a forward (current map.npz), i.e. episodes that fail at their first step."""
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from nav4.bank import TrailBank
from nav4.common import rotate, wrap
from nav4.env import TrailEnv

cfg = json.load(open("nav4/config/train.json"))
bank = TrailBank(cfg["manifest"], cfg["maps"])
n = 20000
env = TrailEnv(bank, n, cfg["env"])
t = env.cfg["task"]
k = jax.random.split(jax.random.PRNGKey(0), 3)
mids = jnp.zeros(n, jnp.int32)
s = env.start_station(k[0], mids)
offset = jax.random.uniform(k[1], (n,), minval=-t["start_lateral"], maxval=t["start_lateral"])
p, yaw, _ = bank.route_at(mids, s)
pos = p[:, :2]+rotate(jnp.stack([jnp.zeros(n), offset], -1), yaw)
yaw = wrap(yaw+jax.random.normal(k[2], (n,))*t["start_yaw_std"])
state = dict(mids=mids, obstacles=jnp.zeros((n, 0, 6)))
ok = np.asarray(env.feasible(state, pos, yaw))
s = np.asarray(s)
at_start = s == t["route_start"]
print(json.dumps(dict(map_sha256=hashlib.sha256(open("data/trail_a/forward/map.npz", "rb").read()).hexdigest(),
    samples=n, infeasible_fraction=float(1-ok.mean()), race_start_samples=int(at_start.sum()),
    race_start_infeasible_fraction=float(1-ok[at_start].mean()),
    infeasible_fraction_by_100m={f"{a}-{a+100}": float(1-ok[(s >= a) & (s < a+100)].mean()) for a in range(0, 800, 100)
                                 if ((s >= a) & (s < a+100)).any()})))
