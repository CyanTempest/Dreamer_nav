"""Attribute C7 assist failures to map feasibility: is the centreline (any lateral offset within +-0.5 m)
envelope-infeasible within 1.5 m ahead of each failure station?"""
import json
import jax
import jax.numpy as jnp
import numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
bank = TrailBank('data/trail_a', ['forward'])
env = TrailEnv(bank, 1)
state = env.reset(jax.random.PRNGKey(0), start_s=2.)[0]
s = np.arange(0, 727.0, .1, dtype=np.float32)
offsets = np.array([0., -.1, .1, -.2, .2, -.3, .3, -.4, .4, -.5, .5], np.float32)
p, yaw, _ = bank.route_at(jnp.zeros(1, jnp.int32), jnp.asarray(s)[None])
yaw = jnp.broadcast_to(yaw[..., None], yaw.shape+offsets.shape)
pos = p[..., None, :2]+jnp.stack([-jnp.sin(yaw)*offsets, jnp.cos(yaw)*offsets], -1)
ok = np.asarray(env.feasible(state, pos, yaw))[0]
centre, anyoff = ok[:, 0], ok.any(1)
r = json.load(open('audit_20261116/nav4_dev/c7_planner_forward_assist.json'))
fails = np.array([f[1] for row in r['rows'] for f in row['failures']])
codes = np.array([f[0] for row in r['rows'] for f in row['failures']])
def near(mask, a, b):
    idx = np.searchsorted(s, fails)
    return np.array([(~mask[max(0, i+int(a/.1)):i+int(b/.1)+1]).any() for i in idx])
blocked = near(anyoff, -.5, 1.5)
centre_bad = near(centre, -.5, 1.5)
hist, edges = np.histogram(fails, bins=np.arange(0, 800, 100))
print('C7_ATTRIB', json.dumps(dict(
    failures=len(fails), within_1p5m_of_station_with_no_feasible_offset=float(blocked.mean()),
    within_1p5m_of_centre_infeasible=float(centre_bad.mean()),
    centre_feasible_fraction=float(centre.mean()), no_offset_feasible_fraction=float(1-anyoff.mean()),
    per_100m=dict(zip([int(e) for e in edges[:-1]], hist.tolist())),
    codes_where_feasible={str(c): int(((codes == c) & ~centre_bad).sum()) for c in (1, 2)},
    unique_failure_spots_0p5m=int(len(np.unique(np.round(fails/.5)))))))
