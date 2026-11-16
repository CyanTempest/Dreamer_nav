"""Envelope feasibility along the trail_a forward centreline (truth layers), at lateral offsets."""
import json
import jax
import jax.numpy as jnp
import numpy as np
from nav4.bank import TrailBank
from nav4.env import TrailEnv
bank = TrailBank('data/trail_a', ['forward'])
env = TrailEnv(bank, 1)
state = env.reset(jax.random.PRNGKey(0), start_s=2.)[0]
s = np.arange(0, float(bank.final(jnp.zeros(1, jnp.int32))[0]), .2, dtype=np.float32)
offsets = np.array([0., -.1, .1, -.2, .2, -.3, .3, -.5, .5], np.float32)
p, yaw, width = bank.route_at(jnp.zeros(1, jnp.int32), jnp.asarray(s)[None])
yaw = jnp.broadcast_to(yaw[..., None], yaw.shape+offsets.shape)
pos = p[..., None, :2]+jnp.stack([-jnp.sin(yaw)*offsets, jnp.cos(yaw)*offsets], -1)
ok = np.asarray(env.feasible(state, pos, yaw))[0]
foot = env.foot
xy = p[0, :, None, :2]+jnp.stack([jnp.cos(yaw[0, :, 0])[:, None]*foot[:, 0]-jnp.sin(yaw[0, :, 0])[:, None]*foot[:, 1],
                                   jnp.sin(yaw[0, :, 0])[:, None]*foot[:, 0]+jnp.cos(yaw[0, :, 0])[:, None]*foot[:, 1]], -1)
data = np.asarray(bank.sample(jnp.zeros(1, jnp.int32), xy.reshape(1, -1, 2))).reshape(len(s), -1, 7)
unsupported = (data[..., 2] > .025).any(-1)
collide = (data[..., 5] < 0).any(-1)
centre = ok[:, 0]
anyoff = ok.any(1)
runs = np.split(s[~centre], np.flatnonzero(np.diff(s[~centre]) > .21)+1) if (~centre).any() else []
print('C7_FEASIBLE', json.dumps(dict(
    stations=len(s), centre_feasible=float(centre.mean()), any_offset_within_0p5_feasible=float(anyoff.mean()),
    centre_unsupported=float(unsupported.mean()), centre_collision=float(collide.mean()),
    infeasible_runs=len(runs), runs_without_any_feasible_offset=sum(1 for r in runs if not anyoff[np.isin(s, r)].any()),
    first_runs=[[float(r[0]), float(r[-1])] for r in runs[:25]],
    blocked_stretches=[[float(r[0]), float(r[-1])] for r in np.split(s[~anyoff], np.flatnonzero(np.diff(s[~anyoff]) > .21)+1)][:30] if (~anyoff).any() else [],
    width_min=float(np.asarray(width).min()))))
