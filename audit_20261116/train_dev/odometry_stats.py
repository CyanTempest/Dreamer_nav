"""Measured vs configured odometry error statistics (the quantities test_env_odometry.NoiseModel asserts), n=20000."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import QUIET, load

n = 20000
env = TrailEnv(load({"m": synthetic.build("flat")}), n, {**QUIET, "route_inputs": False, "odometry_inputs": True})
o = env.cfg["odometry"]
mids = jnp.zeros(n, jnp.int32)
s = jnp.full(n, 27.)
p, yaw, _ = env.bank.route_at(mids, s)
st = jax.jit(lambda e, k: e.odometry_start(k, mids, s, p[:, :2], yaw, yaw))(env, jax.random.PRNGKey(1))
D = 25.
err = np.asarray(st["odo_pos"]-p[:, :2])
out = dict(samples=n, training_start_distance_m=D,
           distance_scale=dict(measured_std=float(np.std(np.asarray(st["odo_dist"])/D-1)), configured=o["distance_scale_std"]),
           heading=dict(measured_std=float(np.std(np.asarray(st["odo_err"]))), model=o["heading_walk_std"]*np.sqrt(D)),
           along_track_m=dict(measured_std=float(err[:, 0].std()), model=o["distance_scale_std"]*D),
           cross_track_m=dict(measured_std=float(err[:, 1].std()), model=o["heading_walk_std"]*np.sqrt(D**3/3)))
z = jnp.zeros(n)
state = dict(pos=jnp.tile(jnp.array([5., 0.]), (n, 1)), yaw=z, mids=mids, odo_err=z, odo_yaw0=z, odo_dist=z)
r = np.asarray(jax.jit(lambda e, k, x: e.odometry_reading(k, x))(env, jax.random.PRNGKey(3), state))
out["imu"] = dict(pitch_std=float(r[:, 3].std()), roll_std=float(r[:, 4].std()), configured=o["imu_std"])
print(json.dumps(out))
