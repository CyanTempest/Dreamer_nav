"""Shared fixtures for nav4 unit tests (CPU)."""
import tempfile

import jax
import numpy as np

from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.tests import synthetic

# Deterministic world: no sensor noise, slip, drift, start jitter; nominal gain and planner-equal delay.
QUIET = {'sensor.height_std': [0., 0.], 'sensor.dropout_base': [0., 0.], 'sensor.dropout_distance': [0., 0.],
         'sensor.velocity_std': 0., 'dynamics.slip_std': 0., 'dynamics.gain': [1., 1.],
         'dynamics.delay_s': [.15, .15], 'localization.position_walk_std': 0., 'localization.yaw_walk_std': 0.,
         'task.start_lateral': 0., 'task.start_yaw_std': 0.}


def load(maps, names=None):
    """TrailBank over {name: map dict} written to a temporary directory (removed again; the bank is in memory)."""
    with tempfile.TemporaryDirectory(prefix='nav4_test_') as root:
        return TrailBank(synthetic.write(root, maps), names)


def bank(*kinds, **kwargs):
    """TrailBank over freshly built synthetic maps (one per kind)."""
    return load({k: synthetic.build(k, **kwargs) for k in kinds})


def env(kinds, n=None, **config):
    b = bank(*kinds) if isinstance(kinds, tuple) else kinds
    return TrailEnv(b, n or b.count, {**QUIET, **config})


def reset(env, start, offset=0., seed=0):
    return env.reset(jax.random.PRNGKey(seed), start_s=start, offset=offset)


def drive(env, state, action, steps):
    """Step with a constant action; returns the list of (state, obs, extra)."""
    out = []
    for _ in range(steps):
        state, obs, extra = env.step(state, np.broadcast_to(np.asarray(action, np.float32), (env.n, 3)))
        out.append((state, obs, extra))
    return out
