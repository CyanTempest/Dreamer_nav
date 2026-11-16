"""One tracking law: env substeps, planner rollout and support-filter prediction move identically."""
import unittest

import jax.numpy as jnp
import numpy as np

from nav4.common import action_target, rotate, target_action
from nav4.planner import Planner
from nav4.tests.helpers import env as make_env, reset


def trajectory(pos, yaw, vel):
    return pos, yaw, vel


class TrackerConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(('flat',), n=2)
        cls.planner = Planner(cls.env.cfg)

    def test_env_planner_filter_identical(self):
        env, planner = self.env, self.planner
        state, obs, _ = reset(env, 5.)
        action = jnp.array([[.2, .5, .6], [-.4, -.8, .1]], jnp.float32)
        local, speed = action_target(action, env.cfg)
        world = state['pos']+rotate(local, state['yaw'])
        pos0, yaw0 = state['pos'], state['yaw']
        positions = []
        for _ in range(4):                                    # 4 policy steps = 16 substeps, same world target
            act = target_action(rotate(world-state['pos'], -state['yaw']), speed, env.cfg)
            state, obs, _ = env.step(state, act)
            positions.append((state['pos'], state['yaw'], state['vel']))
        fixed = planner.rollout(jnp.zeros((2, 3)), lambda p: local, speed, 16, trajectory)
        steps = jnp.zeros((2, 1, 3))                           # planner candidate layout [N,K]
        candidates = planner.rollout(steps[:, 0], lambda p: local[:, None], speed[:, None], 16, trajectory)
        for k, (pos, yaw, vel) in enumerate(positions):
            t = 4*k+3
            for p, y, v in ((fixed[0][t], fixed[1][t], fixed[2][t]),
                            (candidates[0][t][:, 0], candidates[1][t][:, 0], candidates[2][t][:, 0])):
                np.testing.assert_allclose(np.asarray(pos0+rotate(p, yaw0)), np.asarray(pos), atol=1e-5)
                np.testing.assert_allclose(np.asarray(yaw0+y), np.asarray(yaw), atol=1e-5)
                np.testing.assert_allclose(np.asarray(v), np.asarray(vel), atol=1e-5)

    def test_filter_prediction_is_the_shared_rollout(self):
        state, obs, _ = reset(self.env, 5.)
        action = jnp.array([[1., .3, 1.], [0., 0., -1.]], jnp.float32)
        local, speed = action_target(action, self.env.cfg)
        expected = self.planner.rollout(obs['state'][:, 7:10], lambda p: local, speed, 16, trajectory)
        got = self.planner.predict(obs, action, 16, trajectory)
        for a, b in zip(expected, got):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_configured_yaw_gain_is_used(self):
        """R9 ignored its configured yaw_tracking_gain; nav4 env and planner both read tracker.yaw_gain."""
        fast = make_env(('flat',), n=2, **{'tracker.yaw_gain': 4.})
        action = jnp.array([[.2, .5, .6], [.2, -.5, .6]], jnp.float32)
        yaws = []
        for env in (self.env, fast):
            state, _, _ = reset(env, 5.)
            for _ in range(4):
                state, _, _ = env.step(state, action)
            yaws.append(np.asarray(state['yaw']))
        self.assertTrue(np.all(np.abs(yaws[1]) > np.abs(yaws[0])+.02))


if __name__ == '__main__':
    unittest.main()
