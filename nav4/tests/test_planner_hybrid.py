"""Planner port avoids an observed wall; hybrid takes over on a risky policy action and holds."""
import unittest

import jax.numpy as jnp
import numpy as np

from nav4.planner import Hybrid, Planner
from nav4.tests.helpers import env as make_env, reset


class PlannerAndHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(('wall',), n=1)                  # wall x 12..12.4, y 0..1.6
        cls.planner = Planner(cls.env.cfg)

    def test_planner_passes_the_wall(self):
        state, obs, _ = reset(self.env, 8.)
        lowest = 0.
        for _ in range(200):
            state, obs, _ = self.env.step(state, self.planner(obs))
            lowest = min(lowest, float(state['pos'][0, 1]))
            if float(state['s'][0]) > 14. or bool(state['done'][0]):
                break
        self.assertEqual(int(state['failure'][0]), 0)
        self.assertGreater(float(state['s'][0]), 14.)
        self.assertLess(lowest, -.3)                         # went round on the open (right) side

    def test_takeover_and_hold(self):
        state, obs, _ = reset(self.env, 9.5)
        state, obs, _ = self.env.step(state, jnp.array([[-1., 0., -1.]]))   # is_first resets the hold counter
        hybrid = Hybrid(self.planner)
        executed = hybrid(jnp.array([[1., 0., 1.]]), obs)    # 3 m straight into the wall at full speed
        self.assertTrue(bool(hybrid.takeover[0]))
        np.testing.assert_allclose(np.asarray(executed), np.asarray(self.planner(obs)))
        hold = self.planner.cfg['filter']['hold_steps']
        flags = []
        for _ in range(hold+2):
            hybrid(jnp.array([[-1., 0., -1.]]), obs)         # 0.5 m ahead, zero speed: safe
            flags.append(bool(hybrid.takeover[0]))
        self.assertEqual(flags, [True]*hold+[False]*2)


if __name__ == '__main__':
    unittest.main()
