"""(d) Waypoints and corridor channel come from the ESTIMATED pose; extra route quantities stay TRUE."""
import unittest

import jax.numpy as jnp
import numpy as np

from nav4.common import bev_xy, decode
from nav4.tests.helpers import env as make_env, reset


class EstimatedPose(unittest.TestCase):
    def test_waypoints_and_corridor_follow_the_estimate(self):
        env = make_env(('flat',), n=1)
        y = bev_xy(env.cfg)[0, :, 1]
        state, obs, extra = reset(env, 5.)
        np.testing.assert_allclose(np.asarray(obs['state'][0, :6]).reshape(3, 2)*4, [[1, 0], [2, 0], [3, 0]], atol=1e-4)
        state = {**state, 'drift': jnp.array([[.6, .3, 0.]])}          # localisation error: 0.6 m ahead, 0.3 m left
        state, obs, extra = env.observe(state)
        drift = np.asarray(state['drift'][0])
        goals = np.asarray(obs['state'][0, :6]).reshape(3, 2)*4
        # Estimated station = true + 0.6 m, so the waypoints are 1/2/3 m ahead of the ESTIMATED pose
        # (legacy used the true station: 0.4/1.4/2.4 m).
        np.testing.assert_allclose(goals, [[1, -drift[1]], [2, -drift[1]], [3, -drift[1]]], atol=2e-3)
        self.assertAlmostEqual(float(state['est_s'][0]), 5+drift[0], places=3)
        row = int(np.argmin(np.abs(bev_xy(env.cfg)[:, 0, 0]-1.)))       # cells 1 m ahead
        corridor = y[np.asarray(decode(obs['bev'], env.cfg))[0, row, :, 4] > .5]
        self.assertLess(abs(corridor.min()-(-1-drift[1])), .13)         # corridor shifted by the estimate error
        self.assertLess(abs(corridor.max()-(1-drift[1])), .13)
        truth = y[np.asarray(extra['route_mask'])[0, row]]
        self.assertLess(abs(truth.min()+1), .13)                         # privileged mask uses the true pose
        self.assertLess(abs(truth.max()-1), .13)
        self.assertAlmostEqual(float(extra['lateral'][0]), 0., places=4)
        self.assertAlmostEqual(float(extra['heading_error'][0]), 0., places=4)


if __name__ == '__main__':
    unittest.main()
