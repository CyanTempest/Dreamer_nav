"""(a) support/drop from local geometry, downhill is not a drop; (b) no wrap-around at the BEV edge."""
import unittest

import jax.numpy as jnp
import numpy as np

from nav4.common import bev_xy, decode, load_config
from nav4.planner import Planner
from nav4.sensor import support_mask
from nav4.tests.helpers import env as make_env, reset

CFG = load_config('env')
XY = bev_xy(CFG)
X = XY[..., 0]


def mask(height, obstacle=None):
    h = jnp.asarray(height, jnp.float32)[None]
    ob = jnp.zeros_like(h) if obstacle is None else jnp.asarray(obstacle, jnp.float32)[None]
    return np.asarray(support_mask(h, jnp.ones(h.shape, bool), ob, CFG))[0]


def legacy_gradient_ok(h):
    """nav3/env.py:173-175 slope term: jnp.roll central difference over +-2 cells (wraps around)."""
    dx = (np.roll(h, -2, 0)-np.roll(h, 2, 0))/.5
    dy = (np.roll(h, -2, 1)-np.roll(h, 2, 1))/.5
    return np.hypot(dx, dy) < .8


class SupportMask(unittest.TestCase):
    def test_planes_supported_up_to_the_bev_edge(self):
        for deg in (10, 20, 30):
            for axis in (0, 1):
                for sign in (-1, 1):
                    h = sign*np.tan(np.deg2rad(deg))*XY[..., axis]
                    self.assertTrue(mask(h).all(), (deg, axis, sign))
        h = np.tan(np.deg2rad(20))*X
        old = legacy_gradient_ok(h)
        self.assertTrue(old[2:-2].all())
        self.assertFalse(old[[0, 1, -2, -1]].any())        # legacy wrap-around: border rows falsely unsupported

    def test_downhill_is_not_a_drop(self):
        h = -np.tan(np.deg2rad(20))*X
        self.assertTrue((h < -.55).any())                  # legacy rule (ground > -0.55) flags these cells
        self.assertTrue(mask(h).all())

    def test_steep_plane_unsupported(self):
        self.assertTrue(mask(np.tan(np.deg2rad(40))*X)[1:-1].all())
        self.assertFalse(mask(np.tan(np.deg2rad(48))*X)[1:-1].any())  # confirmed limit: 47 deg / 1 m

    def test_ledge_step_and_obstacle(self):
        m = mask(np.where(X > 2, -.5, 0.))
        edge = np.abs(X-2) < .1
        self.assertFalse(m[edge].any())
        self.assertTrue(m[~edge].all())
        self.assertTrue(mask(np.where(X > 2, -.15, 0.)).all())     # a 0.15 m step is walkable
        obstacle = np.zeros(X.shape)
        obstacle[30:33, 30:33] = CFG['terrain']['collision_height']
        m = mask(np.zeros(X.shape), obstacle)
        self.assertFalse(m[30:33, 30:33].any())
        self.assertEqual(int(m.sum()), 64*64-9)

    def test_stairs_with_017_risers_supported(self):
        stairs = np.floor(np.clip(X, 0, None)/.3)*.17
        self.assertTrue(mask(stairs).all())
        rng = np.random.default_rng(0)
        self.assertGreater(min(mask(stairs+rng.normal(0, .025, X.shape)).mean() for _ in range(5)), .98)
        ramp = np.tan(np.deg2rad(20))*X
        self.assertGreater(min(mask(ramp+rng.normal(0, .025, X.shape)).mean() for _ in range(5)), .99)
        self.assertTrue(mask(np.floor(np.clip(X, 0, None)/.3)*.25)[(X > 0) & (X < 5)].all())  # within 0.28 m
        self.assertFalse(mask(np.floor(np.clip(X, 0, None)/.3)*.40)[(X > 0) & (X < 5)].all())


class Observed(unittest.TestCase):
    """End to end: env BEV, planner assess and support filter on a 20 deg descent and at a 0.5 m step."""
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(('ramp_down', 'step_up', 'ledge'), n=3)
        cls.planner = Planner(cls.env.cfg)
        cls.state, cls.obs, _ = reset(cls.env, 9.)          # 1 m into the descent / 1 m before the step
        cls.bev = np.asarray(decode(cls.obs['bev'], cls.env.cfg))

    def test_descent_is_supported_and_safe(self):
        b = self.bev[0]
        ahead = (b[..., 3] < .5) & (X > 2.2) & (np.abs(XY[..., 1]) < 1.)
        self.assertGreater(int(ahead.sum()), 100)
        self.assertTrue((b[..., 0][ahead] < -.55).any())
        self.assertTrue((b[..., 2][ahead] > .5).all())
        risk, _, _ = self.planner.assess(jnp.asarray(self.bev), jnp.full((3, 1, 2), jnp.array([3., 0.])))
        self.assertEqual(float(risk[0, 0]), 0.)
        straight = jnp.tile(jnp.array([[1., 0., 1.]]), (3, 1))
        self.assertFalse(bool(self.planner.support_risk(self.obs, straight)[0]))

    def test_step_is_unsupported(self):
        b = self.bev[1]
        edge = (b[..., 3] < .5) & (np.abs(X-1) < .1) & (np.abs(XY[..., 1]) < 1.)
        self.assertGreater(int(edge.sum()), 4)
        self.assertTrue((b[..., 2][edge] < .5).all())
        risk, _, _ = self.planner.assess(jnp.asarray(self.bev), jnp.full((3, 1, 2), jnp.array([3., 0.])))
        self.assertGreater(float(risk[1, 0]), 0.)

    def test_occluded_drop_reads_unknown(self):
        """Below a 0.5 m drop the terrain is in the lidar shadow: unknown, never 'supported'."""
        b = self.bev[2]
        shadow = (X > 1.05) & (X < 1.3) & (np.abs(XY[..., 1]) < .5)
        self.assertTrue((b[..., 3][shadow] > .5).all())


if __name__ == '__main__':
    unittest.main()
