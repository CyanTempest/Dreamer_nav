"""Confirmed M1 stair-mode capabilities, including scale and boundary counterexamples."""
import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.common import bev_xy, load_config
from nav4.sensor import support_mask

CFG = load_config('env')
XY = bev_xy(CFG)
X, Y = XY[..., 0], XY[..., 1]


def support(h, known=None, obstacle=None, cfg=CFG):
    h = jnp.asarray(h, jnp.float32)[None]
    k = jnp.ones_like(h, bool) if known is None else jnp.asarray(known)[None]
    ob = jnp.zeros_like(h) if obstacle is None else jnp.asarray(obstacle, jnp.float32)[None]
    return np.asarray(support_mask(h, k, ob, cfg))[0]


class Capability(unittest.TestCase):
    def test_028_riser_both_directions_supported(self):
        for sign in (-1, 1):
            self.assertTrue(support(np.where(X >= 2, sign*.28, 0.)).all())

    def test_040_ledge_rejected_even_though_metre_grade_is_low(self):
        for sign in (-1, 1):
            m = support(np.where(X >= 2, sign*.40, 0.))
            self.assertFalse(m[np.abs(X-2) <= CFG['bev']['res']].any())
            self.assertTrue(m[np.abs(X-2) > 2*CFG['bev']['res']].all())

    def test_two_cell_smeared_040_ledge_rejected(self):
        h = .4*np.clip((X-2)/(.25), 0, 1)
        m = support(h)
        self.assertFalse(m[(X > 2) & (X < 2.25)].any())

    def test_47_degree_metre_plane_supported_at_all_headings(self):
        for yaw in (0, np.pi/4, np.pi/2):
            for sign in (-1, 1):
                h = sign*np.tan(np.deg2rad(47))*(X*np.cos(yaw)+Y*np.sin(yaw))
                self.assertTrue(support(h)[5:-5, 5:-5].all(), (yaw, sign))

    def test_48_degree_metre_plane_rejected_by_grade(self):
        m = support(np.tan(np.deg2rad(48))*X)
        self.assertFalse(m[5:-5, 5:-5].any())

    def test_hard_post_remains_collision(self):
        ob = np.zeros_like(X); ob[30:34, 30:34] = .6
        m = support(np.zeros_like(X), obstacle=ob)
        self.assertFalse(m[30:34, 30:34].any())
        self.assertEqual(m.sum(), m.size-16)
        ob[30:34, 30:34] = CFG['terrain']['collision_height']
        self.assertFalse(support(np.zeros_like(X), obstacle=ob)[30:34, 30:34].any())

    def test_unknown_void_is_not_filled(self):
        k = np.ones_like(X, bool); k[28:36, :] = False
        m = support(np.zeros_like(X), known=k)
        self.assertFalse(m[~k].any())
        self.assertTrue(m[k].all())

    def test_no_wraparound_on_bev_edges(self):
        self.assertTrue(support(np.tan(np.deg2rad(30))*X).all())

    def test_saved_legacy_config_keeps_old_limits(self):
        old = copy.deepcopy(CFG)
        old['terrain'] = dict(max_step=.2, max_grade_deg=35., collision_height=.22)
        self.assertFalse(support(np.where(X >= 2, .28, 0.), cfg=old).all())
        self.assertFalse(support(np.tan(np.deg2rad(40))*X, cfg=old)[1:-1].any())

    def test_mapping_and_environment_share_capability_and_scale(self):
        from nav4.mapping.common import load_config as mapping_config
        c, _ = mapping_config()
        self.assertEqual(c['terrain']['max_step_m'], CFG['terrain']['max_step'])
        self.assertEqual(c['terrain']['max_grade_deg'], CFG['terrain']['max_grade_deg'])
        self.assertEqual(2*c['terrain']['grade_radius_m'], CFG['terrain']['grade_baseline_m'])
        self.assertEqual(c['obstacles']['body_band_m'][0], CFG['terrain']['collision_height'])
        self.assertTrue(c['terrain']['trail_terrain_exemption'])  # A-trail site prior, not sensor capability
        self.assertAlmostEqual(CFG['speed_limit']['steep_rise']/(2*CFG['speed_limit']['steep_probe']), .12/.84)


if __name__ == '__main__':
    unittest.main()
