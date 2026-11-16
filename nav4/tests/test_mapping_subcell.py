"""Subcell obstacle distances: physical gap, legacy fallback and sampling contract."""
import copy
import unittest

import numpy as np

from nav4.mapping.common import RES, bilinear, cell_index, footprint_ok, load_config, obstacle_distance, signed_distance


class SubcellDistance(unittest.TestCase):
    def setUp(self):
        self.cfg, _ = load_config()
        self.origin = np.array([-2., -2.])
        self.shape = (41, 41)

    def poles(self):
        # Two 8 cm diameter posts, inner faces exactly 0.62 m apart.
        # Body-band returns on their cylindrical surfaces; no corridor prior.
        th, z = np.meshgrid(np.linspace(0, 2*np.pi, 361), np.linspace(.23, .79, 10))
        cloud = np.concatenate([np.c_[.04*np.cos(th.ravel()),
                                      cy+.04*np.sin(th.ravel()), z.ravel()] for cy in (-.35, .35)])
        ij = cell_index(cloud[:, :2], self.origin)
        mask = np.zeros(self.shape, bool)
        mask[ij[:, 0], ij[:, 1]] = True
        return cloud, mask

    def test_062_gap_for_058_envelope(self):
        cloud, mask = self.poles()
        new = obstacle_distance(mask, cloud[:, :2], self.origin, self.cfg)
        old = signed_distance(mask)
        support = np.full(self.shape, -1.)
        xy = np.c_[np.linspace(-.6, .6, 25), np.zeros(25)]
        yaw = np.zeros(len(xy))
        self.assertTrue(footprint_ok(support, new, self.origin, xy, yaw).all())
        self.assertFalse(footprint_ok(support, old, self.origin, xy, yaw, obstacle_linear=False).all())
        self.assertFalse(footprint_ok(support, old, self.origin, xy, yaw).all())
        self.assertTrue((new[mask] < 0).all())

    def test_far_distances_agree_within_one_cell(self):
        cloud, mask = self.poles()
        new = obstacle_distance(mask, cloud[:, :2], self.origin, self.cfg)
        old = signed_distance(mask)
        far = old > .5
        self.assertLessEqual(np.max(np.abs(new[far]-old[far])), RES)
        cfg = copy.deepcopy(self.cfg)
        cfg['obstacles']['distance_method'] = 'grid'
        np.testing.assert_array_equal(obstacle_distance(mask, cloud[:, :2], self.origin, cfg), old)

    def test_empty_and_exact_return_are_well_defined(self):
        mask = np.zeros(self.shape, bool)
        self.assertTrue((obstacle_distance(mask, np.empty((0, 2)), self.origin, self.cfg) == 100).all())
        mask[20, 20] = True
        sdf = obstacle_distance(mask, np.array([[0., 0.]]), self.origin, self.cfg)
        self.assertLess(sdf[20, 20], 0)
        self.assertAlmostEqual(float(sdf[30, 20]), 1.-self.cfg['obstacles']['distance_noise_m'], places=6)

    def test_obstacle_bilinear_support_still_nearest(self):
        origin = np.full(2, -1.)
        obs = np.broadcast_to(-1.+np.arange(30)*.1-.32, (30, 30)).copy()
        self.assertAlmostEqual(float(bilinear(obs, origin, np.array([1., .34]))), .02)
        # Envelope upper edge .29 at centre y=.045 => y=.335, inside an obstacle y<.32.
        # Use the opposite sign so its upper edge determines collision.
        sup = np.full_like(obs, -1.)
        self.assertTrue(footprint_ok(sup, -obs, origin, np.array([[1., .025]]), np.zeros(1))[0])
        self.assertFalse(footprint_ok(sup, -obs, origin, np.array([[1., .045]]), np.zeros(1))[0])


if __name__ == '__main__':
    unittest.main()
