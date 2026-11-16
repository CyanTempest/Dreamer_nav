"""Synthetic-cloud tests for the segment-A map fixes A-D (nav4/mapping).

A  sparse returns vs a solid trunk at equal range from the walked line
B  stair flight on a bend with one 0.28 m riser (robot-scale cross grade, measured-riser connectivity)
C  1 m-deep ditch with a deck (and one without)
D  route with a feasible path offset from the centreline (and a fully blocked place)

Run: cd <project root> && PYTHONDONTWRITEBYTECODE=1 ../envs/dreamer/bin/python -m unittest discover -s nav4/tests -p "test_mapping_*.py"
"""
import copy
import unittest

import numpy as np

from nav4.mapping.common import RES, footprint_ok, load_config, signed_distance
from nav4.mapping.core import run_model
from nav4.mapping.grid import CellIndex
from nav4.mapping.model import lowest_ground_layer
from nav4.mapping.route import build_route, route_distance, station_monotonicity, turning_radius
from nav4.tests.test_mapping_terrain import DENSITY, Scene, flat, surface_points

CFG, _ = load_config()


def run_scene(sc, zfun, path_xy, h_dev=1.8, cfg=CFG, path_yaw=None, lateral_yaw=None, walked_z=None):
    """Like Scene.run but with an arbitrary walked path (N,2) and per-cell route heading."""
    x, y, z, nz, cls = [np.concatenate(v) for v in zip(*sc.parts)]
    origin = np.array([0., 0.])
    shape = (int(sc.size[0] / RES) + 1, int(sc.size[1] / RES) + 1)
    cells = CellIndex(x, y, z, origin, shape)
    z0, n0 = lowest_ground_layer(cells, cfg)
    wz = zfun(path_xy[:, 0], path_xy[:, 1]) + h_dev if walked_z is None else walked_z
    walked = np.c_[path_xy, wz]
    if lateral_yaw is None:
        from scipy.spatial import cKDTree
        gx, gy = np.meshgrid(np.arange(shape[0]) * RES, np.arange(shape[1]) * RES, indexing='ij')
        _, k = cKDTree(path_xy).query(np.c_[gx.ravel(), gy.ravel()])
        lateral_yaw = path_yaw[k].reshape(shape)
    r = run_model(cells, cells.take(x), cells.take(y), np.abs(cells.take(nz)), cells.take(cls), origin, shape,
                  np.ones(shape, bool), z0, n0, walked, walked[:, :2], lateral_yaw, h_dev, cfg)
    gx, gy = np.meshgrid(np.arange(shape[0]) * RES, np.arange(shape[1]) * RES, indexing='ij')
    r['truth'] = zfun(gx, gy)
    r['gx'], r['gy'] = gx, gy
    return r


class TestSparseVsSolidAtEqualRange(unittest.TestCase):
    """Fix A: isolated low-density return clusters (leaf/grass tufts: 4 returns in one 0.1 m sub-bin, one
    per 0.3 m voxel, so no shape evidence) and a solid trunk, both 1.0 m from the walked line."""

    @classmethod
    def setUpClass(cls):
        cfg = copy.deepcopy(CFG)
        cfg['obstacles']['solid_density_min_reference_cells'] = 3     # the scene has ~10 trunk cells
        sc = Scene()
        sc.ground(flat)
        rng = sc.rng
        cx, cy, rad = 5.0, 2.0, .15
        n = int(2 * np.pi * rad * 3.0 * DENSITY)
        th = rng.uniform(0, 2 * np.pi, n)
        sc.add(cx + rad * np.cos(th), cy + rad * np.sin(th), rng.uniform(0, 3., n), np.zeros(n, np.float32))
        cls.tufts = []
        for tx, ty in [(1.5, 4.0), (2.1, 4.0), (2.7, 4.0), (3.3, 4.0), (1.5, 2.0), (2.1, 2.0), (2.7, 2.0)]:
            sc.add(tx + rng.uniform(-.03, .03, 4), ty + rng.uniform(-.03, .03, 4), rng.uniform(.40, .48, 4),
                   rng.uniform(-1, 1, 4).astype(np.float32))
            cls.tufts.append((tx, ty))
        cls.r = run_scene(sc, flat, np.c_[np.arange(.2, 7.8, .1), np.full(76, 3.)], cfg=cfg,
                          path_yaw=np.zeros(76))

    def cell(self, x, y):
        return int(round(x / RES)), int(round(y / RES))

    def test_tufts_sparse_visible_not_colliding(self):
        r = self.r
        for tx, ty in self.tufts:
            i, j = self.cell(tx, ty)
            self.assertTrue(r['obs']['obstacle'][i, j], 'tuft is an obstacle cell (visible)')
            self.assertFalse(r['obs']['hard_evidence'][i, j])
            self.assertTrue(r['obs']['hard_default'][i, j], 'v5 rule made it hard')
            self.assertTrue(r['sparse'][i, j] and not r['hard'][i, j])
            self.assertEqual(r['flag'][i, j], 2)
            self.assertGreaterEqual(r['sdf']['obstacle_b'][i, j], 0.)

    def test_trunk_hard_with_reference_density(self):
        r = self.r
        ring = np.hypot(r['gx'] - 5.0, r['gy'] - 2.0) <= .25
        ob = r['obs']['obstacle'] & ring
        self.assertTrue(ob.any())
        self.assertTrue(r['hard'][ob].all())
        self.assertFalse(r['sparse'][ring].any())
        self.assertTrue((r['sdf']['obstacle_c'][ob] < 0).all())
        tab = r['obs']['solid_density_table']
        tau = r['obs']['solid_density_threshold']
        i, j = self.cell(*self.tufts[0])
        self.assertGreater(tau[i, j], r['feat']['peak_subbin'][i, j])
        self.assertIsNotNone(tab['threshold'])

    def test_no_reference_disables_sparse(self):
        """Without enough reference cells the sparse class is off (v5 rule kept)."""
        cfg = copy.deepcopy(CFG)
        cfg['obstacles']['solid_density_min_reference_cells'] = 10 ** 6
        sc = Scene()
        sc.ground(flat)
        sc.add(np.full(4, 2.1), np.full(4, 4.0), np.linspace(.4, .45, 4), np.zeros(4, np.float32))
        r = run_scene(sc, flat, np.c_[np.arange(.2, 7.8, .1), np.full(76, 3.)], cfg=cfg, path_yaw=np.zeros(76))
        self.assertFalse(r['sparse'].any())
        self.assertTrue(r['hard'][21, 40])


class TestStairsOnBendWith028Riser(unittest.TestCase):
    """Fix B: stairs ascending around a 90 deg bend (risers radial from the bend centre), riser 0.17 m with
    one 0.28 m riser, treads 0.35 m at the walked radius; route heading given OFF deg off the true tangent
    (a smoothed centreline cutting the bend).  A 0.45 m ledge beside the entry path, inside the trail zone,
    stays a barrier (deeper than the measured maximum riser)."""
    OFF = 60.
    C = np.array([3.0, 4.0])
    R = 2.0
    RISERS = [.17, .17, .28, .17, .17, .17]

    @classmethod
    def zf(cls, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ang = np.arctan2(y - cls.C[1], x - cls.C[0])           # -90 deg at the bend entry, 0 deg at exit
        u = np.clip(ang + np.pi / 2, 0, np.pi / 2) * cls.R      # arc length along the walked radius
        u = np.where(x < cls.C[0], 0., u)
        u = np.where((y > cls.C[1]) & (x >= cls.C[0]), np.pi / 2 * cls.R, u)
        tread = .35
        k = np.clip(np.floor((u - .4) / tread) + 1, 0, len(cls.RISERS))
        cum = np.r_[0, np.cumsum(cls.RISERS)]
        z = cum[k.astype(int)]
        low = (x > .3) & (x < 2.6) & (y > 2.45) & (y < 3.6)    # 0.45 m ledge 0.45 m beside the entry path
        return np.where(low, z - .45, z)

    @classmethod
    def setUpClass(cls):
        sc = Scene(size=(8., 8.))
        sc.ground(cls.zf)
        rng = sc.rng
        # vertical riser faces along the radial lines (dense, like real risers)
        cum = np.r_[0, np.cumsum(cls.RISERS)]
        for k, h in enumerate(cls.RISERS):
            ang = (.4 + k * .35) / cls.R - np.pi / 2
            n = int(4.0 * h * DENSITY)
            rr = rng.uniform(.5, 4.0, n)
            sc.add(cls.C[0] + rr * np.cos(ang), cls.C[1] + rr * np.sin(ang), cum[k] + rng.uniform(0, h, n), np.zeros(n, np.float32))
        a = np.linspace(-np.pi / 2, 0, 40)
        path = np.r_[np.c_[np.arange(.3, 3.0, .1), np.full(27, 2.0)],
                     np.c_[cls.C[0] + cls.R * np.cos(a), cls.C[1] + cls.R * np.sin(a)],
                     np.c_[np.full(30, 5.0), np.arange(4.1, 7.1, .1)]]
        tang = np.unwrap(np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])))
        cls.path = path
        cls.r = run_scene(sc, cls.zf, path, path_yaw=tang + np.deg2rad(cls.OFF))
        on_arc = (np.abs(np.hypot(cls.r['gx'] - cls.C[0], cls.r['gy'] - cls.C[1]) - cls.R) <= .29) & \
            (cls.r['gx'] > cls.C[0]) & (cls.r['gy'] < cls.C[1])
        cls.band = on_arc

    def test_stair_band_supported_with_fix_b(self):
        r = self.r
        self.assertGreaterEqual(r['step'][self.band].max(), .26, 'the 0.28 m riser is in the band')
        self.assertTrue(r['support'][self.band].mean() > .97, f"{r['support'][self.band].mean():.3f}")
        self.assertLess(np.nanmax(r['cross'][self.band]), 35.)

    def test_trail_prior_is_enabled_and_capability_is_retained(self):
        r = self.r
        self.assertTrue(CFG['terrain']['trail_terrain_exemption'])
        self.assertTrue(r['support_cap'][self.band].all())
        self.assertTrue(r['support'][r['support_cap']].all())
        self.assertTrue(r['exempt'][self.band].any())
        self.assertFalse(r['exempt'][~r['zone_conn']].any())

    def test_footprint_follows_the_bend(self):
        r = self.r
        p = self.path[27:67]
        yaw = np.unwrap(np.arctan2(np.gradient(p[:, 1]), np.gradient(p[:, 0])))
        ok = footprint_ok(r['sdf']['support'], r['sdf']['obstacle_c'], np.array([0., 0.]), p, yaw)
        self.assertTrue(ok.all(), f'{ok.mean():.2f}')

    def test_ledge_inside_zone_is_a_barrier(self):
        """The lower side of a 0.45 m ledge (> 0.28 m riser) 0.45-0.75 m from the walked line is not reachable
        through traversable steps (not trail terrain), and a footprint straddling the ledge is infeasible."""
        r = self.r
        low = (r['gx'] > .8) & (r['gx'] < 2.2) & (r['gy'] >= 2.5) & (r['gy'] <= 2.7)
        self.assertTrue((r['dist_walk'][low] <= .75).all(), 'lower side inside the trail zone')
        self.assertFalse(r['zone_conn'][low].any())
        org = np.array([0., 0.])
        ok = footprint_ok(r['sdf']['support'], r['sdf']['obstacle_c'], org, np.array([[1.5, 2.45], [1.5, 2.0]]), np.zeros(2))
        self.assertFalse(ok[0])
        self.assertTrue(ok[1], 'the path itself (0.45 m from the ledge) stays feasible')


class TestDeckOverDitch(unittest.TestCase):
    """Fix C: a 1 m-deep ditch (45 deg banks, bottom connected to the trail surface through the banks beside
    the deck, so the v5 two-layer rule keeps the bottom) crossed by a 1.2 m-wide deck at trail level that
    rests on the bank tops; under the deck only returns deeper than 0.5 m are seen (30 %, through gaps /
    from the side).  The walker crossed at trail level (prior z = 0).  A second, 0.7 m-deep ditch without
    a deck stays low and is reported."""
    DECK = (1.55, 4.45, 2.4, 3.6)

    @staticmethod
    def zf(x, y):
        x = np.asarray(x, float)
        d1 = -np.clip(1.5 - np.abs(x - 3.0), 0, 1.0)           # bottom |x-3| <= 0.5 at -1.0, banks to 1.5
        d2 = -np.clip(1.35 - np.abs(x - 6.6), 0, .7)           # bottom at -0.7 (|x-6.6| <= 0.65), no deck
        return d1 + d2

    @classmethod
    def setUpClass(cls):
        sc = Scene(size=(8.5, 6.))
        rng = sc.rng
        x, y, z, nz = surface_points(rng, 0., 8.5, 0., 6., cls.zf)
        x0, x1, y0, y1 = cls.DECK
        under = (x > x0) & (x < x1) & (y > y0) & (y < y1)
        keep = ~under | (z < -.5) & (rng.uniform(size=len(x)) < .3)
        sc.add(x[keep], y[keep], z[keep], nz[keep])
        x, y, z, nz = surface_points(rng, x0, x1, y0, y1, lambda a, b: np.full_like(a, .02), density=DENSITY * .6)
        sc.add(x, y, z, nz)
        px = np.arange(.2, 8.3, .1)
        path = np.c_[px, np.full(len(px), 3.)]
        cls.r = run_scene(sc, cls.zf, path, path_yaw=np.zeros(len(px)), walked_z=np.full(len(px), 1.8))

    def test_deck_is_ground(self):
        r = self.r
        x0, x1, y0, y1 = self.DECK
        inner = (r['gx'] > x0 + .15) & (r['gx'] < x1 - .15) & (r['gy'] > y0 + .15) & (r['gy'] < y1 - .15)
        self.assertGreater(r['deck_info']['seeds'].sum(), 0)
        self.assertTrue(r['deck'][inner & (r['ground_v5'] < -.3)].all())
        self.assertGreater(r['deck'][inner].sum(), 50)
        self.assertLess(np.abs(r['ground'][inner] - .02).max(), .04)

    def test_v5_ground_was_the_ditch(self):
        r = self.r
        on = (np.abs(r['gx'] - 3.0) < .25) & (np.abs(r['gy'] - 3.0) < .2)
        self.assertLess(r['ground_v5'][on].max(), -.5)

    def test_deck_supported_to_its_edges_and_cut_there(self):
        r = self.r
        x0, x1, y0, y1 = self.DECK
        inner = (r['gx'] > x0 + .15) & (r['gx'] < x1 - .15) & (r['gy'] >= y0 + .15) & (r['gy'] <= y1 - .15)
        self.assertTrue(r['support'][inner].all(), f"{r['support'][inner].mean():.3f}")
        org = np.array([0., 0.])
        xs = np.arange(1.0, 5.0, .05)
        ok = footprint_ok(r['sdf']['support'], r['sdf']['obstacle_c'], org, np.c_[xs, np.full(len(xs), 3.)], np.zeros(len(xs)))
        self.assertTrue(ok.all(), 'robot crosses the ditch on the deck')
        # straddling a deck edge over the 1 m void is infeasible
        edge = footprint_ok(r['sdf']['support'], r['sdf']['obstacle_c'], org, np.array([[3.0, y0], [3.0, y1]]), np.zeros(2))
        self.assertFalse(edge.any())

    def test_ditch_without_deck_kept_and_reported(self):
        r = self.r
        on = (np.abs(r['gx'] - 6.6) < .15) & (np.abs(r['gy'] - 3.0) < .1)
        self.assertTrue(r['deck_info']['void_without_deck'][on].any())
        self.assertLess(r['ground'][on].max(), -.6)
        self.assertFalse(r['deck'][on].any())


class TestRouteOffsetFromCentreline(unittest.TestCase):
    """Fix D: a hard block covers the centreline (x 10-12 m, y -0.6..0.25); the only feasible lane is
    offset to +0.5 m.  The saved route must shift there smoothly (R >= 1 m), pass the own-heading swept
    check, come back to the centreline, and keep the station layer monotone.  A full-width wall at
    x 20 m gives a blocked (least-violation) passage that is reported."""

    @classmethod
    def setUpClass(cls):
        shape = (321, 81)                         # x -1..31 m, y -4..4 m
        cls.origin = np.array([-1., -4.])
        gx, gy = np.meshgrid(cls.origin[0] + np.arange(shape[0]) * RES, cls.origin[1] + np.arange(shape[1]) * RES, indexing='ij')
        support = np.ones(shape, bool)
        support[[0, 1, -2, -1], :] = False
        support[:, [0, 1, -2, -1]] = False
        hard = (gx >= 10) & (gx <= 12) & (gy >= -.6) & (gy <= .25)
        hard |= (gy >= 1.25) & (gx > 8) & (gx < 14)          # bank on the left beyond the lane
        cls.hard0 = hard.copy()
        cls.sup = signed_distance(support)
        cls.obs = signed_distance(hard)
        s = np.arange(0, 30.0001, .2)
        cls.cxy = np.c_[s, np.zeros_like(s)]
        cls.cyaw = np.zeros_like(s)
        cls.cs = s
        cls.rt = build_route(cls.sup, cls.obs, cls.origin, cls.cxy, cls.cyaw, cls.cs, CFG)
        wall = hard | ((gx >= 20) & (gx <= 20.2))
        cls.rt_blocked = build_route(cls.sup, signed_distance(wall), cls.origin, cls.cxy, cls.cyaw, cls.cs, CFG)
        cls.shape = shape

    def test_route_shifts_to_the_lane_and_back(self):
        rt = self.rt
        self.assertTrue(rt['dp_connected'])
        at = lambda x: rt['shift'][np.argmin(np.abs(rt['xy'][:, 0] - x))]
        self.assertGreater(at(11.0), .35)
        self.assertLess(abs(at(3.0)), .1)
        self.assertLess(abs(at(27.0)), .1)

    def test_route_swept_feasible_and_smooth(self):
        from nav4.mapping.analysis import validate_path
        rt = self.rt
        v = validate_path(self.sup, self.obs, self.origin, rt['xy'], rt['yaw'], np.zeros(len(rt['s'])))
        self.assertTrue(v['swept_ok'], v['fine_fail_s'][:10])
        self.assertGreaterEqual(turning_radius(rt['xy']).min(), CFG['route']['min_turn_radius_m'])
        sp = np.diff(rt['s'])
        self.assertTrue(np.all(sp > 0) and abs(sp.mean() - CFG['route']['station_spacing_m']) < .01)

    def test_station_layer_monotone(self):
        rt = self.rt
        dist, near = route_distance(self.origin, self.shape, rt['xy'])
        mono = station_monotonicity(rt['s'][near], self.origin, rt['xy'], rt['s'])
        self.assertTrue(mono['station_samples_monotone'])
        self.assertEqual(mono['jumps_over_1m'], 0)
        self.assertLessEqual(mono['max_backward_step_m'], .2 + 1e-6)

    def test_blocked_place_reported(self):
        rt = self.rt_blocked
        self.assertFalse(rt['dp_connected'])
        self.assertTrue(rt['blocked'][np.abs(rt['xy'][:, 0] - 20.1) < .5].any())
        self.assertFalse(rt['blocked'][np.abs(rt['xy'][:, 0] - 5.0) < 1.].any())
        self.assertGreater(rt['shift'][np.argmin(np.abs(rt['xy'][:, 0] - 11.0))], .35, 'lane still used')


if __name__ == '__main__':
    unittest.main()
