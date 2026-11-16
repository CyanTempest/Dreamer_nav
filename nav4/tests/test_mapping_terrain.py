"""Synthetic-cloud tests for the segment-A terrain model (nav4/mapping).

Run: cd <project root> && PYTHONDONTWRITEBYTECODE=1 ../envs/dreamer/bin/python -m unittest discover -s nav4/tests -p "test_mapping_*.py"
"""
import sys
import unittest
from pathlib import Path

import numpy as np

from nav4.mapping.common import BANDS, RES, footprint_ok, load_config
from nav4.mapping.core import run_model
from nav4.mapping.grid import CellIndex, band_atlas
from nav4.mapping.model import lowest_ground_layer

CFG, _ = load_config()
DENSITY = 6000.          # points per m^2 on surfaces (field data: ~50 per 0.1 m cell)
RNG_SEED = 7


def surface_points(rng, x0, x1, y0, y1, zfun, density=DENSITY, normal=(0, 0, 1)):
    n = int((x1 - x0) * (y1 - y0) * density)
    x = rng.uniform(x0, x1, n)
    y = rng.uniform(y0, y1, n)
    z = zfun(x, y) + rng.normal(0, .005, n)
    nz = np.full(n, normal[2], np.float32)
    return x, y, z, nz


class Scene:
    """Accumulates synthetic points on an 8 m x 6 m grid; path along y=3 m."""

    def __init__(self, size=(8., 6.)):
        self.rng = np.random.default_rng(RNG_SEED)
        self.size = size
        self.parts = []

    def add(self, x, y, z, nz, cls=1):
        self.parts.append((x, y, z, nz, np.full(len(x), cls, np.uint8)))

    def ground(self, zfun, holes=(), x0=0., x1=None, y0=0., y1=None):
        x1 = self.size[0] if x1 is None else x1
        y1 = self.size[1] if y1 is None else y1
        x, y, z, nz = surface_points(self.rng, x0, x1, y0, y1, zfun)
        keep = np.ones(len(x), bool)
        for hx0, hx1, hy0, hy1 in holes:
            keep &= ~((x >= hx0) & (x < hx1) & (y >= hy0) & (y < hy1))
        self.add(x[keep], y[keep], z[keep], nz[keep])

    def run(self, ground_fun, path_y=3.0, h_dev=1.8):
        x, y, z, nz, cls = [np.concatenate(v) for v in zip(*self.parts)]
        origin = np.array([0., 0.])
        shape = (int(self.size[0] / RES) + 1, int(self.size[1] / RES) + 1)
        cells = CellIndex(x, y, z, origin, shape)
        z0, n0 = lowest_ground_layer(cells, CFG)
        px = np.arange(.2, self.size[0] - .2, .1)
        py = np.full(len(px), path_y)
        walked = np.c_[px, py, ground_fun(px, py) + h_dev]
        r = run_model(cells, cells.take(x), cells.take(y), np.abs(cells.take(nz)), cells.take(cls), origin, shape,
                      np.ones(shape, bool), z0, n0, walked, walked[:, :2], np.zeros(shape), h_dev, CFG)
        gx, gy = np.meshgrid(np.arange(shape[0]) * RES, np.arange(shape[1]) * RES, indexing='ij')
        r['truth'] = ground_fun(gx, gy)
        r['gx'], r['gy'] = gx, gy
        r['cells'] = cells
        return r


def flat(x, y):
    return np.zeros_like(np.asarray(x, float))


class TestFlatGroundWithGrass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sc = Scene()
        rng = sc.rng
        # 30 % of ground returns survive under the grass patch, full density elsewhere
        patch = (2.0, 4.0, 0.6, 2.0)
        sc.ground(flat, holes=[patch])
        x, y, z, nz = surface_points(rng, 2.0, 4.0, 0.6, 2.0, flat, density=DENSITY * .3)
        sc.add(x, y, z, nz)
        n = int(2.0 * 1.4 * .5 * 9000)    # volumetric grass 0-0.5 m, random normals
        sc.add(rng.uniform(2.0, 4.0, n), rng.uniform(.6, 2.0, n), rng.uniform(0., .5, n),
               rng.uniform(-1, 1, n).astype(np.float32))
        cls.r = sc.run(flat)
        cls.patch = (cls.r['gx'] > 2.25) & (cls.r['gx'] < 3.75) & (cls.r['gy'] > .85) & (cls.r['gy'] < 1.75)

    def test_ground_recovered_under_grass(self):
        err = np.abs(self.r['ground'] - self.r['truth'])[self.patch]
        self.assertLess(np.percentile(err, 95), .03)
        self.assertTrue(self.r['known'][self.patch].all())

    def test_grass_is_soft_not_hard(self):
        obst = self.r['obs']['obstacle'][self.patch]
        self.assertGreater(obst.mean(), .9)
        self.assertGreater(self.r['soft'][self.patch].mean(), .8)
        self.assertLess(self.r['hard'][self.patch].mean(), .2)
        # soft-only does not collide in the saved obstacle layer
        self.assertTrue((self.r['sdf']['obstacle_c'][self.patch & self.r['soft']] >= 0).all())

    def test_flat_supported(self):
        inner = (self.r['gx'] > .5) & (self.r['gx'] < 7.5) & (self.r['gy'] > .5) & (self.r['gy'] < 5.5)
        self.assertTrue(self.r['support_cap'][inner].all())


class TestRamp30(unittest.TestCase):
    @staticmethod
    def zf(x, y):
        x = np.asarray(x, float)
        return np.tan(np.deg2rad(30)) * np.clip(x - 2., 0, 4.)

    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(cls.zf)
        cls.r = sc.run(cls.zf)

    def test_ground_and_support(self):
        on = (self.r['gx'] > 2.6) & (self.r['gx'] < 5.4) & (self.r['gy'] > .6) & (self.r['gy'] < 5.4)
        err = np.abs(self.r['ground'] - self.r['truth'])[on]
        self.assertLess(np.percentile(err, 95), .03)
        self.assertTrue(self.r['support_cap'][on].all(), 'a 30 deg ramp is within the 35 deg grade limit')
        self.assertLess(np.abs(self.r['grade'][on] - 30).max(), 3.)
        self.assertFalse(self.r['obs']['obstacle'][on].any())


class TestStairs017(unittest.TestCase):
    RISER, TREAD, X0, N = .17, .30, 2.0, 8

    @classmethod
    def zf(cls, x, y):
        x = np.asarray(x, float)
        k = np.clip(np.floor((x - cls.X0) / cls.TREAD) + 1, 0, cls.N)
        return k * cls.RISER

    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(cls.zf)
        rng = sc.rng
        for k in range(1, cls.N + 1):      # vertical riser faces
            xr = cls.X0 + (k - 1) * cls.TREAD
            n = int(6. * cls.RISER * DENSITY)
            sc.add(np.full(n, xr) + rng.normal(0, .003, n), rng.uniform(0, 6., n),
                   (k - 1) * cls.RISER + rng.uniform(0, cls.RISER, n), np.zeros(n, np.float32))
        cls.r = sc.run(cls.zf)

    def test_treads_recovered(self):
        rel = (self.r['gx'] - self.X0) % self.TREAD
        on = (self.r['gx'] > self.X0) & (self.r['gx'] < self.X0 + self.N * self.TREAD) & (rel > .08) & (rel < .22) \
            & (self.r['gy'] > .5) & (self.r['gy'] < 5.5)
        err = np.abs(self.r['ground'] - self.r['truth'])[on]
        self.assertLess(np.percentile(err, 95), .03)

    def test_stairs_supported_by_capability(self):
        on = (self.r['gx'] > self.X0 - .3) & (self.r['gx'] < self.X0 + self.N * self.TREAD - .3) & \
            (self.r['gy'] > .6) & (self.r['gy'] < 5.4)
        self.assertTrue(self.r['support_cap'][on].all(), '0.17 m risers <= 0.20 m step, sustained ~30 deg <= 35 deg')
        self.assertLessEqual(self.r['step'][on].max(), .20)
        self.assertFalse(self.r['obs']['obstacle'][on].any(), 'riser faces are terrain, not obstacles')


class TestTreeTrunk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(flat)
        rng = sc.rng
        cx, cy, rad = 5.0, 1.2, .15
        n = int(2 * np.pi * rad * 3.0 * DENSITY)
        th = rng.uniform(0, 2 * np.pi, n)
        sc.add(cx + rad * np.cos(th), cy + rad * np.sin(th), rng.uniform(0, 3., n), np.zeros(n, np.float32))
        cls.r = sc.run(flat)
        cls.trunk = np.hypot(cls.r['gx'] - cx, cls.r['gy'] - cy) <= rad

    def test_trunk_hard_and_colliding(self):
        self.assertTrue(self.r['obs']['obstacle'][self.trunk].any())
        ring = np.hypot(self.r['gx'] - 5.0, self.r['gy'] - 1.2) <= .25
        ob = self.r['obs']['obstacle'] & ring
        self.assertTrue(self.r['hard'][ob].all())
        self.assertFalse(self.r['soft'][ring].any())
        self.assertTrue((self.r['sdf']['obstacle_c'][ob] < 0).all())

    def test_ground_known_around_trunk(self):
        self.assertTrue(self.r['known'][self.trunk].all())


class TestLedge03(unittest.TestCase):
    @staticmethod
    def zf(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where((x >= 4.) & (y < 2.), .3, 0.)

    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(cls.zf)
        rng = sc.rng
        n = int(2. * .3 * DENSITY)
        sc.add(np.full(n, 4.) + rng.normal(0, .003, n), rng.uniform(0, 2., n), rng.uniform(0, .3, n), np.zeros(n, np.float32))
        n = int(4. * .3 * DENSITY)
        sc.add(rng.uniform(4., 8., n), np.full(n, 2.) + rng.normal(0, .003, n), rng.uniform(0, .3, n), np.zeros(n, np.float32))
        cls.r = sc.run(cls.zf)       # path at y=3 m: the ledge is outside the trail zone

    def test_ledge_edge_unsupported_flat_supported(self):
        gx, gy = self.r['gx'], self.r['gy']
        band = (np.abs(gx - 4.) <= .1) & (gy > .5) & (gy < 1.5)
        # every row across the ledge has unsupported cells at the discontinuity
        cut = ~self.r['support'] & band
        for j in np.unique(np.argwhere(band)[:, 1]):
            self.assertTrue(cut[:, j].any(), '0.3 m ledge > 0.20 m step must be cut')
        self.assertGreaterEqual(self.r['step'][band].max(), .28)
        low = (gx > 1.5) & (gx < 3.7) & (gy > .5) & (gy < 1.5)
        top = (gx > 4.3) & (gx < 7.5) & (gy > .5) & (gy < 1.5)
        self.assertTrue(self.r['support'][low].all())
        self.assertTrue(self.r['support'][top].all())
        err = np.abs(self.r['ground'] - self.r['truth'])[top | low]
        self.assertLess(np.percentile(err, 95), .03)

    def test_robot_cannot_cross_ledge(self):
        sup, obs = self.r['sdf']['support'], self.r['sdf']['obstacle_c']
        xs = np.arange(2.6, 5.5, .05)
        ok = footprint_ok(sup, obs, np.array([0., 0.]), np.c_[xs, np.full(len(xs), 1.0)], np.zeros(len(xs)))
        self.assertTrue(ok[xs < 3.3].all() and ok[xs > 4.7].all())
        self.assertFalse(ok[(xs > 3.6) & (xs < 4.4)].any())


class TestUnobservedUnsupported(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(flat, holes=[(5.0, 6.0, 4.0, 5.0)])
        cls.r = sc.run(flat)

    def test_hole_unknown_and_unsupported(self):
        gx, gy = self.r['gx'], self.r['gy']
        core = (gx > 5.25) & (gx < 5.75) & (gy > 4.25) & (gy < 4.75)
        self.assertFalse(self.r['known'][core].any())
        self.assertFalse(self.r['support'][core].any())
        self.assertTrue((self.r['sdf']['support'][core] > .025).all())
        # the 0.2 m interpolation still closes the hole's rim
        rim = (gx > 5.05) & (gx < 5.15) & (gy > 4.3) & (gy < 4.7)
        self.assertTrue(self.r['known'][rim].all())


class TestSubSurfaceReflection(unittest.TestCase):
    """A dense mirror layer 1.2 m below a walked floor must not become the ground."""

    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(flat)
        rng = sc.rng
        x, y, z, nz = surface_points(rng, 1.0, 7.0, 2.0, 4.0, lambda a, b: np.full_like(a, -1.2), density=DENSITY * .5)
        sc.add(x, y, z, nz)
        cls.r = sc.run(flat)

    def test_floor_kept(self):
        gx, gy = self.r['gx'], self.r['gy']
        m = (gx > 1.5) & (gx < 6.5) & (gy > 2.5) & (gy < 3.5)
        self.assertLess(np.abs(self.r['ground'][m]).max(), .05)
        self.assertTrue(self.r['support'][m].all())


class TestTrailExemptionAndCorridor(unittest.TestCase):
    """A 0.25 m riser is within the confirmed capability, both on and off the path.  A post 0.2 m from the path is hard; it collides
    without the walked-corridor prior (b) and not with it (c)."""
    H = .25

    @classmethod
    def zf(cls, x, y):
        return np.where(np.asarray(x, float) >= 4., cls.H, 0.)

    @classmethod
    def setUpClass(cls):
        sc = Scene()
        sc.ground(cls.zf)
        rng = sc.rng
        n = int(6. * cls.H * DENSITY)
        sc.add(np.full(n, 4.) + rng.normal(0, .003, n), rng.uniform(0, 6., n), rng.uniform(0, cls.H, n), np.zeros(n, np.float32))
        n = int(2 * np.pi * .04 * 1.2 * DENSITY * 2)
        th = rng.uniform(0, 2 * np.pi, n)
        sc.add(2.0 + .04 * np.cos(th), 3.2 + .04 * np.sin(th), rng.uniform(0, 1.2, n), np.zeros(n, np.float32), cls=13)
        cls.r = sc.run(cls.zf)

    def test_025_riser_supported_without_location_exemption(self):
        gx, gy = self.r['gx'], self.r['gy']
        edge = (np.abs(gx - 4.) <= .1) & (gy > .5) & (gy < 5.5)
        self.assertTrue(self.r['support_cap'][edge].all(), '0.25 m <= confirmed 0.28 m')
        self.assertTrue(self.r['support'][edge].all())
        self.assertFalse(self.r['exempt'][~self.r['zone_conn']].any())

    def test_post_hard_and_walked_corridor_prior(self):
        gx, gy = self.r['gx'], self.r['gy']
        post = np.hypot(gx - 2.0, gy - 3.2) <= .05
        self.assertTrue(self.r['hard'][post].all())
        self.assertTrue((self.r['sdf']['obstacle_b'][post] < 0).all())
        self.assertTrue((self.r['sdf']['obstacle_c'][post] >= 0).all())
        self.assertTrue((self.r['flag'][post] == 2).all())


class TestContracts(unittest.TestCase):
    def test_atlas_matches_build_scan_patches(self):
        """band_atlas reproduces the r9 build_scan_patches atlas loop exactly."""
        rng = np.random.default_rng(3)
        shape = (12, 9)
        origin = np.array([10., -5.])
        n = 4000
        x = rng.uniform(origin[0] - .05, origin[0] + 1.15, n)
        y = rng.uniform(origin[1] - .05, origin[1] + .85, n)
        z = rng.uniform(-.2, 4., n).astype(np.float32)
        ground = rng.uniform(-.1, .1, shape).astype(np.float32)
        known = rng.uniform(size=shape) > .3
        cells = CellIndex(x, y, z, origin, shape)
        atlas, _ = band_atlas(cells, ground, known)
        # --- verbatim r9 logic
        ij = np.floor((np.stack([x, y], -1) - origin) / RES + .5).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
        ij, zz = ij[ok], z[ok]
        flat_ = np.ravel_multi_index(ij.T, shape)
        relative = zz - ground[ij[:, 0], ij[:, 1]]
        ref = np.zeros((*shape, 20), np.float32); ref[..., 0] = ground; ref[..., 19] = known
        for b, (lower, upper) in enumerate(BANDS):
            mask = (relative >= lower) & (relative < upper)
            bottom = np.full(np.prod(shape), np.inf, np.float32)
            top = np.full(np.prod(shape), -1, np.float32)
            count = np.zeros(np.prod(shape), np.int32)
            np.minimum.at(bottom, flat_[mask], relative[mask]); np.maximum.at(top, flat_[mask], relative[mask])
            np.add.at(count, flat_[mask], 1)
            ref[..., 1 + 3 * b] = np.where(count > 0, bottom, 0).reshape(shape)
            ref[..., 2 + 3 * b] = top.reshape(shape)
            ref[..., 3 + 3 * b] = (count > 0).reshape(shape)
        np.testing.assert_allclose(atlas, ref, atol=1e-6)

    def test_footprint_matches_qualify_routes(self):
        root = Path(__file__).resolve().parents[2] / 'r9_official_runner'
        if not (root / 'qualify_routes.py').exists():
            self.skipTest('r9 qualify_routes.py not available')
        sys.path.insert(0, str(root))
        try:
            import qualify_routes
        finally:
            sys.path.pop(0)
        rng = np.random.default_rng(5)
        layers = np.zeros((60, 50, 7), np.float32)
        layers[..., 2] = rng.choice([-.2, .0, .02, .03, .3], size=(60, 50))
        layers[..., 5] = rng.choice([-.1, 0., .05, 1.], size=(60, 50), p=[.02, .08, .2, .7])
        data = dict(layers=layers, origin=np.array([1., 2.], np.float32))
        xy = rng.uniform([1.5, 2.5], [6.5, 6.5], (300, 2))
        yaw = rng.uniform(-np.pi, np.pi, 300)
        ref = qualify_routes.footprint_ok(data, xy, yaw)
        ours = footprint_ok(layers[..., 2], layers[..., 5], data['origin'], xy, yaw, obstacle_linear=False)
        np.testing.assert_array_equal(ours, ref)


if __name__ == '__main__':
    unittest.main()
