"""CPU geometry, atomic memory, delayed registration and privilege tests for BEV v2."""
import copy
import json
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nav4 import sensor
from nav4.common import bev_xy, decode, load_config, rotate
from nav4.env import TrailEnv
from nav4.tests.helpers import QUIET, load
from nav4.tests import synthetic

CFG = load_config('env', {'bev.version': 'v2'})
XY = bev_xy(CFG)


def terrain(ground, top=None, known=None):
    g = jnp.asarray(ground, jnp.float32)[None]
    t = g if top is None else jnp.asarray(top, jnp.float32)[None]
    k = jnp.ones_like(g, bool) if known is None else jnp.asarray(known)[None]
    return np.asarray(sensor.terrain_v2(g, t, k, CFG))[0]


def scene(rise=0., slope=0., pole=True):
    m = synthetic.build('flat', length=15.)
    x, y = np.meshgrid(m['origin'][0]+np.arange(m['layers'].shape[0])*synthetic.RES,
                       m['origin'][1]+np.arange(m['layers'].shape[1])*synthetic.RES, indexing='ij')
    g = slope*(x-7.)+rise*(x >= 8.)
    h = np.where((np.abs(x-7.6) < .16) & (np.abs(y-1.) < .16), .5, 0.) if pole else np.zeros_like(x)
    m['layers'][..., 0] = m['atlas'][..., 0] = g
    m['layers'][..., 1] = g+h
    for i, (lo, hi) in enumerate(synthetic.BANDS):
        hit = h > lo
        m['atlas'][..., 1+3*i] = np.where(hit, lo, 0.)
        m['atlas'][..., 2+3*i] = np.where(hit, np.minimum(h, hi), -1.)
        m['atlas'][..., 3+3*i] = hit
    return m


class Geometry(unittest.TestCase):
    def test_ramps_steps_and_adjacent_pole(self):
        for deg in (-35, 20, 46):
            g = np.tan(np.deg2rad(deg))*XY[..., 0]
            b = terrain(g)
            self.assertTrue((b[..., 1] == 0).all())
            self.assertTrue((b[..., 2] > .5).all())
        for rise, supported in ((.25, True), (.40, False)):
            g = rise*(XY[..., 0] > 1.)
            t = g.copy()
            # 0.5 m pillar immediately beside the higher tread. Own-ground cutoff
            # and fixed terrain-face exclusion must BOTH retain this obstacle.
            px = .9375 if rise == .25 else 1.0625
            pillar = (np.abs(XY[..., 0]-px) < .01) & (np.abs(XY[..., 1]-.5625) < .01)
            t[pillar] += .5
            b = terrain(g, t)
            self.assertTrue((b[..., 1][pillar] >= .3).all())
            self.assertTrue((b[..., 2][pillar] == 0).all())
            edge = (np.abs(XY[..., 0]-1.) < .1) & (XY[..., 1] < -.5)
            self.assertEqual(bool((b[..., 2][edge] > .5).all()), supported)
            self.assertTrue((b[..., 1][~pillar] == 0).all())

    def test_unseen_high_ground_cannot_erase_a_pole(self):
        g = np.zeros((64, 64), np.float32); t = g.copy(); k = np.ones_like(g, bool)
        t[30, 30] = .5
        g[31, 30] = t[31, 30] = 20.; k[31, 30] = False
        b = terrain(g, t, k)
        self.assertGreater(b[30, 30, 1], .49)
        self.assertEqual(b[31, 30, 2], 0.)

    def test_partial_pole_return_uses_the_same_median_ground_as_support(self):
        # Only the upper 0.4..0.5 m interval of one pole is visible; surrounding
        # ground returns are zero. Its low return is not itself a ground surface.
        low = np.zeros((64, 64), np.float32)
        top = low.copy()
        low[30, 30], top[30, 30] = .4, .5
        b = terrain(low, top)
        self.assertEqual(b[30, 30, 0], 0.)
        self.assertEqual(b[30, 30, 1], .5)
        self.assertEqual(b[30, 30, 2], 0.)

    def test_end_to_end_visible_geometry(self):
        maps = {'s25': scene(.25), 's40': scene(.40), 'ramp': scene(slope=np.tan(np.deg2rad(20)))}
        e = TrailEnv(load(maps), 3, {**QUIET, 'bev.version': 'v2'})
        _, o, _ = e.reset(jax.random.PRNGKey(1), start_s=6.)
        b = np.asarray(decode(o['bev'], e.cfg))
        for i in range(3):
            pole = (np.abs(XY[..., 0]-1.6) < .25) & (np.abs(XY[..., 1]-1.) < .25)
            self.assertTrue(((b[i, ..., 1] > .3) & (b[i, ..., 3] < .5))[pole].any())
        edge = (np.abs(XY[..., 0]-2.) < .15) & (np.abs(XY[..., 1]) < .7)
        self.assertTrue((b[0, ..., 2][edge & (b[0, ..., 3] < .5)] > .5).all())
        self.assertTrue((b[1, ..., 2][edge & (b[1, ..., 3] < .5)] < .5).any())


class Memory(unittest.TestCase):
    def test_atomic_source_and_latest_acquisition(self):
        memory = jnp.zeros((1, 4, 4, 3)).at[..., 2].set(-jnp.inf)
        ij = jnp.array([[[[1, 1], [1, 1]]]])
        # Old separate maxima would yield ground=.25, relative obstacle=.5,
        # i.e. the nonexistent top=.75. Atomic low/top selection forbids that.
        samples = jnp.array([[[[0., .5], [.25, .30]]]])
        keep = jnp.ones((1, 1, 2), bool)
        m = sensor.update_memory_v2(memory, ij, samples, keep, jnp.array([1.]))
        np.testing.assert_array_equal(m[0, 1, 1], [0., .5, 1.])
        m2 = sensor.update_memory_v2(m, ij, samples[..., ::-1, :], keep, jnp.array([2.]))
        np.testing.assert_allclose(m2[0, 1, 1], [.25, .30, 2.])
        stale = sensor.update_memory_v2(m2, ij, samples, keep, jnp.array([1.]))
        np.testing.assert_array_equal(stale, m2)
        invalid = sensor.update_memory_v2(m2, ij+20, samples, keep, jnp.array([3.]))
        np.testing.assert_array_equal(invalid, m2)

    def test_translation_turn_and_drift_registration(self):
        e = TrailEnv(load({'pole': scene()}), 1, {**QUIET, 'bev.version': 'v2', 'memory.observation_delay': 2})
        s, _, _ = e.reset(jax.random.PRNGKey(2), start_s=6.)
        # No new returns: must reproject the stationary obstacle held in memory.
        for delta, yaw, drift in (([.5, 0.], .4, [0., 0., 0.]), ([.75, .1], -.2, [.04, -.03, .01])):
            moved = {**s, 'first': jnp.array([False]), 'pos': jnp.array([[6.+delta[0], delta[1]]]),
                     'yaw': jnp.array([yaw]), 'drift': jnp.array([drift]),
                     'noise': jnp.array([[0., 1., 0.]])}
            moved, o, _ = e.observe(moved)
            b = np.asarray(decode(o['bev'], e.cfg))[0]
            mask = (b[..., 1] > .3) & (b[..., 3] < .5)
            self.assertTrue(mask.any())
            estimated_pos = moved['pos']+moved['drift'][:, :2]
            estimated_yaw = moved['yaw']+moved['drift'][:, 2]
            expected = np.asarray(rotate(jnp.array([[7.6, 1.]])-estimated_pos, -estimated_yaw))[0]
            error = np.linalg.norm(XY[mask].mean(0)-expected)
            self.assertLess(error, np.sqrt(2)*e.cfg['bev']['res'])
            # Physical-centre error includes known differential registration drift.
            true_local = np.asarray(rotate(jnp.array([[7.6, 1.]])-moved['pos'], -moved['yaw']))[0]
            bound = np.sqrt(2)*e.cfg['bev']['res']+np.linalg.norm(drift[:2])+np.linalg.norm(true_local)*abs(drift[2])
            self.assertLess(np.linalg.norm(XY[mask].mean(0)-true_local), bound)

    def test_actual_measurement_delay_age_and_expiration(self):
        e = TrailEnv(load({'flat': scene(pole=False)}), 1,
                     {**QUIET, 'bev.version': 'v2', 'memory.observation_delay': 2, 'obstacles.count': 1})
        s = e._reset(jax.random.PRNGKey(4), start_s=6., observe=False)
        s['obstacles'] = jnp.zeros_like(s['obstacles'])
        s, _, _ = e.observe(s)
        s['obstacles'] = jnp.array([[[7.6, 1., .16, .16, 0., .5]]])
        seen = []
        for _ in range(3):
            s['first'] = jnp.array([False])
            s, o, _ = e.observe(s)
            b = np.asarray(decode(o['bev'], e.cfg))[0]
            seen.append(bool((b[..., 1] > .3).any()))
        self.assertEqual(seen, [False, False, True])
        hazard = (b[..., 1] > .3) & (b[..., 3] < .5)
        self.assertTrue((b[..., 5][hazard] >= .4/8.-1/255).all())
        # Expire by acquisition time, not delivery time; no hidden indefinite support.
        s = {**s, 'bev_time': s['bev_time']+9., 'keep_queue': jnp.zeros_like(s['keep_queue']),
             'noise': jnp.array([[0., 1., 0.]])}
        s, o, _ = e.observe(s)
        b = np.asarray(decode(o['bev'], e.cfg))[0]
        self.assertTrue((b[..., 3] > .5).all())
        self.assertTrue((b[..., 2] == 0).all())


class Contract(unittest.TestCase):
    def test_no_privileged_map_labels_in_v2_geometry(self):
        a = scene(.25)
        b = copy.deepcopy(a)
        for i in (2, 3, 4, 5, 6):
            b['layers'][..., i] = 777.+i
        cfg = {**QUIET, 'bev.version': 'v2', 'route_inputs': False, 'priv_route': True}
        envs = [TrailEnv(load({'same': m}), 1, cfg) for m in (a, b)]
        out = [e.reset(jax.random.PRNGKey(10), start_s=6.) for e in envs]
        for key in ('bev', 'state'):
            np.testing.assert_array_equal(out[0][1][key], out[1][1][key])
        self.assertFalse(np.array_equal(out[0][1]['priv_trav'], out[1][1]['priv_trav']))
        for _ in range(3):
            out = [e.observe({**v[0], 'first': jnp.array([False])}) for e, v in zip(envs, out)]
            np.testing.assert_array_equal(out[0][1]['bev'], out[1][1]['bev'])
        self.assertEqual(out[0][1]['bev'].shape, (1, 64, 64, 6))
        self.assertEqual(out[0][1]['state'].shape, (1, 13))

    def test_v1_bitwise_archived_regression(self):
        from audit_20261116.nav4_bev2.v1_snapshot import snapshot
        root = Path(__file__).resolve().parents[2]
        golden = json.loads((root/'audit_20261116/nav4_bev2/v1_golden.json').read_text())
        current = snapshot()
        self.assertEqual(current['records'], golden['records'])
        self.assertEqual(current['sha256'], golden['sha256'])

    def test_invalid_version(self):
        with self.assertRaisesRegex(ValueError, 'bev.version'):
            TrailEnv(load({'flat': scene(pole=False)}), 1, {'bev.version': 'v3'})


if __name__ == '__main__':
    unittest.main()
