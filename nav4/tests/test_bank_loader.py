"""TrailBank: float16/float32 maps, real arc length, projection, outside = unsupported, reversed route."""
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.bank import TrailBank
from nav4.planner import Planner
from nav4.tests import synthetic
from nav4.tests.helpers import QUIET, reset
from nav4.env import TrailEnv


class Loader(unittest.TestCase):
    def test_float16_and_float32_maps(self):
        maps = dict(half=synthetic.build('ramp_up', dtype=np.float16),
                    full=synthetic.build('stairs', length=24., dtype=np.float32))
        manifest = synthetic.write(tempfile.mkdtemp(), maps)
        for source in (manifest, manifest.parent):
            bank = TrailBank(source)
            self.assertEqual(bank.names, ('half', 'full'))
            self.assertEqual(bank.layers.dtype, jnp.float32)
            self.assertEqual(bank.ground.dtype, jnp.float32)
            for i, m in enumerate(maps.values()):
                x, y = m['layers'].shape[:2]
                np.testing.assert_array_equal(np.asarray(bank.layers[i, :x, :y]), m['layers'].astype(np.float32))
                np.testing.assert_array_equal(np.asarray(bank.ground[i, :x, :y]), m['atlas'][..., 0].astype(np.float32))
            # padding of the shorter map reads as outside: unsupported, no ground knowledge
            self.assertTrue(np.all(np.asarray(bank.layers[1, -1, :, 2]) > .025))
        self.assertEqual(TrailBank(manifest, names=['full']).names, ('full',))

    def test_outside_is_unsupported(self):
        bank = TrailBank(synthetic.write(tempfile.mkdtemp(), {'flat': synthetic.build('flat')}))
        far = jnp.array([[1e4, -1e4]])
        data = bank.sample(jnp.zeros(1, jnp.int32), far)
        self.assertGreater(float(data[0, 2]), .025)
        self.assertGreater(float(data[0, 3]), 0)
        self.assertEqual(float(data[0, 5]), 100.)
        ground, low, high, cover, known = bank.atlas(jnp.zeros(1, jnp.int32), far, 3)
        self.assertEqual(float(known[0]), 0.)
        self.assertTrue(np.all(np.asarray(high) < 0))

    def test_route_at_uses_real_arc_length(self):
        steps = np.tile([.1, .35, .05, .5], 20)              # irregular spacing, total 20 m
        m = synthetic.build('flat', length=20., spacing=steps)
        bank = TrailBank(synthetic.write(tempfile.mkdtemp(), {'irregular': m}))
        s = np.random.default_rng(0).uniform(0, m['s'][-1], 200).astype(np.float32)
        p, yaw, width = bank.route_at(jnp.zeros(1, jnp.int32), jnp.asarray(s)[None])
        np.testing.assert_allclose(np.asarray(p[0, :, 0]), np.interp(s, m['s'], m['route'][:, 0]), atol=1e-5)
        xy = np.stack([np.interp(s, m['s'], m['route'][:, 0]), np.full_like(s, .3)], -1)
        station, lateral = bank.project(jnp.zeros(200, jnp.int32), jnp.asarray(xy), jnp.asarray(s+.4), 8)
        np.testing.assert_allclose(np.asarray(station), s, atol=1e-4)
        np.testing.assert_allclose(np.asarray(lateral), .3, atol=1e-5)

    def test_reversed_route(self):
        maps = dict(forward=synthetic.build('ramp_up'), reverse=synthetic.build('ramp_up', reverse=True))
        bank = TrailBank(synthetic.write(tempfile.mkdtemp(), maps))
        p, yaw, _ = bank.route_at(jnp.array([0, 1]), jnp.zeros(2))
        np.testing.assert_allclose(np.asarray(p[1, :2]), maps['forward']['route'][-1, :2], atol=1e-5)
        self.assertAlmostEqual(float(yaw[1]), np.pi, places=5)
        np.testing.assert_allclose(np.asarray(bank.final(jnp.array([0, 1]))), [29., 29.])
        env = TrailEnv(bank, 2, QUIET)
        state, obs, _ = reset(env, 2.)
        np.testing.assert_allclose(np.asarray(state['pos']), [[2., 0.], [28., 0.]], atol=1e-4)
        planner = Planner(env.cfg)
        for _ in range(60):
            state, obs, _ = env.step(state, planner(obs))
        self.assertTrue(np.all(np.asarray(state['failure']) == 0))
        self.assertTrue(np.all(np.asarray(state['s']) > 6.))
        self.assertLess(float(state['pos'][1, 0]), 24.)       # the reverse slot drove towards -x (downhill)


if __name__ == '__main__':
    unittest.main()
