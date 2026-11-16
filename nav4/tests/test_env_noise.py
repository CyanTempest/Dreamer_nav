"""(c) Configured, per-episode randomised sensor noise; obstacle and support channels come from the same
noisy, dropout-masked measurement as the height channel (legacy R8/R9 ran with height noise 0)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4 import sensor
from nav4.common import load_config
from nav4.env import TrailEnv
from nav4.tests.helpers import QUIET, bank


class Noise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = bank('flat')

    def test_nominal_values(self):
        s = load_config('env')['sensor']
        self.assertAlmostEqual(np.mean(s['height_std']), .025)
        self.assertAlmostEqual(np.mean(s['dropout_base']), .03)        # metrics/r8_p1_noise_fit.json
        self.assertAlmostEqual(np.mean(s['dropout_distance']), .00025)

    def test_per_episode_randomisation(self):
        env = TrailEnv(self.bank, 32)
        state = env.reset(jax.random.PRNGKey(3), start_s=5.)[0]
        noise = np.asarray(state['noise'])
        for i, key in enumerate(('height_std', 'dropout_base', 'dropout_distance')):
            lo, hi = env.cfg['sensor'][key]
            self.assertTrue(((noise[:, i] >= lo) & (noise[:, i] <= hi)).all())
            self.assertGreater(noise[:, i].std(), (hi-lo)/10)

    def test_channels_share_one_noisy_measurement(self):
        env = TrailEnv(self.bank, 2, {**QUIET, 'sensor.height_std': [.03, .03], 'sensor.dropout_base': [.2, .2]})
        state = env.reset(jax.random.PRNGKey(0), start_s=5.)[0]
        measured, keep, seen = map(np.asarray, jax.jit(lambda s, k: sensor.frame(env, s, k))(state, jax.random.PRNGKey(1)))
        support = sensor.support_mask(jnp.asarray(measured[..., 0]), jnp.asarray(keep), jnp.asarray(measured[..., 1]),
                                      env.cfg)
        np.testing.assert_array_equal(measured[..., 2] > .5, np.asarray(support))
        self.assertFalse((measured[..., 2] > .5)[~keep].any())
        self.assertAlmostEqual(float(measured[..., 0][keep].std()), .03, delta=.003)   # flat: residual = noise
        obstacle = measured[..., 1][keep]
        self.assertGreater(float((obstacle > 0).mean()), .3)                        # noise reaches obstacles
        self.assertLess(float(obstacle.max()), env.cfg['terrain']['collision_height'])
        self.assertAlmostEqual(float(keep.sum()/seen.sum()), .8, delta=.03)


if __name__ == '__main__':
    unittest.main()
