"""CPU synthetic fields: only static obstacle truth changes interpolation."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import QUIET, load


class ObstacleSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m = synthetic.build(length=8., margin=2.)
        y = m['origin'][1]+np.arange(m['layers'].shape[1])*float(m['res'])
        m['layers'][..., 5] = .32-y[None, :]
        cls.bank = load({'linear_field': m})

    def env(self, mode=None):
        config = dict(QUIET)
        if mode is not None:
            config['robot.obstacle_sampling'] = mode
        return TrailEnv(self.bank, 1, config)

    def state(self):
        return dict(mids=jnp.zeros(1, jnp.int32), obstacles=jnp.zeros((1, 0, 6)))

    def test_truth_interpolates_only_layer_five(self):
        xy = jnp.array([[[2., .34]]])
        near, linear = self.env(), self.env('linear')
        self.assertEqual(near.cfg['robot']['obstacle_sampling'], 'nearest')
        a, b = np.asarray(near.truth(self.state(), xy)), np.asarray(linear.truth(self.state(), xy))
        self.assertAlmostEqual(float(a[0, 0, 5]), .02, places=5)
        self.assertAlmostEqual(float(b[0, 0, 5]), -.02, places=5)
        np.testing.assert_array_equal(a[..., [0, 1, 2, 3, 4, 6]], b[..., [0, 1, 2, 3, 4, 6]])
        np.testing.assert_array_equal(a, np.asarray(self.env('nearest').truth(self.state(), xy)))

    def test_grazing_envelope_feasibility_and_step_collision(self):
        # Upper edge y=.29+.045=.335: nearest sees y=.30 as free;
        # bilinear sees the 1.5 cm overlap with the wall at y=.32.
        for mode, fits, failure in [('nearest', True, 0), ('linear', False, 2)]:
            with self.subTest(mode=mode):
                env = self.env(mode)
                pos, yaw = jnp.array([[2., .045]]), jnp.zeros(1)
                self.assertEqual(bool(env.feasible(self.state(), pos, yaw)[0]), fits)
                state = env._reset(jax.random.PRNGKey(0), start_s=2., offset=.045, observe=False)
                state, _, _ = env.step(state, jnp.array([[-1., 0., -1.]]))
                self.assertEqual(int(state['failure'][0]), failure)

    def test_dynamic_boxes_still_use_exact_distance(self):
        state = {**self.state(), 'obstacles': jnp.array([[[2., 0., .1, .1, 0., .5]]])}
        xy = jnp.array([[[2., 0.]]])
        for mode in ('nearest', 'linear'):
            self.assertAlmostEqual(float(self.env(mode).truth(state, xy)[0, 0, 5]), -.1, places=6)

    def test_invalid_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'obstacle_sampling'):
            self.env('cubic')


if __name__ == '__main__':
    unittest.main()
