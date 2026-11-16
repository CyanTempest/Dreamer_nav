"""priv_route=true: privileged decoder targets from the true route; the policy inputs stay route-free."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import load


class PrivRoute(unittest.TestCase):
    def run_pair(self, config):
        """Same physics layers and robot pose; the second map's route arrays (and corridor/station layers) are
        shifted 1 m sideways. Default (noisy) config apart from `config`."""
        envs = [TrailEnv(load({'m': synthetic.build('ramp_up', route_y=y)}), 2, config) for y in (0., 1.)]
        out = [env.reset(jax.random.PRNGKey(7), start_s=5., offset=offset) for env, offset in zip(envs, (0., -1.))]
        steps = [out]
        action = jnp.array([[.5, .1, .4], [0., -.2, .8]], jnp.float32)
        for _ in range(6):
            out = [env.step(state, action) for env, (state, _, _) in zip(envs, out)]
            steps.append(out)
        return steps

    def test_route_shift_changes_only_the_priv_route_targets(self):
        for config in ({'route_inputs': False, 'priv_route': True},
                       {'route_inputs': False, 'priv_route': True, 'odometry_inputs': True}):
            for (a, oa, ea), (b, ob, eb) in self.run_pair(config):
                np.testing.assert_array_equal(np.asarray(a['pos']), np.asarray(b['pos']))
                for key in ('bev', 'state', 'priv_trav'):
                    np.testing.assert_array_equal(np.asarray(oa[key]), np.asarray(ob[key]), err_msg=key)
                self.assertFalse(np.array_equal(np.asarray(oa['priv_route']), np.asarray(ob['priv_route'])))
                self.assertFalse(np.array_equal(np.asarray(oa['priv_progress']), np.asarray(ob['priv_progress'])))
                for o, e in ((oa, ea), (ob, eb)):
                    self.assertEqual((o['priv_route'].dtype, o['priv_route'].shape), (jnp.uint8, (2, 64, 64, 1)))
                    self.assertEqual((o['priv_progress'].dtype, o['priv_progress'].shape), (jnp.float32, (2, 4)))
                    np.testing.assert_array_equal(np.asarray(o['priv_route'][..., 0]),
                                                  np.asarray(e['route_mask']).astype(np.uint8)*255)
                    he = np.asarray(e['heading_error'])
                    np.testing.assert_allclose(np.asarray(o['priv_progress']), np.stack(
                        [np.asarray(e['progress']), np.cos(he), np.sin(he), np.clip(np.asarray(e['lateral']), -2, 2)],
                        -1), atol=1e-6)

    def test_default_has_no_route_targets(self):
        env = TrailEnv(load({'m': synthetic.build('flat')}), 1)
        _, obs, _ = env.reset(jax.random.PRNGKey(0))
        self.assertEqual(sorted(obs), ['bev', 'is_first', 'is_last', 'is_terminal', 'priv_trav', 'reward', 'state'])


if __name__ == '__main__':
    unittest.main()
