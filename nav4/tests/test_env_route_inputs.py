"""route_inputs=false: no map-frame quantity reaches obs['bev'] / obs['state']; extra keeps the true ones."""
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.bank import TrailBank
from nav4.env import TrailEnv
from nav4.tests import synthetic


class RouteInputs(unittest.TestCase):
    def run_pair(self, route_inputs):
        """Same physics and robot pose; the second map's centreline (route arrays, corridor and station
        layers) is shifted 1 m sideways. Default (noisy) config apart from route_inputs."""
        envs = [TrailEnv(TrailBank(synthetic.write(tempfile.mkdtemp(), {'m': synthetic.build('ramp_up', route_y=y)})),
                         2, {'route_inputs': route_inputs}) for y in (0., 1.)]
        out = [env.reset(jax.random.PRNGKey(7), start_s=5., offset=offset) for env, offset in zip(envs, (0., -1.))]
        steps = [out]
        action = jnp.array([[.5, .1, .4], [0., -.2, .8]], jnp.float32)
        for _ in range(6):
            out = [env.step(state, action) for env, (state, _, _) in zip(envs, out)]
            steps.append(out)
        return steps

    def test_route_shift_invisible_without_route_inputs(self):
        for (a, oa, ea), (b, ob, eb) in self.run_pair(False):
            np.testing.assert_array_equal(np.asarray(a['pos']), np.asarray(b['pos']))
            np.testing.assert_array_equal(np.asarray(oa['bev']), np.asarray(ob['bev']))
            np.testing.assert_array_equal(np.asarray(oa['state']), np.asarray(ob['state']))
            self.assertFalse(np.array_equal(np.asarray(ea['route_mask']), np.asarray(eb['route_mask'])))
            np.testing.assert_allclose(np.asarray(eb['lateral'])-np.asarray(ea['lateral']), -1., atol=1e-4)
            self.assertFalse(np.asarray(oa['bev'][..., 4]).any())
            self.assertFalse(np.asarray(oa['state'][:, :7]).any())

    def test_route_shift_visible_with_route_inputs(self):
        (a, oa, ea), (b, ob, eb) = self.run_pair(True)[0]
        self.assertFalse(np.array_equal(np.asarray(oa['bev']), np.asarray(ob['bev'])))
        self.assertFalse(np.array_equal(np.asarray(oa['state']), np.asarray(ob['state'])))


if __name__ == '__main__':
    unittest.main()
