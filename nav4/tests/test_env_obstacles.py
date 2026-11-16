"""(g) Optional episode obstacles (default off) leave a feasible passage of envelope width + 0.10 m."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.common import bev_xy, decode
from nav4.tests.helpers import env as make_env, reset


class Obstacles(unittest.TestCase):
    def test_default_off(self):
        env = make_env(('flat',), n=1)
        self.assertEqual(env.cfg['obstacles']['count'], 0)
        self.assertEqual(reset(env, 2.)[0]['obstacles'].shape[1], 0)

    def test_passage_and_visibility(self):
        env = make_env(('causeway',), n=6, **{'obstacles.count': 3})     # 1.6 m wide path, drops both sides
        state, _, _ = env.reset(jax.random.PRNGKey(5), start_s=2.)
        ob = np.asarray(state['obstacles'])
        self.assertTrue((ob[..., 2] > 0).all())
        length, width = env.cfg['robot']['envelope']
        passage = width+env.cfg['obstacles']['passage_margin']
        lateral = np.arange(-1.2, 1.2, .01)
        for k in range(ob.shape[1]):
            x, y, along, across = ob[:, k, 0], ob[:, k, 1], ob[:, k, 2], ob[:, k, 3]
            self.assertTrue((np.abs(y)-across < width/2).all())             # in the robot's way on the centreline
            stations = x[:, None]+np.linspace(-1, 1, 12)*(along[:, None]+length/2)
            pts = np.stack(np.broadcast_arrays(stations[..., None], lateral), -1)
            data = np.asarray(env.truth(state, jnp.asarray(pts, jnp.float32)))
            free = np.all((data[..., 2] <= .025) & (data[..., 5] >= 0), 1)
            for i in range(env.n):
                runs = np.split(lateral[free[i]], np.flatnonzero(np.diff(lateral[free[i]]) > .015)+1)
                best = max(runs, key=len)
                self.assertGreaterEqual(best[-1]-best[0]+.01, passage-.03)
                centre = jnp.asarray(np.stack([stations[i], np.full(12, (best[0]+best[-1])/2)], -1), jnp.float32)
                fits = env.feasible({**state, 'obstacles': state['obstacles'][i:i+1].repeat(env.n, 0)},
                                    jnp.tile(centre[None], (env.n, 1, 1)), jnp.zeros((env.n, 12)))
                self.assertTrue(bool(fits.all()))                           # the envelope passes beside it
        near = {**state, 'pos': jnp.stack([ob[:, 0, 0]-2.5, np.zeros(env.n)], -1).astype(np.float32),
                'yaw': jnp.zeros(env.n)}
        near['mempos'] = env.memory_origin(near['pos'])
        bev = np.asarray(decode(env.observe(near)[1]['bev'], env.cfg))
        xy = bev_xy(env.cfg)
        for i in range(env.n):
            cells = (np.abs(xy[..., 0]-2.5) < ob[i, 0, 2]) & (np.abs(xy[..., 1]-ob[i, 0, 1]) < ob[i, 0, 3])
            self.assertGreater(bev[i, ..., 1][cells].max(), .2)              # the sensor sees the obstacle


if __name__ == '__main__':
    unittest.main()
