"""odometry_inputs / localization.mode='odometry': onboard dead reckoning (distance, heading, IMU attitude).

GOLDEN is the digest of `rollout_digest()` computed with the pre-odometry env.py (commit f22abe9) on this
server (JAX 0.4.33, CPU, XLA_FLAGS of audit_20261116/nav4_dev/env.sh). It is machine/JAX-version specific.
"""
import hashlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import QUIET, load

GOLDEN = 'bf3701a54cb84a6fea16e01b5f0c99618b9d7c26fd70c059fd8774abb3aaaf93'
ROUTE_START = 2.0
ZERO = {'odometry.distance_scale_std': 0., 'odometry.heading_walk_std': 0., 'odometry.imu_std': 0.}
ODO = {'route_inputs': False, 'odometry_inputs': True}


def rollout(config=None, steps=40):
    """Default-noise rollout on two synthetic maps: random training starts, failures with resets, and one
    assist restart. Returns the list of (state, obs, extra) as numpy dicts."""
    bank = load({'a': synthetic.build('ledge'), 'b': synthetic.build('stairs')})
    env = TrailEnv(bank, 4, config)
    out = [env.reset(jax.random.PRNGKey(5))]
    action = jnp.array([[1., 0., 1.], [.6, .3, .8], [1., -.1, 1.], [0., .5, .5]], jnp.float32)
    for i in range(steps):
        state = out[-1][0]
        if i == 20:
            state = {**state, 'restart': jnp.array([[6., .1], [np.nan, np.nan], [9., 0.], [np.nan, np.nan]])}
        out.append(env.step(state, action))
    return [tuple(jax.device_get(x) for x in step) for step in out]


def rollout_digest(config=None):
    h = hashlib.sha256()
    for step in rollout(config):
        for part in step:
            for key in sorted(part):
                h.update(key.encode())
                h.update(np.ascontiguousarray(part[key]).tobytes())
    return h.hexdigest()


def curved(radius=40., length=30., margin=15.):
    """Flat map whose route is a left arc of the given radius (route arrays only; odometry never reads the
    corridor/station layers)."""
    m = synthetic.build('flat', length=length, margin=margin)
    theta = m['s'].astype(np.float64)/radius
    m['route'] = np.stack([radius*np.sin(theta), radius*(1-np.cos(theta)), 0*theta], -1).astype(np.float32)
    m['yaw'] = theta.astype(np.float32)
    return m


def drive(env, state, action, steps):
    action = jnp.broadcast_to(jnp.asarray(action, jnp.float32), (env.n, 3))
    for _ in range(steps):
        state, obs, extra = env.step(state, action)
        assert not np.asarray(state['done']).any()
    return state, obs, extra


class Default(unittest.TestCase):
    def test_default_is_bit_identical_to_the_pre_odometry_code(self):
        self.assertEqual(rollout_digest(), GOLDEN)

    def test_odometry_changes_only_state_slots_0_to_6(self):
        """Same rollout with and without odometry_inputs (both route_inputs=false): every other output, the
        physical state and the random streams are unchanged."""
        base, odo = rollout({'route_inputs': False}), rollout(ODO)
        for (sa, oa, ea), (sb, ob, eb) in zip(base, odo):
            for k in sa:
                np.testing.assert_array_equal(sa[k], sb[k], err_msg=k)
            for k in ea:
                np.testing.assert_array_equal(ea[k], eb[k], err_msg=k)
            self.assertEqual(set(oa), set(ob))
            for k in oa:
                if k != 'state':
                    np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)
            np.testing.assert_array_equal(oa['state'][:, 7:], ob['state'][:, 7:])
            self.assertFalse(oa['state'][:, :7].any())
            self.assertTrue((ob['state'][:, 0] > 0).any())
            np.testing.assert_array_equal(ob['state'][:, 5:7], 0.)

    def test_odometry_inputs_require_route_inputs_off(self):
        with self.assertRaises(ValueError):
            TrailEnv(load({'m': synthetic.build('flat')}), 1, {'odometry_inputs': True})


class ZeroNoise(unittest.TestCase):
    def test_matches_true_path_length_and_heading_change(self):
        env = TrailEnv(load({'m': synthetic.build('flat')}), 2, {**QUIET, **ODO, **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=ROUTE_START, offset=0.)
        yaw0 = np.asarray(state['yaw'])
        np.testing.assert_array_equal(np.asarray(obs['state'][:, :7]), [[0, 1, 0, 0, 0, 0, 0]]*2)
        action = jnp.array([[1., .15, 1.], [.5, -.15, .6]], jnp.float32)
        for _ in range(15):
            state, obs, _ = drive(env, state, action, 1)
            o = np.asarray(obs['state'])
            np.testing.assert_allclose(o[:, 0]*1000, np.asarray(state['distance']), rtol=1e-5)
            turn = np.asarray(state['yaw'])-yaw0
            np.testing.assert_allclose(o[:, 1:3], np.stack([np.cos(turn), np.sin(turn)], -1), atol=1e-5)
            np.testing.assert_allclose(o[:, 3:5], 0., atol=1e-6)          # flat ground
            np.testing.assert_array_equal(o[:, 5:7], 0.)
        self.assertGreater(float(np.min(state['distance'])), 1.)
        self.assertGreater(float(np.max(np.abs(turn))), .05)

    def test_training_start_integrates_the_route_from_the_race_start(self):
        env = TrailEnv(load({'m': curved()}), 1, {**QUIET, **ODO, **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=22., offset=0.)
        o = np.asarray(obs['state'][0])
        self.assertAlmostEqual(o[0]*1000, 20., places=3)                  # driven route arc 2 -> 22 m
        self.assertAlmostEqual(np.arctan2(o[2], o[1]), 20/40, places=4)   # heading change along the arc
        np.testing.assert_allclose(np.asarray(state['odo_pos']), np.asarray(state['pos']), atol=1e-4)

    def test_pitch_and_roll_from_the_ground_plane(self):
        env = TrailEnv(load({'m': synthetic.build('ramp_up')}), 1, {**QUIET, **ODO, **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=11., offset=0.)
        grade = np.deg2rad(20.)
        np.testing.assert_allclose(np.asarray(obs['state'][0, 3:5]), [-grade, 0.], atol=2e-3)   # uphill: nose up
        state, obs, _ = env.observe({**state, 'yaw': jnp.array([np.pi/2], jnp.float32)})
        np.testing.assert_allclose(np.asarray(obs['state'][0, 3:5]), [0., -grade], atol=2e-3)   # left side down

    def test_assist_restart_continues_the_odometry(self):
        env = TrailEnv(load({'m': synthetic.build('flat')}), 2, {**QUIET, **ODO, **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=ROUTE_START, offset=0.)
        state, obs, _ = drive(env, state, [1., 0., 1.], 10)
        before, odo = np.asarray(state['pos']), np.asarray(obs['state'][:, 0])*1000
        restart = jnp.array([[float(state['s'][0])+1., .1], [np.nan, np.nan]], jnp.float32)
        state, obs, _ = env.step({**state, 'done': jnp.ones(2, bool), 'restart': restart}, jnp.zeros((2, 3)))
        o = np.asarray(obs['state'])
        jump = np.linalg.norm(np.asarray(state['pos'][0])-before[0])
        self.assertAlmostEqual(o[0, 0]*1000, odo[0]+jump, places=3)        # slot 0: assist, odometry continues
        self.assertAlmostEqual(o[1, 0]*1000, float(state['s'][1])-ROUTE_START, places=3)   # slot 1: new episode
        np.testing.assert_array_equal(np.asarray(state['distance']), 0.)


class NoiseModel(unittest.TestCase):
    """Error statistics of the odometry functions on 20000 samples (relative standard error of a std ~0.5%)."""
    n = 20000

    def assertStd(self, x, expected, rel=.04):
        x = np.asarray(x, np.float64)
        self.assertLess(abs(x.std()/expected-1), rel, (x.std(), expected))
        self.assertLess(abs(x.mean()), 4*expected/np.sqrt(len(x)), (x.mean(), expected))

    def setUp(self):
        self.env = TrailEnv(load({'m': synthetic.build('flat')}), self.n, {**QUIET, **ODO})
        self.o = self.env.cfg['odometry']
        self.mids = jnp.zeros(self.n, jnp.int32)

    def test_training_start_error(self):
        s = jnp.full(self.n, 27.)
        p, yaw, _ = self.env.bank.route_at(self.mids, s)
        start = jax.jit(lambda e, k: e.odometry_start(k, self.mids, s, p[:, :2], yaw, yaw))(
            self.env, jax.random.PRNGKey(1))
        distance = 27.-ROUTE_START
        self.assertStd(start['odo_dist']/distance-1, self.o['distance_scale_std'])
        np.testing.assert_allclose(np.asarray(start['odo_scale'])*distance, np.asarray(start['odo_dist']), rtol=1e-5)
        self.assertStd(start['odo_err'], self.o['heading_walk_std']*np.sqrt(distance))
        error = np.asarray(start['odo_pos']-p[:, :2])
        self.assertStd(error[:, 0], self.o['distance_scale_std']*distance, rel=.05)       # along track: scale
        self.assertStd(error[:, 1], self.o['heading_walk_std']*np.sqrt(distance**3/3), rel=.05)   # integrated walk

    def test_per_step_error(self):
        z = jnp.zeros(self.n)
        state = dict(odo_err=z, odo_dist=z, odo_scale=1+jax.random.normal(jax.random.PRNGKey(2), (self.n,))*.02,
                     odo_pos=jnp.zeros((self.n, 2)))
        step = jax.jit(lambda e, k, st: e.odometry_advance(k, st, jnp.tile(jnp.array([.1, 0.]), (self.n, 1))))
        for i in range(50):                                                 # 50 x 0.1 m = 5 m
            state = step(self.env, jax.random.PRNGKey(100+i), state)
        self.assertStd(state['odo_err'], self.o['heading_walk_std']*np.sqrt(5.))
        np.testing.assert_allclose(np.asarray(state['odo_dist']), 5*np.asarray(state['odo_scale']), rtol=1e-5)

    def test_imu_noise(self):
        z = jnp.zeros(self.n)
        state = dict(pos=jnp.tile(jnp.array([5., 0.]), (self.n, 1)), yaw=z, mids=self.mids, odo_err=z,
                     odo_yaw0=z, odo_dist=z)
        reading = jax.jit(lambda e, k, st: e.odometry_reading(k, st))(self.env, jax.random.PRNGKey(3), state)
        self.assertStd(reading[:, 3], self.o['imu_std'])
        self.assertStd(reading[:, 4], self.o['imu_std'])
        self.assertLess(abs(float(np.corrcoef(reading[:, 3], reading[:, 4])[0, 1])), .03)


class DeadReckoning(unittest.TestCase):
    def test_estimated_pose_is_the_odometry(self):
        """localization.mode='odometry': waypoints/corridor use the dead-reckoned pose (route inputs ON)."""
        env = TrailEnv(load({'m': synthetic.build('flat')}), 1, {**QUIET, 'localization.mode': 'odometry', **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=ROUTE_START, offset=0.)
        start = np.asarray(state['pos'][0])
        state = {**state, 'odo_scale': jnp.array([1.1], jnp.float32)}      # 10% distance over-estimate
        state, obs, _ = drive(env, state, [1., 0., 1.], 15)
        pos, est = np.asarray(state['pos'][0]), np.asarray(state['odo_pos'][0])
        np.testing.assert_allclose(est, start+1.1*(pos-start), atol=1e-4)
        np.testing.assert_allclose(np.asarray(state['drift'][0]), [*(est-pos), 0.], atol=1e-6)
        self.assertAlmostEqual(float(state['est_s'][0]), ROUTE_START+1.1*(float(state['s'][0])-ROUTE_START), places=3)
        goals = np.asarray(obs['state'][0, :6]).reshape(3, 2)*env.cfg['task']['waypoint_scale']
        np.testing.assert_allclose(goals[:, 0], [1., 2., 3.], atol=2e-3)   # ahead of the ESTIMATED pose
        np.testing.assert_allclose(goals[:, 1], -est[1], atol=2e-3)
        self.assertGreater(float(state['s'][0])-ROUTE_START, 1.)

    def test_assist_restart_keeps_the_dead_reckoning_estimate(self):
        """The estimate keeps its error through an assist (no re-localisation), here larger than the 5 m
        projection window, and tracks the assist displacement with the odometry scale."""
        env = TrailEnv(load({'m': synthetic.build('flat')}), 1, {**QUIET, 'localization.mode': 'odometry', **ZERO})
        state, obs, _ = env.reset(jax.random.PRNGKey(0), start_s=ROUTE_START, offset=0.)
        state = {**state, 'odo_scale': jnp.array([4.], jnp.float32)}
        state, obs, _ = drive(env, state, [1., 0., 1.], 15)
        error = float(state['est_s'][0]-state['s'][0])
        self.assertGreater(error, env.cfg['task']['project_window'])
        restart = jnp.array([[float(state['s'][0])+1., 0.]], jnp.float32)
        state, obs, _ = env.step({**state, 'done': jnp.ones(1, bool), 'restart': restart}, jnp.zeros((1, 3)))
        self.assertAlmostEqual(float(state['s'][0]), float(restart[0, 0]), places=4)
        self.assertAlmostEqual(float(state['est_s'][0]-state['s'][0]), error+3., places=3)    # + (4-1) x 1 m jump
        np.testing.assert_allclose(np.asarray(state['drift'][0, :2]),
                                   np.asarray(state['odo_pos'][0]-state['pos'][0]), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
