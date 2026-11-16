"""(e) trail task definition (start stations, arrival, failure on the arrival step) and (f) terrain speed limit."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4.tests.helpers import drive, env as make_env, reset

FORWARD = [1., 0., 1.]                                   # 3 m straight ahead at full speed


class Task(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(('flat', 'endwall'), n=2)
        cls.final = float(cls.env.bank.final(jnp.zeros(1, jnp.int32))[0])

    def test_training_start_distribution(self):
        t = self.env.cfg['task']
        s = np.asarray(self.env.start_station(jax.random.PRNGKey(0), jnp.zeros(20000, jnp.int32)))
        first = s == t['route_start']
        self.assertAlmostEqual(first.mean(), t['train_route_start_p'], delta=.01)
        rest, end = s[~first], self.final-t['train_min_remaining']
        self.assertTrue((rest >= t['route_start']).all() and (rest <= end).all())
        self.assertAlmostEqual(rest.mean(), (t['route_start']+end)/2, delta=.3)
        self.assertGreater(np.histogram(rest, 10)[0].min(), .08*len(rest))   # uniform

    def test_evaluation_starts_at_route_start(self):
        state = reset(self.env, self.env.cfg['task']['route_start'])[0]
        np.testing.assert_allclose(np.asarray(state['s']), 2.)
        self.assertEqual(np.asarray(state['gates']).tolist(), [0, 0])

    def test_gates_are_progress_checkpoints(self):
        gate = float(self.env.bank.gates[0, 0])
        state, obs, _ = reset(self.env, gate-.4)
        self.assertEqual(float(obs['state'][0, 6]), 0.)
        for state, obs, _ in drive(self.env, state, FORWARD, 12):
            if int(state['gates'][0]) == 1:
                self.assertGreater(float(obs['reward'][0]), self.env.cfg['reward']['gate']-1)
                break
        self.assertEqual(float(obs['state'][0, 6]), 1/8)

    def test_success_is_arrival_at_the_final_station(self):
        state = reset(self.env, self.final-1.2)[0]
        for state, obs, _ in drive(self.env, state, FORWARD, 30):
            if bool(state['done'][0]):
                break
        self.assertTrue(bool(state['arrived'][0]))
        self.assertEqual(int(state['failure'][0]), 0)
        self.assertEqual(int(state['gates'][0]), 8)
        self.assertTrue(bool(obs['is_last'][0]) and bool(obs['is_terminal'][0]))

    def test_failure_on_the_arrival_step_is_a_failure(self):
        """Both slots cross the final station in this step; slot 1 also touches a wall (legacy: success)."""
        state = reset(self.env, self.final-.05)[0]
        moving = jnp.array([.5, 0., 0.])
        state = {**state, 'vel': jnp.tile(moving, (2, 1)), 'queue': jnp.tile(moving, (2, state['queue'].shape[1], 1))}
        state, obs, _ = self.env.step(state, jnp.tile(jnp.array([FORWARD]), (2, 1)))
        self.assertTrue((np.asarray(state['s']) >= self.final).all())
        self.assertEqual(np.asarray(state['failure']).tolist(), [0, 2])
        self.assertEqual(np.asarray(state['arrived']).tolist(), [True, False])
        reward = np.asarray(obs['reward'])
        self.assertGreater(reward[0], self.env.cfg['reward']['gate']/2)          # arrival bonus
        self.assertLess(reward[1], -self.env.cfg['reward']['failure']+1)         # no arrival bonus


class AssistRestart(unittest.TestCase):
    def test_restart_pose_is_first_feasible(self):
        """Wall x 12..12.4, y 0..1.6: at station 11.5 the first feasible offset in assist.offsets is -0.3 m;
        past the end the station is clamped to final - end_clearance; a restart keeps the obstacles."""
        env = make_env(('wall',), n=3)
        state = reset(env, 2.)[0]
        pose, found = env.restart_pose(state, jnp.array([11.5, 5., 40.]))
        final = float(env.bank.final(jnp.zeros(1, jnp.int32))[0])
        np.testing.assert_allclose(np.asarray(pose), [[11.5, -.3], [5., 0.], [final-.2, 0.]], atol=1e-5)
        self.assertTrue(np.asarray(found).all())
        state = {**state, 'done': jnp.array([True, False, False]), 'restart': jnp.array([[11.5, -.3]]+[[np.nan]*2]*2)}
        state, obs, _ = env.step(state, jnp.zeros((3, 3)))
        np.testing.assert_allclose(np.asarray(state['pos'][0]), [11.5, -.3], atol=1e-4)
        self.assertTrue(bool(obs['is_first'][0]) and not bool(obs['is_first'][1]))
        self.assertTrue(bool(env.feasible(state, state['pos'], state['yaw'])[0]))
        self.assertTrue(np.isnan(np.asarray(state['restart'][0])).all())      # consumed


class SpeedLimit(unittest.TestCase):
    def test_steep_and_stair_cells_limit_speed(self):
        for limit, config in ((.28, {}), (.2, {'speed_limit.steep': .2})):
            env = make_env(('ramp_up', 'stairs', 'flat'), n=3, **config)
            state = reset(env, 8.6)[0]
            speed = np.array([np.linalg.norm(np.asarray(s['vel'][:, :2]), axis=-1) for s, _, _ in
                              drive(env, state, FORWARD, 15)])
            self.assertLessEqual(speed[:, :2].max(), limit+1e-4)
            self.assertGreater(speed[:, :2].max(), limit-.02)
            self.assertGreater(speed[:, 2].max(), .6)


if __name__ == '__main__':
    unittest.main()
