"""CPU synthetic-map coverage of the route-free safety boundary and fallback/hold behavior."""
import unittest

import jax.numpy as jnp
import numpy as np

from nav4.common import action_target, bev_xy, decode, encode, grid, rotate
from nav4.planner import Guard, Planner
from nav4.tests.helpers import env as make_env, reset


class RouteFreeGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(('flat', 'wall'), n=2, route_inputs=False, odometry_inputs=True)
        cls.planner = Planner(cls.env.cfg)
        # The wall at x=12 must lie in the 0.8 s tracker horizon, not just near the distant goal.
        _, cls.obs, _ = reset(cls.env, 11.4)
        cls.action = jnp.array([[1., 0., 1.], [1., 0., 1.]], jnp.float32)

    def clear(self):
        # Fully observed synthetic flat geometry, independent of sensor occlusion, for isolated hazards.
        bev = jnp.zeros_like(self.obs['bev'], dtype=jnp.float32).at[..., 2].set(1.)
        return {**self.obs, 'bev': encode(bev, self.env.cfg)}

    def test_wall_triggers_and_flat_odometry_passes_exactly(self):
        self.assertFalse(np.asarray(self.obs['bev'][..., 4]).any())
        self.assertTrue(np.asarray(self.obs['state'][:, :7]).any())
        guard = Guard(self.planner)
        executed = guard(self.action, self.obs)
        np.testing.assert_array_equal(guard.takeover, [False, True])
        np.testing.assert_array_equal(executed[0], self.action[0])
        self.assertTrue(float(executed[1, 1]) != 0. or float(executed[1, 2]) == -1.)
        if float(executed[1, 2]) != -1.:
            # The chosen detour must itself pass the very same check without a held takeover.
            _, _, taken = self.planner.guard(self.obs, executed, jnp.zeros(2, jnp.int32))
            self.assertFalse(bool(taken[1]))

    def test_route_and_privileged_inputs_cannot_change_action_hold_or_takeover(self):
        rng = np.random.default_rng(73)
        other = {**self.obs,
                 'state': self.obs['state'].at[:, :7].set(rng.normal(size=(2, 7))*1000),
                 'bev': self.obs['bev'].at[..., 4].set(rng.integers(0, 256, size=(2, 64, 64), dtype=np.uint8)),
                 'priv_trav': jnp.full_like(self.obs['priv_trav'], 255),
                 'priv_route': jnp.zeros((2, 64, 64, 1), jnp.uint8),
                 'priv_progress': jnp.full((2, 4), jnp.nan),
                 'extra': {'route': jnp.full((2, 3, 2), jnp.nan)}}
        for hold in (jnp.zeros(2, jnp.int32), jnp.full(2, 3, jnp.int32)):
            for first in (True, False):
                original = {**self.obs, 'is_first': jnp.full(2, first)}
                shuffled = {**other, 'is_first': jnp.full(2, first)}
                expected = self.planner.guard(original, self.action, hold)
                actual = self.planner.guard(shuffled, self.action, hold)
                for a, b in zip(expected, actual):
                    np.testing.assert_array_equal(a, b)

    def test_all_unsafe_brakes_to_action_minimum(self):
        obs = self.clear()
        bev = decode(obs['bev'], self.env.cfg).at[..., 1].set(.6)
        obs['bev'] = encode(bev, self.env.cfg)
        guard = Guard(self.planner)
        executed = guard(self.action, obs)
        self.assertTrue(np.asarray(guard.takeover).all())
        _, speed = action_target(executed, self.env.cfg)
        np.testing.assert_array_equal(speed, self.env.cfg['action']['speed'][0])
        np.testing.assert_array_equal(executed[:, :2], self.action[:, :2])

    def test_unknown_expired_memory_and_known_unsupported_trigger(self):
        for channel, value in ((3, 1.), (5, 1.), (2, 0.)):
            with self.subTest(channel=channel):
                obs = self.clear()
                bev = decode(obs['bev'], self.env.cfg).at[..., channel].set(value)
                obs['bev'] = encode(bev, self.env.cfg)
                guard = Guard(self.planner)
                guard(self.action, obs)
                self.assertTrue(np.asarray(guard.takeover).all())

    def test_measured_sideways_motion_checks_swept_envelope(self):
        obs = self.clear()
        xy = bev_xy(self.env.cfg)
        # Obstacle beside the body: outside the straight-path samples but in the inertial lateral sweep.
        side = (np.abs(xy[..., 0]) < .4) & (xy[..., 1] > .4) & (xy[..., 1] < .9)
        bev = decode(obs['bev'], self.env.cfg).at[..., 1].set(jnp.where(side, .6, 0.))
        obs['bev'] = encode(bev, self.env.cfg)
        stationary = self.planner.guard(obs, self.action, jnp.zeros(2, jnp.int32))
        self.assertFalse(np.asarray(stationary[2]).any())
        obs['state'] = obs['state'].at[:, 8].set(.8)
        moving = self.planner.guard(obs, self.action, jnp.zeros(2, jnp.int32))
        self.assertTrue(np.asarray(moving[2]).all())

    def test_out_of_bev_in_tracker_prediction_triggers(self):
        obs = self.clear()
        # An extreme measured lateral velocity isolates BEV bounds even with clear clipped edge cells.
        obs['state'] = obs['state'].at[:, 8].set(10.)
        guard = Guard(self.planner)
        guard(self.action, obs)
        self.assertTrue(np.asarray(guard.takeover).all())

    def test_narrow_bend_turns_instead_of_braking(self):
        obs = self.clear()
        xy = bev_xy(self.env.cfg)

        def free(points):
            x, y = points[..., 0], points[..., 1]
            # A 1.25 m approach enters a left bend; the inner corner blocks straight travel.
            approach = (x < .75) & (jnp.abs(y) < .625)
            exit_lane = (x > -.625) & (x < 1.125) & (y > -.25)
            return approach | exit_lane

        bev = decode(obs['bev'], self.env.cfg)
        bev = bev.at[..., 2].set(free(xy)).at[..., 1].set(jnp.where(free(xy), 0., .6))
        obs['bev'] = encode(bev, self.env.cfg)
        obs['state'] = obs['state'].at[:, 7:10].set(jnp.array([.4, 0., 0.]))
        action = jnp.zeros((2, 3), jnp.float32)
        guard = Guard(self.planner)
        executed = guard(action, obs)
        self.assertTrue(np.asarray(guard.takeover).all())
        self.assertTrue(np.all(np.asarray(executed[:, 1]) > 0.))
        self.assertTrue(np.all(np.asarray(action_target(executed, self.env.cfg)[1]) > 0.))

        # The round-1 full-radius chord rejects every bearing/speed at this bend.
        candidates = self.planner._guard_candidates(action).at[..., 0].set(action[:, None, 0])
        g = self.planner.cfg['guard']
        target, _ = action_target(candidates, self.env.cfg)
        risk, unknown, _ = self.planner._assess(bev, target, corridor=False, config=g)
        swept = self.planner._support_risk(obs, candidates, corridor=False, config=g, bev=bev)
        self.assertFalse(np.asarray((risk <= g['max_risk']) & (unknown <= g['max_unknown']) & ~swept).any())

        # Independent dense footprint against the analytic corridor, not the guard's sampled risk.
        foot = grid(*self.env.cfg['robot']['envelope'], [43, 25])

        def collision(pos, yaw, vel):
            return ~jnp.all(free(pos[..., None, :]+rotate(foot, yaw[..., None])), -1)

        steps = self.planner.steps(g['horizon_s'])
        self.assertTrue(np.asarray(self.planner.predict(obs, action, steps, collision)).any())
        self.assertFalse(np.asarray(self.planner.predict(obs, executed, steps, collision)).any())

    def test_slow_action_checks_reachable_distance_and_preserves_inertia(self):
        obs = self.clear()
        wall = bev_xy(self.env.cfg)[..., 0] >= .875
        bev = decode(obs['bev'], self.env.cfg).at[..., 1].set(jnp.where(wall, .6, 0.))
        obs['bev'] = encode(bev, self.env.cfg)
        obs['state'] = obs['state'].at[:, 7:10].set(jnp.array([.4, 0., 0.]))
        actions = jnp.broadcast_to(jnp.array([[0., 0., 0.], [0., 0., -.5]]), (2, 2, 3))
        safe = self.planner._guard_safe(obs, bev, actions)
        np.testing.assert_array_equal(safe, [[False, True], [False, True]])
        # A stop command with large measured forward speed must still test its delayed coasting path.
        obs['state'] = obs['state'].at[:, 7].set(.8)
        stop = actions[:, :1].at[..., 2].set(-1.)
        self.assertFalse(np.asarray(self.planner._guard_safe(obs, bev, stop)).any())

    def test_hold_duration_and_episode_reset(self):
        guard = Guard(self.planner)
        guard(self.action, self.obs)
        clear = {**self.clear(), 'is_first': jnp.zeros(2, bool)}
        flags = []
        for _ in range(self.planner.cfg['guard']['hold_steps']+2):
            executed = guard(self.action, clear)
            flags.append(bool(guard.takeover[1]))
            np.testing.assert_array_equal(executed, self.action)
        self.assertEqual(flags, [True]*self.planner.cfg['guard']['hold_steps']+[False]*2)
        guard(self.action, self.obs)
        guard(self.action, self.clear())  # is_first clears the old hold for each slot.
        self.assertFalse(np.asarray(guard.takeover).any())


if __name__ == '__main__':
    unittest.main()
