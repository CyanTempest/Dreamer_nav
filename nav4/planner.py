"""Observation-only sampling planner, support-risk filter and hybrid takeover (port of nav3/lwy_control_v2,
nav3/control.assess, nav3/lwy_support_filter_r7 and the diagnose_policy takeover).

All forward predictions use the shared 20 Hz tracker (nav4.tracker) with the planner's nominal delay/gain.
Inputs are obs['bev'] and obs['state'] only.
"""
import jax
import jax.numpy as jnp
import numpy as np

from . import tracker
from .common import action_target, bev_index, bev_xy, control_dt, decode, grid, load_config, rotate, target_action


class Planner:
    def __init__(self, env_cfg, config=None):
        """config: {'key': value} overrides of nav4/config/planner.json."""
        self.env_cfg, self.cfg = env_cfg, load_config('planner', config)
        self.plan = jax.jit(self._plan)
        self.support_risk = jax.jit(self._support_risk)
        self.assess = jax.jit(self._assess)
        self.hybrid = jax.jit(self._hybrid)

    def __call__(self, obs):
        return self.plan(obs)[0]

    def steps(self, seconds):
        return int(round(seconds/control_dt(self.env_cfg)))

    def rollout(self, vel, target_fn, speed, steps, visit):
        """Shared-tracker prediction from the robot-frame origin; returns visit(pos, yaw, vel) per substep.

        vel [N,3] measured velocity (also fills the command queue), speed [N,...], target_fn(pos) -> target."""
        delay = tracker.delay_steps(self.cfg['prediction_delay_s'], self.env_cfg)
        shape = speed.shape
        vel = jnp.broadcast_to(vel.reshape(vel.shape[:1]+(1,)*(len(shape)-1)+(3,)), shape+(3,))
        init = (jnp.zeros(shape+(2,)), jnp.zeros(shape), vel, jnp.repeat(vel[..., None, :], delay+1, -2))
        delay, gain = jnp.full(shape, delay), jnp.full(shape, self.cfg['prediction_gain'])

        def body(carry, _):
            carry = tracker.substep(*carry, target_fn(carry[0]), speed, delay, gain, self.env_cfg)
            return carry, visit(*carry[:3])
        return jax.lax.scan(body, init, None, length=steps)[1]

    def lookup(self, bev, xy):
        """BEV values at robot-frame points xy [N,...,2] (nearest cell, clipped) and whether inside the grid."""
        size = self.env_cfg['bev']['size']
        ij = bev_index(xy, self.env_cfg)
        inside = jnp.all((ij >= 0) & (ij < size), -1)
        ij = jnp.clip(ij, 0, size-1)
        batch = jnp.arange(len(bev)).reshape((-1,)+(1,)*(xy.ndim-2))
        return bev[batch, ij[..., 0], ij[..., 1]], inside

    def _plan(self, obs):
        """Lateral-offset candidates around the route waypoints; cost = observed obstacle risk + corridor
        border + unknown (time integrals over the rollout, seconds) + offset^2. Brake when the best is risky."""
        p, e = self.cfg, self.env_cfg
        bev = decode(obs['bev'], e)
        n = len(bev)
        spacing = e['task']['waypoints'][1]-e['task']['waypoints'][0]
        goals = obs['state'][:, :6].reshape(n, 3, 2)*e['task']['waypoint_scale']
        direction = goals[:, 2]-goals[:, 0]
        direction = direction/jnp.maximum(jnp.linalg.norm(direction, axis=-1, keepdims=True), .01)
        normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
        origin = 2*goals[:, 0]-goals[:, 1]
        points = jnp.concatenate([origin[:, None], goals], 1)
        offsets = jnp.linspace(-p['max_offset'], p['max_offset'], p['candidates'])
        rows = jnp.arange(n)

        def target_at(pos):
            along = jnp.clip(jnp.sum((pos-origin[:, None])*direction[:, None], -1)+p['lookahead'], 0., p['path_length'])
            along = along/spacing
            idx = jnp.minimum(along.astype(jnp.int32), 2)
            u = along-idx
            a, b = points[rows[:, None], idx], points[rows[:, None], idx+1]
            return a*(1-u[..., None])+b*u[..., None]+offsets[None, :, None]*normal[:, None]
        xy = bev_xy(e)
        measured = jnp.where(bev[..., 3] < .5, bev[..., 1], 0.)
        nearby = jnp.any((measured > p['obstacle_height']) & (np.abs(xy[..., 1]) < p['nearby_y'])
                         & (xy[..., 0] > p['nearby_x'][0]) & (xy[..., 0] < p['nearby_x'][1]), axis=(1, 2))
        speed = jnp.where(nearby, p['slow_speed'], p['speed'])
        margin = 2*p['footprint_margin']
        foot = grid(e['robot']['envelope'][0]+margin, e['robot']['envelope'][1]+margin, p['footprint_grid'])
        dt = control_dt(e)

        def visit(pos, yaw, vel):
            value, _ = self.lookup(bev, pos[..., None, :]+rotate(foot, yaw[..., None]))
            obstacle = jnp.where(value[..., 3] < .5, value[..., 1], 0.)
            return jnp.stack([jnp.mean(obstacle > p['obstacle_height'], -1), jnp.mean(value[..., 4] < .5, -1),
                              jnp.mean(value[..., 3], -1)], -1)*dt
        cost = jnp.sum(self.rollout(obs['state'][:, 7:10], target_at, jnp.broadcast_to(speed[:, None], (n, len(offsets))),
                                    self.steps(p['horizon_s']), visit), 0)
        risk, border, unknown = cost[..., 0], cost[..., 1], cost[..., 2]
        w = p['weights']
        score = w['risk']*risk+w['border']*border+w['unknown']*unknown+w['offset']*offsets[None]**2
        best = jnp.argmin(score, -1)
        selected = risk[rows, best]+border[rows, best]
        speed = jnp.where(selected > p['brake_threshold'], jnp.minimum(speed, p['brake_speed']), speed)
        return target_action(target_at(jnp.zeros((n, len(offsets), 2)))[rows, best], speed, e), dict(
            selected_risk=risk[rows, best], selected_border=border[rows, best], offset=offsets[best],
            min_risk=jnp.min(risk, -1))

    def predict(self, obs, action, steps, visit):
        """Motion commanded by a policy action (fixed target for the step) from the observed velocity."""
        target, speed = action_target(action, self.env_cfg)
        return self.rollout(obs['state'][:, 7:10], lambda pos: target, speed, steps, visit)

    def _support_risk(self, obs, action):
        """True if the motion `action` commands puts the margin-dilated envelope on an observed unsupported
        cell or outside the route corridor within filter.horizon_s."""
        f, e = self.cfg['filter'], self.env_cfg
        bev = decode(obs['bev'], e)
        dilate = np.array([[0., 0.], [1., 0.], [-1., 0.], [0., 1.], [0., -1.]])*f['margin']
        foot = (grid(*e['robot']['envelope'], f['footprint_grid'])[:, None]+dilate).reshape(-1, 2)

        def visit(pos, yaw, vel):
            value, _ = self.lookup(bev, pos[:, None]+rotate(foot, yaw[:, None]))
            unsupported = (value[..., 2] < .5) & (value[..., 3] < .5)
            return jnp.any(unsupported | (value[..., 4] < .5), -1)
        return jnp.any(self.predict(obs, action, self.steps(f['horizon_s']), visit), 0)

    def _assess(self, bev, target):
        """Observed straight-path check towards target [N,K,2] -> (risk, unknown, off-corridor) fractions [N,K].

        Risk = known obstacle above filter.obstacle_height, known unsupported cell (local-geometry support,
        not a height drop), or outside the BEV."""
        f = self.cfg['filter']
        fractions = jnp.array(f['path_fractions'])
        theta = jnp.arctan2(target[..., 1], target[..., 0])
        foot = np.stack(np.meshgrid(f['assess_x'], f['assess_y'], indexing='ij'), -1).reshape(-1, 2)
        points = (target[:, :, None]*fractions[:, None])[..., None, :]+rotate(foot, (theta[..., None]*fractions)[..., None])
        value, inside = self.lookup(bev, points)
        known = value[..., 3] < .5
        risk = known & ((value[..., 1] > f['obstacle_height']) | (value[..., 2] < .5))
        body = (jnp.abs(points[..., 0]) < f['body'][0]) & (jnp.abs(points[..., 1]) < f['body'][1])
        return (jnp.mean(risk | ~inside, (-1, -2)), jnp.mean(~known & ~body, (-1, -2)),
                jnp.mean(value[..., 4] < .5, (-1, -2)))

    def _hybrid(self, obs, action, hold):
        """Planner takes over when the check on `action` fires, and keeps control for filter.hold_steps."""
        f = self.cfg['filter']
        target, _ = action_target(action, self.env_cfg)
        risk, unknown, corridor = self._assess(decode(obs['bev'], self.env_cfg), target[:, None])
        fire = ((risk[:, 0] > f['max_risk']) | (unknown[:, 0] > f['max_unknown']) | (corridor[:, 0] > f['max_corridor'])
                | self._support_risk(obs, action))
        hold = jnp.where(obs['is_first'], 0, hold)
        takeover = fire | (hold > 0)
        executed = jnp.where(takeover[:, None], self._plan(obs)[0], action)
        return executed, jnp.where(fire, f['hold_steps'], jnp.maximum(0, hold-1)), takeover


class Hybrid:
    """hybrid(policy_action, obs) -> executed action; keeps the per-slot hold counter and last takeover mask."""

    def __init__(self, planner):
        self.planner, self.hold, self.takeover = planner, None, None

    def __call__(self, policy_action, obs):
        if self.hold is None:
            self.hold = jnp.zeros(len(policy_action), jnp.int32)
        action, self.hold, self.takeover = self.planner.hybrid(obs, policy_action, self.hold)
        return action
