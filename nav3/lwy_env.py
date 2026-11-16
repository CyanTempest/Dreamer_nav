"""Opt-in LWY environment: same policy contract and sensing, dense footprint."""
from functools import partial
import jax
import jax.numpy as jnp
from .env import FastEnv, rotate


@jax.tree_util.register_pytree_node_class
class LwyEnv(FastEnv):
    def __init__(self, bank, num_envs=1024, random_obstacles=False, direct=False, config=None):
        super().__init__(bank, num_envs, random_obstacles, direct, config={'max_steps': 18000, **(config or {})})
        x, y = jnp.meshgrid(jnp.linspace(-.515, .515, 43), jnp.linspace(-.29, .29, 25), indexing='ij')
        self.foot = jnp.stack([x, y], -1).reshape(-1, 2)
        self.reset = partial(jax.jit(LwyEnv._reset), self)
        self.step = partial(jax.jit(LwyEnv._step), self)
        self.observe = partial(jax.jit(LwyEnv._observe), self)

    def _reset(self, key, mids=None, start_s=None, observe=True):
        if not self.random_obstacles:
            return super()._reset(key, mids, start_s, observe)
        state = super()._reset(key, mids, start_s, observe=False)
        keys = jax.random.split(jax.random.fold_in(key, 9307), 3)
        count = self.cfg['random_obstacles']; n = self.n
        length = self.bank.arcs[state['mids'], self.bank.lengths[state['mids']]-1]
        # Three separated portions of the whole route, not the old 8..190 m
        # window. The >1.7 m legacy activation rule disables every LWY obstacle.
        fraction = (jnp.arange(count)[None]+.5+jax.random.uniform(keys[0], (n, count), minval=-.25, maxval=.25))/count
        station = 8+(length[:, None]-16)*fraction
        pos, yaw, width = self.bank.route_at(state['mids'], station)
        # Keep at least .58 m envelope + .10 m lateral passage on the other
        # side. Sample feasible dimensions; never shrink the robot envelope.
        lateral_max = (width-.03-.58-.10)/2
        factor = jnp.where((jnp.arange(count)[None]%2 == 1)&(lateral_max >= .216), 1.8, 1.)
        maximum = jnp.minimum(.24, lateral_max/factor)
        radius = .12+jax.random.uniform(keys[1], (n, count))*jnp.maximum(0., maximum-.12)
        radius = jnp.where(maximum >= .12, radius, 0.)
        side = jnp.where(jax.random.bernoulli(keys[2], .5, (n, count)), 1., -1.)
        lateral = side*(width/2-radius*factor-.03)
        xy = pos[..., :2]+rotate(jnp.stack([jnp.zeros_like(lateral), lateral], -1), yaw)
        state['obstacles'] = jnp.concatenate([xy, radius[..., None], (radius*factor)[..., None],
            yaw[..., None], jnp.full((n, count, 1), .34)], -1)
        if not observe: return state
        state, obs, extra = self._observe(state)
        state['obs_queue'] = jnp.repeat(obs['bev'][:, None], 2, axis=1)
        return state, obs, extra

    def _step(self, state, action):
        look = state['pos'][:, None]+rotate(jnp.array([[[0., 0.], [.4, 0.]]]), state['yaw'])
        root = jnp.any(self.bank.sample(state['mids'], look, 6) == 2, -1)
        speed_action = jnp.where(root, jnp.minimum(action[:, 2], self.cfg['terrain_speed']/.4-1), action[:, 2])
        return super()._step(state, action.at[:, 2].set(speed_action))
