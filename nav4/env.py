"""TrailEnv: batched JAX trail-navigation environment (self-contained successor of nav3 FastEnv/ScanEnv).

Contract (unchanged for the Dreamer agent):
  reset(key, mids=None, start_s=None, offset=None) -> (state, obs, extra)
  step(state, action[N,3] in [-1,1]) -> (state, obs, extra); a slot whose obs had is_last is reset
      by the next step call (its action is ignored).
  obs: bev uint8 [N,64,64,6] (ground, obstacle, support, unknown, corridor, age), state float32 [N,13]
       (3 route waypoints in the estimated robot frame /waypoint_scale, gates/8, measured velocity, previous
       action), priv_trav uint8 [N,64,64,1] (privileged decoder target), reward, is_first, is_last, is_terminal.
  Action: (target radius, bearing, speed); 5 Hz policy step = timing.substeps tracker substeps.
  failure: 1 fall/unsupported, 2 collision, 3 off route / reversing, 5 stuck / timeout.
"""
import json
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import sensor, tracker
from .common import action_target, bev_xy, control_dt, encode, grid, load_config, rotate, wrap


@jax.tree_util.register_pytree_node_class
class TrailEnv:
    def __init__(self, bank, num_envs, config=None):
        """config: {'section.key': value} overrides of nav4/config/env.json."""
        self.bank, self.n, self.cfg = bank, num_envs, load_config('env', config)
        self._setup()
        for name in ('reset', 'step', 'observe', 'feasible', 'restart_pose'):
            setattr(self, name, partial(jax.jit(getattr(TrailEnv, '_'+name)), self))

    def _setup(self):
        c, s = self.cfg, self.cfg['sensor']
        self.local = bev_xy(c)
        window = c['task']['priv_window']
        self.extended = bev_xy(c, (window[0]//2, window[1]//2))
        angle = (np.arange(s['polar_angles'])+.5)*2*np.pi/s['polar_angles']-np.pi
        self.radii = ((np.arange(s['polar_bins'])+.5)*s['polar_res']).astype(np.float32)
        self.polar = np.stack([np.cos(angle)[:, None]*self.radii, np.sin(angle)[:, None]*self.radii], -1)
        self.foot = grid(*c['robot']['envelope'], c['robot']['footprint_grid'])
        beams = np.array(s['lidar_vertical_deg'])
        # next_beam[ceil(deg*100)+100] = lowest nominal beam elevation >= deg.
        self.next_beam = jnp.asarray(np.r_[beams, 1e3][np.searchsorted(beams, np.arange(-100, 9101)*.01)])
        self.delays = [tracker.delay_steps(x, c) for x in c['dynamics']['delay_s']]
        self.window = int(np.ceil(c['task']['project_window']/self.bank.min_spacing))
        self.gate_count = int(self.bank.gates.shape[1])

    def tree_flatten(self):
        return (self.bank,), (self.n, json.dumps(self.cfg, sort_keys=True))

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        obj = cls.__new__(cls)
        (obj.bank,), obj.n, obj.cfg = leaves, aux[0], json.loads(aux[1])
        obj._setup()
        return obj

    # ---------------------------------------------------------------- world truth
    def obstacle_geometry(self, obstacles, xy):
        """Signed distance to the nearest episode obstacle (box) and obstacle height at xy [N,...,2]."""
        ob = obstacles.reshape((len(xy),)+(1,)*(xy.ndim-2)+obstacles.shape[1:])
        q = jnp.abs(rotate(xy[..., None, :]-ob[..., :2], -ob[..., 4]))-ob[..., 2:4]
        sd = jnp.linalg.norm(jnp.maximum(q, 0), axis=-1)+jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0)
        enabled = ob[..., 2] > 0
        return (jnp.min(jnp.where(enabled, sd, jnp.inf), -1, initial=jnp.inf),
                jnp.max(jnp.where(enabled & (sd < 0), ob[..., 5], 0.), -1, initial=0.))

    def truth(self, state, xy):
        data = self.bank.sample(state['mids'], xy)
        distance, rise = self.obstacle_geometry(state['obstacles'], xy)
        data = data.at[..., 1].set(jnp.maximum(data[..., 1], data[..., 0]+rise))
        return data.at[..., 5].set(jnp.minimum(data[..., 5], distance))

    def _feasible(self, state, pos, yaw):
        """Envelope supported and collision-free in truth, poses pos [N,...,2], yaw [N,...]."""
        xy = pos[..., None, :]+rotate(jnp.asarray(self.foot), yaw[..., None])
        data = self.truth(state, xy.reshape(len(pos), -1, 2)).reshape(xy.shape[:-1]+(7,))
        return jnp.all((data[..., 2] <= self.cfg['robot']['support_limit']) & (data[..., 5] >= 0), -1)

    def _restart_pose(self, state, station):
        """Assist restart: first feasible (station, lateral offset) at or after `station`, and whether found."""
        a = self.cfg['assist']
        stations = jnp.minimum(station[:, None]+np.arange(0, a['search']+1e-6, a['station_step']),
                               self.bank.final(state['mids'])[:, None]-a['end_clearance'])
        offsets = jnp.array(a['offsets'], jnp.float32)
        p, yaw, _ = self.bank.route_at(state['mids'], stations)
        yaw = jnp.broadcast_to(yaw[..., None], yaw.shape+offsets.shape)
        pos = p[..., None, :2]+rotate(jnp.stack([jnp.zeros_like(offsets), offsets], -1), yaw)
        ok = self._feasible(state, pos, yaw).reshape(self.n, -1)
        first = jnp.argmax(ok, -1)
        rows = jnp.arange(self.n)
        return jnp.stack([stations[rows, first//len(offsets)], offsets[first % len(offsets)]], -1), ok.any(-1)

    def speed_limit(self, state):
        """Terrain speed limit on steep / stair ground: fore-aft rise over 2*steep_probe above steep_rise."""
        v = self.cfg['speed_limit']
        probe = state['pos'][:, None]+rotate(jnp.array([[[-v['steep_probe'], 0.], [v['steep_probe'], 0.]]]),
                                             state['yaw'])
        z = self.bank.sample(state['mids'], probe, 0, linear=True)
        return jnp.where(jnp.abs(z[:, 1]-z[:, 0]) > v['steep_rise'], v['steep'], v['max'])

    def priv_trav(self, state):
        """Privileged target: cells where the whole envelope-sized window is supported and obstacle-free."""
        t = self.cfg['task']
        ext = self.truth(state, state['pos'][:, None, None]+rotate(self.extended[None], state['yaw']))
        free = (ext[..., 2] < -t['priv_margin']) & (ext[..., 5] > t['priv_margin'])
        window = (1,)+tuple(t['priv_window'])
        return jax.lax.reduce_window(free.astype(jnp.int32), 1, jax.lax.min, window, (1, 1, 1), 'VALID') > 0

    # ---------------------------------------------------------------- episodes
    def start_station(self, key, mids):
        """Training start: route_start with probability train_route_start_p, else uniform up to the end."""
        t = self.cfg['task']
        k1, k2 = jax.random.split(key)
        end = jnp.maximum(self.bank.final(mids)-t['train_min_remaining'], t['route_start'])
        s = jax.random.uniform(k1, mids.shape, minval=t['route_start'], maxval=end)
        return jnp.where(jax.random.uniform(k2, mids.shape) < t['train_route_start_p'], t['route_start'], s)

    def _obstacles(self, key, mids, start):
        """Episode obstacles [N,count,6] = (x, y, half length, half width, yaw, height); size 0 = absent.

        Stations are stratified between start and the end. Each obstacle is put beside a passage of
        envelope width + passage_margin at one end of the truth-free lateral interval through the
        centreline (checked over the obstacle length plus the robot length), on a random side."""
        o, n = self.cfg['obstacles'], len(mids)
        count = o['count']
        if count == 0:
            return jnp.zeros((n, 0, 6))
        k = jax.random.split(key, 3)
        lo = start+o['end_margin'][0]
        hi = jnp.maximum(self.bank.final(mids)-o['end_margin'][1], lo)
        station = lo[:, None]+(hi-lo)[:, None]*(jnp.arange(count)+jax.random.uniform(k[0], (n, count)))/count
        p, yaw, _ = self.bank.route_at(mids, station)
        half = jax.random.uniform(k[1], (n, count), minval=o['half_size'][0], maxval=o['half_size'][1])
        across = jnp.where(jnp.arange(count) % 2 == 1, half*o['log_factor'], half)   # logs lie across the route
        length, width = self.cfg['robot']['envelope']
        reach = o['half_size'][1]+length/2
        along = np.linspace(-reach, reach, int(np.ceil(2*reach/self.bank.res))+1)
        lateral = np.arange(-o['lateral_search'], o['lateral_search']+1e-6, o['lateral_step']).astype(np.float32)
        points = np.stack(np.meshgrid(along, lateral, indexing='ij'), -1).astype(np.float32)
        xy = p[..., None, None, :2]+rotate(points[None, None], yaw[..., None, None])
        data = self.bank.sample(mids, xy.reshape(n, -1, 2)).reshape(xy.shape[:-1]+(7,))
        free = jnp.all((data[..., 2] <= self.cfg['robot']['support_limit']) & (data[..., 5] >= 0), 2)
        step, edge = o['lateral_step'], o['lateral_search']+o['lateral_step']
        left = jnp.max(jnp.where(~free & (lateral <= 0), lateral, -edge), -1)+step
        right = jnp.min(jnp.where(~free & (lateral >= 0), lateral, edge), -1)-step
        passage = width+o['passage_margin']
        centre = jnp.where(jax.random.bernoulli(k[2], .5, (n, count)), left+passage+across, right-passage-across)
        ok = free[..., int(np.argmin(np.abs(lateral)))] & (right-left >= passage)
        xy = p[..., :2]+rotate(jnp.stack([jnp.zeros_like(centre), centre], -1), yaw)
        return jnp.concatenate([xy, jnp.where(ok, half, 0.)[..., None], jnp.where(ok, across, 0.)[..., None],
                                yaw[..., None], jnp.full((n, count, 1), o['height'])], -1)

    def memory_origin(self, pos):
        res, size = self.cfg['bev']['res'], self.cfg['memory']['size']
        return jnp.floor(pos/res)*res-size*res/2

    def _reset(self, key, mids=None, start_s=None, offset=None, observe=True):
        c, t, n = self.cfg, self.cfg['task'], self.n
        k = jax.random.split(key, 9)
        mids = jnp.arange(n) % self.bank.count if mids is None else mids
        s = (self.start_station(k[0], mids) if start_s is None
             else jnp.broadcast_to(jnp.asarray(start_s, jnp.float32), (n,)))
        if offset is None:
            offset = jax.random.uniform(k[1], (n,), minval=-t['start_lateral'], maxval=t['start_lateral'])
        p, yaw, _ = self.bank.route_at(mids, s)
        pos = p[:, :2]+rotate(jnp.stack([jnp.zeros(n), jnp.broadcast_to(offset, (n,))], -1), yaw)
        ranges = [c['sensor'][x] for x in ('height_std', 'dropout_base', 'dropout_distance')]
        noise = jnp.stack([jax.random.uniform(kk, (n,), minval=lo, maxval=hi)
                           for kk, (lo, hi) in zip(jax.random.split(k[2], 3), ranges)], -1)
        size, image = c['memory']['size'], c['bev']['size']
        zeros, never = jnp.zeros(n, jnp.int32), jnp.zeros(n, bool)
        state = dict(
            key=k[3], mids=mids, pos=pos, yaw=wrap(yaw+jax.random.normal(k[4], (n,))*t['start_yaw_std']),
            vel=jnp.zeros((n, 3)), prev=jnp.zeros((n, 3)), queue=jnp.zeros((n, self.delays[1]+1, 3)),
            delay=jax.random.randint(k[5], (n,), self.delays[0], self.delays[1]+1),
            gain=jax.random.uniform(k[6], (n,), minval=c['dynamics']['gain'][0], maxval=c['dynamics']['gain'][1]),
            noise=noise, drift=jnp.zeros((n, 3)), s=s, est_s=s, best_s=s, stuck=zeros, steps=zeros,
            max_steps=jnp.ceil((self.bank.final(mids)-s)/(t['timeout_speed']*c['timing']['dt'])).astype(jnp.int32),
            gates=jnp.sum(s[:, None] >= self.bank.gates[mids], -1).astype(jnp.int32),
            obstacles=self._obstacles(k[7], mids, s), ep_return=jnp.zeros(n), distance=jnp.zeros(n),
            memory=jnp.zeros((n, size, size, 3)), age=jnp.full((n, size, size), jnp.inf),
            mempos=self.memory_origin(pos),
            obs_queue=jnp.zeros((n, c['memory']['observation_delay']+1, image, image, 6), jnp.uint8),
            first=jnp.ones(n, bool), done=never, arrived=never, failure=zeros, reward=jnp.zeros(n),
            restart=jnp.full((n, 2), jnp.nan))
        if not observe:
            return state
        state, obs, extra = self._observe(state)
        state['obs_queue'] = jnp.repeat(obs['bev'][:, None], state['obs_queue'].shape[1], axis=1)
        return state, obs, extra

    def _step(self, state, action):
        c, t, r = self.cfg, self.cfg['task'], self.cfg['reward']
        n, mids, dt = self.n, state['mids'], control_dt(c)
        resetting = state['done']
        key, slipkey, resetkey = jax.random.split(state['key'], 3)
        action = jnp.clip(action, -1, 1)
        local, speed = action_target(action, c)
        target = state['pos']+rotate(local, state['yaw'])
        speed = jnp.minimum(speed, self.speed_limit(state))
        slips = jax.random.normal(slipkey, (c['timing']['substeps'], n, 2))*c['dynamics']['slip_std']

        def substep(carry, slip):
            pos, yaw, vel, queue, fall, hit = carry
            new, yaw_new, vel, queue = tracker.substep(pos, yaw, vel, queue, target, speed, state['delay'],
                                                       state['gain'], c)
            scale = jnp.minimum(1., jnp.linalg.norm(vel[:, :2], axis=-1)/c['dynamics']['slip_full_speed'])
            pos = new+rotate(slip*scale[:, None], yaw)*dt
            data = self.truth(state, pos[:, None]+rotate(jnp.asarray(self.foot)[None], yaw_new))
            fall |= jnp.any(data[..., 2] > c['robot']['support_limit'], -1)
            hit |= jnp.any(data[..., 5] < 0, -1)
            return (pos, yaw_new, vel, queue, fall, hit), None
        init = (state['pos'], state['yaw'], state['vel'], state['queue'], jnp.zeros(n, bool), jnp.zeros(n, bool))
        (pos, yaw, vel, queue, fall, hit), _ = jax.lax.scan(substep, init, slips)
        s, lateral = self.bank.project(mids, pos, state['s'], self.window)
        progress = jnp.clip(s-state['s'], -r['progress_clip'], r['progress_clip'])
        # Ordered gate crossing: centre inside the route width while moving past the next gate.
        gate_s = self.bank.gates[mids, jnp.minimum(state['gates'], self.gate_count-1)]
        _, _, width = self.bank.route_at(mids, s)
        crossing = (state['s'] < gate_s) & (s >= gate_s) & (jnp.abs(lateral) < width/2) & (state['gates'] < self.gate_count)
        stuck = jnp.where(s > state['best_s']+t['stuck_progress'], 0, state['stuck']+1)
        steps = state['steps']+1
        wrong = (jnp.abs(lateral) > width/2+t['off_route_margin']) | (s < state['best_s']-t['max_reverse'])
        timeout = (steps >= state['max_steps']) | (stuck*c['timing']['dt'] >= t['stuck_seconds'])
        failure = jnp.where(fall, 1, jnp.where(hit, 2, jnp.where(wrong, 3, jnp.where(timeout, 5, 0)))).astype(jnp.int32)
        arrived = (s >= self.bank.final(mids)) & (failure == 0)
        centre = self.truth(state, pos)
        near = jnp.maximum(0, r['clearance_distance']-jnp.minimum(-centre[:, 2], centre[:, 5]))
        reward = (r['progress']*progress+r['gate']*crossing*(failure == 0)-r['time']
                  - r['smooth']*jnp.mean((action-state['prev'])**2, -1)-r['clearance']*near
                  - r['lateral']*(lateral/jnp.maximum(width/2, r['lateral_floor']))**2-r['failure']*(failure > 0))
        state = {**state, 'key': key, 'pos': pos, 'yaw': yaw, 'vel': vel, 'queue': queue, 'prev': action, 's': s,
                 'best_s': jnp.maximum(s, state['best_s']), 'stuck': stuck, 'steps': steps,
                 'gates': state['gates']+crossing.astype(jnp.int32), 'first': jnp.zeros(n, bool),
                 'reward': reward, 'done': (failure > 0) | arrived, 'arrived': arrived, 'failure': failure,
                 'ep_return': state['ep_return']+reward,
                 'distance': state['distance']+jnp.linalg.norm(pos-state['pos'], axis=-1)}

        def reset_some(current):
            # Slots with a finite `restart` (assist) restart there and keep their obstacles.
            k1, k2, k3 = jax.random.split(resetkey, 3)
            restart = jnp.isfinite(current['restart'][:, 0])
            start = jnp.where(restart, current['restart'][:, 0], self.start_station(k1, mids))
            offset = jnp.where(restart, current['restart'][:, 1],
                               jax.random.uniform(k2, (n,), minval=-t['start_lateral'], maxval=t['start_lateral']))
            fresh = self._reset(k3, mids, start, offset, observe=False)
            fresh['obstacles'] = jnp.where(restart[:, None, None], current['obstacles'], fresh['obstacles'])
            pick = lambda a, b: jnp.where(resetting.reshape((n,)+(1,)*(a.ndim-1)), b, a)
            return {k: current[k] if k == 'key' else pick(current[k], fresh[k]) for k in current}
        state = jax.lax.cond(jnp.any(resetting), reset_some, lambda x: x, state)
        return self._observe(state)

    def _observe(self, state):
        c, t, m = self.cfg, self.cfg['task'], self.cfg['memory']
        n, mids, res, size = self.n, state['mids'], self.cfg['bev']['res'], self.cfg['memory']['size']
        key, framekey, driftkey, speedkey = jax.random.split(state['key'], 4)
        state = {**state, 'key': key}
        loc = c['localization']
        drift = (1-loc['reversion'])*state['drift']+jax.random.normal(driftkey, (n, 3))*jnp.array(
            [loc['position_walk_std'], loc['position_walk_std'], loc['yaw_walk_std']])
        estpos, estyaw = state['pos']+drift[:, :2], state['yaw']+drift[:, 2]
        est_s, _ = self.bank.project(mids, estpos, state['est_s'], self.window)
        measured, keep, seen = sensor.frame(self, state, framekey)
        # World-aligned rolling memory keyed by the ESTIMATED pose; integer-cell shifts only.
        world = estpos[:, None, None]+rotate(self.local[None], estyaw)
        origin = self.memory_origin(estpos)
        shift = jnp.rint((origin-state['mempos'])/res).astype(jnp.int32)
        old = jnp.asarray(np.indices((size, size)).transpose(1, 2, 0))[None]+shift[:, None, None]
        inside = jnp.all((old >= 0) & (old < size), -1)
        old = jnp.clip(old, 0, size-1)
        batch = jnp.arange(n)[:, None, None]
        memory = state['memory'][batch, old[..., 0], old[..., 1]]
        age = jnp.where(inside, state['age'][batch, old[..., 0], old[..., 1]]+c['timing']['dt'], jnp.inf)
        znow = self.bank.sample(mids, state['pos'], 0, linear=True)
        ij = jnp.clip(jnp.floor((world-origin[:, None, None])/res).astype(jnp.int32), 0, size-1)
        fresh = jnp.zeros((n, size, size), bool).at[batch, ij[..., 0], ij[..., 1]].max(keep)
        absolute = measured.at[..., 0].add(znow[:, None, None])
        updates = jnp.full_like(memory, -jnp.inf).at[batch, ij[..., 0], ij[..., 1]].max(
            jnp.where(keep[..., None], absolute, -jnp.inf))
        memory = jnp.where(fresh[..., None], updates, memory)
        age = jnp.where(fresh, 0., age)
        local_age = age[batch, ij[..., 0], ij[..., 1]]
        known = local_age < m['seconds']
        terrain = memory[batch, ij[..., 0], ij[..., 1]].at[..., 0].add(-znow[:, None, None])
        route, _, _ = self.bank.route_at(mids, est_s[:, None]+jnp.array(t['waypoints']))
        parts = dict(terrain=jnp.where(known[..., None], terrain, 0.), unknown=~known,
                     corridor=self.bank.sample(mids, world, 3, linear=True) < 0,
                     age=jnp.clip(local_age/m['seconds'], 0, 1),
                     waypoints=rotate(route[..., :2]-estpos[:, None], -estyaw)/t['waypoint_scale'],
                     progress=state['gates']/self.gate_count,
                     velocity=state['vel']+jax.random.normal(speedkey, (n, 3))*c['sensor']['velocity_std'])
        trav = self.priv_trav(state)
        obs, queue = self.assemble(state, parts, trav)
        state.update(memory=memory, age=age, mempos=origin, obs_queue=queue, drift=drift, est_s=est_s)
        # True route quantities (privileged targets / evaluation only, never policy inputs).
        true_s, lateral = self.bank.project(mids, state['pos'], state['s'], self.window)
        _, route_yaw, _ = self.bank.route_at(mids, true_s)
        body = state['pos'][:, None, None]+rotate(self.local[None], state['yaw'])
        extra = dict(occluded=~seen, clean_trav=trav, route_mask=self.bank.sample(mids, body, 3, linear=True) < 0,
                     progress=true_s/self.bank.arcs[mids, self.bank.lengths[mids]-1],
                     heading_error=wrap(route_yaw-state['yaw']), lateral=lateral)
        return state, obs, extra

    def assemble(self, state, parts, trav):
        """Policy observation from the simulator's measurements. Input variants belong here only.

        route_inputs=false removes everything that needs a map-frame pose (corridor channel, waypoints,
        gate progress) by zeroing it; shapes are unchanged. Returns (obs, new observation-delay queue)."""
        c, n = self.cfg, self.n
        if not c['route_inputs']:
            parts = {**parts, 'corridor': jnp.zeros_like(parts['corridor']),
                     'waypoints': jnp.zeros_like(parts['waypoints']), 'progress': jnp.zeros_like(parts['progress'])}
        image = encode(jnp.concatenate([parts['terrain'], parts['unknown'][..., None], parts['corridor'][..., None],
                                        parts['age'][..., None]], -1), c)
        queue = jnp.concatenate([image[:, None], state['obs_queue'][:, :-1]], 1)
        vec = jnp.concatenate([parts['waypoints'].reshape(n, -1), parts['progress'][:, None], parts['velocity'],
                               state['prev']], -1).astype(jnp.float32)
        return dict(bev=jnp.where(state['first'][:, None, None, None], image, queue[:, -1]), state=vec,
                    priv_trav=(trav[..., None]*255).astype(jnp.uint8), reward=state['reward'],
                    is_first=state['first'], is_last=state['done'],
                    is_terminal=state['done'] & (state['failure'] != 5)), queue
