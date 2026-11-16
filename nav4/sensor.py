"""Range-sensor model on the static atlas (port of the R8/R9 sensor, nav3/r8_p1_online.py).

Two hemispherical 96-beam lidars and one forward depth camera see candidate surfaces (ground, the
lowest `atlas_bands` height bands, episode obstacles) through a polar terrain-horizon occlusion test.
The lowest visible surface (after a ground opening) is the ground return, the highest is the top.
Per-episode noise: height noise on both returns and range-dependent dropout. Obstacle height and the
support flag are derived from that same noisy, dropout-masked measurement.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .common import rotate


def opening(x, size):
    """Morphological opening (min then max filter) with edge padding."""
    p = size//2
    for op, init in ((jax.lax.min, jnp.inf), (jax.lax.max, -jnp.inf)):
        x = jax.lax.reduce_window(jnp.pad(x, ((0, 0), (p, p), (p, p)), mode='edge'), init, op,
                                  (1, size, size), (1, 1, 1), 'VALID')
    return x


def neighbours(x):
    """The 8 neighbours of every cell, order (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1).

    Edge-padded: a border cell's missing neighbour is a copy of itself (no wrap-around)."""
    p = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), mode='edge')
    n, m = x.shape[1], x.shape[2]
    return [p[:, 1+i:1+i+n, 1+j:1+j+m] for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]


def minmod(a, b):
    return jnp.where(a*b > 0, jnp.sign(a)*jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.)


def support_mask(height, known, obstacle, cfg):
    """Support from local geometry only (never from height relative to the robot).

    Unsupported: a step to a known 8-neighbour above terrain.max_step, a grade above terrain.max_grade_deg,
    or an obstacle at or above terrain.collision_height. Heights are first 3x3-median filtered over known
    cells (keeps steps, removes per-cell noise). The grade per axis is the minmod of the forward and
    backward differences, so an isolated step is judged by the step test, a continuous slope by the grade
    test. Unknown neighbours and the BEV edge contribute no difference.
    """
    t, res = cfg['terrain'], cfg['bev']['res']
    near = neighbours(known)
    h = jnp.sort(jnp.stack([height]+[jnp.where(k, v, height) for v, k in zip(neighbours(height), near)], -1),
                 -1)[..., 4]
    diff = [jnp.where(k & known, v-h, 0.) for v, k in zip(neighbours(h), near)]
    step = jnp.max(jnp.abs(jnp.stack(diff)), 0)
    grade = jnp.hypot(minmod(diff[6], -diff[1]), minmod(diff[4], -diff[3]))/res
    return (known & (step <= t['max_step']) & (grade <= np.tan(np.deg2rad(t['max_grade_deg'])))
            & (obstacle < t['collision_height']))


def gradient(x, res):
    """Central differences with edge padding."""
    p = jnp.pad(x, ((0, 0), (1, 1), (1, 1)), mode='edge')
    return (p[:, 2:, 1:-1]-p[:, :-2, 1:-1])/(2*res), (p[:, 1:-1, 2:]-p[:, 1:-1, :-2])/(2*res)


def horizon_ok(env, state, origin, zsensor, rel, rr, delta):
    """Polar terrain horizon from the sensor origin: is each candidate surface above it?"""
    s = env.cfg['sensor']
    pos, yaw = state['pos'], state['yaw']
    xy = (pos+rotate(jnp.broadcast_to(origin, pos.shape), yaw))[:, None, None]+rotate(env.polar[None], yaw)
    top = env.bank.atlas_ground(state['mids'], xy)+env.obstacle_geometry(state['obstacles'], xy)[1]
    slope = (top-zsensor[:, None, None])/np.maximum(env.radii, s['horizon_min_range'])
    horizon = jax.lax.associative_scan(jnp.maximum, slope, axis=-1)
    horizon = jnp.concatenate([jnp.full_like(horizon[..., :1], -jnp.inf), horizon[..., :-1]], -1)
    angle = np.arctan2(rel[..., 1], rel[..., 0])
    ai = np.floor((angle+np.pi)/(2*np.pi)*s['polar_angles']).astype(np.int32) % s['polar_angles']
    ri = np.clip(np.floor(rr/s['polar_res']).astype(np.int32), 0, s['polar_bins']-1)
    h = horizon[:, ai, ri]
    return delta/np.maximum(rr, s['horizon_min_range'])[..., None] >= h[..., None]-s['horizon_tolerance']


def frame(env, state, key):
    """One noisy frame in the robot frame.

    Returns measured [N,S,S,3] = (ground height relative to the terrain under the robot, obstacle
    height, support flag), keep [N,S,S] (returned and not dropped) and seen [N,S,S] (visible)."""
    cfg, s = env.cfg, env.cfg['sensor']
    res = cfg['bev']['res']
    pos, yaw, mids = state['pos'], state['yaw'], state['mids']
    world = pos[:, None, None]+rotate(env.local[None], yaw)
    g, low, high, cover, known = env.bank.atlas(mids, world, s['atlas_bands'])
    distance, rise = env.obstacle_geometry(state['obstacles'], world)
    z0 = env.bank.sample(mids, pos, 0, linear=True)
    target = jnp.concatenate([g[..., None], g[..., None]+high, (g+rise)[..., None]], -1)
    lower = jnp.concatenate([g[..., None], g[..., None]+low, g[..., None]], -1)
    present = jnp.concatenate([(known > .5)[..., None], (high >= 0) & (cover >= s['band_coverage']),
                               (distance < 0)[..., None]], -1)
    gx, gy = gradient(g, res)
    vis = jnp.zeros(present.shape, bool)
    for lidar in s['lidars']:
        origin = np.array(lidar['xy'], np.float32)
        zsensor = z0+lidar['z']
        axis = lidar['axis']
        rel = env.local-origin
        rr = np.linalg.norm(rel, axis=-1)
        rx, ry = rel[..., 0, None], rel[..., 1, None]
        delta = target-zsensor[:, None, None, None]
        r3 = jnp.sqrt(rr[..., None]**2+delta**2)
        fov = (rx*axis[0]+ry*axis[1]+delta*axis[2] >= 0) & (r3 >= s['lidar_range'][0]) & (r3 <= s['lidar_range'][1])
        # Does any of the nominal elevation angles fall inside the surface's angular footprint?
        dz = (target+lower)*.5-zsensor[:, None, None, None]
        proj = rx*axis[0]+ry*axis[1]+dz*axis[2]
        r2 = rr[..., None]**2+dz**2
        denom = jnp.maximum(r2*jnp.sqrt(jnp.maximum(r2-proj**2, 1e-8)), 1e-8)
        ax, ay, az = [(axis[k]*r2-proj*q)/denom for k, q in enumerate((rx, ry, dz))]
        width = res/2*(jnp.abs(ax+az*gx[..., None])+jnp.abs(ay+az*gy[..., None]))+(target-lower)*.5*jnp.abs(az)
        angle = jnp.arcsin(jnp.clip(proj/jnp.sqrt(jnp.maximum(r2, 1e-8)), -1, 1))
        first = jnp.clip(jnp.ceil(jnp.rad2deg(angle-width)*100).astype(jnp.int32)+100, 0, len(env.next_beam)-1)
        fov &= env.next_beam[first] <= jnp.rad2deg(angle+width)
        vis |= fov & horizon_ok(env, state, origin, zsensor, rel, rr, delta)
    cam = s['camera']
    origin = np.array(cam['xy'], np.float32)
    zsensor = z0+cam['z']
    rel = env.local-origin
    rr = np.linalg.norm(rel, axis=-1)
    delta = target-zsensor[:, None, None, None]
    forward = rel[..., 0, None]
    fov = ((forward >= cam['range'][0]) & (forward <= cam['range'][1])
           & (np.abs(rel[..., 1, None]) <= forward*np.tan(np.deg2rad(cam['half_hfov_deg'])))
           & (jnp.abs(delta) <= forward*np.tan(np.deg2rad(cam['half_vfov_deg']))))
    vis |= fov & horizon_ok(env, state, origin, zsensor, rel, rr, delta)
    blind = (np.abs(env.local[..., 0]) < s['blind'][0]) & (np.abs(env.local[..., 1]) < s['blind'][1])
    vis &= present & ~blind[..., None]
    seen = jnp.any(vis, -1)
    ground = jnp.where(seen, jnp.min(jnp.where(vis, lower, jnp.inf), -1), g)
    ground = opening(ground, s['ground_opening'])
    top = jnp.max(jnp.where(vis, target, -jnp.inf), -1)
    k_drop, k_ground, k_top = jax.random.split(key, 3)
    std, base, per_m2 = [state['noise'][:, i, None, None] for i in range(3)]
    dropout = jnp.clip(base+per_m2*np.sum(env.local**2, -1), 0, 1)
    keep = seen & (jax.random.uniform(k_drop, seen.shape) >= dropout)
    height = ground-z0[:, None, None]+jax.random.normal(k_ground, seen.shape)*std
    top = top-z0[:, None, None]+jax.random.normal(k_top, seen.shape)*std
    obstacle = jnp.maximum(0., top-height)
    support = support_mask(height, keep, obstacle, cfg)
    return jnp.stack([height, obstacle, support.astype(jnp.float32)], -1), keep, seen
