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


def support_mask(height, known, obstacle, cfg, *, filtered=False):
    """Support from measured geometry, independent of the robot's relative height.

    v8: fit a plane to known, 3x3-median-filtered samples in a disc whose diameter
    is terrain.grade_baseline_m (1 m). A separate 8-neighbour / two-cell step test
    prevents the metre-scale fit from averaging away ledges. Unknown cells never
    gain support. At the BEV boundary the disc is clipped, without wrapping or
    inventing samples; a nondegenerate fit is still required.

    filtered=True accepts the same median ground already computed by v2, so its
    ground channel, collision reference and support use ONE surface, filtered once.
    Explicit configs without grade_baseline_m retain the legacy one-cell minmod
    rule when passed directly to this function (e.g. archived configurations).
    """
    t, res = cfg['terrain'], cfg['bev']['res']
    near = neighbours(known)
    h = jnp.sort(jnp.stack([height]+[jnp.where(k, v, height) for v, k in zip(neighbours(height), near)], -1),
                 -1)[..., 4]
    if filtered:
        h = height
    diff = [jnp.where(k & known, v-h, 0.) for v, k in zip(neighbours(h), near)]
    step = jnp.max(jnp.abs(jnp.stack(diff)), 0)
    if 'grade_baseline_m' not in t:
        grade = jnp.hypot(minmod(diff[6], -diff[1]), minmod(diff[4], -diff[3]))/res
        return (known & (step <= t['max_step']) & (grade <= np.tan(np.deg2rad(t['max_grade_deg'])))
                & (obstacle < t['collision_height']))

    # Match mapping.max_step: catch a riser spread across two cells, too.
    for plus, minus in ((6, 1), (4, 3)):
        jump = jnp.where(known & near[plus] & near[minus], jnp.abs(diff[plus]-diff[minus]), 0.)
        step = jnp.maximum(step, jump)
    radius = t['grade_baseline_m']/2
    if radius < res:
        raise ValueError('terrain.grade_baseline_m must span at least two BEV cells')
    r = int(np.ceil(radius/res))
    u, v = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)*res
    disc = (u*u+v*v <= radius*radius+1e-9).astype(np.float32)
    # Simultaneous weighted moments, SAME padding contributes zero known weight.
    kernel = np.zeros((*disc.shape, 2, 9), np.float32)
    for i, term in enumerate((disc, u*disc, v*disc, u*u*disc, v*v*disc, u*v*disc)):
        kernel[..., 0, i] = term
    for i, term in enumerate((disc, u*disc, v*disc), 6):
        kernel[..., 1, i] = term
    inp = jnp.stack([known.astype(h.dtype), jnp.where(known, h, 0.)], -1)
    moments = jax.lax.conv_general_dilated(inp, jnp.asarray(kernel), (1, 1), 'SAME',
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                         precision=jax.lax.Precision.HIGHEST)
    n, sx, sy, sxx, syy, sxy, sz, sxz, syz = [moments[..., i] for i in range(9)]
    den = jnp.maximum(n, 1.)
    xx, yy, xy = sxx-sx*sx/den, syy-sy*sy/den, sxy-sx*sy/den
    xz, yz = sxz-sx*sz/den, syz-sy*sz/den
    det = xx*yy-xy*xy
    good = (n >= 3) & (det > jnp.finfo(h.dtype).eps)
    det = jnp.where(good, det, 1.)
    grade = jnp.hypot((yy*xz-xy*yz)/det, (xx*yz-xy*xz)/det)
    # Float32 convolution roundoff only (<4 micrometres at a riser), not a physical noise margin.
    eps = 32*jnp.finfo(h.dtype).eps
    return (known & good & (step <= t['max_step']+eps)
            & (grade <= np.tan(np.deg2rad(t['max_grade_deg']))+eps)
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


def horizon_ok_v2(env, state, origin, zsensor, rel, rr, delta):
    """Same physical horizon as v1, indexed by per-environment world-grid queries."""
    s = env.cfg['sensor']
    xy = (state['pos']+rotate(jnp.broadcast_to(origin, state['pos'].shape), state['yaw']))[:, None, None]
    xy = xy+rotate(env.polar[None], state['yaw'])
    top = env.bank.atlas_ground(state['mids'], xy)+env.obstacle_geometry(state['obstacles'], xy)[1]
    slope = (top-zsensor[:, None, None])/np.maximum(env.radii, s['horizon_min_range'])
    horizon = jax.lax.associative_scan(jnp.maximum, slope, axis=-1)
    horizon = jnp.concatenate([jnp.full_like(horizon[..., :1], -jnp.inf), horizon[..., :-1]], -1)
    ai = jnp.floor((jnp.arctan2(rel[..., 1], rel[..., 0])+np.pi)/(2*np.pi)*s['polar_angles']).astype(jnp.int32)
    ri = jnp.clip(jnp.floor(rr/s['polar_res']).astype(jnp.int32), 0, s['polar_bins']-1)
    h = horizon[jnp.arange(len(xy))[:, None, None], ai % s['polar_angles'], ri]
    return delta/jnp.maximum(rr, s['horizon_min_range'])[..., None] >= h[..., None]-s['horizon_tolerance']


def frame_v2(env, state, key, local):
    """Simulate visible return intervals at world-cell centres expressed in the sensor frame.

    Only this forward sensor model accesses the geometry atlas. The bottom/top of
    visible intervals model the low/high ray returns, exactly as in v1; there is
    no true-ground fill in unseen cells and no opening. Absolute z uses the body
    vertical pose (the simulator has no vertical odometry error model).
    Returns (low z, top z), keep, seen, body ground datum. No labels or support.
    """
    cfg, s = env.cfg, env.cfg['sensor']
    res = cfg['bev']['res']
    pos, yaw, mids = state['pos'], state['yaw'], state['mids']
    world = pos[:, None, None]+rotate(local, yaw)
    g, low, high, cover, known = env.bank.atlas(mids, world, s['atlas_bands'])
    distance, rise = env.obstacle_geometry(state['obstacles'], world)
    z0 = env.bank.sample(mids, pos, 0, linear=True)
    target = jnp.concatenate([g[..., None], g[..., None]+high, (g+rise)[..., None]], -1)
    lower = jnp.concatenate([g[..., None], g[..., None]+low, g[..., None]], -1)
    present = jnp.concatenate([(known > .5)[..., None], (high >= 0) & (cover >= s['band_coverage']),
                               (distance < 0)[..., None]], -1)
    axis = jnp.eye(2, dtype=jnp.float32)*res
    probe = world[..., None, :]+rotate(jnp.concatenate([axis, -axis]), yaw[:, None, None, None])
    surface = env.bank.atlas_ground(mids, probe)
    gx, gy = (surface[..., 0]-surface[..., 2])/(2*res), (surface[..., 1]-surface[..., 3])/(2*res)
    vis = jnp.zeros(present.shape, bool)
    for lidar in s['lidars']:
        origin = np.array(lidar['xy'], np.float32)
        zsensor = z0+lidar['z']
        axis = lidar['axis']
        rel = local-origin
        rr = jnp.linalg.norm(rel, axis=-1)
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
        vis |= fov & horizon_ok_v2(env, state, origin, zsensor, rel, rr, delta)
    cam = s['camera']
    origin = np.array(cam['xy'], np.float32)
    zsensor = z0+cam['z']
    rel = local-origin
    rr = jnp.linalg.norm(rel, axis=-1)
    delta = target-zsensor[:, None, None, None]
    forward = rel[..., 0, None]
    fov = ((forward >= cam['range'][0]) & (forward <= cam['range'][1])
           & (jnp.abs(rel[..., 1, None]) <= forward*np.tan(np.deg2rad(cam['half_hfov_deg'])))
           & (jnp.abs(delta) <= forward*np.tan(np.deg2rad(cam['half_vfov_deg']))))
    vis |= fov & horizon_ok_v2(env, state, origin, zsensor, rel, rr, delta)
    blind = (jnp.abs(local[..., 0]) < s['blind'][0]) & (jnp.abs(local[..., 1]) < s['blind'][1])
    vis &= present & ~blind[..., None]
    seen = jnp.any(vis, -1)
    ground = jnp.min(jnp.where(vis, lower, jnp.inf), -1)
    top = jnp.max(jnp.where(vis, target, -jnp.inf), -1)
    k_drop, k_ground, k_top = jax.random.split(key, 3)
    std, base, per_m2 = [state['noise'][:, i, None, None] for i in range(3)]
    dropout = jnp.clip(base+per_m2*jnp.sum(local**2, -1), 0, 1)
    keep = seen & (jax.random.uniform(k_drop, seen.shape) >= dropout)
    ground = ground+jax.random.normal(k_ground, seen.shape)*std
    top = top+jax.random.normal(k_top, seen.shape)*std
    return jnp.where(keep[..., None], jnp.stack([ground, top], -1), 0.), keep, seen, z0


def terrain_v2(ground, top, known, cfg):
    """Infer terrain solely from delivered measurements, at the final BEV resolution.

    Mapping v8 uses top > own_ground+0.30 AND top > max3_ground+0.22.
    Express their intersection as a height relative to one effective reference;
    keep the six-channel contract and the unchanged 0.30 m support cutoff.
    The same known-only 3x3 median as support_mask estimates ground from low
    returns: an isolated elevated return must not raise the neighbourhood's
    collision reference. Filter once, then use that surface for ALL three terrain
    channels. Unknown neighbours contribute nothing. No opening or label is used.
    """
    near = neighbours(known)
    ground = jnp.sort(jnp.stack([ground]+[jnp.where(k, v, ground)
                                        for v, k in zip(neighbours(ground), near)], -1), -1)[..., 4]
    nearby = jax.lax.reduce_window(jnp.where(known, ground, -jnp.inf), -jnp.inf, jax.lax.max,
                                   (1, 3, 3), (1, 1, 1), 'SAME')
    lift = cfg['terrain']['collision_height']-cfg['bev']['v2']['terrain_face_height']
    reference = jnp.maximum(ground, nearby-lift)
    obstacle = jnp.where(known, jnp.maximum(0., top-reference), 0.)
    support = support_mask(ground, known, obstacle, cfg, filtered=True)
    return jnp.stack([jnp.where(known, ground, 0.), obstacle, support.astype(jnp.float32)], -1)


def update_memory_v2(memory, indices, measured, keep, stamp):
    """Whole-tuple latest-acquisition update; duplicate queries choose the first valid source.

    All queries assigned to one cell were already sampled AT that world centre;
    only independent measurement noise can differ. A source-index reduction
    selects one tuple, never separate extrema of its height channels.
    """
    n, size = memory.shape[:2]
    count = keep.shape[1]*keep.shape[2]
    batch = jnp.arange(n)[:, None, None]
    inside = jnp.all((indices >= 0) & (indices < size), -1)
    ids = jnp.arange(count, dtype=jnp.int32).reshape(1, *keep.shape[1:])
    ij = jnp.clip(indices, 0, size-1)
    source = jnp.full((n, size, size), count, jnp.int32).at[batch, ij[..., 0], ij[..., 1]].min(
        jnp.where(keep & inside, ids, count))
    incoming = measured.reshape(n, count, 2)[batch, jnp.minimum(source, count-1)]
    timestamp = jnp.broadcast_to(stamp[:, None, None], (n, size, size))
    incoming = jnp.concatenate([incoming, timestamp[..., None]], -1)
    fresh = (source < count) & (timestamp >= memory[..., 2])
    return jnp.where(fresh[..., None], incoming, memory)
