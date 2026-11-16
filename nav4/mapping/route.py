"""Fix D: swept-feasible, smoothed route for the saved map.

1. Feasibility of the 1.03 x 0.58 m footprint at lateral offsets around the scanner centreline
   (lateral_feasibility, robot aligned with the centreline).
2. Path: corridor_path (feasible DP within +-acceptance_band_m).  Where no feasible path exists
   (blocked stations) the DP is re-run with a soft cost (number of failing footprint samples), so the
   route passes blocked places at the least-violating offset; those stations are reported.
3. Smoothing inside the feasible tube (contiguous feasible offsets around the DP path): minimise the
   squared 2-D second difference of the path (curvature energy) plus a weak pull to the tube middle,
   with bounds = tube (exact convex QP); stations whose radius stays below min_turn_radius_m are reported.
4. Validation with the path's own heading (same checks as validate_path: 5 cm footprints + swept poses);
   where it fails away from blocked places, the tube is narrowed locally and step 3 is repeated; the
   iteration with the fewest such failures (then fewest tight-radius stations) is kept.
5. Resampling at ~station_spacing_m arc length (uniform, exact end points), yaw from the path.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.analysis import corridor_path, lateral_grid
from nav4.mapping.common import RES, footprint_ok, footprint_world, swept_poses

STEP = .05     # lateral offset grid (m), same as the report and qualify_routes


def lateral_violations(support_sdf, obstacle_sdf, origin, xy, yaw, offsets, chunk=256):
    """Failing footprint samples (support sdf > 0.025 or obstacle sdf < 0 or outside) per (station, offset)."""
    cand = lateral_grid(xy, yaw, offsets)
    n, k = cand.shape[:2]
    pts = cand.reshape(-1, 2)
    ang = np.broadcast_to(yaw[:, None], (n, k)).reshape(-1)
    out = np.zeros(len(pts), np.int32)
    shape = support_sdf.shape
    for a in range(0, len(pts), chunk):
        w = footprint_world(pts[a:a + chunk], ang[a:a + chunk])
        ij = np.floor((w - origin) / RES + .5).astype(np.int64)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < shape[1])
        i = ij[..., 0].clip(0, shape[0] - 1)
        j = ij[..., 1].clip(0, shape[1] - 1)
        bad = ~inside | (support_sdf[i, j] > .025) | (obstacle_sdf[i, j] < 0)
        out[a:a + chunk] = bad.sum(1)
    return out.reshape(n, k)


def soft_path(viol, offsets, max_shift_cells, band):
    """DP with cost = 1000 * failing samples + 0.05 * |offset change| + 0.01 * |offset| per station."""
    use = np.abs(offsets) <= band + 1e-9
    v = viol[:, use].astype(np.float64)
    off = offsets[use]
    n, k = v.shape
    cost = 1000. * v[0] + np.abs(off) * .01
    parent = np.zeros((n, k), np.int32)
    for i in range(1, n):
        best = np.full(k, np.inf)
        arg = np.zeros(k, np.int32)
        for d in range(-max_shift_cells, max_shift_cells + 1):
            src = np.arange(k) - d
            ok = (src >= 0) & (src < k)
            c = np.full(k, np.inf)
            c[ok] = cost[src[ok]] + abs(d) * .05
            better = c < best
            best[better] = c[better]
            arg[better] = src[better]
        cost = best + 1000. * v[i] + np.abs(off) * .01
        parent[i] = arg
    j = int(np.argmin(cost))
    idx = [j]
    for i in range(n - 1, 0, -1):
        j = int(parent[i, j])
        idx.append(j)
    idx = np.array(idx[::-1])
    return off[idx], v[np.arange(n), idx].astype(np.int32)


def feasible_tube(feas, offsets, path, band):
    """Contiguous run of feasible offsets (within the band) containing the path offset at each station;
    stations whose path offset is infeasible get lo = hi = path."""
    use = np.abs(offsets) <= band + 1e-9
    off = offsets[use]
    f = feas[:, use]
    lo = path.copy()
    hi = path.copy()
    for i in range(len(path)):
        j = int(np.argmin(np.abs(off - path[i])))
        if not f[i, j]:
            continue
        a = j
        while a > 0 and f[i, a - 1]:
            a -= 1
        b = j
        while b < len(off) - 1 and f[i, b + 1]:
            b += 1
        lo[i], hi[i] = off[a], off[b]
    return lo, hi


def path_check(support_sdf, obstacle_sdf, origin, pts):
    """Same checks as analysis.validate_path on the polyline pts: footprints every 5 cm with the path's own
    heading, plus swept poses.  Returns arc positions (m along pts) of failing fine and swept samples."""
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    fs = np.arange(0, arc[-1], .05)
    fine = np.stack([np.interp(fs, arc, pts[:, d]) for d in range(2)], -1)
    fyaw = np.unwrap(np.arctan2(np.gradient(fine[:, 1]), np.gradient(fine[:, 0])))
    good = footprint_ok(support_sdf, obstacle_sdf, origin, fine, fyaw)
    sp, sa = swept_poses(fine, fyaw)
    sok = footprint_ok(support_sdf, obstacle_sdf, origin, sp, sa)
    _, k = cKDTree(fine).query(sp[~sok]) if (~sok).any() else (None, np.zeros(0, int))
    return dict(arc=arc, fine_fail_arc=fs[~good], swept_fail_arc=fs[k], fine_pass_fraction=float(good.mean()),
                swept_pass_fraction=float(sok.mean()), swept_ok=bool(good.all() and sok.all()))


def turning_radius(xy):
    """Unsigned turning radius at each point of a polyline (circumradius of consecutive triples)."""
    a, b, c = xy[:-2], xy[1:-1], xy[2:]
    ab = np.linalg.norm(b - a, axis=1); bc = np.linalg.norm(c - b, axis=1); ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    with np.errstate(invalid='ignore', divide='ignore'):
        r = ab * bc * ca / np.maximum(2 * cross, 1e-12)
    return np.r_[np.inf, r, np.inf]


def smooth_in_tube(cxy, cyaw, o0, lo, hi, h, pull_scale, weights, mid=None, max_rounds=200):
    """min (1/h^3) sum w |D2 P|^2 + mu h sum (o - mid)^2,  P = c + n o,  lo <= o <= hi;  mu = 1 / pull_scale^4.
    Convex QP with a pentadiagonal Hessian, solved exactly by a primal active-set loop (sparse solves)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    n = len(o0)
    nx, ny = -np.sin(cyaw), np.cos(cyaw)
    mid = (lo + hi) / 2 if mid is None else mid
    mu = 1. / pull_scale ** 4
    A = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n), format='csr')
    W = sparse.diags(weights)
    AtWA = (A.T @ W @ A).tocsr()
    Nx, Ny = sparse.diags(nx), sparse.diags(ny)
    H = ((Nx @ AtWA @ Nx + Ny @ AtWA @ Ny) / h ** 3 + mu * h * sparse.eye(n)).tocsc()
    b = -(nx * (AtWA @ cxy[:, 0]) + ny * (AtWA @ cxy[:, 1])) / h ** 3 + mu * h * mid
    fixed = lo >= hi - 1e-12
    val = np.where(fixed, lo, 0.)
    o = np.clip(o0, lo, hi)
    for _ in range(max_rounds):
        free = ~fixed
        rhs = b[free] - H[free][:, fixed] @ val[fixed]
        o = val.copy()
        o[free] = spsolve(H[free][:, free].tocsc(), rhs)
        over, under = free & (o > hi + 1e-9), free & (o < lo - 1e-9)
        # release bound variables whose multiplier has the wrong sign (gradient pushes them inside)
        g = H @ o - b
        release = fixed & (lo < hi - 1e-12) & (((val <= lo + 1e-12) & (g < -1e-12)) | ((val >= hi - 1e-12) & (g > 1e-12)))
        if not over.any() and not under.any() and not release.any():
            break
        fixed = (fixed & ~release) | over | under
        val = np.where(over, hi, np.where(under, lo, val))
    return np.clip(o, lo, hi)


def _route_pass(support_sdf, obstacle_sdf, origin, cxy, cyaw, cs, cfg, lim_lo, lim_hi, log):
    c, rc = cfg['corridor'], cfg['route']
    band = c['acceptance_band_m']
    offsets = np.arange(-band, band + 1e-9, STEP)
    viol = lateral_violations(support_sdf, obstacle_sdf, origin, cxy, cyaw, offsets)
    outside = (offsets[None] < lim_lo[:, None] - 1e-9) | (offsets[None] > lim_hi[:, None] + 1e-9)
    viol = np.where(outside, 10 ** 6, viol)
    feas = viol == 0
    shift = int(round(c['dp_max_shift_per_station_m'] / STEP))
    connected, reach, path, last = corridor_path(feas, offsets, shift, band)
    if connected:
        path_viol = np.zeros(len(cs), np.int32)
        method = 'corridor_path (feasible DP)'
    else:
        path, path_viol = soft_path(viol, offsets, shift, band)
        method = 'soft DP (least failing footprint samples) - no continuous feasible path'
    log(f'route DP: {method}; stations with failing samples {int((path_viol > 0).sum())}')
    lo, hi = feasible_tube(feas, offsets, path, band)
    # blocked stations: not pinned to the DP offset (that zig-zags), free within +-blocked_freedom_m of it
    blk = path_viol > 0
    lo = np.where(blk, np.maximum(path - rc['blocked_freedom_m'], lim_lo), lo)
    hi = np.where(blk, np.minimum(path + rc['blocked_freedom_m'], lim_hi), hi)
    mid = np.where(blk, path, (lo + hi) / 2)
    h = float(np.median(np.diff(cs)))
    w = np.ones(len(cs) - 2)
    margin = np.zeros(len(cs))
    history = []
    best = None
    nrm = np.stack([-np.sin(cyaw), np.cos(cyaw)], -1)
    for it in range(rc['max_iterations']):
        a = np.minimum(lo + margin, (lo + hi) / 2)
        b = np.maximum(hi - margin, (lo + hi) / 2)
        wk = w
        o = path
        for _ in range(rc['reweight_rounds']):
            o = smooth_in_tube(cxy, cyaw, o, a, b, h, rc['pull_scale_m'], wk, mid=np.clip(mid, a, b))
            # sum |D2 P|^2 / h^3 equals the curvature energy sum kappa^2 h only for station spacing h; offsets
            # change the local spacing h_i (inside a bend it shrinks), so reweight by (h / h_i)^3
            seg = np.linalg.norm(np.diff(cxy + nrm * o[:, None], axis=0), axis=1)
            hi_ = np.maximum((seg[:-1] + seg[1:]) / 2, 1e-3)
            wk = w * np.clip((h / hi_) ** 3, .1, 1e3)
        pts = cxy + nrm * o[:, None]
        R = turning_radius(pts)
        v = path_check(support_sdf, obstacle_sdf, origin, pts)
        fail_arc = np.r_[v['fine_fail_arc'], v['swept_fail_arc']]
        bad = np.zeros(len(cs), bool)
        if len(fail_arc):
            k = np.clip(np.searchsorted(v['arc'], fail_arc), 0, len(cs) - 1)
            bad[k] = True
        # stations within reach of a blocked station fail anyway (the footprint overlaps the blocked place)
        bad &= ~ndi.binary_dilation(path_viol > 0, iterations=int(np.ceil(rc['repair_halfwindow_m'] / h)) + 3)
        tight = R < rc['min_turn_radius_m']
        history.append(dict(iteration=it, min_radius_m=float(R.min()), stations_radius_below=int(tight.sum()),
                            fine_fail_samples=int(len(v['fine_fail_arc'])), swept_fail_samples=int(len(v['swept_fail_arc'])),
                            swept_pass_fraction=v['swept_pass_fraction'], unblocked_fail_stations=int(bad.sum()),
                            unblocked_fail_s_m=[round(float(x), 1) for x in cs[bad]][:40]))
        log(f'route smoothing it {it}: min R {R.min():.2f} m ({int(tight.sum())} below {rc["min_turn_radius_m"]} m), '
            f'fine/swept fail samples {len(v["fine_fail_arc"])}/{len(v["swept_fail_arc"])}, '
            f'stations failing away from blocked places {int(bad.sum())}')
        key = (int(bad.sum()), int(tight.sum()))
        if best is None or key < best[0]:
            best = (key, o.copy(), pts.copy(), v, it)
        if not bad.any():
            break
        grow = ndi.binary_dilation(bad, iterations=int(round(rc['repair_halfwindow_m'] / h)))
        margin = np.where(grow, margin + STEP, margin)
    _, o, pts, v, best_it = best
    # resample at ~station spacing (uniform, exact end points)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    n_new = int(round(arc[-1] / rc['station_spacing_m']))
    s_new = np.linspace(0, arc[-1], n_new + 1)
    xy_new = np.stack([np.interp(s_new, arc, pts[:, d]) for d in range(2)], -1)
    yaw_new = np.unwrap(np.arctan2(np.gradient(xy_new[:, 1]), np.gradient(xy_new[:, 0])))
    blocked_new = np.interp(s_new, arc, (path_viol > 0).astype(float)) > 0
    return dict(xy=xy_new, yaw=yaw_new, s=s_new, blocked=blocked_new, method=method, dp_connected=bool(connected),
                chosen_iteration=best_it, history=history, blocked_reference_stations=int((path_viol > 0).sum()))


def _project(line_xy, line_yaw, line_s, pts):
    """Signed lateral offset (+left) and station of pts relative to a reference polyline (nearest station)."""
    _, k = cKDTree(line_xy).query(pts)
    d = pts - line_xy[k]
    n = np.stack([-np.sin(line_yaw[k]), np.cos(line_yaw[k])], -1)
    t = np.stack([np.cos(line_yaw[k]), np.sin(line_yaw[k])], -1)
    return np.sum(d * n, 1), line_s[k] + np.sum(d * t, 1)


def build_route(support_sdf, obstacle_sdf, origin, cxy, cyaw, cs, cfg, log=lambda m: None):
    """Pass 1 on the scanner centreline (offsets within +-acceptance_band_m); passes >= 2 re-run the same
    procedure on the previous route (its own heading for the footprint tube), offsets limited to
    +-refine_halfwidth_m around it and to +-acceptance_band_m from the centreline."""
    rc = cfg['route']
    band = cfg['corridor']['acceptance_band_m']
    ref_xy, ref_yaw, ref_s = cxy, cyaw, cs
    lim_lo, lim_hi = np.full(len(cs), -band), np.full(len(cs), band)
    passes = []
    for p in range(rc['refine_passes']):
        log(f'route pass {p + 1}')
        r = _route_pass(support_sdf, obstacle_sdf, origin, ref_xy, ref_yaw, ref_s, cfg, lim_lo, lim_hi, log)
        passes.append(dict(method=r['method'], dp_connected=r['dp_connected'], chosen_iteration=r['chosen_iteration'],
                           blocked_reference_stations=r['blocked_reference_stations'], history=r['history']))
        ref_xy, ref_yaw, ref_s = r['xy'], r['yaw'], r['s']
        shift, _ = _project(cxy, cyaw, cs, ref_xy)
        lim_lo = np.maximum(-rc['refine_halfwidth_m'], -band - shift)
        lim_hi = np.minimum(rc['refine_halfwidth_m'], band - shift)
    shift, centre_s = _project(cxy, cyaw, cs, r['xy'])
    # offset of the route at each centreline station (where it crosses the centreline normal, approx.)
    o_c, _ = _project(r['xy'], r['yaw'], r['s'], cxy)
    R_new = turning_radius(r['xy'])
    return dict(xy=r['xy'], yaw=r['yaw'], s=r['s'], shift=shift, centre_s=centre_s, blocked=r['blocked'],
                offsets_at_centre=-o_c, method=r['method'], dp_connected=passes[0]['dp_connected'],
                min_radius_m=float(R_new.min()), radius_below_min=int((R_new < rc['min_turn_radius_m']).sum()),
                passes=passes, history=[h for q in passes for h in q['history']])


def route_distance(origin, shape, route_xy, workers=8):
    """Distance from every cell centre to the nearest route station and that station's index."""
    gx = origin[0] + np.arange(shape[0]) * RES
    gy = origin[1] + np.arange(shape[1]) * RES
    tree = cKDTree(route_xy)
    dist = np.empty(shape, np.float32)
    near = np.empty(shape, np.int32)
    for a in range(0, shape[0], 256):
        xx, yy = np.meshgrid(gx[a:a + 256], gy, indexing='ij')
        d, k = tree.query(np.stack([xx.ravel(), yy.ravel()], -1), workers=workers)
        dist[a:a + 256] = d.reshape(xx.shape)
        near[a:a + 256] = k.reshape(xx.shape)
    return dist, near


def station_monotonicity(station_layer, origin, route_xy, route_s, step=.05):
    """Sample the station layer along the route every `step` m: largest backwards step and the number of
    jumps > 1 m (a jump means the nearest-station assignment switched to another part of the route)."""
    from nav4.mapping.analysis import lookup
    seg = np.linalg.norm(np.diff(route_xy, axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    fs = np.arange(0, arc[-1], step)
    pts = np.stack([np.interp(fs, arc, route_xy[:, d]) for d in range(2)], -1)
    v = lookup(station_layer, origin, pts, np.nan)
    d = np.diff(v)
    at_st = lookup(station_layer, origin, route_xy, np.nan)
    return dict(samples=int(len(fs)), max_backward_step_m=float(max(0., -d.min())), jumps_over_1m=int((np.abs(d) > 1.).sum()),
                station_samples_monotone=bool(np.all(np.diff(at_st) >= -1e-6)),
                station_backward_steps=int((np.diff(at_st) < -1e-6).sum()),
                max_abs_error_vs_route_s_m=float(np.nanmax(np.abs(v - np.interp(fs, arc, route_s)))))
