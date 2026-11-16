"""Route-level analysis shared by the old-proxy diagnosis and the acceptance report."""
import numpy as np

from nav4.mapping.common import RES, footprint_ok, footprint_world, swept_poses

ROBOT_W = .58
ROBOT_L = 1.03


def lookup(grid2d, origin, xy, fill):
    ij = np.floor((np.asarray(xy) - origin) / RES + .5).astype(np.int64)
    sh = grid2d.shape
    inside = (ij[..., 0] >= 0) & (ij[..., 0] < sh[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < sh[1])
    out = np.full(ij.shape[:-1], fill, dtype=np.result_type(grid2d.dtype, type(fill)))
    out[inside] = grid2d[ij[..., 0][inside], ij[..., 1][inside]]
    return out


def lateral_grid(route_xy, yaw, offsets):
    normal = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    return route_xy[:, None, :] + normal[:, None, :] * offsets[None, :, None]


def lateral_feasibility(support_sdf, obstacle_sdf, origin, route_xy, yaw, offsets):
    cand = lateral_grid(route_xy, yaw, offsets)
    ang = np.broadcast_to(yaw[:, None], cand.shape[:2]).reshape(-1)
    ok = footprint_ok(support_sdf, obstacle_sdf, origin, cand.reshape(-1, 2), ang)
    return ok.reshape(cand.shape[:2])


def feasible_width(feas, offsets, centre_band=.75):
    """Width of the contiguous run of feasible centre offsets that overlaps |o|<=centre_band,
    plus the robot width (the physical clear lane the robot can use).  0 if none."""
    step = offsets[1] - offsets[0]
    out = np.zeros(len(feas))
    near = np.abs(offsets) <= centre_band + 1e-9
    for i, row in enumerate(feas):
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        best = 0.
        for a, b in zip(edges[::2], edges[1::2]):
            if near[a:b].any():
                best = max(best, (b - a - 1) * step + ROBOT_W)
        out[i] = best
    return out


def corridor_path(feas, offsets, max_shift_cells=2, centre_band=.75):
    """DP over (station, offset) nodes with |offset|<=centre_band.  Edges between
    consecutive stations change the offset by <=max_shift_cells.  Returns
    (connected, reach_fraction, path_offsets or None, last_reached_station)."""
    use = np.abs(offsets) <= centre_band + 1e-9
    f = feas[:, use]
    off = offsets[use]
    n, k = f.shape
    reach = np.zeros((n, k), bool)
    parent = np.full((n, k), -1, np.int32)
    cost = np.full((n, k), np.inf)
    reach[0] = f[0]
    cost[0] = np.where(f[0], np.abs(off), np.inf)
    last = 0 if f[0].any() else -1
    for i in range(1, n):
        prev = cost[i - 1]
        best = np.full(k, np.inf)
        arg = np.full(k, -1)
        for d in range(-max_shift_cells, max_shift_cells + 1):
            shifted = np.full(k, np.inf)
            src = np.arange(k) - d
            ok = (src >= 0) & (src < k)
            shifted[ok] = prev[src[ok]] + abs(d) * .05
            better = shifted < best
            best[better] = shifted[better]
            arg[better] = src[better]
        best = np.where(f[i], best + np.abs(off) * .01, np.inf)
        cost[i] = best
        parent[i] = arg
        reach[i] = np.isfinite(best)
        if reach[i].any():
            last = i
        else:
            break
    connected = bool(reach[-1].any())
    path = None
    if connected:
        j = int(np.argmin(cost[-1]))
        idx = [j]
        for i in range(n - 1, 0, -1):
            j = int(parent[i, j])
            idx.append(j)
        path = off[np.array(idx[::-1])]
    return connected, (last + 1) / n, path, last


def blocking_runs(ok, s, xy, min_len=1):
    """Contiguous runs of False -> list of dicts with station range and coordinates."""
    runs = []
    edges = np.flatnonzero(np.diff(np.r_[0, (~ok).astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_len:
            mid = (a + b - 1) // 2
            runs.append(dict(first_station=int(a), last_station=int(b - 1), s_from_m=float(s[a]), s_to_m=float(s[b - 1]),
                             length_m=float(s[b - 1] - s[a] + .2), xy_mid=[float(xy[mid, 0]), float(xy[mid, 1])]))
    return runs


def validate_path(support_sdf, obstacle_sdf, origin, route_xy, yaw, path_offsets):
    """Fine (5 cm) footprint + swept check along the extracted DP path, with its own heading."""
    normal = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    pts = route_xy + normal * path_offsets[:, None]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    fs = np.arange(0, arc[-1], .05)
    fine = np.stack([np.interp(fs, arc, pts[:, d]) for d in range(2)], -1)
    fyaw = np.unwrap(np.arctan2(np.gradient(fine[:, 1]), np.gradient(fine[:, 0])))
    good = footprint_ok(support_sdf, obstacle_sdf, origin, fine, fyaw)
    sp, sa = swept_poses(fine, fyaw)
    swept = footprint_ok(support_sdf, obstacle_sdf, origin, sp, sa)
    return dict(fine_pass_fraction=float(good.mean()), swept_pass_fraction=float(swept.mean()),
                fine_fail_s=fs[~good].tolist()[:200], swept_ok=bool(swept.all() and good.all()),
                path_xy=pts, fine_ok=good, fine_s=fs)


def footprint_cells(origin, shape, xy, yaw):
    world = footprint_world(xy, yaw)
    ij = np.floor((world - origin) / RES + .5).astype(np.int64)
    inside = (ij[..., 0] >= 0) & (ij[..., 0] < shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < shape[1])
    ij[..., 0] = ij[..., 0].clip(0, shape[0] - 1)
    ij[..., 1] = ij[..., 1].clip(0, shape[1] - 1)
    return ij, inside
