"""Ground surface for segment A: lowest coherent layer per cell, outlier tests,
connectivity from the walked path, walked-path prior (scanner z - device height).

All thresholds come from the config section 'ground' (see trail_map_config.json).
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from nav4.mapping.common import RES


def lowest_layer(cells, cand, thickness, k_min):
    """Per cell: z of the lowest candidate point that has >= k_min candidate points
    within [z, z+thickness] in the same cell (rejects isolated low noise), plus
    count / z-spread / median |nz| of the points in that layer."""
    flat = cells.flat[cand]
    z = cells.z[cand]
    zmm = np.round((z.astype(np.float64) + 50.) * 1000.).astype(np.int64)
    key = flat * (1 << 18) + zmm
    upper = np.searchsorted(key, key + int(round(thickness * 1000)), side='right')
    count = upper - np.arange(len(key))
    ok = count >= k_min
    size = cells.size
    flat_ok = flat[ok]
    z_low = np.full(size, np.nan, np.float32)
    n_low = np.zeros(size, np.int32)
    if ok.any():
        start = np.r_[0, np.flatnonzero(np.diff(flat_ok)) + 1]
        z_low[flat_ok[start]] = z[ok][start]
        n_low[flat_ok[start]] = count[ok][start]
    return z_low.reshape(cells.shape), n_low.reshape(cells.shape)


def layer_stats(cells, cand, z_low, thickness, absnz):
    """Spread (std of z) and mean |nz| of the candidate points in the lowest layer."""
    zl = z_low.reshape(-1)[cells.flat]
    member = cand & np.isfinite(zl) & (cells.z >= zl - 1e-4) & (cells.z <= zl + thickness)
    n = cells.count(member).astype(np.float64)
    s1 = cells.sum(cells.z.astype(np.float64) - np.nan_to_num(zl), member)
    s2 = cells.sum((cells.z.astype(np.float64) - np.nan_to_num(zl)) ** 2, member)
    snz = cells.sum(absnz.astype(np.float64), member)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s1 / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        mnz = snz / n
    return std.astype(np.float32), mnz.astype(np.float32)


def nan_median(values, size):
    """Median over a size x size window ignoring NaN (via nearest fill of NaNs first)."""
    valid = np.isfinite(values)
    if not valid.any():
        return values.copy()
    _, ind = ndi.distance_transform_edt(~valid, return_indices=True)
    filled = values[tuple(ind)]
    return ndi.median_filter(filled, size=size)


def connected_from_seeds(z, valid, seeds, max_dz):
    """Cells reachable from seed cells through 8-neighbour steps with |dz|<=max_dz."""
    shape = z.shape
    idx = np.arange(z.size).reshape(shape)
    rows, cols = [], []
    H, W = shape
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        c0, c1 = max(0, -dj), W - max(0, dj)
        a = (slice(0, H - di), slice(c0, c1))
        b = (slice(di, H), slice(c0 + dj, c1 + dj))
        ok = valid[a] & valid[b] & (np.abs(z[a] - z[b]) <= max_dz)
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(z.size, z.size)).tocsr()
    _, labels = connected_components(g, directed=False)
    labels = labels.reshape(shape)
    seed_labels = np.unique(labels[seeds & valid])
    return np.isin(labels, seed_labels) & valid


def connected_two_layer(za, va, zb, vb, seeds_a, seeds_b, max_dz):
    """Connectivity on a two-layer graph (node = cell x layer); edges between 8-neighbour
    nodes of any layer pair with |dz| <= max_dz.  Returns (conn_a, conn_b)."""
    shape = za.shape
    n = za.size
    idx = np.arange(n).reshape(shape)
    H, W = shape
    rows, cols = [], []
    layers = ((za, va, 0), (zb, vb, n))
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        c0, c1 = max(0, -dj), W - max(0, dj)
        a = (slice(0, H - di), slice(c0, c1))
        b = (slice(di, H), slice(c0 + dj, c1 + dj))
        for z1, v1, o1 in layers:
            for z2, v2, o2 in layers:
                ok = v1[a] & v2[b] & (np.abs(z1[a] - z2[b]) <= max_dz)
                rows.append(idx[a][ok] + o1)
                cols.append(idx[b][ok] + o2)
    # same cell, both layers: never connected (they are >= gap apart by construction)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(2 * n, 2 * n)).tocsr()
    _, labels = connected_components(g, directed=False)
    seeds = np.r_[(seeds_a & va).ravel(), (seeds_b & vb).ravel()]
    keep = np.isin(labels, np.unique(labels[seeds]))
    return keep[:n].reshape(shape) & va, keep[n:].reshape(shape) & vb


def walked_cells(origin, shape, xy, radius):
    """Boolean mask of cells whose centre lies within radius of any walked pose."""
    mask = np.zeros(shape, bool)
    ij = np.floor((xy - origin) / RES + .5).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    mask[ij[ok, 0], ij[ok, 1]] = True
    r = int(np.ceil(radius / RES))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ndi.binary_dilation(mask, structure=(xx ** 2 + yy ** 2) * RES ** 2 <= radius ** 2 + 1e-9)


def polyline_distance_mask(origin, shape, xy, radius):
    """Cells within radius of the polyline through xy (dense resampling at RES/2)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    pts = [xy[:1]]
    for a, b, L in zip(xy[:-1], xy[1:], seg):
        if L > RES / 2:
            u = np.arange(1, int(np.ceil(L / (RES / 2))) + 1) / np.ceil(L / (RES / 2))
            pts.append(a + u[:, None] * (b - a))
        else:
            pts.append(b[None])
    dense = np.concatenate(pts)
    seedmask = np.zeros(shape, bool)
    ij = np.floor((dense - origin) / RES + .5).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    seedmask[ij[ok, 0], ij[ok, 1]] = True
    dist = ndi.distance_transform_edt(~seedmask) * RES
    return dist <= radius, dist


def device_height(z_low, origin, poses_xyz, radius):
    """Per pose: scanner z minus median observed lowest-layer ground within radius."""
    shape = z_low.shape
    r = int(np.ceil(radius / RES))
    offs = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if (a * a + b * b) * RES ** 2 <= radius ** 2 + 1e-9]
    ij = np.floor((poses_xyz[:, :2] - origin) / RES + .5).astype(np.int64)
    vals = []
    for a, b in offs:
        i = np.clip(ij[:, 0] + a, 0, shape[0] - 1)
        j = np.clip(ij[:, 1] + b, 0, shape[1] - 1)
        vals.append(z_low[i, j])
    vals = np.stack(vals, 1)
    with np.errstate(all='ignore'):
        g = np.nanmedian(vals, 1)
    return poses_xyz[:, 2] - g, g
