"""Dense map grid over segment A and fast per-cell reductions on the cropped cloud."""
import json

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import DATA, RES

CROP = DATA / 'cache' / 'crop'


def load_centreline():
    with np.load(DATA / 'segment' / 'centreline.npz') as f:
        return {k: f[k] for k in f.files}


def make_grid(route_xy, margin, coverage_radius, workers=8):
    """Origin/shape exactly like build_scan_patches (floor/ceil to RES, dims rounded to 32)."""
    lo = np.floor((route_xy.min(0) - margin) / RES) * RES
    hi = np.ceil((route_xy.max(0) + margin) / RES) * RES
    dims = (np.ceil((hi - lo) / RES).astype(int) + 1 + 31) // 32 * 32
    shape = tuple(int(d) for d in dims)
    gx = lo[0] + np.arange(shape[0]) * RES
    gy = lo[1] + np.arange(shape[1]) * RES
    tree = cKDTree(route_xy)
    dist = np.empty(shape, np.float32)
    near = np.empty(shape, np.int32)
    for a in range(0, shape[0], 256):
        xx, yy = np.meshgrid(gx[a:a + 256], gy, indexing='ij')
        d, k = tree.query(np.stack([xx.ravel(), yy.ravel()], -1), workers=workers)
        dist[a:a + 256] = d.reshape(xx.shape)
        near[a:a + 256] = k.reshape(xx.shape)
    return dict(origin=lo.astype(np.float64), shape=shape, dist=dist, near=near,
                coverage=dist <= coverage_radius)


def load_points(fields=('x', 'y', 'z', 'nz', 'cls'), mmap=True):
    return {k: np.load(CROP / f'{k}.npy', mmap_mode='r' if mmap else None) for k in fields}


class CellIndex:
    """Points sorted by (cell, z).  Subset reductions keep that order, so the
    per-cell min/max of any subset is the first/last element of its group."""

    def __init__(self, x, y, z, origin, shape):
        ij0 = np.floor((np.asarray(x, np.float64) - origin[0]) / RES + .5).astype(np.int64)
        ij1 = np.floor((np.asarray(y, np.float64) - origin[1]) / RES + .5).astype(np.int64)
        inside = (ij0 >= 0) & (ij0 < shape[0]) & (ij1 >= 0) & (ij1 < shape[1])
        idx = np.flatnonzero(inside)
        flat = ij0[idx] * shape[1] + ij1[idx]
        del ij0, ij1
        zz = np.asarray(z)[idx]
        order = np.lexsort((zz, flat))
        self.index = idx[order]          # original point index, sorted by (cell, z)
        self.flat = flat[order]
        self.z = zz[order].astype(np.float32)
        self.shape = shape
        self.size = shape[0] * shape[1]

    def take(self, arr):
        return np.asarray(arr)[self.index]

    def count(self, mask=None):
        flat = self.flat if mask is None else self.flat[mask]
        return np.bincount(flat, minlength=self.size).reshape(self.shape)

    def first(self, values, mask=None, fill=np.nan):
        """Per-cell value of the lowest-z point of the subset."""
        flat = self.flat if mask is None else self.flat[mask]
        vals = values if mask is None else values[mask]
        out = np.full(self.size, fill, np.float32)
        if len(flat):
            start = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
            out[flat[start]] = vals[start]
        return out.reshape(self.shape)

    def last(self, values, mask=None, fill=np.nan):
        flat = self.flat if mask is None else self.flat[mask]
        vals = values if mask is None else values[mask]
        out = np.full(self.size, fill, np.float32)
        if len(flat):
            end = np.r_[np.flatnonzero(np.diff(flat)), len(flat) - 1]
            out[flat[end]] = vals[end]
        return out.reshape(self.shape)

    def sum(self, values, mask=None):
        flat = self.flat if mask is None else self.flat[mask]
        vals = values if mask is None else values[mask]
        return np.bincount(flat, weights=vals, minlength=self.size).reshape(self.shape)


def band_atlas(cells, ground, known):
    """build_scan_patches atlas: ch0 ground, 1+3b bottom, 2+3b top, 3+3b coverage, ch19 known."""
    from nav4.mapping.common import BANDS
    shape = cells.shape
    relative = cells.z - ground.reshape(-1)[cells.flat]
    atlas = np.zeros((*shape, 20), np.float32)
    atlas[..., 0] = ground
    atlas[..., 19] = known
    for b, (lower, upper) in enumerate(BANDS):
        m = (relative >= lower) & (relative < upper)
        cnt = cells.count(m)
        bottom = cells.first(relative, m, fill=np.inf)
        top = cells.last(relative, m, fill=-1)
        atlas[..., 1 + 3 * b] = np.where(cnt > 0, bottom, 0)
        atlas[..., 2 + 3 * b] = top
        atlas[..., 3 + 3 * b] = cnt > 0
    return atlas, relative


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')
