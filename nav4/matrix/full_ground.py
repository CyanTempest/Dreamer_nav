"""Use a shared XY lattice for observed ground, marked fills and segment seams."""
from dataclasses import dataclass
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import bilinear, write_json
from nav4.matrix.full_scan import FULL, WORK, TILE, files, read_tiles, route
from nav4.matrix.register import voxel_ids
from nav4.matrix.survey import Corridor, V8

PITCH = .05
MAX_AXIS_GRADE = 2.30
LOCAL_SUPPORT_M = .15
PILOT = FULL / "pilot_0_30_rebuild"


@dataclass
class V8Map:
    layers: np.ndarray
    origin: np.ndarray
    resolution: float

    @classmethod
    def load(cls):
        with np.load(V8 / "map.npz") as f:
            return cls(f["layers"], f["origin"].astype(float), float(f["res"]))

    def z(self, xy):
        return bilinear(self.layers[..., 0], self.origin, xy, res=self.resolution)

    def classes(self, xy):
        ij = np.rint((xy - self.origin) / self.resolution).astype(np.int32)
        valid = ((ij >= 0) & (ij < self.layers.shape[:2])).all(1)
        i, j = ij.clip(0, np.array(self.layers.shape[:2]) - 1).T
        return np.where(valid, self.layers[i, j, 6], 0).astype(np.uint8)


def candidate_returns(raw, ply, v8):
    """Admit dense returns near a PLY ground patch; hold-out flags travel unchanged."""
    prior = v8.z(ply[:, :2])
    p = ply[(abs(ply[:, 2] - prior) < .36) & (abs(ply[:, 5]) > .62), :3]
    if not len(p) or not len(raw):
        return raw[:0], np.empty(0)
    p = p[voxel_ids(p, .024)]
    possible = np.flatnonzero(abs(raw[:, 2] - v8.z(raw[:, :2])) < .48)
    d, j = cKDTree(p).query(raw[possible, :3], distance_upper_bound=.095, workers=4)
    keep = np.isfinite(d)
    match = raw[possible[keep]]
    dz = np.abs(match[:, 2] - p[j[keep], 2])
    return match, dz


def median_cells(p, base, shape):
    ij = np.rint((p[:, :2] - base) / PITCH).astype(np.int32)
    use = ((ij >= 0) & (ij < shape)).all(1)
    key = np.ravel_multi_index(ij[use].T, shape)
    z = p[use, 2]
    order = np.lexsort((z, key))
    u, first, n = np.unique(key[order], return_index=True, return_counts=True)
    sorted_z = z[order]
    return u, (sorted_z[first + (n - 1) // 2] + sorted_z[first + n // 2]) * .5


def bounded_grades(z, pitch=PITCH, grade=MAX_AXIS_GRADE):
    """Average the lower and upper Lipschitz envelopes on the grid.

    The lower envelope is the highest surface below all observations
    satisfying the axis grade limit. The upper envelope is its dual.
    """
    low, high = envelopes(z, pitch, grade)
    return (low + high) * .5


def envelopes(z, pitch=PITCH, grade=MAX_AXIS_GRADE, anchors=None):
    low, high = z.copy(), -z.copy()
    if anchors is not None:
        low[~anchors] = np.inf
        high[~anchors] = np.inf
    for axis, n in enumerate(z.shape):
        shape = [-1, 1] if axis == 0 else [1, -1]
        delta = (np.arange(n, dtype=np.float32) * (pitch * grade)).reshape(shape)
        for value in (low, high):
            value[:] = np.minimum.accumulate(value - delta, axis=axis) + delta
            reverse = np.flip(value, axis=axis)
            reverse[:] = np.minimum.accumulate(reverse - delta, axis=axis) + delta
    return -high, low


def fill_patch(observed, prior):
    present = np.isfinite(observed)
    smooth_prior = ndi.gaussian_filter(prior, .40 / PITCH)
    if not present.any():
        return smooth_prior, np.full(observed.shape, np.inf), 0
    distance, near = ndi.distance_transform_edt(~present, return_indices=True)
    near_z = observed[tuple(near)]
    median = ndi.median_filter(near_z, size=3)
    support = ndi.convolve(present.astype(np.uint8), np.ones((3, 3), np.uint8))
    rejected = present & (support >= 5) & (abs(observed - median) > .10)
    present &= ~rejected
    distance, near = ndi.distance_transform_edt(~present, return_indices=True)
    gap_distance = distance * PITCH
    stable = np.where(present, observed, 0.)
    sigma = .34 / PITCH
    evidence = ndi.gaussian_filter(present.astype(np.float32), sigma)
    residual = np.where(present, stable - smooth_prior, 0.)
    correction = ndi.gaussian_filter(residual, sigma) / np.maximum(evidence, .006)
    large_fill = smooth_prior + correction
    # Let measured edges influence the hole boundary, then decay into the prior.
    w = np.clip(1. - gap_distance / .35, 0, 1) ** 2
    near_z = np.nan_to_num(observed[tuple(near)])
    filled = (1 - w) * large_fill + w * near_z
    return np.where(present, observed, filled).astype(np.float32), gap_distance, int(rejected.sum())


def lattice(path):
    lower = np.floor((path.min(0) - 6.7) / PITCH).astype(int)
    upper = np.ceil((path.max(0) + 6.7) / PITCH).astype(int)
    return lower * PITCH, tuple(upper - lower + 1)


def grid_points(base, lo, hi):
    return (np.indices(hi - lo, dtype=np.float32).reshape(2, -1).T + lo) * PITCH + base


def patches(shape, size):
    for i in range(0, shape[0], size):
        for j in range(0, shape[1], size):
            lo = np.array([i, j])
            hi = np.minimum(lo + size, shape)
            yield lo, hi


def region(lo, hi):
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def grid_geometry(z, offset):
    """A single winding across all tiles. No raster discontinuity faces."""
    n, m = z.shape
    xy = np.indices(z.shape).reshape(2, -1).T * PITCH + offset
    v = np.c_[xy, z.ravel()]
    i, j = np.indices((n - 1, m - 1)).reshape(2, -1)
    a = i * m + j
    b, c, d = a + m, a + m + 1, a + 1
    flip = abs(z[:-1, :-1] - z[1:, 1:]) > abs(z[1:, :-1] - z[:-1, 1:])
    flip = flip.ravel()
    f = np.stack([np.c_[a, b, np.where(flip, d, c)],
                  np.c_[np.where(flip, b, a), c, d]], axis=1)
    return v, f


def grid_sample(z, base, xy):
    """Interpolate the same diagonal used in the actual height mesh."""
    ij = np.floor((xy - base) / PITCH).astype(int)
    valid = ((ij >= 0) & (ij < np.array(z.shape) - 1)).all(1)
    p = (xy - base) / PITCH - ij
    x, y = p.T
    i, j = ij.clip(0, np.array(z.shape) - 2).T
    a, b, c, d = z[i, j], z[i+1, j], z[i+1, j+1], z[i, j+1]
    flip = abs(a - c) > abs(b - d)
    ac = np.where(y <= x, a + x*(b-a) + y*(c-b), a + y*(d-a) + x*(c-d))
    bd = np.where(x+y <= 1, a + x*(b-a) + y*(d-a), c + (1-x)*(d-c) + (1-y)*(b-c))
    return np.where(valid, np.where(flip, bd, ac), np.nan)


def source_patch(low, high):
    for name, width in (("las", 4), ("ply", 7)):
        arr = read_tiles(files(WORK / name, low, high), width)
        mask = ((arr[:, :2] >= low) & (arr[:, :2] < high)).all(1)
        yield arr[mask]


def edge_statistics(vertices, faces):
    xyz = vertices[faces.reshape(-1, 3)]
    cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    area = np.linalg.norm(cross, axis=1)
    near_vertical = (area > 1e-10) & (abs(cross[:, 2]) < area * np.cos(np.deg2rad(75)))
    return dict(faces=len(xyz), near_vertical=int(near_vertical.sum()),
                near_vertical_area_m2=float(area[near_vertical].sum() / 2),
                surface_m2=float(area.sum() / 2))


def make_ground(path, v8, pilot_source=None):
    start = time.monotonic()
    base, shape = lattice(path)
    obs = np.full(shape, np.nan, np.float32)
    samples = []
    corridor = Corridor(path, width=7.25)
    step = round(TILE / PITCH)
    initial_checks = cKDTree(path[::4])
    path_tree = cKDTree(path)

    for lo, hi in patches(shape, step):
        lower = base + (lo - 1) * PITCH
        upper = base + (hi + 1) * PITCH
        if initial_checks.query((lower + upper) / 2)[0] > 7.3 + np.linalg.norm(upper - lower) / 2:
            continue
        if pilot_source is None:
            raw, ply = source_patch(lower - .13, upper + .13)
            # Compute ownership from the rounding convention used by median_cells.
            ij = np.rint((raw[:, :2] - base) / PITCH).astype(np.int32)
            owned = ((ij >= lo) & (ij < hi)).all(1)
            raw = raw[owned]
            match, dz = candidate_returns(raw, ply, v8)
        else:
            raw, ply = pilot_source
            ij = np.rint((raw[:, :2] - base) / PITCH).astype(np.int32)
            raw = raw[((ij >= lo) & (ij < hi)).all(1)]
            ply = ply[((ply[:, :2] >= lower - .13) & (ply[:, :2] < upper + .13)).all(1)]
            match, dz = candidate_returns(raw, ply, v8)
        train = (match[:, 3] == 0) & (dz < .055)
        indexes, z = median_cells(match[train], base + lo * PITCH, hi - lo)
        own = np.unravel_index(indexes, hi - lo)
        obs[own[0] + lo[0], own[1] + lo[1]] = z
        samples.append(match[match[:, 3] > 0, :3])
    print("ground observations", int(np.isfinite(obs).sum()), "seconds", time.monotonic()-start, flush=True)

    filled = np.zeros(shape, np.float32)
    gap = np.full(shape, np.inf, np.float32)
    active = np.zeros(shape, bool)
    station = np.full(shape, -1., np.float32)
    rejects = 0
    for lo, hi in patches(shape, step):
        lower, upper = base + lo * PITCH, base + hi * PITCH
        if initial_checks.query((lower + upper) / 2)[0] > 6.3 + np.linalg.norm(upper - lower) / 2:
            continue
        halo = round(2.0 / PITCH)
        l, h = np.maximum(lo - halo, 0), np.minimum(hi + halo, shape)
        xy = grid_points(base, l, h)
        prior = v8.z(xy).astype(np.float32).reshape(h - l)
        # v8 is only the low-frequency prior. Out-of-grid raster cells may
        # occur in the padding, and must not create a 100 m perimeter wall.
        prior[prior > 85] = np.nan
        valid = np.isfinite(prior)
        if not valid.all():
            if valid.any():
                inds = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
                prior = prior[tuple(inds)]
            else:
                raise ValueError(f"No ground prior in patch at {lower}")
        local, dist, removed = fill_patch(obs[region(l, h)], prior)
        own = region(lo - l, hi - l)
        filled[region(lo, hi)] = local[own]
        gap[region(lo, hi)] = dist[own]
        cxy = grid_points(base, lo, hi) + PITCH / 2
        s, lateral, keep = corridor.project(cxy)
        # Corridor uses its true Euclidean width plus flat-ended clipping.
        use = keep & (np.abs(lateral) <= 6) & (path_tree.query(cxy, workers=4)[0] <= 6.00006)
        active[region(lo, hi)] = use.reshape(hi - lo)
        station[region(lo, hi)] = s.reshape(hi - lo)
        rejects += removed
    return base, filled, gap, active, station, np.concatenate(samples), rejects


def apply_grades(z, active, anchors=None):
    """Pad outside the modeled corridor by a smooth continuation before grading."""
    # Only local neighborhoods can influence a face; run both envelopes over
    # the full lattice to match the two sides of every chunk boundary exactly.
    pad = ndi.maximum_filter(active, size=11)
    nearest = ndi.distance_transform_edt(~pad, return_distances=False, return_indices=True)
    extended = z[tuple(nearest)]
    del nearest
    safe = bounded_grades(extended)
    if anchors is not None:
        lower, upper = envelopes(extended, anchors=anchors)
        # The midpoint of the two bounds forms a mutually feasible set of
        # route anchors. Recompute bounds from those consistent heights.
        central = (lower + upper) * .5
        del lower, upper
        lower, upper = envelopes(central, anchors=anchors)
        safe = np.minimum(np.maximum(safe, lower), upper)
    return safe


def route_anchors(path, origin, active, gap):
    keep = np.argwhere(active & (gap <= .10))
    ds = cKDTree(path).query(origin + keep*PITCH, workers=4)[0]
    anchors = np.zeros(active.shape, bool)
    anchors[tuple(keep[ds < .78].T)] = True
    return anchors
