"""Triangulate a centre-sampled height raster with optional vertical discontinuities."""

import numpy as np


# Cells around a corner, counter-clockwise, and corners around a cell.
CORNER_CELLS = np.array([[-1, -1], [0, -1], [0, 0], [-1, 0]])
CELL_CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def jump_edges(z, jump):
    """Detect a step in excess of the local one-sided slopes on both treads."""
    edges = []
    for a in (z, z.T):
        d = np.diff(a, axis=0)
        padded = np.pad(d, ((1, 1), (0, 0)), mode="edge")
        before, after = padded[:-2], padded[2:]
        sign = np.sign(d)
        background = np.maximum(np.maximum(before * sign, after * sign), 0)
        # A plane, including a 47-degree uphill plane, has zero residual.
        edges.append(np.abs(d) - background >= jump)
    return edges[0], edges[1].T


def corner_regions(z, active, jump):
    """At each corner, blend only cells reachable without crossing a riser."""
    bx, by = jump_edges(z, jump)
    h = np.pad(z, 1, mode="edge")
    m = np.pad(active, 1)
    heights = np.stack([h[:-1, :-1], h[1:, :-1], h[1:, 1:], h[:-1, 1:]], -1)
    present = np.stack([m[:-1, :-1], m[1:, :-1], m[1:, 1:], m[:-1, 1:]], -1)
    cross = np.stack([
        np.pad(bx, ((1, 1), (1, 0)), constant_values=True),
        np.pad(by, ((0, 1), (1, 1)), constant_values=True),
        np.pad(bx, ((1, 1), (0, 1)), constant_values=True),
        np.pad(by, ((1, 0), (1, 1)), constant_values=True),
    ], -1)
    link = ~cross & present & np.roll(present, -1, axis=-1)
    component = np.broadcast_to(np.arange(4, dtype=np.int8), heights.shape).copy()
    for _ in range(3):
        component = np.minimum(component, np.where(link, np.roll(component, -1, axis=-1), 4))
        component = np.minimum(component, np.where(np.roll(link, 1, axis=-1),
                                                    np.roll(component, 1, axis=-1), 4))
    same = component[..., :, None] == component[..., None, :]
    blended = (same * heights[..., None, :]).sum(-1) / same.sum(-1)
    return blended.astype(z.dtype), component, (bx, by)


def triangulate(z, active, select, res, jump, centre_tolerance):
    """Mesh selected cells plus risers on their +x/+y edges.

    A halo of active cells makes height and edge decisions agree across tiles.
    Coordinates are local to raster centre (0,0). Upper surfaces face up.
    """
    heights, region, jumps = corner_regions(z, active, jump)
    ij = np.argwhere(select)
    i, j = ij.T
    nx, ny = z.shape
    ncorner = (nx + 1) * (ny + 1) * 4
    corner_xyz = []
    corner_ids = []
    for k, (di, dj) in enumerate(CELL_CORNERS):
        quadrant = (k + 2) % 4
        coord_i, coord_j = i + di, j + dj
        rep = region[coord_i, coord_j, quadrant]
        corner_ids.append(4 * (coord_i * (ny + 1) + coord_j) + rep)
        corner_xyz.append(heights[coord_i, coord_j, quadrant])
    ids = np.stack(corner_ids, -1)
    hz = np.stack(corner_xyz, -1)
    diag_a = (hz[:, 0] + hz[:, 2]) / 2
    diag_b = (hz[:, 1] + hz[:, 3]) / 2
    error_a, error_b = abs(diag_a - z[i, j]), abs(diag_b - z[i, j])
    detail = np.minimum(error_a, error_b) > centre_tolerance
    a = (error_a <= error_b) & ~detail
    b = (error_a > error_b) & ~detail
    centre_ids = ncorner + i[detail] * ny + j[detail]
    faces = [
        ids[a][:, [0, 1, 2]], ids[a][:, [0, 2, 3]],
        ids[b][:, [0, 1, 3]], ids[b][:, [1, 2, 3]],
    ]
    for k in range(4):
        faces.append(np.column_stack([ids[detail, k], ids[detail, (k + 1) % 4], centre_ids]))
    riser_edges = []
    for axis, (edge, opposite) in enumerate((((1, 2), (0, 3)), ((2, 3), (1, 0)))):
        di, dj = AXIS_STEP[axis]
        low = (i + di < nx) & (j + dj < ny)
        ii, jj = i[low], j[low]
        hit = active[ii + di, jj + dj] & jumps[axis][ii, jj]
        ii, jj = ii[hit], jj[hit]
        corners = ij[low][hit, None, :] + CELL_CORNERS[list(edge)]
        ci, cj = corners[..., 0], corners[..., 1]
        q0, q1 = (np.array(edge) + 2) % 4, (np.array(opposite) + 2) % 4
        id0 = 4 * (ci * (ny + 1) + cj) + region[ci, cj, q0]
        id1 = 4 * (ci * (ny + 1) + cj) + region[ci, cj, q1]
        # Reverse the upper surface's edge winding to face the lower tread.
        owner_high = (z[ii, jj] > z[ii + di, jj + dj])[:, None]
        upper, lower = np.where(owner_high, id0, id1), np.where(owner_high, id1, id0)
        upper = np.where(owner_high, upper, upper[:, ::-1])
        lower = np.where(owner_high, lower, lower[:, ::-1])
        for k, f in enumerate((np.c_[upper[:, 1], upper[:, 0], lower[:, 0]],
                               np.c_[upper[:, 1], lower[:, 0], lower[:, 1]])):
            valid = upper[:, k] != lower[:, k]
            faces.append(f[valid])
        riser_edges.append(np.c_[ii, jj, np.full(len(ii), axis)])
    unique, faces = np.unique(np.concatenate(faces).ravel(), return_inverse=True)
    vc, vf = unique[unique < ncorner], unique[unique >= ncorner] - ncorner
    ci, cj = np.divmod(vc // 4, ny + 1)
    vi, vj = np.divmod(vf, ny)
    vertices = np.vstack([np.c_[ci - .5, cj - .5, heights[ci, cj, vc % 4]],
                          np.c_[vi, vj, z[vi, vj]]])
    vertices[:, :2] *= res
    return vertices, faces.reshape(-1, 3), np.concatenate(riser_edges)


AXIS_STEP = ((1, 0), (0, 1))
