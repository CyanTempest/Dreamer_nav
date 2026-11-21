"""Conservative refinement of occupied surface cells, without changing frames."""
import numpy as np


def integer_bounds(boxes, resolution):
    edges = np.c_[boxes[:, :3] - boxes[:, 3:], boxes[:, :3] + boxes[:, 3:]]
    bounds = np.rint(edges / resolution).astype(np.int64)
    if not np.allclose(bounds * resolution, edges, atol=1e-7, rtol=0):
        raise ValueError('Parent boxes do not align with the refinement grid')
    if np.any(bounds[:, 3:] <= bounds[:, :3]):
        raise ValueError('Degenerate parent box')
    return bounds


def child_cells(boxes, resolution):
    parts = []
    for bound in integer_bounds(boxes, resolution):
        lower, upper = bound[:3], bound[3:]
        cells = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lower, upper)),
                                    indexing='ij'), -1).reshape(-1, 3)
        parts.append(cells)
    return np.unique(np.concatenate(parts), axis=0) if parts else np.empty((0, 3), np.int64)


def conservative_surface_mask(distances, resolution):
    distances = np.asarray(distances)
    if resolution <= 0 or np.any(~np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError('Finite nonnegative distances and a positive resolution required')
    # Every cube intersected by a source triangle/point has its centre at most
    # half a cube diagonal from that source. Retaining this superset cannot cut
    # away a measured surface inside the original occupied envelope.
    return distances <= np.sqrt(3.) * resolution / 2 + 1e-9


def merge_cells(cells, resolution):
    bounds = np.c_[cells, cells + 1]
    for _ in range(3):
        before = len(bounds)
        for axis in (0, 1, 2):
            others = [i for i in range(6) if i not in (axis, axis + 3)]
            groups = {}
            for box in bounds:
                groups.setdefault(tuple(box[others]), []).append(box)
            merged = []
            for group in groups.values():
                group.sort(key=lambda b: b[axis]); current = group[0].copy()
                for nxt in group[1:]:
                    if current[axis + 3] == nxt[axis]:
                        current[axis + 3] = nxt[axis + 3]
                    else:
                        merged.append(current); current = nxt.copy()
                merged.append(current)
            bounds = np.asarray(merged, np.int64).reshape(-1, 6)
        if len(bounds) == before:
            break
    if np.prod(bounds[:, 3:] - bounds[:, :3], axis=1).sum() != len(cells):
        raise AssertionError('Merging changed the occupied cell volume')
    return np.c_[(bounds[:, :3] + bounds[:, 3:]) * resolution / 2,
                 (bounds[:, 3:] - bounds[:, :3]) * resolution / 2]
