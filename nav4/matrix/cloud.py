"""Sample the original planning PLY and fit collision shapes in v8 hard cells."""

from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

from nav4.mapping.common import cell_index, open_ply
from nav4.mapping.grid import CellIndex
from nav4.mapping.model import collision_returns, lowest_layer
from nav4.matrix.compose import write_obj


@dataclass
class Scan:
    body: np.ndarray
    continuation: np.ndarray
    foliage: np.ndarray
    ground: np.ndarray
    source_points: int


def voxel_first(points, size, group=None):
    if not len(points):
        return points
    keys = np.floor(points[:, :3] / size).astype(np.int32)
    if group is not None:
        keys = np.column_stack((keys, group))
    _, ids = np.unique(keys, axis=0, return_index=True)
    return points[ids]


def scan_source(path, layers, aux, active, origin, res, cfg, map_cfg):
    z = layers[..., 0]
    hard = (layers[..., 6] == 1) & active
    visible = layers[..., 6] == 2
    known = aux["known"]
    gmax = ndi.maximum_filter(np.where(known, z, -np.inf), 3)
    gmax = np.where(np.isfinite(gmax), gmax, z)
    cloud = open_ply(path)
    body, continuation, foliage, ground = [], [], [], []
    cfg_cloud = cfg["cloud"]
    chunk = cfg_cloud["read_chunk"]
    for start in range(0, len(cloud), chunk):
        block = cloud[start:start + chunk]
        xy = np.column_stack((block["x"], block["y"]))
        ij = cell_index(xy, origin, res)
        i, j = ij.T
        valid = (i >= 0) & (j >= 0) & (i < z.shape[0]) & (j < z.shape[1])
        idx = np.flatnonzero(valid)
        idx = idx[active[i[idx], j[idx]]]
        i, j = i[idx], j[idx]
        p = np.column_stack((xy[idx], block["z"][idx]))
        rel = p[:, 2] - z[i, j]
        terrain_rel = p[:, 2] - gmax[i, j]
        hits = hard[i, j] & collision_returns(rel, terrain_rel, map_cfg)
        label = block["classification"][idx]
        body.append(voxel_first(np.c_[p[hits], label[hits]], cfg_cloud["voxel_m"],
                                label[hits]))
        tail = hard[i, j] & ~hits & (rel >= .04) & (rel < cfg["hard"]["max_extra_height_m"])
        continuation.append(voxel_first(np.c_[p[tail], label[tail]], cfg_cloud["voxel_m"] * 1.5))
        near = aux["route_dist"][i, j]
        soft = (visible[i, j] & (rel > .10) & (rel < .8)) | (
            (rel >= .8) & (rel < 8) & (terrain_rel > .6) & (near > .75))
        kind = np.where(rel < .4, 0, np.where(rel < 1.1, 1, 2))
        foliage.append(voxel_first(np.c_[p[soft], kind[soft]], cfg_cloud["vegetation_voxel_m"]))
        # Ground samples are selected by raw normals and within-cell vertical coherence.
        # For the mapped overpass the desired surface is the identified bridge deck.
        deck = aux["deck"][i, j]
        raw_ground = (near <= 1) & aux["observed"][i, j] & (
            np.abs(block["normal_z"][idx]) >= cfg_cloud["ground_normal_min"])
        raw_ground &= ~deck | (np.abs(p[:, 2] - z[i, j]) <= cfg_cloud["ground_window_m"])
        ground.append(p[raw_ground])
    return Scan(voxel_first(np.concatenate(body), cfg_cloud["voxel_m"]),
                voxel_first(np.concatenate(continuation), cfg_cloud["voxel_m"] * 1.5),
                voxel_first(np.concatenate(foliage), cfg_cloud["vegetation_voxel_m"]),
                np.concatenate(ground), len(cloud))


def reference_ground(points, origin, shape, cfg):
    """Mean XY/Z of the lowest coherent raw-ground layer in each observed cell."""
    cells = CellIndex(*points.T, origin, shape)
    width = cfg["cloud"]["ground_window_m"]
    minimum = cfg["cloud"]["ground_min_points"]
    lo, _ = lowest_layer(cells, np.ones(len(cells.z), bool), width, minimum)
    low = lo.ravel()[cells.flat]
    use = (cells.z >= low) & (cells.z <= low + width)
    ids = cells.flat[use]
    xyz = points[cells.index[use]]
    unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    means = np.add.reduceat(xyz.astype(np.float64), starts, axis=0) / counts[:, None]
    return means, np.column_stack(np.unravel_index(unique, shape)), counts


def hull(points, thickness):
    """Convex hull with finite thickness for the unobserved back of a face."""
    middle = points.mean(0)
    local = points - middle
    _, basis = np.linalg.eigh(local.T @ local)
    for direction in basis.T:
        extent = np.ptp(local @ direction)
        if extent < thickness:
            offset = direction * (thickness - extent) / 2
            local = np.vstack((local - offset, local + offset))
    try:
        shape = ConvexHull(local)
    except QhullError:
        return None
    faces = shape.simplices.copy()
    a, b, c = local[faces].transpose((1, 0, 2))
    reverse = (np.cross(b - a, c - a) * shape.equations[:, :3]).sum(1) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    keep, faces = np.unique(faces, return_inverse=True)
    return middle, local[keep], faces.reshape(-1, 3)


def column(points, hit, axis, padding):
    """Cylinder from the measured axial interval and XY cover of the returns."""
    center = (hit.min(0) + hit.max(0)) / 2
    rel = hit - center
    t = rel @ axis
    radial = rel - t[:, None] * axis
    radius = max(np.linalg.norm(radial, axis=1).max(), padding / 2)
    t = (points - center) @ axis
    ends = [center + (t.min() - padding / 2) * axis, center + (t.max() + padding / 2) * axis]
    return dict(type="cylinder", size=[radius], fromto=np.ravel(ends))


def groups(keys):
    """Yield stable index groups using array keys, with no per-group full scan."""
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    splits = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, splits)


def fit_obstacles(scan, layers, active, origin, res, cfg, map_cfg, out):
    hard = (layers[..., 6] == 1) & active
    components, n_components = ndi.label(hard, np.ones((3, 3), int))
    observation = scan.body
    body_cell = cell_index(observation[:, :2], origin, res)
    label = components[tuple(body_cell.T)]
    if np.any(label == 0):
        raise ValueError("Hard evidence escaped the saved collision mask")
    record = Counter()
    shapes = []
    extra = scan.continuation
    extra_cell = cell_index(extra[:, :2], origin, res)
    extra_key = extra_cell[:, 0] * hard.shape[1] + extra_cell[:, 1]
    ordered = np.argsort(extra_key)
    sorted_key = extra_key[ordered]
    rail_codes = map_cfg["obstacles"]["railing_codes"]
    settings = cfg["hard"]
    pad = settings["min_thickness_m"]
    patch = settings["patch_m"]
    for comp in groups(label):
        p = observation[comp, :3]
        wide = np.max(np.ptp(p[:, :2], axis=0)) > patch
        partitions = (groups(np.floor((p[:, :2] - p.min(0)[:2]) / patch).astype(int))
                      if wide else [np.arange(len(comp))])
        for idx in partitions:
            ii = comp[idx]
            hit = observation[ii, :3]
            ij = body_cell[ii]
            cell_keys = np.unique(ij[:, 0] * hard.shape[1] + ij[:, 1])
            starts = np.searchsorted(sorted_key, cell_keys)
            ends = np.searchsorted(sorted_key, cell_keys, side="right")
            neighbors = np.concatenate([ordered[a:b] for a, b in zip(starts, ends)])
            tail = extra[neighbors, :3]
            widths = np.ptp(hit[:, :2], axis=0)
            rail = np.isin(observation[ii, 3], rail_codes).mean() >= settings["rail_code_fraction"]
            both = np.vstack([hit, tail])
            tall = np.ptp(both[:, 2]) >= settings["tree_height_m"]
            hx = hit[:, :2] - hit[:, :2].mean(0)
            eigen = np.linalg.eigvalsh(hx.T @ hx)
            compact = eigen[-1] <= settings["tree_max_flatness"] * max(eigen[0], pad**2)
            stem = tall and compact and not wide and widths.max() <= settings["tree_max_width_m"]
            if stem or rail:
                if rail:
                    hp = hit - hit.mean(0)
                    _, axes = np.linalg.eigh(hp.T @ hp)
                    axis = axes[:, -1]
                    # Rail-code continuations only, to avoid incorporating leaves into a rail.
                    tails = tail[np.isin(extra[neighbors, 3], rail_codes)]
                    # Lateral-only continuation would change the radius without evidence.
                    on_axis = tails - hit.mean(0)
                    dr = on_axis - (on_axis @ axis)[:, None] * axis
                    tails = tails[np.linalg.norm(dr, axis=1) <= max(widths.max() / 2, pad)]
                    geom = column(np.vstack([hit, tails]), hit, axis, pad)
                    material, kind = "rail", "rail_cylinder"
                else:
                    geom = column(both, hit, np.array([0., 0., 1.]), pad)
                    material, kind = "trunk", "trunk_cylinder"
            else:
                lo, hi = map_cfg["obstacles"]["body_band_m"]
                g = layers[ij[:, 0], ij[:, 1], 0]
                tail = tail[(tail[:, 2] >= g.min() - pad) & (tail[:, 2] <= g.max() + hi)]
                # Only connect measured returns in the same hard columns. Avoid
                # extending wall hulls high into vegetation above the body band.
                surface = np.vstack([hit, tail])
                if len(surface) < 2:
                    # A single point lacks an axis. A return-scale convex box is its finite hull.
                    surface = np.vstack([surface, surface + [0., 0., pad]])
                fitted = hull(surface, pad)
                if fitted is not None:
                    center, vertices, faces = fitted
                    geom = dict(type="mesh", pos=center)
                    material, kind = "stone", "rock_wall_hull"
                else:
                    center = (surface.min(0) + surface.max(0)) / 2
                    geom = dict(type="box", pos=center, size=np.maximum(np.ptp(surface, axis=0), pad) / 2)
                    material, kind = "stone", "degenerate_hull_box"
            name = f"hard_{len(shapes):05d}"
            item = dict(name=name, geom=geom, material=material, kind=kind,
                        bb_min=hit.min(0), bb_max=hit.max(0), source_cells=len(cell_keys),
                        source_component=int(label[ii[0]]))
            if geom["type"] == "mesh":
                file = f"hard/{name}.obj"
                write_obj(out / file, vertices, faces)
                item.update(file=file, faces=len(faces))
            shapes.append(item)
            record[kind] += 1
    observed_cells = np.unique(body_cell, axis=0).shape[0]
    return shapes, dict(shape_counts=dict(record), hard_cells=int(hard.sum()),
                        hard_cells_with_fit_evidence=observed_cells,
                        hard_components=int(n_components), body_voxels=len(scan.body))
