"""Reconstruct a dense, locally sampled ground surface and measure hold-out error."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import bilinear
from nav4.matrix.register import summary
from nav4.matrix.survey import Corridor


def coherent_samples(points, ref_ground, coarse, v8_origin, coarse_res):
    coarse_z = bilinear(coarse, v8_origin, points[:, :2], res=coarse_res)
    window = np.flatnonzero(abs(points[:,2]-coarse_z) < .48)
    eligible = points[window]
    d, near = cKDTree(ref_ground).query(eligible, workers=4)
    # A stable return should have a nearby PLY surface. In occluded side
    # terrain use the coarse prior during hole filling, with a separate mask.
    keep = d < .095
    return window[keep], ref_ground[near[keep],2], d[keep]


def ground_raster(points, ref_ground, coarse, origin, coarse_res, path, pitch):
    first = path[0] - (path[1]-path[0]) / np.linalg.norm(path[1]-path[0]) * pitch
    last = path[-1] + (path[-1]-path[-2]) / np.linalg.norm(path[-1]-path[-2]) * pitch
    corridor = Corridor(np.vstack([first,path,last]))
    xy0 = np.floor((path.min(0)-6.04)/pitch)*pitch
    upper = path.max(0)+6.04
    shape = tuple(np.ceil((upper-xy0)/pitch).astype(int)+1)
    cell = np.rint((points[:, :2]-xy0)/pitch).astype(int)
    candidate, z_guide, ref_dist = coherent_samples(points, ref_ground, coarse, origin, coarse_res)
    key = np.ravel_multi_index(cell[candidate].T, shape)
    # Interleaved raw records withheld before constructing the grid.
    test = (candidate * 37) % 11 < 2
    rows = np.column_stack((key[~test], points[candidate[~test], 2], z_guide[~test]))
    order = np.lexsort((rows[:,1],rows[:,0]))
    rows = rows[order]
    ids, begin, count = np.unique(rows[:,0].astype(int), return_index=True, return_counts=True)
    z = np.full(np.prod(shape), np.nan)
    residual = np.abs(rows[:,1]-rows[:,2])
    # Median of returns close to the local observed PLY ground, up to 5 cm.
    # A lone return requires subsequent agreement with neighboring cells.
    good = residual < .055
    rows = rows[good]
    ids, begin, count = np.unique(rows[:,0].astype(int), return_index=True, return_counts=True)
    lo = begin+(count-1)//2
    hi = begin+count//2
    z[ids] = (rows[lo,1]+rows[hi,1])/2
    z = z.reshape(shape)
    ij = np.indices(shape).reshape(2,-1).T
    grid = ij*pitch+xy0
    _, _, active = corridor.project(grid)
    active = active.reshape(shape)
    prior = bilinear(coarse, origin, grid, res=coarse_res).reshape(shape)
    observed = np.isfinite(z)
    distance, nearest = ndi.distance_transform_edt(~observed, return_indices=True)
    nearby = distance*pitch <= .30
    dense = np.where(nearby, z[tuple(nearest)], prior)
    # Fill very small sampling holes, keeping observations intact. Occlusions
    # farther than 30 cm use a smooth v8 prior with a recorded fill mask.
    target = ndi.median_filter(dense, size=3)
    dense = np.where(observed & (abs(dense-target) < .075), dense, target)
    dense = np.where(nearby, dense, ndi.gaussian_filter(prior, .10/pitch))
    withheld = points[candidate[test]]
    return xy0, dense, active, observed, nearby, withheld, ref_dist[test]


def underlay_treads(ground, xy0, pitch, stairs):
    """Keep a continuous terrain base below the fitted tread solids.

    A two-cell overlap keeps all incident raster triangles below the
    precision edges and covers curved staircase/ground seams.
    """
    selection = np.zeros(ground.shape, dtype=bool)
    underlay = ground.copy()
    for step in stairs:
        polygon = step["polygon"]
        i0, j0 = np.maximum(0, np.floor((polygon.min(0)-xy0)/pitch).astype(int)-3)
        i1, j1 = np.minimum(ground.shape, np.ceil((polygon.max(0)-xy0)/pitch).astype(int)+4)
        coords = np.indices((i1-i0,j1-j0)).reshape(2,-1).T+np.array([i0,j0])
        xy = coords*pitch+xy0
        inside = in_polygon(xy, polygon)
        selection[tuple(coords[inside].T)] = True
        near = in_polygon(xy, polygon, margin=pitch*1.60)
        index = tuple(coords[near].T)
        slab = step["vertices"][4:]
        top = np.linalg.lstsq(np.c_[slab[:,:2], np.ones(4)], slab[:,2], rcond=None)[0]
        base = (xy[near]@top[:2]+top[2])-.085
        underlay[index] = np.minimum(underlay[index],base)
        if "before_plane_coeff" in step:
            edge = step["edge"]
            span = edge[1]-edge[0]
            uphill = np.array([span[1],-span[0]])/np.linalg.norm(span)
            d = (xy-edge.mean(0))@uphill
            across = ((xy-edge[0])@span)/(span@span)
            apron = (d >= -2.35*pitch) & (d <= .5*pitch) & (across > -.015) & (across < 1.015)
            index = tuple(coords[apron].T)
            before = step["before_plane_coeff"]
            base = ((xy[apron]-step["before_plane_anchor"])@before[:2]+before[2])
            underlay[index] = np.minimum(underlay[index],base-.012)
    return selection, underlay


def in_polygon(xy, polygon, margin=0):
    delta = np.roll(polygon,-1,axis=0)-polygon
    rel = xy[...,None,:]-polygon
    cross = delta[:,0]*rel[...,1]-delta[:,1]*rel[...,0]
    return (cross >= -margin*np.linalg.norm(delta,axis=1)).all(axis=-1)


def point_mesh_distances(meshes, points, sample=8000):
    """Exact point-to-triangle distances with a conservative XY candidate search."""
    import trimesh

    rng = np.random.default_rng(1180)
    p = points[rng.choice(len(points), min(len(points), sample), replace=False)]
    best = np.full(len(p), np.inf)
    for v, f in meshes:
        lower = v[:,:2].min(0)-.30
        upper = v[:,:2].max(0)+.30
        use = np.flatnonzero(((p[:,:2] >= lower) & (p[:,:2] <= upper)).all(axis=1))
        if not len(use):
            continue
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        for idx in np.array_split(use, 1+len(use)//150):
            _, d, _ = trimesh.proximity.closest_point(mesh, p[idx])
            best[idx] = np.minimum(best[idx], d)
    finite = np.isfinite(best)
    return dict(**summary(best[finite]), sampled=len(best), without_triangle=int((~finite).sum()))
