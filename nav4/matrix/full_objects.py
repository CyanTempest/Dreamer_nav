"""Fit woody volumes, scan-supported rock patches, rails and visual undergrowth."""
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from nav4.matrix.cloud import groups, hull
from nav4.matrix.full_ground import grid_sample, source_patch
from nav4.matrix.full_scan import TILE, WORK, read_tiles
from nav4.matrix.objects import tapered_solid
from nav4.matrix.register import voxel_ids
from nav4.matrix.stairs import PRISM_FACES
from nav4.matrix.survey import Corridor

BODY_HEIGHT = .8
BOX_CORNERS = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                        [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])


def fit_trunk_section(points, seed, rng):
    """RANSAC circle with arc coverage and geometric radial refinement."""
    if len(points) < 19:
        return None
    xy = points - seed
    sample = xy[rng.choice(len(xy), min(375, len(xy)), replace=False)]
    tris = sample[rng.integers(len(sample), size=(175, 3))]
    o = tris[:, 0]
    a, b = tris[:, 1] - o, tris[:, 2] - o
    det = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
    ok = abs(det) > .0008
    a, b, o, det = a[ok], b[ok], o[ok], det[ok]
    aa, bb = (a*a).sum(1), (b*b).sum(1)
    centers = o + np.c_[aa*b[:, 1]-bb*a[:, 1], bb*a[:, 0]-aa*b[:, 0]]/(2*det[:, None])
    radii = np.linalg.norm(centers-o, axis=1)
    good = (radii >= .062) & (radii <= .55) & (np.linalg.norm(centers, axis=1) < .49)
    if not good.any():
        return None
    centers, radii = centers[good], radii[good]
    norms = np.linalg.norm(sample[None] - centers[:, None], axis=-1)
    close = abs(norms - radii[:, None]) < .023
    order = np.argsort(-close.sum(1))[:6]
    best = None
    for idx in order:
        core = sample[close[idx]]
        if len(core) < 13:
            continue
        fit = least_squares(lambda p: np.linalg.norm(core-p[:2], axis=1)-p[2],
                            np.r_[centers[idx], radii[idx]], loss="soft_l1", f_scale=.013, max_nfev=22)
        center, radius = fit.x[:2], fit.x[2]
        delta = xy - center
        errors = abs(np.linalg.norm(delta, axis=1)-radius)
        use = errors < .027
        angles = np.sort(np.arctan2(delta[use, 1], delta[use, 0]))
        if len(angles) < 19:
            continue
        spread = 2*np.pi - np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
        bins = np.unique(np.floor((angles+np.pi)/(.25*np.pi))).size
        if not (.062 <= radius <= .55 and spread >= 1.75 and bins >= 3 and use.mean() >= .29):
            continue
        strength = use.sum() * (1 + spread/(2*np.pi))
        if best is None or strength > best[0]:
            best = strength, (center+seed, float(radius), int(use.sum()), float(np.sqrt(np.mean(errors[use]**2))))
    return None if best is None else best[1]


def mesh_item(name, kind, material, fitted, **evidence):
    center, vertices, faces = fitted
    return dict(name=name, kind=kind, material=material, vertices=vertices, triangles=faces,
                file=f"hard/{name}.obj", geom=dict(type="mesh", pos=center), **evidence)


def foliage_item(name, points):
    """Batch sparse low vegetation into a single visual-only OBJ per source tile."""
    m = len(points)
    radius = np.array([.081, .084, .090])
    center = points.mean(0)
    v = (points[:, None, :]-center+BOX_CORNERS[None]*radius).reshape(-1, 3)
    f = (PRISM_FACES[None]+np.arange(m)[:, None, None]*len(BOX_CORNERS)).reshape(-1, 3)
    return mesh_item(name, "soft", "vegetation", (center, v, f), voxels=m)


def stem_candidates(p, h, norm):
    use = (h > .38) & (h < 3.10) & (abs(norm) < .70)
    points, h = p[use, :2], h[use]
    if not len(points):
        return []
    pitch = .15
    keys = np.floor(points / pitch).astype(np.int32)
    ij0 = keys.min(0)
    keys -= ij0
    shape = keys.max(0)+1
    base = keys[:, 0]*shape[1]+keys[:, 1]
    bit = np.clip((h/.22).astype(np.int32), 1, 15)
    pids = np.unique(base.astype(np.int64)*16+bit)
    cells, counts = np.unique(pids//16, return_counts=True)
    valid = cells[counts >= 7]
    mask = np.zeros(shape, np.uint8)
    mask.ravel()[valid] = counts[counts >= 7]
    peak = mask == ndi.maximum_filter(mask, size=4)
    chosen = np.argwhere(peak & (mask > 0))
    order = sorted(chosen, key=lambda q: -int(mask[tuple(q)]))
    output = []
    for q in order:
        xy = (q+ij0+.5)*pitch
        if all(np.linalg.norm(xy-other) >= .48 for other in output):
            output.append(xy)
    return output


def fit_stem(raw, local_ground, seed, name, rng):
    floor = local_ground(seed[None])[0]
    xy_ok = (abs(raw[:, :2]-seed)<.70).all(1)
    points = raw[xy_ok & (raw[:, 2]>floor+.06) & (raw[:, 2]<floor+3.80)]
    if len(points) < 90:
        return []
    points = points[voxel_ids(points, .024)]
    sections = []
    anchor = seed
    for k, level in enumerate(np.arange(.24, 3.65, .25)):
        z = floor+level
        slab = points[(abs(points[:, 2]-z)<.12) & (np.linalg.norm(points[:, :2]-anchor, axis=1)<.53)]
        result = fit_trunk_section(slab[:, :2], anchor, rng)
        if result is None:
            continue
        xy, radius, n, rms = result
        if sections:
            prev, pr, _, _, old_k = sections[-1]
            if (np.linalg.norm(xy-prev[:2]) > .09+.13*(k-old_k) or
                    abs(radius-pr) > .061+.24*pr or k-old_k > 3):
                continue
        sections.append((np.r_[xy, z], radius, n, rms, k))
        anchor = xy
    if len(sections) < 5 or sections[0][0][2]-floor > .83 or sections[-1][0][2]-sections[0][0][2] < 1.10:
        return []
    core = sections[:min(4, len(sections))]
    bottom_xy = np.median([s[0][:2] for s in core], axis=0)
    ground = local_ground(bottom_xy[None])[0]
    if abs(sections[0][0][2]-ground) > .86:
        return []
    section0 = (np.r_[bottom_xy, ground-.045], np.median([s[1] for s in core]), 0, None, -1)
    chain = [section0, *sections]
    out = []
    for i, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
        if a[0][2] >= b[0][2] or b[0][2]-a[0][2] > .65:
            continue
        fitted = tapered_solid(a[0], b[0], a[1], b[1], 10)
        out.append(mesh_item(f"{name}_{i:02d}", "trunk", "trunk", fitted,
                             evidence_returns=a[2]+b[2], diameters_m=[2*a[1], 2*b[1]],
                             radial_rms_m=[a[3], b[3]], stem_id=name))
    return out


def woody_ridges(raw, local_ground, stem, radius, name):
    d = raw[:, :2]-stem
    r = np.linalg.norm(d, axis=1)
    p = raw[(r>radius*.82) & (r<1.05)]
    if len(p) < 50:
        return []
    p = p[voxel_ids(p, .027)]
    d = p[:, :2]-stem
    h = p[:, 2]-local_ground(p[:, :2])
    phi = np.arctan2(d[:, 1], d[:, 0])
    output = []
    for kind, condition in (("root", (h>.055)&(h<.30)), ("low_branch", (h>.25)&(h<BODY_HEIGHT))):
        hist, _ = np.histogram(phi[condition], bins=np.linspace(-np.pi, np.pi, 25))
        done = []
        for k in np.argsort(-hist)[:4]:
            if hist[k] < 15 or any(min(abs(k-j), 24-abs(k-j)) < 2 for j in done):
                continue
            angle = -np.pi+(k+.5)*np.pi/12
            axis = np.array([np.cos(angle), np.sin(angle)])
            u, cross = d@axis, d@np.array([-axis[1], axis[0]])
            use = condition&(u>radius*.82)&(abs(cross)<.105)
            samples = p[use]
            if len(samples) < 25 or np.ptp(u[use]) < .28 or np.min(u[use]) > radius+.14:
                continue
            occupied = np.unique(np.floor(u[use]/.10).astype(int))
            if len(occupied) < 4 or np.max(np.diff(occupied)) > 1:
                continue
            if kind == "low_branch":
                center = samples.mean(0)
                _, sig, directions = np.linalg.svd(samples-center, full_matrices=False)
                direction = directions[0]
                if sig[0] <= 3*sig[1] or not (.022 <= 2*sig[1]/np.sqrt(len(samples)) <= .10):
                    continue
                proj = (samples-center)@direction
                cap = np.percentile(proj, [8, 92])
                r = float(np.clip(1.3*sig[1]/np.sqrt(len(samples)), .022, .065))
                output.append(dict(name=f"{name}_{kind}_{k:02d}", kind=kind, material="trunk",
                                   geom=dict(type="capsule", fromto=(center+cap[:, None]*direction).ravel(),
                                             size=[r]), evidence_returns=len(samples)))
            else:
                height = h[use]
                for i, (lo, hi) in enumerate(zip(occupied[:-1], occupied[1:])):
                    hits = (u[use] >= lo*.10) & (u[use] < (hi+1)*.10)
                    q = samples[hits]
                    if len(q) >= 11 and np.median(height[hits]) > .064:
                        fitted = hull(np.r_[q, q-[0, 0, .07]], .024)
                        if fitted is not None:
                            output.append(mesh_item(f"{name}_{kind}_{k:02d}_{i:02d}", kind, "trunk", fitted,
                                                    evidence_returns=len(q)))
            done.append(k)
    return output


def supported_rock(visible, normals):
    if len(visible) < 18:
        return None
    centered = visible - visible.mean(0)
    _, sig, axes = np.linalg.svd(centered, full_matrices=False)
    if sig[1] < .11 or sig[2] > .38*sig[1] or abs(axes[-1, 2]) > .68:
        return None
    spread = np.ptp(centered@axes.T, axis=0)
    if spread[0] < .14 or spread[1] < .12:
        return None
    inliers = (abs(centered@axes[-1]) < .041) & (abs(normals@axes[-1]) >= .70)
    if inliers.sum() < 14 or inliers.mean() < .61:
        return None
    return hull(visible[inliers], .036)


def linear_rail(points, name):
    if len(points) < 13:
        return None
    center = points.mean(0)
    _, sig, vectors = np.linalg.svd(points-center, full_matrices=False)
    axis = vectors[0]
    distances = (points-center)@axis
    length = np.ptp(distances)
    if length < .23 or sig[0] < 3.4*sig[1]:
        return None
    transverse = np.linalg.norm(points-center-distances[:, None]*axis, axis=1)
    radius = np.percentile(transverse, 72)
    if not .014 < radius < .065 or np.percentile(transverse, 90) > .087:
        return None
    a, b = np.percentile(distances, [3, 97])
    return dict(name=name, kind="rail_scan", material="rail",
                geom=dict(type="capsule", size=[radius], fromto=np.r_[center+a*axis, center+b*axis]),
                evidence_returns=len(points))


def detect_objects(path, v8, ground, origin, pilot=False):
    start = time.monotonic()
    corridor = Corridor(path)
    route_tree = cKDTree(path)
    rng = np.random.default_rng(8125)
    projection = lambda points: grid_sample(ground, origin, points)
    found, stems = [], []
    for f in sorted((WORK/"ply").glob("*.bin")):
        lo = np.array([int(a) for a in f.stem.split("_")])*TILE
        hi = lo + TILE
        if route_tree.query((lo+hi)*.5)[0] > 6+TILE*.71:
            continue
        annotated = read_tiles([f], 7)
        xyz = annotated[:, :3]
        floor = projection(xyz[:, :2])
        h = xyz[:, 2]-floor
        # Eliminate the crowns from the expensive neighborhood queries first.
        eligible = np.flatnonzero((h>-.15)&(h<4.3))
        st, lat, valid = corridor.project(xyz[eligible, :2])
        keep = eligible[valid]
        p, h = annotated[keep], h[keep]
        st, lat = st[valid], lat[valid]
        codes = v8.classes(p[:, :2])
        label = f"{int(lo[0]/TILE)}_{int(lo[1]/TILE)}".replace("-", "n")
        soft = (codes == 2) & (h > .075) & (h < 1.85) & (np.abs(lat) > np.where(st<20.2, .74, .42))
        v = p[soft, :3]
        if len(v):
            v = v[voxel_ids(v, .335)]
            found.append(foliage_item(f"soft_{label}", v))

        seeds = stem_candidates(p[:, :3], h, p[:, 5])
        if seeds:
            raw, _ = source_patch(lo-1.15, hi+1.15)
            r = raw[raw[:, 3] == 0, :3]
            index = cKDTree(r[:, :2])
            for j, seed in enumerate(seeds):
                s, l, included = corridor.project(seed[None])
                if not included[0] or abs(l[0]) < .42:
                    continue
                if (abs(s[0]-3.64)<.7 and abs(l[0]+3.25)<.7) or (abs(s[0]-5.73)<.55 and abs(l[0]+1.7)<.55):
                    continue
                if any(np.linalg.norm(seed-q) < .56 for q in stems):
                    continue
                nearby = r[index.query_ball_point(seed, 1.24)]
                key = f"trunk_auto_{label}_{j:02d}"
                pieces = fit_stem(nearby, projection, seed, key, rng)
                if not pieces:
                    continue
                stem = pieces[0]["geom"]["pos"][:2]
                if any(np.linalg.norm(stem-q) < .60 for q in stems):
                    continue
                stems.append(stem)
                found.extend(pieces)
                found.extend(woody_ridges(nearby, projection, stem, pieces[0]["diameters_m"][0]/2, key))

        if not pilot:
            rail_mask = np.isin(p[:, 6], [13, 14]) & ((abs(lat)>.36)&(abs(lat)<2.40)) & (h>.15)&(h<1.4)&(st>20.6)
            rail = p[rail_mask, :3]
            rail = rail[voxel_ids(rail, .017)]
            for i, idx in enumerate(groups(np.floor(rail/.46).astype(int))):
                item = linear_rail(rail[idx], f"rail_scan_{label}_{i:04d}")
                if item is not None:
                    found.append(item)
            rocky = (codes != 2) & (abs(p[:, 5]) < .50) & (abs(lat) > .42) & (
                h > .13) & (h < 4.0) & (st > 20.2)
            candidate = p[rocky]
            candidate = candidate[voxel_ids(candidate[:, :3], .026)]
            for i, idx in enumerate(groups(np.floor(candidate[:, :3]/.42).astype(int))):
                q = candidate[idx]
                fitted = supported_rock(q[:, :3], q[:, 3:6])
                if fitted is not None:
                    center = fitted[0]
                    if any(np.linalg.norm(center[:2]-stem) < .41 for stem in stems):
                        continue
                    found.append(mesh_item(f"rock_{label}_{i:05d}", "rock_face", "stone", fitted,
                                           evidence_returns=len(q)))
    print("objects", len(found), "auto stems", len(stems), "seconds", time.monotonic()-start, flush=True)
    return found
