"""Export the standalone scan-coordinate scenes into one forward TrailBank map.

Run from the project root with numpy, scipy, Shapely 2 and a C++17 compiler:
  .venv_lwy/bin/python -m nav4.matrix.export_nav4

The two XML inventories are streamed sequentially. No MuJoCo model is compiled.
"""
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from nav4.mapping.common import BANDS, RES, ROOT, load_config, signed_distance
from nav4.mapping.ground import connected_from_seeds
from nav4.mapping.model import max_step, oriented_cross_grade, plane_gradient


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


class Raster:
    def __init__(self, origin, shape, work):
        src = Path(__file__).with_name("export_raster.cpp")
        so = Path(work) / "raster.so"
        subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
                        str(src), "-o", str(so)], check=True)
        self.lib = ctypes.CDLL(str(so.resolve()))
        ptr = ctypes.c_void_p
        integer, real = ctypes.c_int, ctypes.c_double
        self.lib.raster.argtypes = [ptr, ptr, integer, integer, integer, real, real, real,
                                    ptr, ptr, ptr, ptr, integer, ptr]
        self.lib.distances.argtypes = [ptr, integer, integer, integer, real, real, real, ptr]
        self.lib.raster.restype = self.lib.distances.restype = None
        self.origin = np.asarray(origin, dtype=float)
        self.shape = shape
        self.bands = np.asarray(BANDS)

    def triangles(self, vertices, faces, surface, mode=0, atlas=None, top=None, soft=None):
        v = np.ascontiguousarray(vertices, np.float64)
        f = np.ascontiguousarray(faces, np.int32)
        outputs = [x.ctypes.data if x is not None else None for x in (surface, atlas, top, soft)]
        self.lib.raster(v.ctypes.data, f.ctypes.data, len(f), *self.shape, *self.origin, RES,
                        *outputs, mode, self.bands.ctypes.data)

    def cell_range(self, lo, hi, margin=0.):
        a = np.maximum(0, np.ceil((lo-self.origin)/RES - margin).astype(int))
        b = np.minimum(self.shape, np.floor((hi-self.origin)/RES + margin).astype(int)+1)
        return a, b


def inventory(directory):
    root = ET.parse(directory/"scene_full.xml")
    assets = {a.get("name"): a for a in root.find("asset") if a.tag in ("mesh", "hfield")}
    for g in root.findall("worldbody/geom"):
        if any(k in g.attrib for k in ("quat", "euler", "axisangle", "zaxis", "xyaxes")):
            raise ValueError(f"Unexpected rotated geometry {g.attrib}")
        yield g, assets.get(g.get("mesh") or g.get("hfield"))


def mesh(directory, g, asset):
    if asset.get("scale") or asset.get("refpos") or asset.get("refquat"):
        raise ValueError(f"Unexpected mesh transform {asset.attrib}")
    # Exporter's triangle OBJ files contain only position vertices and indexed faces.
    path = directory/asset.attrib["file"]
    data = path.read_bytes()
    first = data.index(b"\nf ")
    vs = np.fromstring(data[:first].replace(b"v ", b""), sep=" ").reshape(-1, 3)
    fs = np.fromstring(data[first+1:].replace(b"f ", b""), dtype=np.int32, sep=" ").reshape(-1, 3)-1
    if not len(fs) or fs.min()<0 or fs.max()>=len(vs):
        raise ValueError(f"Invalid triangles in {path}")
    return vs + np.fromstring(g.get("pos", "0 0 0"), sep=" "), fs


def capsule(g):
    """Inscribed capsule, 64 meridians and angular spacing pi/32 on the caps."""
    ends = np.fromstring(g.attrib["fromto"], sep=" ").reshape(2, 3)
    radius = float(g.attrib["size"])
    axis = ends[1]-ends[0]
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, np.eye(3)[np.argmin(abs(axis))])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    phi = np.arange(64)*np.pi/32
    circle = np.cos(phi)[:, None]*u + np.sin(phi)[:, None]*v
    pts = [ends[0]-radius*axis, ends[1]+radius*axis]
    for end, angles in zip(ends, (np.arange(-15, 1), np.arange(16))):
        for theta in angles*np.pi/32:
            pts.extend(end + radius*(np.cos(theta)*circle+np.sin(theta)*axis))
    pts = np.asarray(pts)
    return pts, ConvexHull(pts).simplices


def hfield(directory, g, asset, grid, out):
    pos = np.fromstring(g.attrib["pos"], sep=" ")
    half = np.fromstring(asset.attrib["size"], sep=" ")
    with (directory/asset.attrib["file"]).open("rb") as file:
        rows, cols = np.fromfile(file, "<i4", 2)
        z = np.fromfile(file, "<f4").reshape(rows, cols).T
    base = pos[:2]-half[:2]
    a,b = grid.cell_range(base-1e-7, pos[:2]+half[:2]+1e-7)
    if (b<=a).any():
        return
    ij = np.indices(b-a).reshape(2, -1).T+a
    uv = np.clip((grid.origin+ij*RES-base)/(2*half[:2])*(np.array(z.shape)-1),
                 0., np.array(z.shape)-1)
    k = np.minimum(uv.astype(int), np.array(z.shape)-2)
    i,j = k.T
    x,y = (uv-k).T
    # MuJoCo splits each height-field quad along its lower-left to upper-right diagonal.
    norm = ((1-np.maximum(x,y))*z[i,j] + np.maximum(x-y,0)*z[i+1,j]
            + np.maximum(y-x,0)*z[i,j+1] + np.minimum(x,y)*z[i+1,j+1])
    ids = tuple(ij.T)
    out[ids] = np.maximum(out[ids], pos[2]+half[2]*norm)


def support_layers(ground, known, tread, source, terrain, cfg):
    t = cfg["terrain"]
    assert t["max_step_m"] == terrain["max_step"] and t["max_grade_deg"] == terrain["max_grade_deg"]
    assert 2*t["grade_radius_m"] == terrain["grade_baseline_m"]
    def capability(mask):
        gx, gy, _ = plane_gradient(ground, mask, t["grade_radius_m"])
        grade = np.degrees(np.arctan(np.hypot(gx, gy)))
        step = max_step(ground, mask, t["step_radius_m"])
        return mask & (grade <= terrain["max_grade_deg"]) & (step <= terrain["max_step"])
    cap = capability(known)
    exempt = np.zeros_like(known)
    if t["trail_terrain_exemption"]:
        # Only the walked-track location, seeds and orientation come from v8.
        with np.load(source/"forward/aux.npz") as aux:
            zone = (aux["dist_walked"] <= t["trail_zone_halfwidth_m"]) & known
            seeds = aux["walked_cells"]
            yaw = aux["centreline_yaw"][aux["centreline_near"]]
        conn = connected_from_seeds(ground, zone, seeds, t["trail_zone_connect_max_step_m"])
        cross = oriented_cross_grade(ground, known, zone, yaw, t["cross_grade_half_width_m"],
                                     t["cross_grade_half_length_m"])
        exempt = conn & (cross <= t["trail_zone_max_cross_grade_deg"])
        if t["trail_tread_support"]:
            exempt |= conn & capability(tread)
    support = cap | exempt
    support[[0,1,-2,-1],:] = False
    support[:,[0,1,-2,-1]] = False
    return signed_distance(support), cap, exempt


def projected_pieces(v, faces, grid, ground, band):
    """Clip the solid to the local body slab in each overlapped map cell.

    A slab of a convex solid projects to the hull of its in-slab vertices and
    the edge/plane intersections. Clip that footprint to the cell rectangle.
    """
    import shapely as sh
    lo,hi = grid.cell_range(v[:,:2].min(0), v[:,:2].max(0), .5)
    if (hi<=lo).any():
        return []
    edges = np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),1),axis=0)
    p,q = v[edges[:,0]],v[edges[:,1]]
    swap = p[:,2]>q[:,2]
    p[swap],q[swap] = q[swap].copy(),p[swap].copy()
    low_z, high_z = v[:,2].min(), v[:,2].max()
    pieces = []
    cache = {}
    for i in range(lo[0], hi[0]):
        for j in range(lo[1], hi[1]):
            bot,top = ground[i,j]+band
            if bot>=high_z or top<=low_z:
                continue
            key = max(bot,low_z), min(top,high_z)
            if key not in cache:
                pts = [v[(v[:,2]>=bot)&(v[:,2]<=top),:2]]
                for z in key:
                    cut = (p[:,2]<z)&(q[:,2]>z)
                    a,b = p[cut],q[cut]
                    u = ((z-a[:,2])/(b[:,2]-a[:,2]))[:,None]
                    pts.append(a[:,:2]+u*(b[:,:2]-a[:,:2]))
                xy = np.concatenate(pts)
                cache[key] = sh.convex_hull(sh.multipoints(xy)) if len(xy) else None
            polygon = cache[key]
            if polygon is not None:
                x,y = grid.origin + RES*np.array([i,j])
                piece = sh.clip_by_rect(polygon, x-RES/2,y-RES/2,x+RES/2,y+RES/2)
                if piece.area>0:
                    pieces.append(piece)
    return pieces


def obstacle_sdf(grid, polygons):
    import shapely as sh
    result = np.full(grid.shape, 100., np.float32)
    if not polygons:
        return result
    # A nanometre snap eliminates floating-point cracks at shared cell edges.
    union = sh.union_all(polygons, grid_size=1e-9)
    segments = []
    inside = np.zeros(grid.shape, bool)
    for poly in sh.get_parts(union):
        if poly.geom_type != "Polygon":
            raise ValueError(f"Unexpected dissolved footprint {poly.geom_type}")
        for ring in (poly.exterior, *poly.interiors):
            xy = np.asarray(ring.coords)
            segments.append(np.c_[xy[:-1],xy[1:]])
        bounds = np.array(poly.bounds)
        a,b = grid.cell_range(bounds[:2], bounds[2:])
        if (b<=a).any():
            continue
        i,j = np.ogrid[a[0]:b[0],a[1]:b[1]]
        inside[a[0]:b[0],a[1]:b[1]] |= sh.contains_xy(poly,grid.origin[0]+i*RES,grid.origin[1]+j*RES)
    edges = np.ascontiguousarray(np.concatenate(segments), np.float64)
    grid.lib.distances(edges.ctypes.data,len(edges),*grid.shape,*grid.origin,RES,result.ctypes.data)
    result[inside] *= -1
    return result


def build(source, v1, v2, output, work):
    import shapely as sh
    with np.load(source/"forward/map.npz") as old:
        payload = {k:old[k] for k in old.files if k!="atlas"}
    if abs(float(payload["res"])-RES)>1e-7:
        raise ValueError("Expected the v8 0.1 m grid")
    layers = payload["layers"]
    shape = layers.shape[:2]
    grid = Raster(payload["origin"], shape, work)
    ground, stairs, visual = [np.full(shape, -np.inf, np.float32) for _ in range(3)]
    for g,asset in inventory(v2):
        if g.get("group")!="3":
            continue
        if g.get("type")=="hfield":
            hfield(v2,g,asset,grid,ground)
        elif g.get("name").startswith("col_stair_"):
            grid.triangles(*mesh(v2,g,asset),stairs)
    np.maximum(ground, stairs, out=ground)
    tread = np.isfinite(stairs) & (stairs>=ground)
    from_v2 = np.isfinite(ground)
    print("Read v2 support", from_v2.sum(), flush=True)
    for g,asset in inventory(v1):
        if g.get("group") in ("0","2","4"):
            grid.triangles(*mesh(v1,g,asset),visual)
    known = from_v2 | np.isfinite(visual)
    ground[~from_v2] = visual[~from_v2]
    # A nearest reference height outside the mesh permits relative visual bins
    # at scan edges. The known and support bits stay false there.
    nearest = ndi.distance_transform_edt(~known, return_distances=False, return_indices=True)
    ground = np.ascontiguousarray(ground[tuple(nearest)])
    del nearest, visual, stairs
    layers[...,0] = ground
    env = json.loads((ROOT/"nav4/config/env.json").read_text())
    cfg,_ = load_config()
    terrain = env["terrain"]
    layers[...,2], cap, exempt = support_layers(ground,known,tread,source,terrain,cfg)
    print("Computed terrain support",flush=True)

    atlas = np.zeros((*shape,20),np.float32)
    atlas[...,0] = ground
    atlas[...,19] = known
    atlas[...,2:19:3] = -1
    top = ground.copy()
    soft = np.zeros(shape, np.uint8)
    for g,asset in inventory(v1):
        group = g.get("group")
        if group not in ("0","1","2","4","5"):
            raise ValueError(f"Unexpected visual group {group}")
        geom = capsule(g) if g.get("type")=="capsule" else mesh(v1,g,asset)
        grid.triangles(*geom,ground,2 if group=="5" else 1,atlas,top,soft)
    layers[...,1] = top
    payload["atlas"] = atlas
    print("Rasterized visual scene",flush=True)

    band = np.array([terrain["collision_height"],cfg["obstacles"]["body_band_m"][1]])
    polygons = []
    count = 0
    for g,asset in inventory(v2):
        if g.get("group")!="3" or g.get("type")=="hfield" or g.get("name").startswith("col_stair_"):
            continue
        if g.get("type")=="capsule":
            v,faces = capsule(g)
        else:
            v,_ = mesh(v2,g,asset)
            faces = ConvexHull(v).simplices
        bits = projected_pieces(v,faces,grid,ground,band)
        if bits:
            polygons.append(sh.union_all(bits,grid_size=1e-9))
        count += 1
        if count%2000==0:
            print("Clipped hard bodies",count,flush=True)
    sdf = obstacle_sdf(grid,polygons)
    layers[...,5] = sdf
    layers[...,6] = np.where(sdf<0,1,np.where(soft,2,0))
    forward = output/"forward"
    forward.mkdir(parents=True,exist_ok=True)
    path = forward/"map.npz"
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp,**payload)
    temp.replace(path)
    np.savez_compressed(forward/"aux.npz",known=known,v2_support_surface=from_v2,
                        tread=tread,support_cap=cap,exemption=exempt,soft_visual=soft.astype(bool))
    record = dict(name="forward",path="forward/map.npz",map_sha256=digest(path),res=RES,
                  dimensions=list(shape),stations=len(payload["s"]),segment_length_m=float(payload["s"][-1]),
                  hard_bodies=count,known_cells=int(known.sum()),v2_surface_cells=int(from_v2.sum()),
                  v1_fallback_cells=int((known & ~from_v2).sum()),collision_height_band_m=band.tolist(),
                  support_cap_cells=int(cap.sum()),additional_prior_cells=int((exempt & ~cap).sum()))
    write_json(forward/"metadata.json", record)
    write_json(output/"manifest.json",dict(maps=[record],acceptance_direction="forward",
        visual_source=str(v1), collision_source=str(v2), route_source=str(source),
        source_sha256={str(p):digest(p) for p in (v1/"scene_full.xml",v2/"scene_full.xml",source/"forward/map.npz")}))
    shutil.copyfile(ROOT/"nav4/config/env.json", output/"env.json")
    shutil.copyfile(ROOT/"nav4/mapping/trail_map_config.json", output/"trail_map_config.json")
    print(record,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=ROOT/"data/trail_a_v8_frozen")
    p.add_argument("--v1",type=Path,default=ROOT/"data/trail_a_mujoco/full_v1")
    p.add_argument("--v2",type=Path,default=ROOT/"data/trail_a_mujoco/full_v2")
    p.add_argument("--output",type=Path,default=ROOT/"data/trail_a_v9")
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as work:
        build(a.source,a.v1,a.v2,a.output,work)


if __name__=="__main__":
    main()

