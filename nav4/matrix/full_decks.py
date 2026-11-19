"""Add independently scanned lower ground where v8 flagged multiple layers."""
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.compose import place_terrain, save_xml, write_obj
from nav4.matrix.full_ground import (FULL, PITCH, V8Map, bounded_grades, edge_statistics,
                                     grid_geometry, grid_sample, median_cells, source_patch)
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import CONFIG, LIMITS, scene_dir
from nav4.matrix.register import voxel_ids
from nav4.matrix.survey import Corridor, V8


def patch_geometry(surface, xy0, raster_shape, lo_xy, hi_xy):
    raw, ply = source_patch(lo_xy-.15, hi_xy+.20)
    floor = lambda q: grid_sample(surface["z"],surface["origin"],q)
    pz = ply[:, 2] - floor(ply[:, :2])
    reference = ply[(pz < -.30) & (pz > -2.50) & (abs(ply[:, 5]) > .61), :3]
    if len(reference) < 30:
        return None
    reference = reference[voxel_ids(reference,.034)]
    train = raw[(raw[:, 3] == 0), :3]
    rz = train[:, 2] - floor(train[:, :2])
    train = train[(rz < -.29)&(rz > -2.54)]
    if len(train) < 35:
        return None
    d, _ = cKDTree(reference).query(train, distance_upper_bound=.085, workers=4)
    dense = train[np.isfinite(d)]
    pos, zz = median_cells(dense, xy0, raster_shape)
    if len(pos) < 15:
        return None
    z = np.full(raster_shape, np.nan, np.float32)
    z.ravel()[pos] = zz
    observed = np.isfinite(z)
    distance, indices = ndi.distance_transform_edt(~observed, return_indices=True)
    nearest = z[tuple(indices)]
    accepted = distance <= 1.41422
    corners = accepted[:-1,:-1] & accepted[1:,:-1] & accepted[1:,1:] & accepted[:-1,1:]
    if corners.sum() < 6:
        return None
    z = np.where(observed,z,ndi.gaussian_filter(nearest,.85))
    z = bounded_grades(z)
    v, f = grid_geometry(z, xy0)
    f = f[corners.ravel()].reshape(-1, 3)
    idx, faces = np.unique(f, return_inverse=True)
    return v[idx], faces.reshape(-1, 3), int(corners.sum()), len(dense)


def main():
    import json
    start = time.monotonic()
    output = FULL/"lower_layers.json"
    if output.exists():
        return
    v8 = V8Map.load()
    deck = np.load(V8/"aux.npz")["deck"]
    all_segments = json.loads((FULL/"segments.json").read_text())
    labels, n = ndi.label(deck)
    _, path = route()
    corridor = Corridor(path)
    with np.load(FULL/"surface.npz") as scene:
        surface = {"z": scene["z"], "origin": scene["origin"]}
    items = []
    for k in range(1,n+1):
        ij = np.argwhere(labels == k)
        if len(ij) < 15:
            continue
        xy = v8.origin+ij*v8.resolution
        s, _, _ = corridor.project(xy)
        xy0 = np.floor((xy.min(0)-.15)/PITCH)*PITCH
        hi_xy = np.ceil((xy.max(0)+.15)/PITCH)*PITCH
        shape = np.rint((hi_xy-xy0)/PITCH).astype(int)+1
        fitted = patch_geometry(surface, xy0, shape, xy0, hi_xy)
        if fitted is None:
            continue
        vertices, faces, cells, samples = fitted
        station = float(np.median(s))
        i = int(np.searchsorted(LIMITS[1:-1],station))
        out = scene_dir(FULL,i)
        name = f"s{i:02d}_lower_layer_{k:02d}"
        scene_path = out/"scene.xml"
        scene = ET.parse(scene_path).getroot()
        center = vertices.mean(0)
        rel = f"terrain/{name}.obj"
        write_obj(out/rel,vertices-center,faces)
        if scene.find(f"./worldbody/geom[@name='{name}']") is None:
            place_terrain(scene,name,rel,center,CONFIG["terrain"]["collision_skin_m"])
            scene.find(f"./worldbody/geom[@name='{name}']").set("rgba",".44 .43 .39 1")
            save_xml(scene,scene_path)
        stat = edge_statistics(vertices,faces)
        meta = all_segments[i]
        label = "lower_observed_ground"
        existing = meta["surface"].setdefault(label,{})
        for key,value in stat.items():
            existing[key] = existing.get(key,0)+value
        meta["bytes"] = scene_path.stat().st_size+sum(p.stat().st_size for p in out.rglob("*.obj"))
        write_json(out/"geometry.json",meta)
        difference = grid_sample(surface["z"],surface["origin"],vertices[:,:2])-vertices[:,2]
        items.append(dict(station_m=station, station_span_m=[s.min(),s.max()], segment=i, mesh=rel,
                          v8_deck_seed_cells=len(ij), scan_returns=samples, lower_midpoint_separation_m=np.median(difference),
                          lower_projected_area_m2=cells*PITCH**2, **stat))
    write_json(FULL/"segments.json",all_segments)
    write_json(output,dict(patches=items,seconds=time.monotonic()-start))
    print("lower independently observed patches",len(items),flush=True)


if __name__ == "__main__":
    main()
