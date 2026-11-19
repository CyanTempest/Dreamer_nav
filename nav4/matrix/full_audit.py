"""Measure serialized section seams and classify the original pilot's steep strips."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.full_ground import FULL, PILOT, PITCH
from nav4.matrix.full_scene import LIMITS, load_pilot_objects, raw_obj, scene_dir
from nav4.matrix.ground_mesh import in_polygon
from nav4.matrix.survey import OUT


def sections(path):
    root = ET.parse(path).getroot()
    assets = {n.get("name"): n.get("file") for n in root.findall("./asset/mesh")}
    for node in root.findall("./worldbody/geom"):
        name = node.get("name")
        if name in assets:
            yield node, path.parent/assets[name]


def pilot_near_vertical():
    stairs, items = load_pilot_objects()
    tree = cKDTree([x["geom"]["pos"][:2] for x in items
                    if x["kind"] in {"retaining_wall","wall_rock_or_root"}])
    tally = {}
    for label, scene in (("original",OUT/"scene.xml"),("rebuild",PILOT/"scene.xml")):
        near = {}
        for node, file in sections(scene):
            name = node.get("name")
            if not name.startswith(("terrain_","s00_filled_ground","s00_observed_ground")):
                continue
            vertices,faces = raw_obj(file)
            vertices += np.fromstring(node.get("pos"),sep=" ")
            v = vertices[faces]
            normal = np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
            area = np.linalg.norm(normal,axis=1)
            steep = (area>1e-10)&(abs(normal[:,2])<area*np.cos(np.deg2rad(75)))
            if not steep.any():
                continue
            selected = v[steep]
            center = selected[:,:,:2].mean(1)
            stair = np.zeros(len(center), bool)
            for step in stairs:
                stair |= in_polygon(center,step["polygon"],margin=.10)
            masonry = tree.query(center,workers=4)[0] <= .24
            span_z = np.ptp(selected[:,:,2],axis=1)
            long_axis = np.max(np.linalg.norm(selected[:,:,:2]-np.roll(selected[:,:,:2],1,axis=1),axis=2),axis=1)
            strip = (span_z>=.10) & (long_axis<=.04)
            for key,mask in (("stair_footprint_extended",stair),
                             ("near_masonry_or_rock",~stair&masonry),
                             ("other_slope",~stair&~masonry)):
                stat = near.setdefault(key,dict(near_vertical=0,steep_thin_strips=0))
                stat["near_vertical"] += int(mask.sum())
                stat["steep_thin_strips"] += int((mask&strip).sum())
        tally[label] = {k:near.get(k,dict(near_vertical=0,steep_thin_strips=0)) for k in
                        ("stair_footprint_extended","near_masonry_or_rock","other_slope")}
    write_json(PILOT/"slope_artifacts_by_region.json",tally)
    print(tally,flush=True)


def seam_vertices(active, station):
    owner = np.searchsorted(LIMITS[1:-1], station[:-1,:-1], side="right")
    owner[~active[:-1,:-1]] = -1
    vertices = {}
    for axis in (0,1):
        g = owner if axis==0 else owner.T
        a,b = g[:-1],g[1:]
        ii,jj = np.nonzero((a>=0)&(b>=0)&(a!=b))
        for i,j in zip(ii,jj):
            pair = tuple(sorted((int(a[i,j]),int(b[i,j]))))
            if axis==0:
                v = ((i+1,j),(i+1,j+1))
            else:
                v = ((j,i+1),(j+1,i+1))
            vertices.setdefault(pair, set()).update(v)
    return vertices


def seam_export():
    d = np.load(FULL/"surface.npz")
    base = d["origin"]
    by_pair = seam_vertices(d["active"],d["station"])
    needed = {}
    for pair, keys in by_pair.items():
        for i in pair:
            needed.setdefault(i,set()).update(keys)
    actual = {}
    for i, keys in needed.items():
        sub = scene_dir(FULL,i)/"scene.xml"
        point_dict = actual.setdefault(i,{})
        for node,file in sections(sub):
            if not node.get("name").startswith((f"s{i:02d}_observed_ground",f"s{i:02d}_filled_ground")):
                continue
            with file.open() as stream:
                world_offset = np.fromstring(node.get("pos"),sep=" ")
                vertices = []
                for line in stream:
                    if not line.startswith("v "):
                        break
                    vertices.append(np.fromstring(line[2:],sep=" ")+world_offset)
            p = np.asarray(vertices)
            ids = np.rint((p[:,:2]-base)/PITCH).astype(int)
            for q,xyz in zip(map(tuple,ids),p):
                if q in keys:
                    point_dict.setdefault(q,[]).append(xyz)
    reports = []
    for (a,b),keys in sorted(by_pair.items()):
        available = keys & actual[a].keys() & actual[b].keys()
        worst = max((np.linalg.norm(np.ptp(np.vstack([*actual[a][q],*actual[b][q]]),axis=0))
                     for q in available),default=None)
        reports.append(dict(segments=[a,b], shared_vertices_expected=len(keys), checked_vertices=len(available),
                            missing_vertices=len(keys)-len(available), max_obj_world_disagreement_m=worst))
    write_json(FULL/"seams.json",reports)
    print("seams",len(reports),"missing",sum(r["missing_vertices"] for r in reports),flush=True)


if __name__ == "__main__":
    import sys
    {"pilot":pilot_near_vertical,"seams":seam_export}[sys.argv[1]]()
