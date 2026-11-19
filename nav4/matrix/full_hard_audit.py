"""Retain ground-connected masonry and close measured small rail gaps."""
from collections import deque
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.compose import place_hard, save_xml
from nav4.matrix.full_ground import FULL, WORK, edge_statistics, grid_sample
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import LIMITS, raw_obj, scene_dir
from nav4.matrix.survey import Corridor


def connected_rock(rocks, ground):
    if not rocks:
        return np.empty(0,bool), []
    center = np.array([r["world"].mean(0) for r in rocks])
    normals, bottoms = [], []
    for rock in rocks:
        q = rock["world"]
        _, axes = np.linalg.eigh((q-q.mean(0)).T@(q-q.mean(0)))
        normals.append(axes[:,0])
        bottoms.append(float(np.min(q[:,2]-ground(q[:,:2]))))
    normals = np.asarray(normals)
    near = cKDTree(center).query_pairs(.66, output_type="ndarray")
    edges = [[] for _ in rocks]
    for a,b in near[abs(np.sum(normals[near[:,0]]*normals[near[:,1]], axis=1)) >= .72]:
        p, q = rocks[a]["world"],rocks[b]["world"]
        if np.min(np.sum((p[:,None]-q[None])**2,axis=2)) <= .115**2:
            edges[a].append(b)
            edges[b].append(a)
    marked = np.asarray(bottoms) <= .16
    todo = deque(np.flatnonzero(marked))
    while todo:
        a = todo.popleft()
        for b in edges[a]:
            if not marked[b]:
                marked[b] = True
                todo.append(b)
    return marked,bottoms


def link_rail(items, corridor):
    recorded = []
    for item in items:
        if item["kind"] != "rail_scan":
            continue
        a = np.asarray(item["geom"]["fromto"]).reshape(2,3)
        length = np.linalg.norm(a[1]-a[0])
        if (length < .19) or (abs(a[1,2]-a[0,2]) / length > .64):
            continue
        s,l,_ = corridor.project(a[:,:2])
        if s[0] > s[1]:
            a,s,l = a[::-1],s[::-1],l[::-1]
        if s[1]-s[0] < .16 or l[0]*l[1] < 0:
            continue
        recorded.append(dict(name=item["name"], xyz=a, s=s, l=l, radius=item["geom"]["size"][0]))
    if not recorded:
        return []
    beginnings = cKDTree([x["xyz"][0] for x in recorded])
    inferred, claimed = [], set()
    for ring in sorted(recorded,key=lambda r:r["s"][0]):
        a,b = ring["xyz"]
        unit = (b-a)/np.linalg.norm(b-a)
        possible = []
        for j in beginnings.query_ball_point(b,1.22):
            next_ = recorded[j]
            if j in claimed:
                continue
            c,d = next_["xyz"]
            one = c-b
            dist = np.linalg.norm(one)
            ds = next_["s"][0]-ring["s"][1]
            if ds<.065 or ds>1.18 or dist<.09 or abs(ring["l"][1]-next_["l"][0])>.19:
                continue
            u = one/dist
            v = (d-c)/np.linalg.norm(d-c)
            if min(u@unit,u@v)<.85 or abs(ring["radius"]-next_["radius"])>.025:
                continue
            # A gap links two rails of consistent direction and lateral side.
            # Reject links with an already observed rail body in the gap.
            mid = (b+c)*.5
            overlaps = beginnings.query_ball_point(mid,.23)
            if any(k != j and abs(recorded[k]["l"][0]-ring["l"][1])<.16 for k in overlaps):
                continue
            possible.append((dist,j,c))
        if not possible:
            continue
        distance,j,c = min(possible,key=lambda q:q[0])
        other = recorded[j]
        claimed.add(j)
        name = f"rail_occlusion_{len(inferred):04d}"
        inferred.append(dict(name=name,kind="rail_inferred",material="rail",
                             geom=dict(type="capsule",fromto=np.r_[b,c],
                                       size=[(ring["radius"]+other["radius"])*.5]),
                             end_scans=[ring["name"],other["name"]],gap_m=distance))
    return inferred


def main():
    start=time.monotonic()
    marker=FULL/"hard_support_audit.json"
    if marker.exists():
        return
    metadata=json.loads((FULL/"segments.json").read_text())
    items=json.loads((FULL/"objects.json").read_text())
    rocks, roots = [], []
    for i in range(len(metadata)):
        directory=scene_dir(FULL,i)
        root=ET.parse(directory/"scene.xml").getroot()
        roots.append(root)
        names={n.get("name"):n for n in root.findall("./asset/mesh")}
        for node in root.findall("./worldbody/geom"):
            key=node.get("name")
            if not key.startswith("rock_"):
                continue
            asset=names[key]
            file=directory/asset.get("file")
            v,f=raw_obj(file)
            p=np.fromstring(node.get("pos"),sep=" ")
            rocks.append(dict(i=i,key=key,node=node,asset=asset,file=file,world=v+p,faces=f))
    with np.load(FULL/"surface.npz") as f:
        z=f["z"]
        origin=f["origin"]
    valid,gap=connected_rock(rocks,lambda xy:grid_sample(z,origin,xy))
    removed=[]
    for r,keep,g in zip(rocks,valid,gap):
        if keep:
            continue
        i=r["i"]
        root=roots[i]
        root.find("worldbody").remove(r["node"])
        root.find("asset").remove(r["asset"])
        segment=metadata[i]
        segment["objects"]["rock_face"]-=1
        stat=edge_statistics(r["world"],r["faces"])
        for k,v in stat.items():
            segment["surface"]["walls_and_rocks"][k]-=v
        archive=WORK/"disconnected_rock"/f"seg_{i:02d}"
        archive.mkdir(parents=True,exist_ok=True)
        r["file"].replace(archive/r["file"].name)
        removed.append(dict(name=r["key"],segment=i,minimum_height_over_ground_m=g))
    names={r["name"] for r in removed}
    items=[i for i in items if i["name"] not in names]
    corridor=Corridor(route()[1])
    extra=link_rail(items,corridor)
    for rail in extra:
        center=rail["geom"]["fromto"].reshape(2,3).mean(0)
        s=corridor.project(center[None,:2])[0][0]
        i=int(np.searchsorted(LIMITS[1:-1],s))
        place_hard(roots[i],rail)
        segment=metadata[i]
        segment["objects"]["rail_inferred"]=segment["objects"].get("rail_inferred",0)+1
    for i,root in enumerate(roots):
        directory=scene_dir(FULL,i)
        scene=directory/"scene.xml"
        save_xml(root,scene)
        metadata[i]["bytes"]=scene.stat().st_size+sum(x.stat().st_size for x in directory.rglob("*.obj"))
        write_json(directory/"geometry.json",metadata[i])
    write_json(FULL/"objects.json",items+extra)
    write_json(FULL/"segments.json",metadata)
    write_json(marker,dict(removed_disconnected_rock=removed, retained_rock=int(valid.sum()),
                           rail_gap_bridges=len(extra), bridged_length_m=sum(a["gap_m"] for a in extra),
                           seconds=time.monotonic()-start))
    print("rock faces retained",int(valid.sum()),"discarded", len(removed), "rail gaps bridged",len(extra),flush=True)


if __name__ == "__main__":
    main()
