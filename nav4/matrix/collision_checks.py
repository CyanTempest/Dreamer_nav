"""CPU acceptance checks on serialized full_v2 geometry."""
import argparse
import copy
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nav4.mapping.common import write_json
from nav4.matrix import full_checks
from nav4.matrix.collision import FULL, V2, LIMITS, raw_obj
from nav4.matrix.full_scan import route
from nav4.matrix.stairs import heading, path_at

DOWN = np.array([0., 0., -1.])
VIS = np.array([1,0,1,0,1,0], np.uint8)
COL = np.array([0,0,0,1,0,0], np.uint8)


@lru_cache(maxsize=2)
def parsed(path):
    return ET.parse(path).getroot()


def subset(file, keep):
    """Absolute file references let the same extracted geoms compile in memory."""
    source = parsed(file)
    root = ET.Element("mujoco")
    for tag in ("compiler","option","visual","statistic"):
        node = source.find(tag)
        if node is not None:
            root.append(copy.deepcopy(node))
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    names = set()
    for node in source.find("worldbody"):
        if node.tag == "light":
            world.append(copy.deepcopy(node))
        elif keep(node):
            world.append(copy.deepcopy(node))
            names.update(node.get(k) for k in ("material","mesh","hfield") if node.get(k))
    for node in source.find("asset"):
        if node.get("name") not in names and node.tag != "texture":
            continue
        new = copy.deepcopy(node)
        if new.get("file"):
            new.set("file", str((file.parent/new.get("file")).resolve()))
        asset.append(new)
    return root


def local(file, xy, radius=7., visual_hard=False):
    def near(node):
        if node.get("group") == "5":
            return False
        if node.get("group") == "1" and not visual_hard:
            return False
        if node.get("name","").startswith("seg_") and visual_hard:
            return True
        if node.get("fromto"):
            ends = np.fromstring(node.get("fromto"), sep=" ").reshape(2,3)
            p = ends.mean(0)
            extent = np.linalg.norm(ends[0,:2]-p[:2])+.15
        else:
            p = np.fromstring(node.get("pos",""),sep=" ")
            extent = 5.  # 6.4 m square hfield, 3.2 m ground mesh or local convex object
        return p.size and np.linalg.norm(p[:2]-xy) <= radius+extent
    return subset(file,near)


def model_of(root):
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))


def ray(m,d,xy,groups):
    key = np.empty(1,np.int32)
    hit = mujoco.mj_ray(m,d,np.r_[xy,150.],DOWN,groups,1,-1,key)
    return 150.-hit if hit >= 0 else np.nan


def ground_geom(node):
    return (node.get("group") in ("0","2","4") or node.get("type")=="hfield"
            or node.get("name","").startswith("col_stair_"))


def fidelity():
    _, path = route()
    rows = []
    totals = []
    for k in range(len(LIMITS)-2):
        low = 0 if k==0 else LIMITS[k]+25
        high = 722.01 if k==13 else LIMITS[k+1]+25
        root = subset(V2/f"window_{k:02d}.xml", ground_geom)
        for node in root.findall("worldbody/geom"):
            node.set("rgba",".5 .5 .5 1")
        m = model_of(root)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        result = []
        for s in np.arange(low, high, .20):
            p, a = path_at(path,s), heading(path,s)
            normal = np.array([-a[1],a[0]])
            for t in np.linspace(-1.5,1.5,17):
                pt = p + t*normal
                z, h = ray(m,d,pt,VIS), ray(m,d,pt,COL)
                result.append([s,t,*pt,z,h])
        rows.extend(result)
        arr = np.array(result)
        diff = arr[:,5]-arr[:,4]
        val = diff[np.isfinite(diff)]
        totals.append(dict(window=k,n=len(val),rms_m=np.sqrt(np.mean(val**2)),p95_m=np.percentile(abs(val),95)))
        print(k,totals[-1],flush=True)
        del m,d
    np.savetxt(V2/"work"/"ground_rays.csv", rows, delimiter=",",fmt="%.8f",
               header="s_m,t_m,x,y,visual_z_m,collision_z_m",comments="")
    a = np.array(rows)
    dz = a[:,5]-a[:,4]
    valid = np.isfinite(dz)
    write_json(V2/"work"/"fidelity.json",dict(samples=len(dz),both=int(valid.sum()),
        visual_missing=int(np.isnan(a[:,4]).sum()), collision_missing=int(np.isnan(a[:,5]).sum()),
        unsupported_visual=int((np.isfinite(a[:,4]) & np.isnan(a[:,5])).sum()),
        rms_m=np.sqrt(np.mean(dz[valid]**2)),p95_m=np.percentile(abs(dz[valid]),95),
        max_m=abs(dz[valid]).max(),per_window=totals))


def hard_and_stairs():
    _, path = route()
    stairs = json.loads((FULL/"stair_geometry.json").read_text())
    with (FULL/"stairs.csv").open() as f:
        table = list(csv.DictReader(f))
    metrics = []
    hard = Counter()
    worst = (0., "")
    for i in range(len(LIMITS)-1):
        rel = Path("segments")/f"seg_{i:02d}"/"scene.xml"
        v1, v2 = parsed(FULL/rel),parsed(V2/rel)
        colliders = {n.get("name"): n for n in v2.findall("worldbody/geom") if n.get("contype")!="0"}
        meshes = {n.get("name"): n.get("file") for n in v1.findall("asset/mesh")}
        mesh2 = {n.get("name"): n.get("file") for n in v2.findall("asset/mesh")}
        for vis in v2.findall("worldbody/geom"):
            if vis.get("group") != "3":
                assert vis.get("contype")==vis.get("conaffinity")=="0"
        for node in v1.findall("worldbody/geom"):
            name = node.get("name")
            if node.get("group") not in ("1","2"):
                continue
            col = colliders.get("col_"+name)
            if col is None:
                hard["cropped"]+=1
                continue
            keys = ("type","pos","quat","size","fromto","mesh")
            assert all(node.get(k)==col.get(k) for k in keys)
            if node.get("mesh"):
                source = (FULL/rel.parent)/meshes[name]
                target = (V2/rel.parent)/mesh2[name]
                assert source.read_bytes()==target.read_bytes()
                v,f = raw_obj(target)
                # For each input face, all vertices must lie on one side of
                # its plane if it is a face of the collision hull.
                poly = v[f]
                normals = np.cross(poly[:,1]-poly[:,0],poly[:,2]-poly[:,0])
                length = np.linalg.norm(normals,axis=1)
                good = length>1e-12
                normals = normals[good]/length[good,None]
                dist = normals@v.T-(normals*poly[good,0]).sum(1)[:,None]
                error = max(0.,float(np.minimum(dist.max(1),-dist.min(1)).max()))
                if error>worst[0]:
                    worst = (error,name)
                if name.startswith("stair_"):
                    j = int(name.split("_")[1])
                    shape = stairs[j]
                    idx = j if "station_m" not in shape else min(range(len(table)),
                              key=lambda k: abs(float(table[k]["station_m"])-shape["station_m"]))
                    row = table[idx]
                    v += np.fromstring(node.get("pos"),sep=" ")
                    top = v[4:]
                    coeff = np.linalg.lstsq(np.c_[top[:,:2],np.ones(4)], top[:,2], rcond=None)[0]
                    xy = path_at(path,float(row["station_m"]))
                    previous = shape["before_plane_coeff"]
                    before = (xy-shape["before_plane_anchor"])@previous[:2]+previous[2]
                    rise = xy@coeff[:2]+coeff[2]-before
                    metrics.append(dict(name=name, station_m=float(row["station_m"]),csv_m=float(row["riser_m"]),
                        rise_m=rise,planarity_m=np.max(abs(np.c_[top[:,:2],np.ones(4)]@coeff-top[:,2])),
                        vertical_xy_m=np.max(np.linalg.norm(v[:4,:2]-top[:,:2],axis=1)),
                        slope_deg=np.degrees(np.arctan(np.linalg.norm(coeff[:2])))))
            hard["retained_"+node.get("type")]+=1
    write_json(V2/"work"/"solids.json",dict(counts=hard,maximum_face_plane_violation_m=worst[0],
               worst_mesh=worst[1],stairs=metrics))


def proxy():
    _, path = route()
    tests = full_checks.stairs_cases(path)+full_checks.narrow_cases(full_checks.load_rows(),path)
    cases = []
    for case in tests:
        root = local(V2/"scene_full.xml",case["xy"])
        result = full_checks.proxy_case(case,root)
        cases.append(dict(case,**result))
        print(case["name"],result.get("max_contact_overlap_m"),flush=True)
        write_json(V2/"physics.json",dict(cases=cases,settings=full_checks.CONFIG["physics"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase",choices=["fidelity","solids","proxy"])
    phase = p.parse_args().phase
    {"fidelity":fidelity,"solids":hard_and_stairs,"proxy":proxy}[phase]()
