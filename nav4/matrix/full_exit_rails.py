"""Fit exit footbridge rail cylinders from unlabelled dense LAS returns."""
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.compose import place_hard, save_xml
from nav4.matrix.full_ground import FULL, grid_sample, source_patch
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import scene_dir
from nav4.matrix.objects import capsule, cylinder_section
from nav4.matrix.stairs import heading, path_at

# Targets were picked from the three LAS height-band projections of the bridge.
SIDES = {
    "left": (np.array([713.96, 714.51, 715.12, 715.70, 716.08]),
             np.array([.80,.745,.78,.87,.98])),
    "right": (np.array([713.77, 714.44, 715.01, 715.66, 716.36]),
              np.array([-.65,-.68,-.64,-.55,-.42])),
}


def fit_sections(cloud, path, sample, knots, edges, level, rng):
    along = np.linspace(knots[0],knots[-1],int(np.ceil(np.ptp(knots)/.19))+1)
    xy = path_at(path,along)
    axis = heading(path,along)
    normal = np.c_[-axis[:,1],axis[:,0]]
    base = sample(xy)
    tree = cKDTree(cloud[:,:2])
    fits = []
    for i, (p, t) in enumerate(zip(xy,along)):
        hint = p+np.interp(t,knots,edges)*normal[i]
        points = cloud[tree.query_ball_point(hint,.24)]
        delta = points[:,:2]-hint
        hits = (abs(delta@axis[i])<.115)&(abs(delta@normal[i])<.115)&(
            abs(points[:,2]-base[i]-level)<.20)
        points = points[hits]
        if len(points)<20:
            fits.append(None)
            continue
        uv = np.c_[(points[:,:2]-hint)@normal[i],points[:,2]-base[i]]
        bins, inverse = np.unique(np.floor(uv/.018).astype(int),axis=0,return_inverse=True)
        freq = np.bincount(inverse)
        best = bins[freq.argmax()]
        near = (abs(np.floor(uv/.018)-best)<2).all(1)
        y,z = np.median(uv[near],axis=0)
        prior = np.r_[hint+y*normal[i],z+base[i]]
        view = points[rng.choice(len(points),min(1500,len(points)),replace=False)]
        center,radius,record = cylinder_section(view,prior,np.r_[axis[i],0.],.026,.075)
        if not record["measured"]:
            center,radius = prior,.023
        if record.get("count",0)<8 and near.sum()<14:
            fits.append(None)
            continue
        fits.append(dict(station_m=t,center=center,radius=radius,section=record,
                         raw_returns=len(points),both_fit=record["measured"]))
    return fits


def main():
    record = FULL/"exit_rails.json"
    if record.exists():
        return
    _, path = route()
    window = path_at(path,np.arange(713.25,717.05,.05))
    raw,_ = source_patch(window.min(0)-1.40,window.max(0)+1.4)
    raw = raw[raw[:,3] == 0,:3]
    with np.load(FULL/"surface.npz") as f:
        z,origin = f["z"],f["origin"]
    sample = lambda p: grid_sample(z,origin,p)
    rng = np.random.default_rng(930)
    rail = []
    evidence = {}
    for side,(knots,lateral) in SIDES.items():
        results = []
        for name,level in (("top",.96),("middle",.47)):
            values = fit_sections(raw,path,sample,knots,lateral,level,rng)
            supported = [v for v in values if v is not None]
            evidence[f"{side}_{name}"] = values
            if len(supported)<5:
                continue
            radius = float(np.clip(np.median([v["radius"] for v in supported]),.020,.039))
            for a,b in zip(supported[:-1],supported[1:]):
                gap = b["station_m"]-a["station_m"]
                if gap > .67 or np.linalg.norm(a["center"]-b["center"])>.76:
                    continue
                shape = capsule(f"rail_exit_{side}_{name}_{len(rail):03d}",a["center"],b["center"],radius,
                                dict(measured_sections=[a["raw_returns"],b["raw_returns"]],gap_m=gap))
                rail.append(dict(shape,kind="rail_exit"))
            if name=="top":
                results=supported
        for i,a in enumerate(results[:1]+results[-1:]):
            head = np.array(a["center"])
            floor = sample(head[None,:2])[0]
            column = raw[((abs(raw[:,:2]-head[:2])<.10).all(1))&(raw[:,2]>floor+.12)&(raw[:,2]<head[2]-.12)]
            if len(np.unique(np.floor(column[:,2]/.06)))<5:
                continue
            xyz = np.median(column[:,:2],axis=0)
            item = capsule(f"rail_exit_{side}_post_{i}",np.r_[xyz,floor-.012],np.r_[xyz,head[2]],.030,
                           dict(vertical_returns=len(column)))
            rail.append(dict(item,kind="rail_exit"))
    out=scene_dir(FULL,14)
    scene=ET.parse(out/"scene.xml").getroot()
    for item in rail:
        place_hard(scene,item)
    save_xml(scene,out/"scene.xml")
    items=json.loads((FULL/"objects.json").read_text())+rail
    write_json(FULL/"objects.json",items)
    metadata=json.loads((FULL/"segments.json").read_text())
    metadata[-1]["objects"]["rail_exit"] = len(rail)
    metadata[-1]["bytes"]=(out/"scene.xml").stat().st_size+sum(p.stat().st_size for p in out.rglob("*.obj"))
    write_json(out/"geometry.json",metadata[-1])
    write_json(FULL/"segments.json",metadata)
    write_json(record,dict(capsules=len(rail), sections=evidence, geometry=rail))
    print("exit rails",len(rail),flush=True)


if __name__ == "__main__":
    main()
