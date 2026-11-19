"""Make measured trunk sections continuous and filter unsupported woody pieces."""
from collections import defaultdict
import json
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.cloud import hull
from nav4.matrix.compose import save_xml, vector, write_obj
from nav4.matrix.full_ground import FULL, PILOT, WORK, edge_statistics, grid_sample
from nav4.matrix.full_scene import raw_obj, scene_dir


def wood_group(item):
    name = item["name"]
    if item["kind"] == "trunk":
        return item.get("stem_id", name.rsplit("_", 1)[0])
    if "_auto_" in name:
        return name.split("_root_")[0].split("_low_branch_")[0]
    if name.startswith("root_"):
        return "trunk_"+"_".join(name.removeprefix("root_").split("_")[:-2])
    return name


def wood_chain(parts, sample):
    ordered = sorted(parts, key=lambda p: p["position"][2])
    first = ordered[0]
    xyz = first["vertices"]
    floor = sample(first["position"][None,:2])[0]
    if not np.isfinite(floor) or np.min(xyz[:,2]-sample(xyz[:,:2])) > .13:
        return []
    if floor < np.percentile(xyz[:,2],20)-.32:
        return []
    keep = [first]
    for piece in ordered[1:]:
        prev = keep[-1]
        delta = piece["position"]-prev["position"]
        gap = np.percentile(piece["vertices"][:,2],22)-np.percentile(prev["vertices"][:,2],78)
        if delta[2] <= 0 or delta[2] > .69 or np.linalg.norm(delta[:2]) > .30 or gap > .22:
            break
        keep.append(piece)

    az = np.linspace(0,2*np.pi,12,endpoint=False)
    ring = np.c_[np.cos(az),np.sin(az),np.zeros(len(az))]
    centers = np.array([part["position"] for part in keep])
    radius = np.array([part["diameter"] for part in keep])/2
    a = centers[0].copy()
    first_interface = np.percentile(keep[0]["vertices"][:,2],18)
    a[2] = min(floor-.025, first_interface)
    if len(keep) > 1:
        delta = centers[1]-centers[0]
        a[:2] += np.clip((a[2]-centers[0,2])/delta[2],-1,0)*delta[:2]
    b = centers[-1].copy()
    b[2] = np.percentile(keep[-1]["vertices"][:,2],86)
    knots = np.vstack([a, (centers[1:]+centers[:-1])*.5, b])
    sizes = np.r_[radius[0,0],(radius[:-1,1]+radius[1:,0])*.5,radius[-1,1]]
    pieces = []
    for k, part in enumerate(keep):
        if knots[k+1,2]-knots[k,2] < .035:
            break
        q = np.vstack([knots[k]+sizes[k]*ring-[0,0,.004],
                       knots[k+1]+sizes[k+1]*ring+[0,0,.004]])
        fitted = hull(q,.025)
        if fitted is None:
            break
        pieces.append((part,fitted,knots[k],knots[k+1]))
    return pieces


def main(pilot):
    start = time.monotonic()
    out = PILOT if pilot else FULL
    marker = out/"woody_support_audit.json"
    if marker.exists():
        return
    source = json.loads((out/"objects.json").read_text())
    items = {item["name"]:item for item in source}
    metadata = json.loads((out/"segments.json").read_text())
    raster = np.load(out/"surface.npz")
    z,b = raster["z"],raster["origin"]
    sample = lambda xy: grid_sample(z,b,xy)
    stacks, roots, stages = defaultdict(list), [], []
    for i in range(len(metadata)):
        directory = scene_dir(out,i)
        scene = ET.parse(directory/"scene.xml").getroot()
        world = scene.find("worldbody")
        assets = scene.find("asset")
        names = {n.get("name"): n for n in assets.findall("mesh")}
        stages.append((scene,directory))
        for node in world.findall("geom"):
            name = node.get("name")
            if name not in items:
                continue
            info = items[name]
            kind = info["kind"]
            if kind in ("trunk","root"):
                g = names[name]
                position = np.fromstring(node.get("pos"),sep=" ")
                file = directory/g.get("file")
                v,f = raw_obj(file)
                parts = dict(section=i, name=name, scene=scene, node=node, asset=g, file=file, kind=kind,
                             position=position, vertices=v+position, faces=f,
                             diameter=info.get("diameters_m",info.get("diameter_m")))
                if kind=="trunk":
                    stacks[wood_group(info)].append(parts)
                else:
                    roots.append(parts)
            elif kind == "low_branch":
                branch = np.fromstring(node.get("fromto"),sep=" ").reshape(2,3)
                roots.append(dict(section=i, name=name, scene=scene, node=node, kind=kind,
                                  vertices=branch, asset=None))
    updates, remove, supported_groups = [], [], {}
    by_name = {}
    for name, parts in stacks.items():
        stems = wood_chain(parts,sample)
        retained = {item["name"] for item,_,_,_ in stems}
        remove.extend((p,"disconnected stem section") for p in parts if p["name"] not in retained)
        if not stems:
            continue
        supported_groups[name] = stems
        for item, fitted, a, b_ in stems:
            center,v,f = fitted
            struct = items[item["name"]]
            old = edge_statistics(item["vertices"],item["faces"])
            new = edge_statistics(v,f)
            stat = metadata[item["section"]]["surface"]["woody_solids"]
            for k in old:
                stat[k] += new[k]-old[k]
            item["node"].set("pos",vector(center))
            write_obj(item["file"],v,f)
            struct["geom"]["pos"] = center
            struct["fitted_axis_knots"] = [a,b_]
            by_name[item["name"]] = struct
            updates.append(item["name"])
    centers = [part["position"] for pieces in supported_groups.values() for part,_,_,_ in pieces]
    nearby_tree = cKDTree(centers) if centers else None
    for p in roots:
        floors = sample(p["vertices"][:,:2])
        group = wood_group(items[p["name"]])
        if p["kind"] == "root" and not np.any(p["vertices"][:,2]-floors > .045):
            remove.append((p,"root completely covered by ground"))
            continue
        if p["kind"] == "root" and np.any(p["vertices"][:,2]-floors < .055):
            if group in supported_groups or nearby_tree is not None and (
                    nearby_tree.query(p["vertices"])[0].min() <= .90):
                continue
        stems = supported_groups.get(group, [])
        attached = any(np.linalg.norm(p["vertices"]-a,axis=1).min() < .22+part["diameter"][0]/2
                       for part,_,a,_ in stems[:4])
        if not attached:
            remove.append((p,"root or branch lacks ground/trunk attachment"))

    audit = []
    ignored = set()
    for item, reason in remove:
        scene = item["scene"]
        scene.find("worldbody").remove(item["node"])
        meta = metadata[item["section"]]
        kind = item["kind"]
        meta["objects"][kind] -= 1
        if item["asset"] is not None:
            scene.find("asset").remove(item["asset"])
            record = edge_statistics(item["vertices"],item["faces"])
            for k,v in record.items():
                meta["surface"]["woody_solids"][k] -= v
            archive = WORK/"excluded_wood"/out.name/f"seg_{item['section']:02d}"
            archive.mkdir(parents=True,exist_ok=True)
            item["file"].replace(archive/item["file"].name)
        ignored.add(item["name"])
        audit.append(dict(name=item["name"],kind=kind,section=item["section"],reason=reason))
    for i,(scene,directory) in enumerate(stages):
        p = directory/"scene.xml"
        save_xml(scene,p)
        metadata[i]["bytes"] = p.stat().st_size+sum(a.stat().st_size for a in directory.rglob("*.obj"))
        write_json(directory/"geometry.json",metadata[i])
    write_json(out/"segments.json",metadata)
    write_json(out/"objects.json",[by_name.get(o["name"],o) for o in source if o["name"] not in ignored])
    write_json(marker,dict(grounded_stems=len(supported_groups), continuous_trunk_pieces=len(updates),
                           removed=audit, seconds=time.monotonic()-start))
    print("wood supports",len(supported_groups),"retained trunk pieces",len(updates),
          "filtered pieces",len(ignored),flush=True)


if __name__ == "__main__":
    import sys
    main("--pilot" in sys.argv)
