"""Build full_v2 from the frozen full_v1 surface and its visual assets."""
import copy
import filecmp
import os
import tempfile
import json
import shutil
import struct
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from nav4.mapping.common import write_json
from nav4.matrix.compose import save_xml, write_obj, xml
from nav4.matrix.full_ground import FULL, PITCH, grid_sample
from nav4.matrix.full_package import shift_world
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import LIMITS, raw_obj
from nav4.matrix.package import prefix_assets
from nav4.matrix.ground_mesh import in_polygon
from nav4.matrix.stairs import path_at, prism

V2 = FULL.with_name("full_v2")
MODEL = Path("matrix/UeSim/Content/model/zgws")
WIDTH = 4.
TILE = 32  # 1.6 m on the original 5 cm lattice
CONTACT = dict(group="3", rgba="0 0 0 0", contype="1", conaffinity="0")


def install_asset(src, dst, link=False):
    """Private copies from v1. Within v2, share identical bytes by hard link."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        same_inode = src.samefile(dst)
        if link and same_inode:
            return dst
        if not link and not same_inode and filecmp.cmp(src,dst,shallow=False):
            return dst
    with tempfile.TemporaryDirectory(dir=dst.parent) as temp:
        tmp = Path(temp)/dst.name
        if link:
            try:
                os.link(src,tmp)
            except OSError:
                shutil.copy2(src,tmp)
        else:
            shutil.copy2(src,tmp)
        os.replace(tmp,dst)
    return dst


def link_package_asset(src,dst):
    return install_asset(src,dst,link=True)


def heightfield(root, directory, name, origin, z, pitch, thickness=.025):
    """z[x,y] in world metres. MuJoCo's binary asset stores rows along Y."""
    span = np.maximum(np.ptp(z), 1.e-6)
    lo = float(z.min())
    rel = Path("collision") / (name + ".bin")
    path = directory / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.ascontiguousarray(((z.T-lo)/span), dtype="<f4")
    path.write_bytes(struct.pack("<ii", *rows.shape) + rows.tobytes())
    half = (np.array(z.shape)-1) * pitch / 2
    xml(root.find("asset"), "hfield", name=name, file=str(rel), size=[*half, span, thickness])
    return xml(root.find("worldbody"), "geom", type="hfield", name=name, hfield=name,
               pos=[*(origin + half), lo], friction="1 .005 .0001", solref=".008 1", **CONTACT)


def hard_copy(node, tree, vertices=None):
    """Retain shapes whose horizontal bounding disk intersects the swept corridor."""
    if node.get("fromto"):
        ends = np.fromstring(node.get("fromto"), sep=" ").reshape(2, 3)[:, :2]
        center = ends.mean(0)
        radius = np.linalg.norm(ends[0]-center) + float(node.get("size"))
    else:
        center = np.fromstring(node.get("pos"), sep=" ")[:2]
        radius = np.linalg.norm(vertices[:, :2], axis=1).max() if vertices is not None else np.linalg.norm(
            np.fromstring(node.get("size"), sep=" "))
    if tree.query(center)[0] > WIDTH + radius + .026:
        return None
    collider = copy.deepcopy(node)
    collider.attrib.pop("material", None)
    collider.attrib.update(CONTACT, name="col_" + node.get("name"))
    return collider


def lower_fields(root, directory, node, obj):
    # Preserve all observed lower cells, including holes. Each contiguous one-cell
    # strip gets its own thin heightfield, independent of the bearing surface.
    vertices, faces = raw_obj(obj)
    vertices += np.fromstring(node.get("pos"), sep=" ")
    origin = vertices[:, :2].min(0)
    ij = np.rint((vertices[:, :2]-origin)/PITCH).astype(int)
    grid = np.full(ij.max(0)+1, np.nan)
    grid[tuple(ij.T)] = vertices[:, 2]
    cells = np.unique(ij[faces].min(axis=1), axis=0)
    for i in np.unique(cells[:, 0]):
        js = cells[cells[:, 0] == i, 1]
        for run in np.split(js, np.flatnonzero(np.diff(js) > 1)+1):
            j, end = run[0], run[-1]+2
            heightfield(root, directory, f"lower_{node.get('name')}_{i}_{j}",
                        origin + np.array([i,j])*PITCH, grid[i:i+2, j:end], PITCH)
    return vertices[:, :2].min(0), vertices[:, :2].max(0)


def clear_stairs(roots, dirs, ground, origin, reference, stairs):
    """Keep the hfield beneath each measured plane all the way to its edge."""
    previous_ends = [np.array(s["polygon"])[[1,2]] for s in stairs]
    sections = {g.get("name"):i for i,root in enumerate(roots)
                for g in root.findall("worldbody/geom") if g.get("group")=="2"}

    patches = []

    def recess(polygon, plane):
        patches.append((polygon,plane))
        # A heightfield triangle can touch a footprint with all three sample
        # vertices outside it. Include the full diagonal of one 5 cm cell.
        lo = np.maximum(0,np.floor((polygon.min(0)-origin-.08)/PITCH).astype(int))
        hi = np.minimum(ground.shape,np.ceil((polygon.max(0)-origin+.08)/PITCH).astype(int)+1)
        ij = np.indices(hi-lo).reshape(2,-1).T+lo
        xy = origin+ij*PITCH
        inside = in_polygon(xy,polygon,margin=.072)
        ids = tuple(ij[inside].T)
        ground[ids] = np.minimum(ground[ids],xy[inside]@plane[:2]+plane[2]-.006)

    for j,shape in enumerate(stairs):
        poly=np.array(shape["polygon"])
        top=np.array(shape["vertices"])[4:]
        plane=np.linalg.lstsq(np.c_[top[:,:2],np.ones(len(top))],top[:,2],rcond=None)[0]
        recess(poly,plane)
        a,b=edge=np.array(shape["edge"])
        if any(np.max(abs(end-edge))<1e-5 for end in previous_ends):
            continue
        # A disconnected flight has no preceding solid to expose the measured
        # before-plane. Cap a short approach using that fit, without changing
        # any of the source tread faces or the original visual ground.
        direction=np.array([b[1]-a[1],a[0]-b[0]])
        direction/=np.linalg.norm(direction)
        if direction@(poly.mean(0)-edge.mean(0))<0:
            direction*=-1
        before=np.array(shape["before_plane_coeff"])
        before[2]-=np.array(shape["before_plane_anchor"])@before[:2]
        # Do not extrapolate a missing landing sideways over a drop where
        # even the lower of the two fitted surfaces has no v1 ground support.
        u=np.linspace(0,1,81)
        test=a+u[:,None]*(b-a)-.04*direction
        supported=grid_sample(reference,origin,test)>=np.minimum(test@before[:2]+before[2],
                                                                 test@plane[:2]+plane[2])-.04
        mid=len(u)//2
        left=right=mid
        while left>0 and supported[left-1]: left-=1
        while right+1<len(u) and supported[right+1]: right+=1
        left=min(left,mid-3); right=max(right,mid+3)
        a,b=a+u[left]*(b-a),a+u[right]*(b-a)
        apron=np.array([a-.12*direction,a,b,b-.12*direction])
        heights=apron@before[:2]+before[2]
        recess(apron,before)
        v,f=prism(apron,heights,heights.min()-.08)
        center=v.mean(0)
        i=sections[f"stair_{j:04d}"]
        name=f"col_stair_approach_{j:04d}"
        rel=f"collision/{name}.obj"
        (dirs[i]/"collision").mkdir(exist_ok=True)
        write_obj(dirs[i]/rel,v-center,f)
        xml(roots[i].find("asset"),"mesh",name=name,file=rel)
        xml(roots[i].find("worldbody"),"geom",type="mesh",mesh=name,name=name,pos=center,**CONTACT)

    return patches


def terrain_tiles(roots, dirs, surface, path, layers):
    origin, ground = surface["origin"], surface["z"]
    reference = ground.copy()
    stairs = json.loads((FULL/"stair_geometry.json").read_text())
    stair_patches = clear_stairs(roots, dirs, ground, origin, reference, stairs)
    bounds = np.array([[poly.min(0),poly.max(0)] for poly,_ in stair_patches])
    # Only boundary patches need a fine underlay; the closed tread supports
    # the whole interior and normal terrain keeps the source lattice.
    edges = Counter(tuple(sorted([tuple(np.round(a,6)),tuple(np.round(b,6))]))
                    for poly,_ in stair_patches for a,b in zip(poly,np.roll(poly,-1,axis=0)))
    outer = np.array([edge for edge,count in edges.items() if count==1])
    outer_lo,outer_hi = outer.min(1),outer.max(1)
    tree = cKDTree(path)
    # Rasterize the corridor at 0.8 m for its outline. Pack adjacent cells
    # into rectangles with identical section membership (including seam overlap).
    cell = TILE // 2
    cell_m = cell * PITCH
    seeds = np.unique(np.floor((path-origin)/cell_m).astype(int), axis=0)
    halo = int(np.ceil(WIDTH/cell_m))+1
    offsets = np.indices((2*halo+1,)*2).reshape(2,-1).T-halo
    cells = np.unique((seeds[:,None,:]+offsets).reshape(-1,2), axis=0)
    owners = [cKDTree(path[int(a*20):int(b*20)+1]) for a,b in zip(LIMITS,LIMITS[1:])]
    active = surface["active"]
    packed = defaultdict(set)

    def emit(lo, hi, ids):
        # Resolve the original 5 cm vertices throughout the walkable band.
        mid = origin+(hi+lo)*PITCH/2
        near_walkable = tree.query(mid)[0] <= 1.55 + np.linalg.norm(hi-lo)*PITCH/2
        stride = 1 if near_walkable or np.min(hi-lo)<2 else 2
        low, high = origin+lo*PITCH, origin+hi*PITCH
        close = np.flatnonzero(np.all(high+.10>=bounds[:,0],axis=1) &
                               np.all(low-.10<=bounds[:,1],axis=1))
        pitch = stride*PITCH
        values = ground[lo[0]:hi[0]+1:stride, lo[1]:hi[1]+1:stride]
        if len(close):
            corners=np.array([low,high,[low[0],high[1]],[high[0],low[1]]])
            if any(in_polygon(corners,stair_patches[j][0],margin=-1e-6).all() for j in close):
                return
            if np.any(np.all(high+.1>=outer_lo,axis=1) & np.all(low-.1<=outer_hi,axis=1)):
                pitch=PITCH/4
                size=(hi-lo)*4+1
                xy=low+np.indices(size).reshape(2,-1).T*pitch
                h=grid_sample(reference,origin,xy)
                for j in close:
                    poly,plane=stair_patches[j]
                    use=in_polygon(xy,poly,margin=pitch*np.sqrt(2)+1e-6)
                    h[use]=np.minimum(h[use],xy[use]@plane[:2]+plane[2]-.006)
                values=h.reshape(size)
        name = "ground_" + "_".join(map(str,[*lo,*hi]))
        for i in ids:
            heightfield(roots[i], dirs[i], name, origin+lo*PITCH, values, pitch)

    def select(lo, size, ids):
        hi = lo+size
        valid = active[lo[0]:hi[0],lo[1]:hi[1]]
        if not valid.any():
            return
        low, high = origin+lo*PITCH, origin+hi*PITCH
        thin = any(np.all(high>=a-.15) and np.all(low<=b+.15) for a,b in layers)
        if (not valid.all() or (thin and size>8)) and size>1:
            for offset in ((0,0),(1,0),(0,1),(1,1)):
                select(lo+np.array(offset)*(size//2), size//2, ids)
        elif np.any(np.all(high+.12>=bounds[:,0],axis=1) & np.all(low-.12<=bounds[:,1],axis=1)):
            emit(lo,hi,ids)
        elif size == cell and not thin:
            packed[ids].add(tuple(lo//cell))
        else:
            emit(lo, hi, ids)

    for ij in cells:
        lo = ij*cell
        if np.any(lo<0) or np.any(lo+cell>=ground.shape):
            continue
        center = origin+(lo+.5*cell)*PITCH
        radius = cell_m/np.sqrt(2)
        dist, near = tree.query(center)
        if dist>WIDTH+radius+.026:
            continue
        ids = tuple(i for i,(a,b) in enumerate(zip(LIMITS,LIMITS[1:]))
                    if a-12<=near*.05<=b+12 and owners[i].query(center)[0]<=WIDTH+radius+.026)
        select(lo, cell, ids)
    for ids, remaining in packed.items():
        for x,y in sorted(remaining):
            if (x,y) not in remaining:
                continue
            nx = ny = 1
            while nx<8 and (x+nx,y) in remaining:
                nx += 1
            while ny<8 and all((j,y+ny) in remaining for j in range(x,x+nx)):
                ny += 1
            remaining.difference_update((j,k) for j in range(x,x+nx) for k in range(y,y+ny))
            emit(np.array([x,y])*cell, np.array([x+nx,y+ny])*cell, ids)


def batch_hard_visuals(root, dst, assets):
    # Different rigid convex scans can share one strictly visual OBJ. Keep
    # the original indices/winding within each component; never hull the batch.
    groups = defaultdict(list)
    world = root.find("worldbody")
    for node in world.findall("geom"):
        if node.get("group") == "1" and node.get("mesh"):
            assert not (set(node.attrib) & {"quat","euler","axisangle"})
            groups[node.get("material")].append(node)
    asset = root.find("asset")
    for material, group in groups.items():
        verts, faces, count = [], [], 0
        for node in group:
            v,f = raw_obj(dst/assets[node.get("mesh")])
            verts.append(v + np.fromstring(node.get("pos"), sep=" "))
            faces.append(f+count)
            count += len(v)
            world.remove(node)
        v = np.concatenate(verts)
        pos = v.mean(0)
        name = f"{dst.name}_visual_{material}"
        file = f"visual/{name}.obj"
        (dst/"visual").mkdir(exist_ok=True)
        write_obj(dst/file, v-pos, np.concatenate(faces))
        xml(asset,"mesh",name=name,file=file,inertia="shell")
        xml(world,"geom",name=name,type="mesh",mesh=name,material=material,group=1,
            pos=pos,contype=0,conaffinity=0)
    used = {n.get("mesh") for n in world.findall("geom")}
    for node in asset.findall("mesh"):
        if node.get("name") not in used:
            asset.remove(node)


def assemble(indices, prefix, shift=None):
    root = None
    unique = set()
    for i in indices:
        seg = f"seg_{i:02d}"
        part = ET.parse(V2/"segments"/seg/"scene.xml").getroot()
        prefix_assets(part, prefix / seg)
        if shift is not None:
            shift_world(part, shift)
        if root is None:
            root = part
            unique.update(x.get("name") for x in part.find("worldbody").findall("geom"))
            continue
        for section, tags in (("asset", ("mesh","hfield")), ("worldbody", ("geom",))):
            root.find(section).extend(x for x in part.find(section)
                                      if x.tag in tags and x.get("name") not in unique)
        unique.update(x.get("name") for x in part.find("worldbody").findall("geom"))
    if len(indices)==2:
        # A fitted solid is owned by its centre. Its tread or rail can extend
        # across the cut of a window. Transfer the neighbours' local colliders
        # at the two outer ends, retaining the original asset and world pose.
        _, path=route()
        for neighbour,station in ((indices[0]-1,LIMITS[indices[0]]),
                                  (indices[-1]+1,LIMITS[indices[-1]+1])):
            if not 0<=neighbour<len(LIMITS)-1:
                continue
            tag=f"seg_{neighbour:02d}"
            source=ET.parse(V2/"segments"/tag/"scene.xml").getroot()
            anchor=path_at(path,station)
            mesh_names=set()
            chosen=[]
            for g in source.findall("worldbody/geom"):
                if g.get("group")!="3" or g.get("type")=="hfield":
                    continue
                if g.get("fromto"):
                    ends=np.fromstring(g.get("fromto"),sep=" ").reshape(2,3)
                    p=ends.mean(0)
                    radius=np.linalg.norm(ends[0]-p)+float(g.get("size"))
                else:
                    p=np.fromstring(g.get("pos"),sep=" ")
                    radius=3.  # local convex parts, including long stair landings
                if np.linalg.norm(p[:2]-anchor)<=WIDTH+1.+radius:
                    chosen.append(g)
                    mesh_names.add(g.get("mesh"))
            prefix_assets(source,prefix/tag)
            if shift is not None:
                shift_world(source,shift)
            root.find("worldbody").extend(chosen)
            root.find("asset").extend(m for m in source.findall("asset/mesh") if m.get("name") in mesh_names)
    return root


def build():
    _, path = route()
    tree = cKDTree(path)
    roots, dirs, layers, stats = [], [], [], []
    manifest = json.loads((FULL/"assembly.json").read_text())
    shift = manifest["shared_shift_m"]
    for i in range(len(LIMITS)-1):
        rel = Path("segments")/f"seg_{i:02d}"
        src, dst = FULL/rel, V2/rel
        dst.mkdir(parents=True, exist_ok=True)
        root = ET.parse(src/"scene.xml").getroot()
        # This flag controls diagnostic BVH coloring only. MuJoCo 3.3.0
        # otherwise clears one byte per BVH node, including the visual meshes,
        # on every collision pass.
        root.find("visual/global").set("bvactive", "false")
        assets = {a.get("name"): a.get("file") for a in root.find("asset").findall("mesh")}
        for sub in ("terrain","stairs","hard"):
            if (src/sub).exists():
                shutil.copytree(src/sub, dst/sub, copy_function=install_asset, dirs_exist_ok=True)
        world = root.find("worldbody")
        counts = Counter()
        flex_faces = Counter()
        for node in list(world):
            name = node.get("name", "")
            if node.tag == "flexcomp":
                obj = src/node.get("file")
                category = next(k for k in ("observed_ground","filled_ground","lower_layer") if k in name)
                with obj.open("rb") as stream:
                    flex_faces[category] += sum(line.startswith(b"f ") for line in stream)
                if category == "lower_layer":
                    layers.append(lower_fields(root, dst, node, obj))
                counts["v1_flexcomp"] += 1
                world.remove(node)
            elif node.tag == "geom" and node.get("contype") != "0":
                v = raw_obj(src/assets[node.get("mesh")])[0] if node.get("mesh") else None
                collider = hard_copy(node, tree, v)
                counts["v1_"+node.get("type")] += 1
                if collider is not None:
                    world.append(collider)
                    counts["v2_"+node.get("type")] += 1
                node.attrib.update(contype="0", conaffinity="0")
            elif node.tag == "geom":
                assert node.get("conaffinity") == "0"
        batch_hard_visuals(root, dst, assets)
        roots.append(root)
        dirs.append(dst)
        stats.append(dict(range_m=LIMITS[i:i+2], geoms=counts, source_flex_faces=flex_faces))
    with np.load(FULL/"surface.npz") as surface:
        terrain_tiles(roots, dirs, surface, path, layers)
    for i,(root,dst,row) in enumerate(zip(roots, dirs, stats)):
        row["heightfields"] = len(root.findall("asset/hfield"))
        save_xml(root, dst/"scene.xml")
        write_json(dst/"geometry.json", row)
        shutil.copytree(dst, V2/MODEL/"atec_full"/dst.name, copy_function=link_package_asset,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("scene.xml","geometry.json"))
    write_json(V2/"segments.json", stats)
    for f in ("stairs.csv","stairs.json","stair_geometry.json","lower_layers.json","objects.json"):
        install_asset(FULL/f, V2/f)
    shutil.copytree(FULL/MODEL/"assets", V2/MODEL/"assets", copy_function=install_asset, dirs_exist_ok=True)
    for label,indices in [(f"window_{i:02d}", [i,i+1]) for i in range(len(roots)-1)] + [("full",range(len(roots)))]:
        static = "scene_full" if label == "full" else label
        save_xml(assemble(indices, Path("segments")), V2/(static+".xml"))
        save_xml(assemble(indices, Path("atec_full"), shift), V2/MODEL/f"atec_scene_{label}.xml")
        for f in (f"zgws_atec_robot_{label}.xml", f"zgws_atec_{label}.xml"):
            shutil.copyfile(FULL/MODEL/f, V2/MODEL/f)
    write_json(V2/"assembly.json", manifest)


if __name__ == "__main__":
    from nav4.matrix.collision_spawn import main as prepare_spawns

    build()
    prepare_spawns()
