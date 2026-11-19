"""Check exposed stair edges, convex closures, and load on the multilayer patches."""
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from nav4.mapping.common import write_json
from nav4.matrix.collision import FULL,V2,raw_obj
from nav4.matrix.collision_checks import COL,VIS,ground_geom,local,model_of,parsed,ray,subset
from nav4.matrix.compose import xml
from nav4.matrix.full_scan import route
from nav4.matrix.stairs import heading,path_at


def points_on(triangle,spacing):
    edge=np.max(np.linalg.norm(triangle-np.roll(triangle,1,axis=0),axis=1))
    n=max(1,int(np.ceil(edge/spacing)))
    yz=np.array([(j/n,k/n) for j in range(n+1) for k in range(n-j+1)])
    return triangle[0]+yz@(triangle[1:]-triangle[0])


def directed_bound(source,target,spacing=.008):
    mesh=trimesh.Trimesh(vertices=target.reshape(-1,3),faces=np.arange(target.size//3).reshape(-1,3),process=False)
    worst=0.
    for triangle in source:
        pts=points_on(triangle,spacing)
        for sample in np.array_split(pts,1+len(pts)//300):
            _,dist,_=trimesh.proximity.closest_point(mesh,sample)
            worst=max(worst,dist.max())
    # Distance to a closed set is 1-Lipschitz and every microtriangle has
    # diameter <= spacing, so this upper bound includes unsampled points.
    return float(worst+spacing)


def audit():
    scopes={}
    bounds=[]
    max_flat=0.
    for i in range(15):
        rel=Path("segments")/f"seg_{i:02d}"/"scene.xml"
        v1=parsed(FULL/rel)
        v2=parsed(V2/rel)
        hits={g.get("name") for g in v2.findall("worldbody/geom") if g.get("contype")!="0"}
        paths={a.get("name"):a.get("file") for a in v1.findall("asset/mesh")}
        stats=Counter()
        for g in v1.findall("worldbody/geom"):
            group=g.get("group")
            typ=g.get("type")
            key=f"group{group}_{typ}"
            stats[key+"_geoms"]+=1
            if typ!="mesh":
                continue
            name=g.get("name")
            obj=FULL/rel.parent/paths[g.get("mesh")]
            if group not in ("1","2"):
                with obj.open("rb") as f:
                    stats[key+"_faces"]+=sum(l.startswith(b"f ") for l in f)
                continue
            v,f=raw_obj(obj)
            stats[key+"_faces"]+=len(f)
            if "col_"+name not in hits:
                continue
            tr=v[f]
            n=np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0])
            l=np.linalg.norm(n,axis=1)
            ok=l>1e-12
            n=n[ok]/l[ok,None]
            a=n@v.T-(n*tr[ok,0]).sum(1)[:,None]
            deviation=np.maximum(0,np.minimum(a.max(1),-a.min(1))).max()
            if deviation>1.e-5:
                hull=v[ConvexHull(v).simplices]
                ab=directed_bound(tr,hull)
                ba=directed_bound(hull,tr)
                bounds.append(dict(mesh=name,visual_to_hull_bound_m=ab,hull_to_visual_bound_m=ba))
            else:
                max_flat=max(max_flat,deviation)
        scopes[str(i)]=stats
    write_json(V2/"work"/"inventory.json",scopes)
    write_json(V2/"work"/"hulls.json",dict(tolerance_m=1.e-5,max_convex_numerical_residual_m=max_flat,
               nonconvex_meshes=bounds,subdivision_spacing_m=.008))


def exposed():
    _,path=route()
    shapes=json.loads((FULL/"stair_geometry.json").read_text())
    solids=json.loads((V2/"work"/"solids.json").read_text())["stairs"]
    rows=[]
    for r in solids:
        j=int(r["name"].split("_")[1])
        edge=np.array(shapes[j]["edge"])
        side=edge[1]-edge[0]
        axis=np.array([side[1],-side[0]])/np.linalg.norm(side)
        s=r["station_m"]
        if axis@heading(path,s)<0: axis=-axis
        xy=path_at(path,s)
        xy=edge[0]+side*np.clip((xy-edge[0])@side/(side@side),.1,.9)
        root=local(V2/"scene_full.xml",xy,radius=.4)
        for g in list(root.findall("worldbody/geom")):
            if not ground_geom(g):
                root.find("worldbody").remove(g)
            else:
                g.set("rgba","1 1 1 1")
        m=model_of(root); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
        row=dict(name=r["name"],s=s,csv=r["csv_m"])
        for k,grp in (("visual",VIS),("collision",COL)):
            row[k]=ray(m,d,xy+.005*axis,grp)-ray(m,d,xy-.005*axis,grp)
        rows.append(row)
        del m,d
    write_json(V2/"work"/"exposed_risers.json",rows)


def decks():
    result=[]
    for patch in json.loads((FULL/"lower_layers.json").read_text())["patches"]:
        source=FULL/"segments"/f"seg_{patch['segment']:02d}"/"scene.xml"
        meshname=Path(patch["mesh"]).stem
        node=parsed(source).find(f"./worldbody/geom[@name='{meshname}']")
        v,f=raw_obj(source.parent/patch["mesh"])
        v+=np.fromstring(node.get("pos"),sep=" ")
        p=v[f].mean(1)
        near=np.argmin(np.linalg.norm(p[:,:2]-v[:,:2].mean(0),axis=1))
        xy=p[near,:2]
        root=local(V2/"scene_full.xml",xy,radius=1.)
        for n in root.findall("worldbody/geom"):
            if n.get("type")=="hfield" or n.get("name","").startswith("col_stair"):
                n.set("rgba","1 1 1 1")
        body=xml(root.find("worldbody"),"body",name="load")
        xml(body,"freejoint")
        xml(body,"geom",name="plate",type="box",size=[.12,.1,.05],mass=30,group=5,
            contype=2,conaffinity=1)
        m=model_of(root); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
        upper=ray(m,d,xy,COL)
        origin=np.r_[xy,p[near,2]+.18]
        k=np.empty(1,np.int32)
        hit=mujoco.mj_ray(m,d,origin,np.array([0.,0.,-1.]),COL,1,-1,k)
        lower_mesh=m.geom(k[0]).name if hit>=0 else None
        heights=[ray(m,d,xy+[x,y],COL) for x in (-.12,.12) for y in (-.1,.1)]
        d.qpos[:3]=[*xy,max(heights)+.06]
        loaded=0
        support_names=Counter()
        for i in range(1000):
            mujoco.mj_step(m,d)
            if i>=500:
                force=np.zeros(6)
                for n,c in enumerate(d.contact):
                    if m.geom("plate").id not in c.geom or c.efc_address<0: continue
                    mujoco.mj_contactForce(m,d,n,force)
                    if abs(c.frame[2])*force[0]>1.:
                        support_names[m.geom(c.geom[0] if c.geom[1]==m.geom("plate").id else c.geom[1]).name]+=1
                loaded+=any(abs(c.frame[2])>.5 and c.efc_address>=0 for c in d.contact)
        result.append(dict(station_m=patch["station_m"],xy=xy,lower_name=lower_mesh,
            lower_z=origin[2]-hit,upper_z=upper,layer_gap_m=upper-(origin[2]-hit),
            settled_center_above_lower_m=d.qpos[2]-(origin[2]-hit),last_second_contact_steps=loaded,
            support_names=dict(support_names),end_speed_mps=np.linalg.norm(d.qvel[:3]),final_xy=d.qpos[:2]))
        result[-1]["end_center_minus_local_top_m"] = d.qpos[2]-ray(m,d,d.qpos[:2],COL)
        del m,d
        # A vertical guide isolates bearing capacity from sliding off a steep
        # bank. It contributes no upward force to the proof mass.
        body.remove(body.find("freejoint"))
        body.set("pos"," ".join(map(str,[*xy,max(heights)+.06])))
        xml(body,"joint",type="slide",axis="0 0 1",limited="false")
        m=model_of(root); d=mujoco.MjData(m)
        forces=[]; names=set()
        for step in range(1000):
            mujoco.mj_step(m,d)
            if step<500:
                continue
            support=0.
            for j,c in enumerate(d.contact):
                if c.efc_address<0 or m.geom("plate").id not in c.geom:
                    continue
                f=np.zeros(6)
                mujoco.mj_contactForce(m,d,j,f)
                world_force=c.frame.reshape(3,3).T@f[:3]
                support+=world_force[2]*(1 if c.geom2==m.geom("plate").id else -1)
                if f[0]>1:
                    names.add(m.geom(c.geom1 if c.geom2==m.geom("plate").id else c.geom2).name)
            forces.append(support)
        result[-1]["vertical_proof"] = dict(mass_kg=30,weight_n=30*9.81,duration_s=2.,
             loaded_last_second_steps=int(np.count_nonzero(np.array(forces)>1)),
             mean_reaction_n=np.mean(forces),final_center_z_m=d.body("load").xpos[2],
             support_names=sorted(names),end_speed_mps=abs(d.qvel[0]))
        del m,d
    write_json(V2/"work"/"decks.json",result)


if __name__=="__main__":
    audit()
    decks()
    exposed()
