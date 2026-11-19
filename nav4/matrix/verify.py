"""Exercise scene loading, rendering and rigid contact using CPU MuJoCo."""
import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from nav4.mapping.common import bilinear, cell_index, write_json
from nav4.matrix.compose import xml, scene_subset
from nav4.matrix.detail import local_v8
from nav4.matrix.register import summary
from nav4.matrix.stairs import heading, path_at
from nav4.matrix.survey import OUT


def height(model, data, xy, group=(1,0,1,0,0,0)):
    origin = np.r_[xy, 25.]
    id_ = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model,data,origin,np.array([0.,0.,-1.]),
                             np.array(group,dtype=np.uint8),1,-1,id_)
    return (origin[2]-distance,int(id_[0])) if distance>=0 else (np.nan,-1)


def place_camera(world, model, data, path, station, settings, idx):
    xy = path_at(path,station)
    course = heading(path,station+.15)
    eye_xy = xy-settings["camera_back_m"]*course
    eye_z = height(model,data,xy)[0]+settings["camera_height_m"]
    aim_xy = path_at(path,station+2.1)
    aim_z = height(model,data,aim_xy)[0]+.30
    forward = np.r_[aim_xy-eye_xy, aim_z-eye_z]
    forward /= np.linalg.norm(forward)
    across = np.cross(forward,[0.,0.,1.])
    across /= np.linalg.norm(across)
    up = np.cross(across,forward)
    xml(world, "camera", name=f"ref_{idx:02d}", pos=np.r_[eye_xy,eye_z], xyaxes=np.r_[across,up],
        fovy=settings["vertical_fov_deg"])
    return dict(eye=np.r_[eye_xy,eye_z], target=np.r_[aim_xy,aim_z], station_m=station)


def render(out, model, data, settings, path):
    # Recompile the same scene with fixed camera poses. Source frames are
    # selected by shared landmarks, since the video's station origin differs.
    root = scene_subset(out/"scene.xml",lambda n:True)
    info = [place_camera(root.find("worldbody"),model,data,path,s,settings,k)
            for k,(_,s) in enumerate(settings["views"])]
    view_model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    view_data = mujoco.MjData(view_model)
    mujoco.mj_forward(view_model,view_data)
    flags = mujoco.MjvOption()
    flags.geomgroup[3] = 0
    flags.flexgroup[3] = 0
    width,height_px = settings["width"],settings["height"]
    outdir = OUT/"figures" if out==OUT else out/"figures"
    outdir.mkdir(exist_ok=True)
    with mujoco.Renderer(view_model,height=height_px,width=width) as renderer:
        for k,(frame,s) in enumerate(settings["views"]):
            renderer.update_scene(view_data,camera=f"ref_{k:02d}",scene_option=flags)
            rendered = Image.fromarray(renderer.render())
            rendered.save(outdir/f"mujoco_{k+1:02d}.png")
            renderer.enable_depth_rendering()
            renderer.update_scene(view_data,camera=f"ref_{k:02d}",scene_option=flags)
            depth = renderer.render()
            np.save(outdir/f"depth_{k+1:02d}_m.npy",depth)
            renderer.disable_depth_rendering()
            ref = Image.open(OUT/"figures"/"reference"/f"ref_{frame:02d}.jpg").convert("RGB")
            sheet = Image.new("RGB",(2*width,height_px+28),(23,26,28))
            ref.thumbnail((width,height_px))
            sheet.paste(ref,((width-ref.width)//2,28+(height_px-ref.height)//2))
            sheet.paste(rendered,(width,28))
            draw=ImageDraw.Draw(sheet)
            draw.text((12,7),f"Reference video | +{frame:02d} s",fill="white")
            draw.text((width+12,7),f"MuJoCo / CPU | v8 station {s:.2f} m",fill="white")
            sheet.save(outdir/f"compare_{k+1:02d}.jpg",quality=92)
    write_json(out/"camera_poses.json",info)


def physics(out, model, data, path, settings):
    half = np.array(settings["box_half_size_m"])
    results = {}
    stair_shapes = json.loads((out/"stair_geometry.json").read_text())
    for name,station,aim,start_offset,lateral in (
        ("entry_riser",.46,1.10,0.,False),
        ("bend_riser",4.38,5.10,0.,False),
        ("turn_side",7.90,8.16,0.,True),
        ("flight2_riser",8.64,9.36,0.,False),
    ):
        xy = path_at(path,station)
        if name == "entry_riser":
            ends = np.array(stair_shapes[0]["edge"])
            span = ends[1]-ends[0]
            tangent = np.array([span[1],-span[0]])
            tangent /= np.linalg.norm(tangent)
            xy = ends.mean(0) - (half[0]+.045)*tangent
        else:
            tangent = path_at(path,aim)-xy
            tangent /= np.linalg.norm(tangent)

        def nearby(node):
            if node.tag == "flexcomp":
                return np.linalg.norm(np.fromstring(node.get("pos"),sep=" ")[:2]-xy) < 3.9
            if node.tag != "geom" or int(node.get("contype",0)) == 0:
                return False
            if node.get("fromto"):
                p = np.fromstring(node.get("fromto"),sep=" ").reshape(2,3).mean(0)
            else:
                p = np.fromstring(node.get("pos"),sep=" ")
            return np.linalg.norm(p[:2]-xy) < 2.35

        root = scene_subset(out/"scene.xml",nearby)
        world = root.find("worldbody")
        box = xml(world,"body",name="contact_proxy")
        xml(box,"freejoint",name="proxy_joint")
        xml(box,"geom",type="box",name="proxy_box",size=half,mass=settings["mass_kg"],
            friction=[.12,.001,.0001],priority=2,rgba=[.1,.4,.6,1],solref=[.007,1.],
            contype=2,conaffinity=1)
        m = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        d = mujoco.MjData(m)
        body = m.body("contact_proxy").id
        geom = m.geom("proxy_box").id
        jnt = m.joint("proxy_joint").id
        ad = m.jnt_qposadr[jnt]
        pos = d.qpos[ad:ad+3]
        quat = d.qpos[ad+3:ad+7]
        side = np.array([-tangent[1],tangent[0]])
        xy += start_offset*side
        yaw = np.arctan2(tangent[1],tangent[0])
        orientation=np.r_[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
        test_xy = np.array([xy + sx*tangent + sy*side
                            for sx in (-half[0],0,half[0]) for sy in (-half[1],0,half[1])])
        heights=np.array([height(model,data,p)[0] for p in test_xy])
        z = float(np.nanmax(heights))
        pos[:] = np.r_[xy,z+half[2]+.015]
        quat[:] = orientation
        initial = pos.copy()
        records=[]
        planar = -side if lateral else tangent
        for duration,push in ((settings["settle_s"],False),(settings["push_s"],True)):
            stats=dict(lowest_contact_m=0.,max_forward_contact_force_n=0.,
                       min_support_contact_z=1e9,max_support_contact_z=-1e9,
                       support_steps=0,blocking_steps=0,blocking_geoms={},
                       max_proxy_height_m=float(pos[2]))
            start=pos.copy()
            for _ in range(round(duration/m.opt.timestep)):
                if push:
                    speed=d.qvel[m.jnt_dofadr[jnt]:m.jnt_dofadr[jnt]+2]@planar
                    force = np.clip((.095-speed)*1400,0,settings["push_n"])
                    d.xfrc_applied[body,:2] = force*planar
                mujoco.mj_step(m,d)
                support, block = False, False
                for i in range(d.ncon):
                    c=d.contact[i]
                    if geom not in (c.geom1,c.geom2) or c.efc_address < 0:
                        continue
                    stats["lowest_contact_m"] = min(stats["lowest_contact_m"],float(c.dist))
                    local=np.zeros(6)
                    mujoco.mj_contactForce(m,d,i,local)
                    normal=c.frame[:3]
                    support |= abs(normal[2])>.72 and local[0]>.2
                    if abs(normal[2])>.72 and local[0]>.2:
                        stats["min_support_contact_z"] = min(stats["min_support_contact_z"],float(c.pos[2]))
                        stats["max_support_contact_z"] = max(stats["max_support_contact_z"],float(c.pos[2]))
                    projected=float(abs(normal[:2]@planar)*local[0])
                    stats["max_forward_contact_force_n"] = max(stats["max_forward_contact_force_n"],projected)
                    if abs(normal[2])<.40 and projected>3:
                        block=True
                        neighbour=c.geom2 if c.geom1==geom else c.geom1
                        label=m.geom(neighbour).name if neighbour>=0 else "triangle_flex"
                        hits=stats["blocking_geoms"]
                        hits[label]=hits.get(label,0)+1
                stats["support_steps"] += support
                stats["blocking_steps"] += block
                stats["max_proxy_height_m"] = max(stats["max_proxy_height_m"],float(pos[2]))
            for key in ("min_support_contact_z","max_support_contact_z"):
                if abs(stats[key])>=1e9:
                    stats[key]=None
            records.append(dict(**stats,delta_m=pos-start,forward_m=float((pos-start)[:2]@planar),
                                final_pose=np.r_[pos,quat]))
        results[name]=dict(station_m=station,initial_center=initial,settle=records[0],push=records[1],
                           local_collision_triangles=m.nflexelem,local_geoms=m.ngeom)
        print(name, "physics done",flush=True)
    write_json(out/"physics.json",dict(proxy_length_width_m=2*half[:2],mass_kg=float(m.body_mass[body]),
                                      cases=results,settings=settings))
    return results


def compare_v8(out, model, data, path, origin, layers, known, res):
    # 5 cm longitudinal sampling plus cross-track comparisons in the height band of v8.
    s=np.arange(len(path))*.05
    expected=bilinear(layers[...,0],origin,path,res=res)
    measured=np.array([height(model,data,p)[0] for p in path])
    records=[]
    for i,st in enumerate(s):
        piece = dict(station_m=st, v8_z_m=expected[i], scene_ray_z_m=measured[i],
                     scene_minus_v8_m=measured[i]-expected[i])
        records.append(piece)
    entries={}
    ii,jj=np.nonzero((layers[...,6]==1) & known)
    v8_hard=np.c_[ii,jj]*res+origin
    tree=cKDTree(v8_hard)
    all_clearance=tree.query(path)[0]
    delta = measured-expected
    for start,stop in ((0,5),(5,7.5),(7.5,9.25),(9.25,15),(15,20),(20,30.01)):
        mask = (s>=start)&(s<stop)&np.isfinite(delta)
        if mask.any():
            ix=np.flatnonzero(mask)[np.argmax(abs(delta[mask]))]
            entries[f"{start:g}-{stop:g} m"]=dict(**summary(delta[mask]),worst_station_m=s[ix],
                                                 worst_signed_m=delta[ix])
    hard_names=[]
    hard_pos=[]
    for i in range(model.ngeom):
        name=model.geom(i).name
        if name.startswith(("solid_","wall_","trunk_","root_")) or "_post_" in name:
            hard_pos.append(data.geom_xpos[i,:2])
            hard_names.append(name)
    distances = tree.query(hard_pos)[0]
    ordered=np.argsort(distances)[-12:][::-1]
    far = [dict(geom=hard_names[i],center_m=hard_pos[i],nearest_v8_hard_cell_center_m=distances[i])
           for i in ordered]
    with (out/"route_vs_v8.csv").open("w") as f:
        f.write("station_m,v8_z_m,scene_z_m,scene_minus_v8_m,v8_hard_distance_m\n")
        for r,v8d in zip(records,all_clearance):
            f.write(",".join(f"{v:.5f}" for v in [*r.values(),v8d])+"\n")
    result = dict(route_ground=entries,missing_rays=int((~np.isfinite(delta)).sum()),
                  hard_geom_center_to_v8_hard_cell=summary(distances),deviating_hard_shapes=far)
    write_json(out/"v8_difference.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--load-only",action="store_true")
    args=parser.parse_args()
    out=OUT/"preview" if args.preview else OUT
    settings=json.loads(Path(__file__).with_name("settings.json").read_text())
    start=time.monotonic()
    model=mujoco.MjModel.from_xml_path(str(out/"scene.xml"))
    load=time.monotonic()-start
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    paths = [out/"scene.xml"]
    for name in ("stairs","terrain","hard"):
        paths.extend((out/name).glob("*.obj"))
    stats=dict(version=mujoco.__version__,load_seconds=load,files=len(paths),
               mesh_faces=model.nmeshface,flex_faces=model.nflexelem,parts=model.ngeom,
               bytes=sum(p.stat().st_size for p in paths))
    write_json(out/f"load_{mujoco.__version__}.json",stats)
    print(stats,flush=True)
    if args.load_only:
        return
    path,origin,layers,known,_,res,_,_=local_v8()
    render(out,model,data,settings["render"],path)
    print("rendered comparison views",flush=True)
    compare_v8(out,model,data,path,origin,layers,known,res)
    cases=physics(out,model,data,path,settings["physics"])
    print({key: (c["push"]["forward_m"],c["push"]["blocking_steps"],c["push"]["lowest_contact_m"])
           for key,c in cases.items()},flush=True)


if __name__=="__main__":
    main()
