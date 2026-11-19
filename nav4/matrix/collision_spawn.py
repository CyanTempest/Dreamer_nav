"""Fit a load-bearing ZGWS home pose, then test free-base PD standing."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from nav4.mapping.common import write_json
from nav4.matrix.collision import FULL, V2, MODEL
from nav4.matrix.collision_bench import hold
from nav4.matrix.collision_checks import COL, local, model_of, ray
from nav4.matrix.compose import save_xml, vector
from nav4.matrix.full_scan import route
from nav4.matrix.stairs import heading, path_at


def rot(q):
    return Rotation.from_quat(np.roll(q,-1))


def quat(r):
    return np.roll(r.as_quat(),1)


def local_model(label, xy, robot_file):
    scene = local(V2/MODEL/f"atec_scene_{label}.xml",xy,radius=3.)
    robot = ET.parse(robot_file).getroot()
    for node in robot.findall("asset/mesh"):
        node.set("file",str((robot_file.parent/node.get("file")).resolve()))
    for tag in ("asset","worldbody"):
        scene.find(tag).extend(robot.find(tag))
    for tag in ("default","actuator","sensor","keyframe","contact"):
        if robot.find(tag) is not None:
            scene.append(robot.find(tag))
    m = model_of(scene)
    m.geom_rgba[m.geom_group==3,3] = 1
    return m


def wheels(m):
    return [i for i in range(m.ngeom) if m.geom_contype[i] and
            m.body(m.geom_bodyid[i]).name.endswith("FOOT_LINK")]


def initialize(m, d, xy, yaw):
    """Fit all four support patches. Preserve vendor joint coordinates/axes."""
    mujoco.mj_resetDataKeyframe(m,d,m.key("home").id)
    joints = d.qpos[7:].copy()
    feet = wheels(m)
    assert len(feet)==4
    vert = []
    for g in feet:
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices = m.mesh_vert[start:start+count].copy()
        # Only the hull vertices can minimize a projected height.
        from scipy.spatial import ConvexHull
        vert.append(vertices[ConvexHull(vertices).vertices])
    d.qpos[:2]=xy
    mujoco.mj_forward(m,d)
    floor = ray(m,d,xy,COL)
    def residual(v):
        d.qpos[2] = floor+v[0]
        d.qpos[3:7] = quat(Rotation.from_euler("ZYX",[yaw,v[1],v[2]]))
        d.qpos[7:]=joints
        d.qpos[9::4] += v[3:]
        mujoco.mj_kinematics(m,d)
        gaps=[]
        for g,vv in zip(feet,vert):
            pts = vv@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            p = pts[np.argmin(pts[:,2])]
            gaps.append(p[2]-ray(m,d,p[:2],COL))
        return np.r_[gaps,.012*v[3:],.001*v[1:3]]
    solution = least_squares(residual,[.56,0,0,0,0,0,0],
        bounds=([.36,-.65,-.4,*[-.65]*4],[.9,.65,.4,*[.65]*4]), diff_step=1e-4)
    residual(solution.x)
    target = d.qpos[7:].copy()
    # Find first physical contact at fixed attitude (wheel widths and slopes
    # matter). A submillimetre preload makes initial contact observable.
    top = d.qpos[2]+.14
    low = d.qpos[2]-.14
    for _ in range(18):
        mid = (top+low)/2
        d.qpos[2]=mid
        mujoco.mj_forward(m,d)
        if any(c.geom1 in feet or c.geom2 in feet for c in d.contact):
            low=mid
        else:
            top=mid
    d.qpos[2]=low-.0006
    mujoco.mj_forward(m,d)
    return target


def tread_pose(m,d,xy,yaw):
    """Place each wheel on a local support patch clear of the adjacent riser."""
    mujoco.mj_resetDataKeyframe(m,d,m.key("home").id)
    q0=d.qpos[7:].copy()
    forward=np.array([np.cos(yaw),np.sin(yaw)])
    side=np.array([-forward[1],forward[0]])
    d.qpos[:2]=xy
    d.qpos[3:7]=quat(Rotation.from_euler("z",yaw))
    mujoco.mj_forward(m,d)
    feet=wheels(m)
    wheel_offsets=[]
    radii=[]
    for g in feet:
        start=m.mesh_vertadr[m.geom_dataid[g]]
        nv=m.mesh_vertnum[m.geom_dataid[g]]
        v=m.mesh_vert[start:start+nv]
        wheel_offsets.append((v.min(0)+v.max(0))/2)
        extent=np.ptp(v,axis=0)
        radii.append(max(extent)/2)
    def centers():
        return np.array([d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@offset
                         for g,offset in zip(feet,wheel_offsets)])
    goals=[]
    for p,r in zip(centers(),radii):
        candidates=[]
        for dx in np.linspace(-.19,.19,21):
            pt=p[:2]+dx*forward
            samples=np.array([[ray(m,d,pt+t*forward+w*side,COL)
                      for w in (-.028,0.,.028)] for t in np.linspace(-r*.95,r*.95,9)])
            if not np.all(np.isfinite(samples)):
                continue
            arc=np.sqrt(r*r-np.linspace(-r*.95,r*.95,9)**2)
            height=np.max(samples+arc[:,None])
            score=np.ptp(samples[1:-1])+.075*abs(dx)+.3*(height-r-np.median(samples))
            candidates.append((score,np.r_[pt,height]))
        goals.append(min(candidates,key=lambda item:item[0])[1])
    goals=np.array(goals)
    floor=ray(m,d,xy,COL)
    hip=np.array([1,2,5,6,9,10,13,14])
    def residual(v):
        d.qpos[2]=floor+v[0]
        d.qpos[3:7]=quat(Rotation.from_euler("ZYX",[yaw,v[1],v[2]]))
        d.qpos[7:]=q0
        d.qpos[7+hip]+=v[3:]
        mujoco.mj_kinematics(m,d)
        return np.r_[(centers()-goals).ravel(), .007*v[3:],.008*v[1:3]]
    v=least_squares(residual,[.55,0,0,*np.zeros(8)],
        bounds=([.32,-.75,-.35,*[-1.1]*8],[.84,.75,.35,*[1.1]*8]),diff_step=1.e-4)
    residual(v.x)
    target=d.qpos[7:].copy()
    mujoco.mj_forward(m,d)
    return target


def contacts(m,d):
    foot=set()
    gap=0.
    force=0.
    wrench=np.zeros(6)
    for i,c in enumerate(d.contact):
        a,b = c.geom
        if a<0 or b<0 or m.geom_bodyid[a]*m.geom_bodyid[b]!=0 or c.efc_address<0:
            continue
        body = m.body(max(m.geom_bodyid[a],m.geom_bodyid[b])).name
        gap=min(gap,c.dist)
        if body.endswith("FOOT_LINK") and abs(c.frame[2])>.45:
            mujoco.mj_contactForce(m,d,i,wrench)
            force+=abs(c.frame[2])*max(0,wrench[0])
            if wrench[0]>1:
                foot.add(body)
    return len(foot),gap,force


def stand(m,d,target,steps=1500):
    mujoco.mj_forward(m,d)
    q0=d.qpos.copy()
    start=contacts(m,d)
    up0=rot(q0[3:7]).apply([0,0,1])
    nfeet=[]; gaps=[]; forces=[]; travel=[]; rise=[]; tilt=[]
    for _ in range(steps):
        hold(m,d,target)
        mujoco.mj_step(m,d)
        n,g,f=contacts(m,d)
        nfeet.append(n); gaps.append(g); forces.append(f)
        travel.append(np.linalg.norm(d.qpos[:2]-q0[:2]))
        rise.append(d.qpos[2]-q0[2])
        z=rot(d.qpos[3:7]).apply([0,0,1])
        tilt.append(np.degrees(np.arccos(np.clip(z@up0,-1,1))))
    return dict(duration_s=steps*m.opt.timestep, initial_loaded_wheels=start[0],
        initial_contact_gap_m=start[1],initial_vertical_contact_n=start[2],
        terminal_loaded_wheels=nfeet[-1], terminal_contact_n=forces[-1],
        supported_fraction=np.mean(np.array(nfeet)>0),min_loaded_wheels=min(nfeet),
        max_overlap_m=-min(gaps),max_xy_drift_m=max(travel),z_range_m=[min(rise),max(rise)],
        max_attitude_change_deg=max(tilt),end_speed_mps=np.linalg.norm(d.qvel[:3]),
        warnings=d.warning.number.tolist())


def set_reference(robot,q,target=None):
    """Preserve forward kinematics for every absolute joint position while
    making mj_resetData and reset-to-home start from the same bent-leg pose."""
    base=robot.find("./worldbody/body[@name='base_link']")
    base.set("pos",vector(q[:3])); base.set("quat",vector(q[3:7]))
    items=[(b,b.find("joint")) for b in base.iter("body") if b.find("joint") is not None]
    assert len(items)==len(q)-7
    for angle,(body,joint) in zip(q[7:],items):
        assert joint.get("ref","0")=="0" and not np.any(np.fromstring(joint.get("pos","0 0 0"),sep=" "))
        axis=np.fromstring(joint.get("axis"),sep=" ")
        prior=rot(np.fromstring(body.get("quat","1 0 0 0"),sep=" "))
        body.set("quat",vector(quat(prior*Rotation.from_rotvec(axis*angle))))
        joint.set("ref",f"{angle:.10g}")
    key=robot.find("./keyframe/key[@name='home']")
    key.set("qpos",vector(q))
    if target is not None:
        torque=np.clip(190*(target-q[7:]),-140,140)
        torque[3::4]=np.clip(torque[3::4],-38,38)
        key.set("ctrl",vector(torque))


def prepare(label,station,path,shift,export,fit_treads=False):
    xy=path_at(path,station)-shift[:2]
    yaw=float(np.arctan2(*heading(path,station)[::-1]))
    file=f"zgws_atec_robot_{label}.xml"
    src=FULL/MODEL/file
    m=local_model(label,xy,src)
    d=mujoco.MjData(m)
    target=(tread_pose if fit_treads else initialize)(m,d,xy,yaw)
    # Settle under the same actuator PD used during verification. Remove any
    # small accumulated yaw drift by rotating the base about the vertical.
    for t in range(500):
        if fit_treads:
            # A positioning fixture is removed before zero-velocity release.
            # Limit planar drift during initialization on short downhill treads.
            d.xfrc_applied[m.body("base_link").id,:2] = 3500*(xy-d.qpos[:2])-130*d.qvel[:2]
        hold(m,d,target)
        mujoco.mj_step(m,d)
    d.xfrc_applied[:]=0
    q=d.qpos.copy()
    euler=rot(q[3:7]).as_euler("ZYX")
    q[3:7]=quat(Rotation.from_euler("ZYX",[yaw,*euler[1:]]))
    # Re-centre exactly on the requested station, then solve vertical preload
    # by matching the total normal reaction to the robot weight at zero speed.
    q[:2]=xy
    mujoco.mj_resetData(m,d)
    d.qpos[:]=q
    weight=m.body_mass.sum()*9.81
    mujoco.mj_forward(m,d)
    floor=ray(m,d,xy,COL)
    assert np.isfinite(floor)
    hi=floor+.95
    # Bracket the first load-bearing contact from above. Deeply interpenetrating
    # wheel meshes can yield a second branch of vertical reaction.
    for z in np.arange(hi, floor+.20, -.002):
        d.qpos[2]=z
        hold(m,d,target)
        mujoco.mj_forward(m,d)
        _, gap, force=contacts(m,d)
        if force>=weight or gap<-.012:
            lo,hi=z,z+.002
            break
    else:
        raise ValueError("No wheel support above the local floor")
    for _ in range(22):
        d.qpos[2]=(lo+hi)/2
        hold(m,d,target)
        mujoco.mj_forward(m,d)
        if contacts(m,d)[2]>weight:
            lo=d.qpos[2]
        else:
            hi=d.qpos[2]
    q=d.qpos.copy()
    trial=stand(m,d,target)
    result=dict(station_m=station,base_pos_m=q[:3],quat_wxyz=q[3:7],yaw_rad=yaw,
                qpos=q,pd_joint_target=target,stand=trial)
    if export and stable(trial):
        robot=ET.parse(src).getroot()
        set_reference(robot,q,target)
        save_xml(robot,V2/MODEL/file)
        # Independent FK equivalence at both the shipped pose and a second pose.
        vm=local_model(label,xy,V2/MODEL/file)
        vd=mujoco.MjData(vm)
        assert np.allclose(vm.qpos0,q,atol=1e-7)
        assert np.allclose(vm.key("home").qpos,q,atol=1e-7)
        delta=0.
        for joints in (q[7:],target+.07):
            d.qpos[:]=q
            d.qpos[7:]=joints
            vd.qpos[:]=d.qpos
            mujoco.mj_kinematics(m,d); mujoco.mj_kinematics(vm,vd)
            delta=max(delta,np.max(abs(d.xpos-vd.xpos)),np.max(abs(d.xmat-vd.xmat)))
        result["max_fk_difference"]=delta
        assert delta<1.e-7
    return result


def stable(row):
    return (row["initial_loaded_wheels"]>=2 and row["supported_fraction"]==1 and
            row["max_overlap_m"]<.015 and row["max_xy_drift_m"]<.05 and
            row["max_attitude_change_deg"]<5 and row["end_speed_mps"]<.02)


def choose(label,station,path,shift,export):
    attempts=[]
    for offset,fit in [(0.,False),(0.,True)]+[(v,True) for v in (.15,.30,.50,.80,1.2,2.)]:
        result=prepare(label,station+offset,path,shift,export,fit)
        row=result["stand"]
        attempts.append(dict(station_m=station+offset,fit_treads=fit,stand=row))
        if stable(row):
            break
    result["placement_trials"]=attempts
    if export and not stable(row):
        raise ValueError(f"Could not establish stable support for {label}: {attempts}")
    return result


def main():
    _,path=route()
    manifest=json.loads((FULL/"assembly.json").read_text())
    shift=np.array(manifest["shared_shift_m"])
    original=json.loads((FULL/MODEL.parent/"config/config_atec_window_00_candidate.json").read_text())
    spawns={}
    for row in manifest["statics"]:
        label=Path(row["window"]).stem
        station=row["station_m"][0]+.4
        result=choose(label,station,path,shift,True)
        spawns[label]=result
        row.update(initial_base_pos_m=result["base_pos_m"],initial_quat_wxyz=result["quat_wxyz"],
                   initial_yaw_rad=result["yaw_rad"],spawn_station_m=result["station_m"])
        config=copy.deepcopy(original)
        config["robot"].update(position=dict(zip("xyz",result["base_pos_m"])),mujoco_model=row["matrix_entry"])
        write_json(V2/MODEL.parent/"config"/f"config_atec_{label}_candidate.json",config)
        print(label,result["stand"],flush=True)
        write_json(V2/"work"/"spawns.json",spawns)
    first=spawns["window_00"]
    # Full and window 00 share the release frame and identical local terrain.
    robot=ET.parse(FULL/MODEL/"zgws_atec_robot_full.xml").getroot()
    set_reference(robot,first["qpos"],first["pd_joint_target"])
    save_xml(robot,V2/MODEL/"zgws_atec_robot_full.xml")
    config=copy.deepcopy(original)
    config["robot"].update(position=dict(zip("xyz",first["base_pos_m"])),mujoco_model=manifest["matrix_full"])
    write_json(V2/MODEL.parent/"config/config_atec_full_candidate.json",config)
    manifest.update(initial_matrix_base_m=first["base_pos_m"],initial_quat_wxyz=first["quat_wxyz"])
    manifest.pop("seconds",None)
    write_json(V2/"assembly.json",manifest)
    checks={}
    for station,label in [(1.75,"window_00"),(10.4,"window_00"),(6.2,"window_00"),
                          (618.,"window_12"),(704.,"window_13")]:
        checks[str(station)]=choose(label,station,path,shift,False)
    write_json(V2/"work"/"stairs_standing.json",checks)


if __name__=="__main__":
    main()
