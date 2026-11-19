"""Compile the actual MATRiX entries and verify both reset paths and standing."""
import argparse
import json
import resource
import time

import mujoco
import numpy as np
from nav4.mapping.common import write_json
from nav4.matrix.collision import MODEL,V2
from nav4.matrix.collision_bench import hold
from nav4.matrix.collision_spawn import contacts,rot,stable,stand


def release(label):
    file=V2/MODEL/f"zgws_atec_{label}.xml"
    start=time.perf_counter()
    m=mujoco.MjModel.from_xml_path(str(file))
    load_s=time.perf_counter()-start
    d=mujoco.MjData(m)
    assert m.nflex==0 and m.opt.timestep==.002 and not m.vis.global_.bvactive
    spawn=json.loads((V2/"work"/"spawns.json").read_text())["window_00" if label=="full" else label]
    q=m.key("home").qpos
    reset_error=np.max(abs(q-m.qpos0))
    assert reset_error<1e-8
    config=json.loads((V2/MODEL.parent/"config"/f"config_atec_{label}_candidate.json").read_text())["robot"]
    assert np.allclose([config["position"][s] for s in "xyz"],q[:3],atol=1e-7)
    assert config["mujoco_model"].endswith(file.name)
    output=dict(load_s=load_s,reset_error=reset_error,ngeom=m.ngeom,nhfield=m.nhfield,nflex=m.nflex,
                ngeom_collision=int(np.count_nonzero(m.geom_contype|m.geom_conaffinity)),
                nbvh=m.nbvh,home_yaw_error_deg=np.degrees(rot(q[3:7]).as_euler("ZYX")[0]-spawn["yaw_rad"]))
    mujoco.mj_resetDataKeyframe(m,d,m.key("home").id)
    mujoco.mj_forward(m,d)
    output["home_before_pd"]=contacts(m,d)
    support_names=set()
    for j,c in enumerate(d.contact):
        if c.efc_address < 0:
            continue
        force=np.zeros(6)
        mujoco.mj_contactForce(m,d,j,force)
        if force[0]>1 and abs(c.frame[2])>.45:
            support_names.update(m.geom(g).name for g in c.geom if g>=0 and m.geom_bodyid[g]==0)
    output["home_support_geoms"]=sorted(support_names)
    scene=m.geom_group==3
    alphas=m.geom_rgba[scene,3].copy()
    m.geom_rgba[scene,3]=1
    origin=np.r_[q[:2],q[2]+2]
    key=np.empty(1,np.int32)
    depth=mujoco.mj_ray(m,d,origin,np.array([0.,0.,-1.]),np.array([0,0,0,1,0,0],np.uint8),1,-1,key)
    output["support_z_under_base_m"]=float(origin[2]-depth) if depth>=0 else None
    m.geom_rgba[scene,3]=alphas
    output["home_stand"]=stand(m,d,np.asarray(spawn["pd_joint_target"]))
    output["pass"]=stable(output["home_stand"])
    if label in ("window_00","full"):
        references=json.loads((V2/"work"/"stairs_standing.json").read_text())
        rows=[]
        poses=[spawn,references["10.4"],references["6.2"]]
        if label=="full":
            poses.extend([references["618.0"],references["704.0"]])
        for pose in poses:
            mujoco.mj_resetData(m,d)
            d.qpos[:]=pose["qpos"]
            target=np.asarray(pose["pd_joint_target"])
            mujoco.mj_forward(m,d)
            for _ in range(100):
                hold(m,d,target)
                mujoco.mj_step(m,d)
            initial=d.qpos.copy()
            spent=0
            counts=[]
            for _ in range(2000):
                hold(m,d,target)
                tick=time.perf_counter_ns()
                mujoco.mj_step(m,d)
                spent+=time.perf_counter_ns()-tick
                counts.append(d.ncon)
            ms=spent/2e9
            rows.append(dict(station_m=pose["station_m"],steps=len(counts),ms=ms,ncon=np.mean(counts),
                             realtime=2/ms,contact_range=[min(counts),max(counts)],
                             xy_drift_m=np.linalg.norm(d.qpos[:2]-initial[:2]),warnings=d.warning.number))
        output["standing_benchmarks"]=rows[:3]
        if label=="full":
            output["descent_benchmarks"]=rows[3:]
    output["peak_rss_kib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(V2/"work"/f"release_{label}.json",output)
    print(label,output["pass"],output["home_stand"]["max_xy_drift_m"],output.get("standing_benchmarks",[]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("label")
    release(parser.parse_args().label)
