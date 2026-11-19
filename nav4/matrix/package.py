"""Stage the verified ZGWS + pilot MJCF in the MATRiX directory layout."""
import json
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from nav4.mapping.common import write_json
from nav4.matrix.compose import save_xml, vector
from nav4.matrix.stairs import heading, path_at
from nav4.matrix.survey import OUT
from nav4.matrix.verify import height

RELEASE = Path("../matrix_sim/extracted/matrix_v1.1.2/MATRiX_v1.1.2/UeSim")


def prefix_assets(root, prefix):
    for section in list(root.findall("asset"))+list(root.findall("worldbody")):
        for node in section.iter():
            if node.get("file"):
                file = Path(node.get("file"))
                if file.is_absolute() or ".." in file.parts:
                    raise ValueError(f"Unexpected path in packaged scene: {file}")
                node.set("file",(prefix/file).as_posix())


def main():
    relative = Path("Content/model")
    vendor = RELEASE / relative / "zgws"
    stage = OUT / "matrix" / "UeSim" / relative
    zgws = stage / "zgws"
    geometry = zgws / "atec_pilot_0_30"
    geometry.mkdir(parents=True,exist_ok=True)
    for name in ("terrain","stairs","hard"):
        shutil.copytree(OUT/name,geometry/name,dirs_exist_ok=True)
    shutil.copytree(vendor/"assets",zgws/"assets",dirs_exist_ok=True)
    original = ET.parse(OUT/"scene.xml").getroot()
    terrain = mujoco.MjModel.from_xml_path(str(OUT/"scene.xml"))
    terrain_data = mujoco.MjData(terrain)
    mujoco.mj_forward(terrain,terrain_data)
    path = np.load(OUT/"route_xy.npy")[:,1:]
    station = .40
    xy = path_at(path,station)
    z,name_id = height(terrain,terrain_data,xy)
    yaw = float(np.arctan2(*heading(path,station)[::-1]))
    if not np.isfinite(z):
        raise ValueError("Cannot initialize ZGWS without measured scene ground")
    robot = ET.parse(vendor/"zgws.xml").getroot()
    compiler = robot.find("compiler")
    meshdir = Path(compiler.attrib.pop("meshdir"))
    prefix_assets(robot,meshdir)
    base = robot.find("./worldbody/body[@name='base_link']")
    pos = np.r_[xy,z+.70]
    quat = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
    shift = np.r_[xy,z]
    for node in original.find("worldbody"):
        if node.get("pos"):
            node.set("pos",vector(np.fromstring(node.get("pos"),sep=" ")-shift))
        if node.get("fromto"):
            ends = np.fromstring(node.get("fromto"),sep=" ").reshape(2,3)
            node.set("fromto",vector((ends-shift).ravel()))
    base.set("pos",vector(pos-shift))
    base.set("quat",vector(quat))
    prefix_assets(original,Path("atec_pilot_0_30"))
    save_xml(original,geometry/"scene.xml")
    save_xml(robot,zgws/"zgws_atec_robot.xml")
    wrapper = ET.Element("mujoco",model="ZGWS A-trail pilot")
    ET.SubElement(wrapper,"include",file="zgws_atec_robot.xml")
    ET.SubElement(wrapper,"include",file="atec_pilot_0_30/scene.xml")
    entry = zgws/"zgws_atec_pilot.xml"
    save_xml(wrapper,entry)

    config = json.loads((RELEASE/relative/"config/config.json").read_text())
    config["robot"].update(mujoco_model=(relative/"zgws"/entry.name).as_posix(),
                           scene_model="", main_body="base_link", inside_mc=False)
    write_json(stage/"config/config_atec_pilot_candidate.json",config)
    begin = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(entry))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    actual = data.body("base_link").xpos
    np.testing.assert_allclose(actual,pos-shift,atol=1e-7)
    report = dict(mujoco_version=mujoco.__version__,seconds=time.monotonic()-begin,
                  init_station_m=station,world_z_ground_m=z,world_yaw_rad=yaw,
                  v8_origin_for_matrix_m=shift, init_v8_body_pos_m=pos,
                  initial_body_pos_m=actual,initial_body_quat=data.body("base_link").xquat,
                  scene_ground_geom=terrain.geom(name_id).name,terrain_triangles=model.nflexelem,
                  meshes=model.nmesh,robot_dof=model.nv,actuators=model.nu,
                  namespace="world_from_v8(xyz) = xyz - v8_origin_for_matrix_m, metres and z up",
                  runtime_ue_validation=False)
    write_json(OUT/"matrix_compile.json",report)
    print(report,flush=True)


if __name__ == "__main__":
    main()
