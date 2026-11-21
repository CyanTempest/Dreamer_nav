"""Import the supplied M1 URDF with MuJoCo; add a torque-limited rolling baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import trimesh

from _bootstrap import ROOT


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    mesh_dir = output / "meshes"
    mesh_dir.mkdir()
    source = ROOT / "assets/robots/m1/urdf/ZG_M1_A0_V1_0/urdf/ZG_M1_A0_V1_0.urdf"
    urdf = ET.parse(source).getroot()
    converted = set()
    for mesh in urdf.findall(".//mesh"):
        path = (source.parent / mesh.get("filename")).resolve()
        dest = mesh_dir / (path.stem + ".obj")
        if path not in converted:
            trimesh.load(path, process=False).export(dest)
            converted.add(path)
        mesh.set("filename", str(dest))
    settings = ET.SubElement(urdf, "mujoco")
    ET.SubElement(settings, "compiler", discardvisual="false", fusestatic="false", balanceinertia="false")
    urdf_path = output / "import.urdf"
    ET.ElementTree(urdf).write(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    path = output / "robot.xml"
    mujoco.mj_saveLastXML(str(path), model)
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", "meshes")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", iterations="30")
    base = root.find("worldbody/body")
    base.set("pos", "0 0 0.56")
    ET.SubElement(base, "freejoint", name="root")
    actuator = ET.SubElement(root, "actuator")
    targets = {}
    for joint in base.findall(".//joint"):
        name = joint.get("name")
        wheel = "FOOT" in name
        joint.set("damping", "0.15" if wheel else "0.5")
        joint.set("armature", "0.005" if wheel else "0.03")
        if wheel:
            joint.attrib.pop("range", None)
            joint.set("limited", "false")
        ET.SubElement(actuator, "motor", name=name, joint=name,
                      ctrllimited="true", ctrlrange="-28.68 28.68" if wheel else "-180 180")
        if not wheel:
            front = name.startswith(("FAR", "FBL"))
            targets[name] = (0.0 if "ABAD" in name else
                             (-0.6 if front else 0.6) if "HIP" in name else
                             (1.2 if front else -1.2))
    for geom in base.findall(".//geom"):
        if geom.get("contype") == "0":
            continue
        geom.set("contype", "1")
        geom.set("conaffinity", "2")
        geom.set("group", "3")
        geom.set("friction", "0.9 0.005 0.0001")
        geom.set("condim", "3")
    ET.indent(root)
    ET.ElementTree(root).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq == 23 and model.nv == 22 and model.nu == 16
    manifest = {"source_urdf": str(source.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "robot_name": "ZG_M1_W", "backend": "mujoco", "version": mujoco.__version__,
                "mass_kg": float(model.body_mass.sum()), "free_base": True,
                "actuated_joints": 16, "wheel_radius_m": 0.0975,
                "stance_targets_rad": targets, "controller": "engineering_PD_stance_and_wheel_velocity",
                "gains": {"leg_kp": 180.0, "leg_kd": 10.0, "wheel_kv": 3.0},
                "assumptions": ["simulation_only_gains_and_stance_not_hardware_calibration",
                                "robot_self_collision_disabled_environment_contacts_enabled",
                                "joint_armature_and_damping_are_engineering_parameters",
                                "friction_0.9_is_unmeasured_simulation_assumption"],
                "validated_motion": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args().output.resolve())
