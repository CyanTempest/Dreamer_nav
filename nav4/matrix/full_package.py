"""Create static and ZGWS assemblies using the same nonduplicated segment assets."""
import copy
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np

from nav4.mapping.common import write_json
from nav4.matrix.compose import save_xml, vector
from nav4.matrix.full_ground import FULL, grid_sample
from nav4.matrix.full_scan import route
from nav4.matrix.full_scene import CONFIG, LIMITS, scene_dir
from nav4.matrix.package import RELEASE, prefix_assets
from nav4.matrix.stairs import heading, path_at


def shift_world(root, shift):
    for node in root.find("worldbody"):
        for key in ("pos", "fromto"):
            values = node.get(key)
            if values:
                vectors = np.fromstring(values, sep=" ").reshape(-1, 3)
                node.set(key, vector((vectors-shift).ravel()))


def assemble(indices, segments, prefix, shift=None):
    from nav4.matrix.compose import new_scene

    root = new_scene("ATEC A trail", CONFIG)
    ET.SubElement(root, "size", memory="512M")
    for i in indices:
        tag = f"seg_{i:02d}"
        source = ET.parse(segments/tag/"scene.xml").getroot()
        prefix_assets(source, prefix/tag)
        if shift is not None:
            shift_world(source, shift)
        root.find("asset").extend(source.find("asset").findall("mesh"))
        root.find("worldbody").extend(node for node in source.find("worldbody") if node.tag != "light")
    return root


def link_or_copy(a, b):
    if Path(b).exists():
        return b
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)
    return b


def matrix_scene(zgws, robot_source, path, sampler, station, indices, label, shift):
    robot = copy.deepcopy(robot_source)
    xy = path_at(path, station)
    body = robot.find("./worldbody/body[@name='base_link']")
    yaw = float(np.arctan2(*heading(path, station)[::-1]))
    init = np.r_[xy, sampler(xy[None])[0]+.7]
    body.set("pos", vector(init-shift))
    body.set("quat", vector([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]))
    robot_file = f"zgws_atec_robot_{label}.xml"
    save_xml(robot, zgws/robot_file)
    root = assemble(indices, FULL/"segments", Path("atec_full"), shift)
    geometry_file = f"atec_scene_{label}.xml"
    save_xml(root, zgws/geometry_file)
    combined = ET.Element("mujoco", model=f"ZGWS A trail {label}")
    ET.SubElement(combined, "include", file=robot_file)
    ET.SubElement(combined, "include", file=geometry_file)
    entry = f"zgws_atec_{label}.xml"
    save_xml(combined, zgws/entry)
    return entry, init-shift, yaw


def main():
    start = time.monotonic()
    n = len(LIMITS)-1
    static = FULL/"segments"
    static_prefix = Path("segments")
    save_xml(assemble(range(n), static, static_prefix), FULL/"scene_full.xml")
    manifest = []
    for i in range(n-1):
        # Keep all paths rooted at full_v1, as with scene_full.xml.
        entry = f"window_{i:02d}.xml"
        save_xml(assemble([i, i+1], static, static_prefix), FULL/entry)
        manifest.append(dict(window=entry, segments=[i, i+1],
                             station_m=[LIMITS[i], LIMITS[i+2]],
                             next_common_m=[LIMITS[i+1], LIMITS[i+2]] if i+2<n else None))

    relative = Path("Content/model")
    vendor = RELEASE/relative/"zgws"
    matrix_root = FULL/"matrix/UeSim"/relative
    zgws = matrix_root/"zgws"
    zgws.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        src = scene_dir(FULL, i)
        dst = zgws/"atec_full"/src.name
        shutil.copytree(src, dst, copy_function=link_or_copy, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("geometry.json", "load.json", "scene.xml"))
    shutil.copytree(vendor/"assets", zgws/"assets", dirs_exist_ok=True, copy_function=link_or_copy)
    robot = ET.parse(vendor/"zgws.xml").getroot()
    prefix_assets(robot, Path(robot.find("compiler").attrib.pop("meshdir")))
    model = np.load(FULL/"surface.npz")
    base, z = model["origin"], model["z"]
    _, path = route()
    sample_z = lambda xy: grid_sample(z, base, xy)
    initial_station = .40
    xy = path_at(path, initial_station)
    shift = np.r_[xy, sample_z(xy[None])[0]]
    config = json.loads((RELEASE/relative/"config/config.json").read_text())
    for row, i in zip(manifest, range(n-1)):
        label = f"window_{i:02d}"
        entry, pos, yaw = matrix_scene(zgws, robot, path, sample_z, LIMITS[i]+.40, [i, i+1], label, shift)
        row.update(matrix_entry=(relative/"zgws"/entry).as_posix(), initial_base_pos_m=pos, initial_yaw_rad=yaw)
    whole, pos, yaw = matrix_scene(zgws, robot, path, sample_z, initial_station, range(n), "full", shift)
    config["robot"].update(mujoco_model=(relative/"zgws"/whole).as_posix(), scene_model="", main_body="base_link",
                           inside_mc=False, position=dict(zip("xyz", pos.tolist())))
    write_json(matrix_root/"config/config_atec_full_candidate.json", config)
    window = manifest[0]
    config["robot"].update(mujoco_model=window["matrix_entry"],
                           position=dict(zip("xyz", np.asarray(window["initial_base_pos_m"]).tolist())))
    write_json(matrix_root/"config/config_atec_window_00_candidate.json", config)
    write_json(FULL/"assembly.json", dict(statics=manifest, matrix_full=(relative/"zgws"/whole).as_posix(),
                                         initial_matrix_base_m=pos, initial_yaw_rad=yaw, shared_shift_m=shift,
                                         seconds=time.monotonic()-start))
    print("assemblies built", len(manifest), flush=True)


if __name__ == "__main__":
    main()
