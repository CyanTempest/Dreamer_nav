"""CPU orthographic diagnostics and same-camera/video comparisons."""
import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from nav4.mapping.common import write_json
from nav4.matrix.compose import scene_subset, xml
from nav4.matrix.full_ground import FULL, PILOT
from nav4.matrix.stairs import heading, path_at
from nav4.matrix.survey import OUT

WIDTH, HEIGHT = 768, 432


def measure_compile(name):
    t = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(name))
    return model, dict(seconds=time.monotonic()-t, version=mujoco.__version__,
                       mesh_faces=model.nmeshface, collision_triangles=model.nflexelem,
                       mesh_assets=model.nmesh, geoms=model.ngeom)


def camera(frame, n, world, fovy=48):
    eye, target = map(np.asarray, (frame["eye"], frame["target"]))
    unit = target-eye
    unit /= np.linalg.norm(unit)
    right = np.cross(unit, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, unit)
    xml(world, "camera", name=f"view_{n:02d}", pos=eye, xyaxes=np.r_[right, up], fovy=fovy)


def images(scene, frames, show_soft=True):
    root = scene_subset(scene, lambda n: True)
    for i, frame in enumerate(frames):
        camera(frame, i, root.find("worldbody"), frame.get("vertical_fov_deg", 48))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    flags = mujoco.MjvOption()
    flags.flexgroup[3] = 0
    flags.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = 0
    flags.geomgroup[4] = 1
    flags.geomgroup[5] = int(show_soft)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH, max_geom=max(12000, model.ngeom+1000)) as renderer:
        for i in range(len(frames)):
            renderer.update_scene(data, camera=f"view_{i:02d}", scene_option=flags)
            yield Image.fromarray(renderer.render().copy())


def panel(items, target, width=WIDTH, height=HEIGHT):
    sheet = Image.new("RGB", (width*len(items), height+29), (28, 31, 31))
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(items):
        pic = im.convert("RGB")
        pic.thumbnail((width, height))
        sheet.paste(pic, (i*width+(width-pic.width)//2, 29+(height-pic.height)//2))
        draw.text((i*width+12, 7), label, fill="white")
    sheet.save(target, quality=92)


def overview(path, data, frames, thumb=256):
    with Image.open(path) as sample:
        ratio = sample.height/sample.width
    height = int(thumb*ratio)
    grid = Image.new("RGB", (4*thumb, int(np.ceil(len(frames)/4))*(height+20)), "#1c2023")
    draw = ImageDraw.Draw(grid)
    folder = path.parent
    for i, (index, caption) in enumerate(frames):
        pic = Image.open(folder/f"{index:05d}.jpg").convert("RGB")
        pic.thumbnail((thumb, height))
        x,y = (i%4)*thumb, (i//4)*(height+20)
        grid.paste(pic, (x,y+20))
        draw.text((x+7,y+4),caption,fill="white")
    grid.save(data, quality=89)


def pilot():
    target = PILOT/"figures"
    target.mkdir(exist_ok=True)
    old_cameras = json.loads((OUT/"camera_poses.json").read_text())
    config = json.loads((Path(__file__).with_name("settings.json")).read_text())
    _, load = measure_compile(PILOT/"scene.xml")
    write_json(PILOT/"load.json",load)
    for i, (image, (ref, station)) in enumerate(zip(images(PILOT/"scene.xml", old_cameras),
                                                   config["render"]["views"]), 1):
        original = Image.open(OUT/"figures"/f"mujoco_{i:02d}.png")
        frame = Image.open(OUT/"figures"/"reference"/f"ref_{ref:02d}.jpg")
        panel([(frame, f"MATRiX video  +{ref:02d}s"), (original, "Original pilot / matched camera"),
               (image, f"Full method pilot  {station:.2f}m / OSMesa")],target/f"compare_{i:02d}.jpg")
        image.save(target/f"mujoco_{i:02d}.png")
    write_json(PILOT/"camera_poses.json", old_cameras)
    print("pilot renders", load, flush=True)


def pilot_tree():
    target = PILOT/"figures"
    target.mkdir(exist_ok=True)
    old = json.loads((OUT/"camera_poses.json").read_text())[0]
    camera = {**old, "eye": np.asarray(old["eye"])+[0,.62,.18], "vertical_fov_deg": 57}
    camera["target"] = np.asarray(old["target"])+[-.8,1.37,.65]
    reference = Image.open(OUT/"figures/reference/ref_07.jpg")
    pic = next(images(PILOT/"scene.xml",[camera],show_soft=False))
    panel([(reference, "MATRiX video +07s"), (pic, "OSMesa  wider angle: terrain, trunk and roots")],
          target/"entry_tree_detail.jpg")
    write_json(target/"entry_tree_camera.json",camera)


def film_grid():
    folder = Path.home()/"ATEC_NAV"/"frames4hz"
    frames = [(i, f"{i:05d}  {(i-1)/4:06.2f}s") for i in np.rint(np.linspace(1, 2863, 48)).astype(int)]
    overview(folder/"00001.jpg", FULL/"work/film_contact.jpg", frames)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pilot", "film", "tree"])
    {"pilot":pilot, "film":film_grid, "tree":pilot_tree}[p.parse_args().command]()
