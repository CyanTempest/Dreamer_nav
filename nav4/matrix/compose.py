"""MJCF writer for paired render/raycast meshes and rigid triangle collision."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def vector(values):
    return " ".join(f"{v:.9g}" for v in values)


def xml(parent, tag, **attributes):
    values = {key: value if isinstance(value, str) else vector(np.atleast_1d(value))
              for key, value in attributes.items()}
    return ET.SubElement(parent, tag, values)


def save_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")


def write_obj(path, vertices, faces, uv=None):
    with path.open("w") as stream:
        np.savetxt(stream, vertices, fmt="v %.7f %.7f %.7f")
        if uv is None:
            np.savetxt(stream, faces + 1, fmt="f %d %d %d")
        else:
            np.savetxt(stream, uv, fmt="vt %.7f %.7f")
            np.savetxt(stream, np.repeat(faces + 1, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def new_scene(name, cfg):
    root = ET.Element("mujoco", model=name)
    xml(root, "compiler", angle="radian", autolimits="true")
    xml(root, "option", timestep=cfg["physics"]["timestep_m"], integrator="implicitfast")
    # Extent determines clipping distances, so keep the near plane independent of route length.
    xml(root, "statistic", extent=10)
    vis = xml(root, "visual")
    xml(vis, "global", offwidth=cfg["render"]["width"], offheight=cfg["render"]["height"])
    xml(vis, "map", znear=.004, zfar=12, fogstart=2.5, fogend=8)
    xml(vis, "headlight", ambient=[.43, .42, .40], diffuse=[.47, .46, .42], specular=[.05]*3)
    assets = xml(root, "asset")
    xml(assets, "texture", type="skybox", builtin="gradient", rgb1=[.55, .67, .76],
        rgb2=[.83, .88, .9], width=256, height=256)
    for name, rgba in cfg["materials"].items():
        xml(assets, "material", name=name, rgba=rgba, specular=.08, shininess=.12)
    world = xml(root, "worldbody")
    xml(world, "light", directional="true", dir=[-.42, -.35, -1],
        diffuse=[.44, .41, .32], specular=[.08]*3, castshadow="false")
    return root


def place_terrain(root, name, obj, pos, radius):
    assets, world = root.find("asset"), root.find("worldbody")
    xml(assets, "mesh", name=name, file=obj, inertia="shell")
    xml(world, "geom", type="mesh", name=name, mesh=name, pos=pos, rgba=[.55, .54, .51, 1],
        contype=0, conaffinity=0, group=0)
    flex = xml(world, "flexcomp", name=f"{name}_collision", type="mesh", file=obj,
               pos=pos, rigid="true", radius=radius, rgba=[0, 0, 0, 0], group=3)
    xml(flex, "contact", contype=1, conaffinity=0, selfcollide="none",
        friction=[1., .005, .0001], solref=[.008, 1.])


def place_hard(root, item):
    attributes = dict(item["geom"], name=item["name"], material=item["material"],
                      contype=1, conaffinity=0, group=item.get("group", 1))
    if item["geom"]["type"] == "mesh":
        xml(root.find("asset"), "mesh", name=item["name"], file=item["file"])
        attributes["mesh"] = item["name"]
    xml(root.find("worldbody"), "geom", **attributes)


def scene_subset(path, predicate):
    """Read full scene with asset paths made absolute before in-memory compilation."""
    root = ET.parse(path).getroot()
    for node in root.iter():
        if "file" in node.attrib:
            node.set("file", str((Path(path).parent / node.get("file")).resolve()))
    world = root.find("worldbody")
    for node in list(world):
        if not predicate(node):
            world.remove(node)
    used = {node.get(attr) for node in world for attr in ("mesh", "material")}
    assets = root.find("asset")
    used |= {node.get("texture") for node in assets if node.get("name") in used}
    for node in list(assets):
        if node.get("name") is not None and node.get("name") not in used:
            assets.remove(node)
    return root
