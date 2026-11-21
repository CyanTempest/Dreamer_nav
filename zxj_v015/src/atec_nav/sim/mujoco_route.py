"""M1 physical rolling baseline on a measured ATEC road heightfield."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def initialize_on_terrain(model, data, controller, route, start_s=0., lateral_offset=0., yaw_offset=0.):
    """Place a reset pose above measured wheel supports, before any dynamics.

    The fitted attitude is an initialization only; all later movement is physical.
    A flat-ground spawn height can embed front shins in the first stair riser.
    """
    from scipy.spatial.transform import Rotation
    controller.initialize(data)
    point = np.array([np.interp(start_s, route[:, 0], route[:, j]) for j in (1, 2, 3)])
    ahead = np.array([np.interp(min(route[-1, 0], start_s+.8), route[:, 0], route[:, j]) for j in (1, 2)])
    yaw = np.arctan2(*(ahead-point[:2])[::-1])
    point[:2] += float(lateral_offset) * np.array([-np.sin(yaw), np.cos(yaw)])
    yaw += float(yaw_offset)
    data.qpos[:3] = point + [0, 0, .8]
    data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
    wheel_bodies = [model.body(name+'_FOOT_LINK').id for name in ('FAR','FBL','RAR','RBL')]
    ground_geom = model.geom('road_collision').id
    direction = np.array([0., 0., -1.])
    def support(xy):
        origin = np.r_[xy, point[2]+4]
        distance = mujoco.mj_rayHfield(model, data, ground_geom, origin, direction)
        if distance < 0:
            raise ValueError('Reset wheel has no measured ground support')
        return origin[2]-distance
    mujoco.mj_forward(model, data)
    wheel_xy = data.xpos[wheel_bodies, :2]
    heights = np.array([support(xy) for xy in wheel_xy])
    plane, _, _, _ = np.linalg.lstsq(np.c_[wheel_xy-point[:2],np.ones(4)],heights,rcond=None)
    normal = np.r_[-plane[:2],1.]; normal /= np.linalg.norm(normal)
    forward = np.array([np.cos(yaw),np.sin(yaw),0.])
    forward -= normal*np.dot(forward,normal); forward /= np.linalg.norm(forward)
    rotation = np.column_stack([forward,np.cross(normal,forward),normal])
    data.qpos[3:7] = Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
    mujoco.mj_forward(model, data)
    # Sample the wheel footprint, not just its center, to avoid placing it through a riser.
    required = []
    for body in wheel_bodies:
        center = data.xpos[body]
        floor = max(support(center[:2]+rotation[:2,:2]@np.array([x,y]))
                    for x in (-.08,0,.08) for y in (-.025,0,.025))
        required.append(floor+controller.radius+.015-center[2])
    data.qpos[2] += max(required)
    mujoco.mj_forward(model, data)


def load_scene(robot, terrain, xml_output=None, matrix_imu=False):
    robot, terrain = Path(robot).resolve(), Path(terrain).resolve()
    root = ET.parse(robot / "robot.xml").getroot()
    root.find("compiler").set("meshdir", str(robot / "meshes"))
    asset = root.find("asset")
    world = root.find("worldbody")
    if matrix_imu:
        ET.SubElement(world.find('body'),'site',name='imu',pos='0 0 0')
        sensors=root.find('sensor')
        if sensors is None:sensors=ET.SubElement(root,'sensor')
        ET.SubElement(sensors,'framequat',name='imu_quat',objtype='site',objname='imu')
        ET.SubElement(sensors,'gyro',name='imu_gyro',site='imu')
        ET.SubElement(sensors,'accelerometer',name='imu_acc',site='imu')
    grid = np.load(terrain / "terrain.npz")
    h, xs, ys = grid["heights"], grid["xs"], grid["ys"]
    zmin, zrange = float(h.min()), float(np.ptp(h))
    field = ET.SubElement(asset, "hfield", name="road", nrow=str(len(ys)), ncol=str(len(xs)),
                          size=f"{np.ptp(xs)/2} {np.ptp(ys)/2} {zrange} .2")
    if xml_output is not None:
        # Standalone scene.xml must load the actual ground, not an empty heightfield.
        field_path = Path(xml_output).resolve().with_suffix(".hfield.bin")
        with field_path.open("wb") as stream:
            np.asarray(h.shape, dtype="<i4").tofile(stream)
            np.asarray((h-zmin)/zrange, dtype="<f4").tofile(stream)
        field.attrib.pop("nrow");field.attrib.pop("ncol")
        field.set("file", str(field_path))
    ET.SubElement(world, "geom", name="road_collision", type="hfield", hfield="road",
                  pos=f"{np.mean(xs)} {np.mean(ys)} {zmin}", contype="2", conaffinity="1",
                  friction="0.9 .005 .0001", group="3")
    # Render exactly the source mesh, while collision uses its sampled ground surface.
    ET.SubElement(asset, "mesh", name="road_visual", file=str(terrain / "visual.obj"), maxhullvert="32")
    ET.SubElement(world, "geom", name="road_visual", type="mesh", mesh="road_visual",
                  contype="0", conaffinity="0", rgba=".48 .52 .46 1", group="1")
    if (terrain / "obstacles.npz").is_file():
        with np.load(terrain / "obstacles.npz") as obstacles:
            for index, box in enumerate(obstacles["boxes"]):
                ET.SubElement(world, "geom", name=f"obstacle_{index}", type="box",
                              pos=" ".join(map(str, box[:3])), size=" ".join(map(str, box[3:])),
                              contype="2", conaffinity="1", friction=".9 .005 .0001", group="3")
    # Engineering camera mount: optical +Z forward along base +X, image down -Z.
    # Position is an assumed virtual sensor, not a calibrated hardware extrinsic.
    base = world.find("body")
    ET.SubElement(base, "camera", name="navigation", pos=".35 0 .15",
                  xyaxes="0 -1 0 0 0 1", fovy="70", ipd="0")
    ET.SubElement(world, "light", pos="0 0 15", dir="0 0 -1", directional="true", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="0 0 10", dir=".3 .4 -1", directional="true", diffuse=".35 .35 .35")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "map", znear=".0001", zfar="15")
    ET.SubElement(visual, "quality", shadowsize="2048")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=".35 .55 .75", rgb2=".85 .9 .95", width="256", height="1536")
    route = np.loadtxt(terrain / "centerline.csv", delimiter=",", skiprows=1)
    for i in range(0, len(route), 25):
        p = route[i, 1:4] + [0, 0, .025]
        ET.SubElement(world, "geom", name=f"route_marker_{i}", type="sphere", size=".035",
                      pos=" ".join(map(str, p)), contype="0", conaffinity="0", rgba=".1 .7 .95 1", group="2")
    ET.indent(root)
    xml = ET.tostring(root).decode()
    if xml_output is not None:
        Path(xml_output).write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    model.hfield_data[:] = ((h - zmin) / zrange).ravel()
    return model, route, grid, json.loads((robot / "manifest.json").read_text())


def route_position(route, xy, previous_index):
    """Ordered local projection: nearby hairpins cannot jump progress."""
    start, end = max(0, previous_index - 5), min(len(route) - 1, previous_index + 30)
    a, b = route[start:end, 1:3], route[start + 1:end + 1, 1:3]
    v = b - a
    fractions = np.clip(np.sum((xy - a) * v, axis=1) / np.sum(v * v, axis=1), 0, 1)
    projected = a + fractions[:, None] * v
    errors = np.linalg.norm(projected - xy, axis=1)
    local = int(np.argmin(errors))
    index = start + local
    progress = route[index, 0] + fractions[local] * (route[index + 1, 0] - route[index, 0])
    return index, float(progress), float(errors[local])


def navigation_command(route, xy, yaw, progress, goal_s, speed=.4, lookahead=1.2, lateral_offset=0., respect_yaw_limit=False):
    if lookahead <= 0 or abs(lateral_offset) > .2:
        raise ValueError('Invalid local reference tracking parameters')
    target_s = min(goal_s, progress + lookahead)
    target = np.array([np.interp(target_s, route[:, 0], route[:, j]) for j in (1, 2)])
    if lateral_offset:
        before = np.array([np.interp(max(route[0,0], target_s-.1), route[:,0], route[:,j]) for j in (1,2)])
        after = np.array([np.interp(min(route[-1,0], target_s+.1), route[:,0], route[:,j]) for j in (1,2)])
        tangent = after-before; tangent /= max(np.linalg.norm(tangent), 1e-9)
        target += lateral_offset*np.array([-tangent[1], tangent[0]])
    delta = target - xy
    angle = (np.arctan2(delta[1], delta[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
    vx = speed * max(.2, np.cos(angle))
    vx = min(vx, max(.08, (goal_s - progress) * .6))
    curvature = 2 * np.sin(angle) / max(.4, np.linalg.norm(delta))
    if respect_yaw_limit and abs(curvature) > 1e-9:
        # Clipping yaw alone widens the commanded turn. Reduce forward speed
        # first so the desired curvature remains feasible at the same yaw limit.
        vx = min(vx, .3 / abs(curvature))
    wz = np.clip(vx * curvature, -.3, .3)
    return float(vx), float(wz)
