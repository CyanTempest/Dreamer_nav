"""Render actual M1 physics states against the native dataset 3D Gaussian scene."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from _bootstrap import ROOT
from atec_nav.sim.mujoco_route import load_scene
from atec_nav.sim.splats import SplatRenderer, mujoco_camera


def contact_shadows(frame, model, data, view, intrinsic):
    """Small soft contact shadows, anchored only to actual wheel/world contacts.

    This is a screen-space lighting approximation; static scene lighting is baked
    into the captured gaussians. It neither moves the robot nor changes physics.
    """
    seen = set()
    for contact in data.contact:
        if contact.dist > .002: continue
        bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        wheels = [b for b in bodies if "FOOT_LINK" in model.body(b).name]
        if 0 not in bodies or not wheels or wheels[0] in seen: continue
        seen.add(wheels[0])
        point = view[:3, :3] @ contact.pos + view[:3, 3]
        if point[2] <= .05: continue
        focal = intrinsic[0, 0]
        uv = focal * point[:2] / point[2] + intrinsic[:2, 2]
        jac = focal / point[2] * np.array([[1, 0, -point[0]/point[2]], [0, 1, -point[1]/point[2]]])
        tangent = jac @ view[:3, :2]
        covariance = tangent @ tangent.T * .065**2 + np.eye(2)
        radius = int(np.ceil(3*np.sqrt(np.linalg.eigvalsh(covariance).max())))
        x0, x1 = max(0, int(uv[0])-radius), min(frame.shape[1], int(uv[0])+radius+1)
        y0, y1 = max(0, int(uv[1])-radius), min(frame.shape[0], int(uv[1])+radius+1)
        if x1 <= x0 or y1 <= y0: continue
        xx, yy = np.meshgrid(np.arange(x0,x1), np.arange(y0,y1))
        delta = np.stack([xx-uv[0], yy-uv[1]], -1)
        distance = np.einsum('...i,ij,...j->...', delta, np.linalg.inv(covariance), delta)
        frame[y0:y1,x0:x1] = (frame[y0:y1,x0:x1] * (1-.45*np.exp(-.5*distance))[...,None]).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--name", default="follow_camera_3dgs.mp4")
    parser.add_argument("--camera-distance", type=float, default=4.2)
    parser.add_argument("--camera-elevation", type=float, default=-24.)
    parser.add_argument("--camera-azimuth-offset", type=float, default=155.)
    args = parser.parse_args()
    out = args.run.resolve()
    destination = out / args.name
    if destination.exists(): raise ValueError("Output exists")
    manifest = json.loads((out / "run_manifest.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    states = np.load(out / "physical_states.npz")
    model, route, _, _ = load_scene(manifest["robot"], manifest["terrain"])
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = .35
    model.vis.headlight.diffuse[:] = .45
    renderer = mujoco.Renderer(model, args.height, args.width)
    splats = SplatRenderer(Path(manifest["terrain"]) / "splats.npz")
    # Robot visuals and source mesh both use group 1 in the physical scene.
    # Move only the static visual to a hidden group for hybrid composition.
    model.geom_group[model.geom("road_visual").id] = 4
    option = mujoco.MjvOption();option.geomgroup[2:] = 0
    camera = mujoco.MjvCamera();camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = args.camera_distance;camera.elevation = args.camera_elevation
    begin = time.monotonic()
    rows = []
    stop = len(states["time"]) if not args.max_frames else min(len(states["time"]), args.start_frame+args.max_frames)
    try:
        with imageio.get_writer(destination, fps=manifest["fps"], codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1, ffmpeg_params=["-crf", "21", "-preset", "fast"]) as writer:
            for i in range(args.start_frame, stop):
                data.qpos[:] = states["qpos"][i];data.qvel[:] = states["qvel"][i];data.time = states["time"][i]
                mujoco.mj_forward(model, data)
                rot = data.xmat[1].reshape(3, 3)
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
                camera.lookat[:] = data.qpos[:3] + [np.cos(yaw)*.4, np.sin(yaw)*.4, .05]
                camera.azimuth = np.degrees(yaw)+args.camera_azimuth_offset
                renderer.update_scene(data, camera, scene_option=option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                view, intrinsic = mujoco_camera(renderer.scene, args.width, args.height)
                robot = renderer.render().copy()
                renderer.enable_depth_rendering();robot_depth = renderer.render().copy();renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering();seg = renderer.render().copy();renderer.disable_segmentation_rendering()
                scene, alpha = splats.render(view, intrinsic, args.width, args.height)
                frame = np.uint8(np.clip(scene[..., :3]*255, 0, 255))
                contact_shadows(frame, model, data, view, intrinsic)
                mask = (seg[..., 0] >= 0) & ((robot_depth <= scene[..., 3]+.03) | (alpha < .1))
                if not (seg[..., 0] >= 0).any():
                    raise RuntimeError("Robot missing from the shared camera view")
                frame[mask] = robot[mask]
                progress = float(states["progress_m"][i]) - float(manifest.get('start_s', 0.))
                cv2.rectangle(frame, (0, 0), (args.width, 76), (22, 28, 30), -1)
                label = 'ATEC | M1 physical wheel baseline'
                if manifest.get('controller_variant') == 'terrain_step':
                    label = 'ATEC | Experimental wheel-lift mobility test'
                if manifest.get('controller_variant') == 'matrix_robot_mc':
                    motion = manifest.get('locomotion','official_ZGWS_RL_stair').rsplit('_',1)[-1]
                    label = 'ATEC | M1 + official MATRiX Robot MC | '+motion+' policy'
                if manifest.get('policy') == 'DreamerV3_checkpoint':
                    label = 'ATEC | DreamerV3 + MATRiX | checkpoint evaluation'
                if manifest.get('teacher_used'):
                    label = 'ATEC | Reference controller + MATRiX | physical rollout'
                if i < len(states['time'])-1:
                    label += ' | running'
                else:
                    label += ' | ' + ('goal reached' if metrics['success'] else
                                       metrics.get('result',metrics.get('reason','incomplete')))
                cv2.putText(frame, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 244, 245), 2, cv2.LINE_AA)
                cv2.putText(frame, f't={data.time:.1f}s   Route {progress:.2f}/{metrics["route_length_m"]:.1f} m   v={np.linalg.norm(data.qvel[:2]):.2f} m/s', (18, 59), cv2.FONT_HERSHEY_SIMPLEX, .62, (110, 230, 245), 2, cv2.LINE_AA)
                writer.append_data(frame)
                rows.append([i-args.start_frame, i, float(data.time), progress])
                if i == args.start_frame or i in (stop//2, stop-1):
                    imageio.imwrite(destination.with_name(destination.stem+f'_{i:06}.png'), frame)
                if (i-args.start_frame) % 100 == 0:
                    print(json.dumps({"frame": i, "total": stop, "wall_s": round(time.monotonic()-begin, 1)}), flush=True)
    finally:
        renderer.close()
    with destination.with_suffix('.frames.csv').open('w') as stream:
        writer = csv.writer(stream);writer.writerow(['encoded_frame','physics_state_index','sim_time_s','progress_m']);writer.writerows(rows)
    report = {"video": destination.name, "frame_count": len(rows), "fps": manifest["fps"], "resolution": [args.width, args.height],
              "duration_s": len(rows)/manifest["fps"], "source": "saved_actual_physics_states",
              "appearance": "native_3DGS", "occlusion": "expected_gaussian_depth_vs_robot_metric_depth",
              "navigation_success": metrics['success'], "navigation_result": metrics.get('result',metrics.get('reason','incomplete')),
              "camera": {"distance": args.camera_distance, "elevation": args.camera_elevation, "azimuth_offset": args.camera_azimuth_offset},
              "dynamic_robot_shadows": "approximate_soft_wheel_contact_shadows", "is_full_run": args.start_frame == 0 and stop == len(states["time"]),
              "render_wall_s": time.monotonic()-begin,
              "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.with_suffix('.recording.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
