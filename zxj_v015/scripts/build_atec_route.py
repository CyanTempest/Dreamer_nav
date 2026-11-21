"""Build a measured 100 m route and collision heightfield from dataset meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
import trimesh

from _bootstrap import ROOT
from inspect_atec_dataset import ply_vertices


def project_ground(mesh, xy, top=1.0):
    origins = np.column_stack([xy, np.broadcast_to(top, len(xy))])
    heights = np.full(len(xy), -np.inf)
    for start in range(0, len(xy), 2000):
        p = origins[start:start + 2000]
        hit, idx, _ = mesh.ray.intersects_location(p, np.tile([0., 0., -1.], (len(p), 1)), multiple_hits=True)
        np.maximum.at(heights, idx + start, hit[:, 2])
    return heights


def dense_ground_roi(path, route):
    # Fixed schema verified by inspect_atec_dataset.py; stream the 4.2 GB cloud.
    with path.open("rb") as f:
        for _ in range(100):
            if f.readline().strip() == b"end_header":break
        else:raise ValueError("Invalid planning PLY header")
        offset = f.tell()
    dtype = np.dtype([(k, '<f4') for k in ['x', 'y', 'z', 'intensity', 'normal_x', 'normal_y', 'normal_z']] + [('time', '<f8'), ('classification', 'u1')])
    points = np.memmap(path, mode='r', dtype=dtype, offset=offset)
    lo, hi = route[:, :2].min(axis=0)-5, route[:, :2].max(axis=0)+5
    parts = []
    for start in range(0, len(points), 2_000_000):
        chunk = points[start:start+2_000_000]
        mask = (chunk['x']>lo[0]) & (chunk['x']<hi[0]) & (chunk['y']>lo[1]) & (chunk['y']<hi[1]) & (chunk['z']>-1) & (chunk['z']<1)
        parts.append(np.column_stack([chunk[k][mask] for k in ['x','y','z']]))
    return np.concatenate(parts)


def build(dataset, output, start_seconds, end_seconds, resolution, native_x_offset, lane_detour):
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps({"status": "incomplete", "navigation_complete": False}))
    traj_path = dataset / "03_Trajectory/developer_data/poses.csv"
    poses = np.loadtxt(traj_path)
    relative = poses[:, 0] - poses[0, 0]
    mask = (relative >= start_seconds) & (relative <= end_seconds)
    xy = savgol_filter(poses[mask, 1:3], 31, 3, axis=0)
    xy[:, 0] += native_x_offset
    lane_times = [2520, 2540, 2550, 2570, 2580, 2595, 2610, 2625, 2645, 2670]
    lane_offsets = [0, -1, -4, -4, 0, 0, 0, .5, .5, 0]
    if lane_detour:
        xy[:, 0] += PchipInterpolator(lane_times, lane_offsets)(relative[mask])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.arange(0, distance[-1], .2)
    xy = np.column_stack([np.interp(s, distance, xy[:, j]) for j in range(2)])
    tree = cKDTree(xy)
    chunks, sources = [], []
    for path in sorted((dataset / "02_3DGS_map/render2/data/mesh").glob("*.ply")):
        mesh = trimesh.load(path, process=False)
        centers = mesh.triangles_center
        keep = (tree.query(centers[:, :2])[0] < 8) & (centers[:, 2] > -2) & (centers[:, 2] < 6)
        if keep.any():
            chunks.append(mesh.submesh([np.flatnonzero(keep)], append=True))
            sources.append({"path": str(path.relative_to(dataset)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    mesh = trimesh.util.concatenate(chunks)
    z = project_ground(mesh, xy, top=1.0)
    if not np.isfinite(z).all():
        raise ValueError("Route has unobserved ground; choose another segment")
    route = np.column_stack([xy, z])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    if arc[-1] < 100:
        raise ValueError("Selected segment is shorter than 100 m")
    s = np.linspace(0, 100, 501)
    route = np.column_stack([np.interp(s, arc, route[:, j]) for j in range(3)])
    native_start = route[0].copy()
    yaw = float(np.arctan2(*(route[5, :2] - route[0, :2])[::-1]))
    rotation = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ native_start
    mesh.apply_transform(transform)
    route = (route - native_start) @ rotation.T
    tree = cKDTree(route[:, :2])
    faces = mesh.faces
    areas = mesh.area_faces
    valid = (areas > 1e-10) & np.isfinite(mesh.vertices[faces]).all(axis=(1, 2))
    mesh = mesh.submesh([np.flatnonzero(valid)], append=True)
    mesh.export(output / "visual.obj")
    _, lidar = ply_vertices(dataset / "01_LiDAR_point_cloud/ds_map/location_map.ply", max_samples=8_000_000)
    native_route = route @ rotation + native_start
    source_tree = cKDTree(native_route[:, :2])
    local_lidar = lidar[(source_tree.query(lidar[:, :2])[0] < 5) & (lidar[:, 2] < 1)]
    dense_lidar = dense_ground_roi(dataset / "01_LiDAR_point_cloud/ds_map/plannar_map.ply", native_route)
    scene_lidar = (dense_lidar - native_start) @ rotation.T
    lidar_xy_tree = cKDTree(scene_lidar[:, :2])
    lower = np.floor((route[:, :2].min(axis=0) - 5) / resolution) * resolution
    upper = np.ceil((route[:, :2].max(axis=0) + 5) / resolution) * resolution
    xs, ys = [np.arange(lo, hi + resolution / 2, resolution) for lo, hi in zip(lower, upper)]
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    corridor_distance = tree.query(grid_xy)[0]
    queried = corridor_distance <= 4.0
    heights = np.full(len(grid_xy), np.nan)
    sampled = project_ground(mesh, grid_xy[queried], top=1.0 - native_start[2])
    heights[queried] = sampled
    observed = np.isfinite(heights)
    from_lidar = np.zeros(len(grid_xy), dtype=bool)
    # Fill mesh sampling holes only with a locally supported measured LiDAR plane.
    for index in np.flatnonzero(queried & ~observed):
        distances, nearest = lidar_xy_tree.query(grid_xy[index], k=12)
        points = scene_lidar[nearest[distances < .30]]
        if len(points) < 5:
            continue
        design = np.column_stack([points[:, :2] - grid_xy[index], np.ones(len(points))])
        coefficients, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
        residual = np.abs(design @ coefficients - points[:, 2])
        if rank == 3 and np.max(residual) < .04 and np.linalg.norm(coefficients[:2]) < .3:
            heights[index] = coefficients[2]
            observed[index] = True
            from_lidar[index] = True
    drive = corridor_distance <= 1.0
    if not observed[drive].all():
        missing = grid_xy[drive & ~observed]
        native_missing = np.column_stack([missing, np.zeros(len(missing))]) @ rotation + native_start
        np.savetxt(output / "unsupported_cells.csv", np.column_stack([missing, corridor_distance[drive & ~observed], native_missing[:, :2]]), delimiter=",", header="x,y,route_distance,native_x,native_y")
        raise ValueError(f"{int((drive & ~observed).sum())} unobserved cells within the driving corridor")
    # Unknown cells stay flagged and are placed below the scene, not interpolated into road.
    heights[~observed] = -3.0
    heights = heights.reshape(xx.shape)
    np.savez_compressed(output / "terrain.npz", heights=heights, observed=observed.reshape(xx.shape),
                        drive=drive.reshape(xx.shape), lidar_repair=from_lidar.reshape(xx.shape), xs=xs, ys=ys)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    np.savetxt(output / "centerline.csv", np.column_stack([arc, route]), delimiter=",",
               header="s_m,x_m,y_m,z_m", comments="", fmt="%.8f")
    # Independent, dense local point-cloud support check at sampled mesh ground points.
    lidar_tree = cKDTree(local_lidar)
    distances, idx = lidar_tree.query(native_route)
    partitions = []
    for lo, hi in [(0, 33), (33, 66), (66, 101)]:
        mask = (arc >= lo) & (arc < hi)
        partitions.append({"s_range_m": [lo, hi], "p50_surface_distance_m": float(np.median(distances[mask])),
                           "p95_surface_distance_m": float(np.percentile(distances[mask], 95)),
                           "median_offset_xyz_m": np.median(local_lidar[idx[mask]] - native_route[mask], axis=0).tolist()})
    # Local agreement, not a claim that the full dataset has been globally registered.
    if np.percentile(distances, 95) > .25:
        raise ValueError("Local lidar/mesh support exceeds 25 cm inspection threshold")
    manifest = {"status": "geometry_ready_physics_pending", "scene_id": "atec_coastal_100m_" + output.name, "dataset_root": str(dataset),
                "sources": sources, "trajectory_sha256": hashlib.sha256(traj_path.read_bytes()).hexdigest(),
                "trajectory_candidate_seconds": [start_seconds, end_seconds],
                "route_native_x_offset_m": native_x_offset,
                "lane_detour": {"time_s": lane_times, "native_x_offset_m": lane_offsets} if lane_detour else None,
                "route_length_m": float(arc[-1]), "native_start_xyz_m": native_start.tolist(),
                "T_scene_native": transform.tolist(), "source_coordinate_assumption": "native_mesh_and_trajectory",
                "frame_validation_scope": "selected_corridor_only_surface_support_against_lidar",
                "local_lidar_alignment": partitions, "lidar_repaired_cells": int(from_lidar.sum()),
                "lidar_repair_source": "01_LiDAR_point_cloud/ds_map/plannar_map.ply",
                "lidar_repair_rule": "at_least_5_points_within_0.30m_plane_max_residual_below_0.04m_slope_below_0.3",
                "units": "meters", "up_axis": "Z", "ground_projection": "highest_surface_below_native_z_1m_on_selected_low_road",
                "collision": "sampled_ground_heightfield", "resolution_m": resolution,
                "unknown_cells_below_scene": True, "drive_corridor_halfwidth_m": 1.0,
                "observed_drive_fraction": float(observed[drive].mean()),
                "visual_faces": len(mesh.faces), "dropped_degenerate_faces": int((~valid).sum()),
                "raw_ground_range_m": [float(route[:, 2].min()), float(route[:, 2].max())],
                "max_centerline_grade": float(np.max(np.abs(np.diff(route[:, 2])) / np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))),
                "physics_validated": False, "navigation_complete": False,
                "limitations": ["low_road_baseline_not_forest_stairs", "unmeasured_friction",
                                "native_3dgs_appearance_not_rendered", "collision_limited_to_ground_corridor"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    plot = axs[0].scatter(grid_xy[observed, 0], grid_xy[observed, 1], c=heights.ravel()[observed], s=1, cmap="terrain", vmin=-.4, vmax=.5)
    axs[0].plot(route[:, 0], route[:, 1], "r-", linewidth=2, label=f"Ground route: {arc[-1]:.3f} m")
    axs[0].scatter(*route[[0, -1], :2].T, c=["lime", "red"], s=60)
    axs[0].set(xlabel="Scene x (m)", ylabel="Scene y (m)", title="ATEC dataset route | observed ground only")
    axs[0].axis("equal");axs[0].legend();fig.colorbar(plot, ax=axs[0], label="Ground z (m)")
    axs[1].plot(arc, route[:, 2]);axs[1].set(xlabel="Ground centerline arc length (m)", ylabel="Ground height (m)", title="Dataset road elevation profile")
    fig.savefig(output / "route_overview.png", dpi=140);plt.close(fig)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-seconds", type=float, default=2520)
    p.add_argument("--end-seconds", type=float, default=2670)
    p.add_argument("--resolution", type=float, default=.15)
    p.add_argument("--native-x-offset", type=float, default=-1.5,
                   help="Offset candidate away from unsupported road-edge cells, then revalidate against measured surfaces")
    p.add_argument("--lane-detour", action="store_true")
    args = p.parse_args()
    build(args.dataset_root.resolve(), args.output.resolve(), args.start_seconds, args.end_seconds, args.resolution, args.native_x_offset, args.lane_detour)
