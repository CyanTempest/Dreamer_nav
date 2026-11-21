"""Refine existing obstacle occupancy against source triangles AND measured LiDAR.

The ground, route, robot, transforms and native visual scene are retained. Only
coarse occupied cells in the requested route interval are subdivided. Fine cells
near either source surface remain conservatively occupied; no obstacle is
removed merely because a navigation policy contacted it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from _bootstrap import ROOT
from atec_nav.sim.surface_voxels import child_cells, conservative_surface_mask, merge_cells


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-s', type=float, default=2.)
    parser.add_argument('--end-s', type=float, default=12.)
    parser.add_argument('--resolution', type=float, default=.02)
    a = parser.parse_args()
    src, dst = a.source.resolve(), a.output.resolve()
    if dst.exists(): raise ValueError('Output must be a new version')
    m = json.loads((src / 'manifest.json').read_text())
    existing_levels = m.get('obstacle_voxel_levels_m', [m['obstacle_voxel_m']])
    if (not 0 < a.resolution <= min(existing_levels) or not any(level > a.resolution for level in existing_levels)
            or not 0 <= a.start_s < a.end_s <= 100):
        raise ValueError('Expected a finer grid and a valid course interval')
    route = np.loadtxt(src / 'centerline.csv', delimiter=',', skiprows=1)
    with np.load(src / 'obstacles.npz') as data: boxes = data['boxes'].copy()
    _, index = cKDTree(route[:, 1:3]).query(boxes[:, :2])
    selected = (route[index, 0] >= a.start_s) & (route[index, 0] <= a.end_s)
    if not selected.any(): raise ValueError('No parent boxes in interval')
    cells = child_cells(boxes[selected], a.resolution)
    centers = (cells + .5) * a.resolution
    pad = np.sqrt(3.) * a.resolution
    lower, upper = centers.min(0) - pad, centers.max(0) + pad
    mesh = trimesh.load(src / 'visual.obj', process=False, force='mesh')
    tri = mesh.triangles
    keep_triangles = (tri.max(1) >= lower).all(1) & (tri.min(1) <= upper).all(1)
    local = mesh.submesh([np.flatnonzero(keep_triangles)], append=True)
    del tri, mesh
    transform = np.asarray(m['T_scene_native'])
    native_corners = np.array(np.meshgrid(*zip(lower, upper))).reshape(3, -1).T
    native_corners = (native_corners - transform[:3, 3]) @ transform[:3, :3]
    cloud = np.load(m['lidar_roi'], mmap_mode='r')
    select_cloud = ((cloud[:, :3] >= native_corners.min(0)) &
                    (cloud[:, :3] <= native_corners.max(0))).all(1)
    lidar = np.asarray(cloud[select_cloud, :3]) @ transform[:3, :3].T + transform[:3, 3]
    del cloud, select_cloud
    lidar_tree = cKDTree(lidar)
    mesh_distances = np.empty(len(centers)); lidar_distances = np.empty(len(centers))
    started = time.monotonic()
    for begin in range(0, len(centers), 2048):
        stop = min(begin + 2048, len(centers))
        _, mesh_distances[begin:stop], _ = trimesh.proximity.closest_point(local, centers[begin:stop])
        lidar_distances[begin:stop] = lidar_tree.query(centers[begin:stop])[0]
        if begin % 20480 == 0:
            print(json.dumps({'processed':stop, 'total':len(centers), 'seconds':time.monotonic()-started}), flush=True)
    retained = conservative_surface_mask(np.minimum(mesh_distances, lidar_distances), a.resolution)
    refined = merge_cells(cells[retained], a.resolution)
    combined = np.concatenate([boxes[~selected], refined])
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.is_file() and f.name not in ('manifest.json', 'obstacles.npz', 'refinement_audit.npz'):
            os.link(f, dst / f.name)
    np.savez_compressed(dst / 'obstacles.npz', boxes=combined)
    np.savez_compressed(dst / 'refinement_audit.npz', child_indices=cells, retained=retained,
                        distance_to_mesh_m=mesh_distances, distance_to_lidar_m=lidar_distances,
                        selected_parent_indices=np.flatnonzero(selected))
    refinement = {'parent':str(src), 'parent_manifest_sha256':sha(src / 'manifest.json'),
        'parent_obstacles_sha256':sha(src / 'obstacles.npz'), 'visual_sha256':sha(src / 'visual.obj'),
        'lidar_sha256':m['lidar_roi_sha256'], 'route_interval_m':[a.start_s, a.end_s],
        'fine_resolution_m':a.resolution, 'parent_resolution_m':m['obstacle_voxel_m'],
        'selected_parent_boxes':int(selected.sum()), 'candidate_fine_cells':len(cells),
        'retained_fine_cells':int(retained.sum()), 'refined_boxes':len(refined),
        'unchanged_boxes':int((~selected).sum()), 'local_source_triangles':len(local.faces),
        'local_lidar_points':len(lidar), 'method':'retain_child_cube_if_center_distance_to_source_mesh_OR_lidar_le_half_diagonal',
        'surface_coverage':'all source surface intersections inside the selected parent envelope retained conservatively',
        'outside_parent_envelope':'unchanged; no new completeness claim',
        'source_script_sha256':sha(__file__), 'wall_seconds':time.monotonic()-started}
    if (src / 'refinement_audit.npz').is_file():
        refinement['parent_refinement_audit_sha256'] = sha(src / 'refinement_audit.npz')
        refinement['parent_obstacle_voxel_levels_m'] = existing_levels
    m.update(scene_id='atec_competition_forest_100m_' + dst.name,
             parent_terrain=str(src), obstacle_boxes=len(combined), obstacle_voxel_m=a.resolution,
             obstacle_voxel_levels_m=sorted({a.resolution, *existing_levels}),
             surface_collision_refinement=refinement, physics_validated=False,
             status='surface_refinement_requires_physical_validation', navigation_complete=False)
    (dst / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    print(json.dumps(refinement, indent=2), flush=True)


if __name__ == '__main__': main()
