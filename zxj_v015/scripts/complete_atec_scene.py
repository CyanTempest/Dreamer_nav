"""Add native 3DGS appearance, full-height context and measured obstacle voxels."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
import trimesh

from _bootstrap import ROOT
from atec_nav.sim.splats import read_sog


def build(dataset, base, output, radius=35., voxel=.12, ground_relative=False):
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((base / "manifest.json").read_text())
    transform = np.asarray(source["T_scene_native"])
    route = np.loadtxt(base / "centerline.csv", delimiter=",", skiprows=1)
    native_route = (route[:, 1:4] - transform[:3, 3]) @ transform[:3, :3]
    native_tree = cKDTree(native_route[:, :2])
    scene_tree = cKDTree(route[:, 1:3])
    if ground_relative:
        with np.load(base / 'terrain.npz') as grid:
            ground_at = RegularGridInterpolator((grid['ys'], grid['xs']), grid['heights'],
                                               bounds_error=False, fill_value=np.nan)
    raw = dataset / "02_3DGS_map/render2"
    lcc = json.loads((raw / "scene.lcc2").read_text())
    files = lcc["root"]["splatFiles"]
    selections = defaultdict(list)
    def visit(node):
        lo, hi = [np.array(node["boundingBox"][k]) for k in ("min", "max")]
        closest = np.clip(native_route[:, :2], lo[:2], hi[:2])
        if np.linalg.norm(closest - native_route[:, :2], axis=1).min() > radius:
            return
        children = node.get("child", {})
        if children:
            for child in children.values(): visit(child)
        elif "3dgs" in node.get("data", {}):
            info = node["data"]["3dgs"]
            selections[info["name"]].append((info["start"], info["count"]))
    visit(lcc["root"])
    selections[lcc["root"]["data"]["env"]["name"]] = [(0, lcc["env"]["splatsCount"])]
    parts = defaultdict(list)
    provenance = []
    scene_rot = Rotation.from_matrix(transform[:3, :3])
    for number, (file_id, ranges) in enumerate(sorted(selections.items())):
        path = raw / files[file_id]
        splats = read_sog(path)
        indices = np.concatenate([np.arange(start, start + count) for start, count in ranges])
        if indices.max() >= len(splats["means"]): raise ValueError("LCC range outside SOG")
        points = splats["means"][indices]
        if file_id != lcc["root"]["data"]["env"]["name"]:
            indices = indices[native_tree.query(points[:, :2])[0] <= radius]
        splats = {k: v[indices] for k, v in splats.items()}
        splats["means"] = splats["means"] @ transform[:3, :3].T + transform[:3, 3]
        splats["quats"] = (scene_rot * Rotation.from_quat(splats["quats"][:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        for key, value in splats.items(): parts[key].append(value.astype(np.float32))
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "ranges": ranges, "selected": len(indices)})
        print(f"3DGS {number+1}/{len(selections)}: {len(indices)} splats", flush=True)
    np.savez(output / "splats.npz", **{k: np.concatenate(v) for k, v in parts.items()})
    del parts
    meshes = []
    for path in sorted((raw / "data/mesh").glob("*.ply")):
        mesh = trimesh.load(path, process=False)
        keep = native_tree.query(mesh.triangles_center[:, :2])[0] <= radius
        if keep.any():
            mesh = mesh.submesh([np.flatnonzero(keep)], append=True)
            mesh.apply_transform(transform)
            meshes.append(mesh)
            provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    mesh = trimesh.util.concatenate(meshes)
    mesh.export(output / "visual.obj")
    # Surface occupancy preserves open passages. Never convexify an entire road/tree mesh.
    # Adaptive barycentric samples are <= voxel/2 apart; boxes conservatively cover samples.
    centers = mesh.triangles_center
    nearest = scene_tree.query(centers[:, :2])
    corridor = 1.5 if ground_relative else 4.
    max_z = route[nearest[1], 3] + 2.5 if ground_relative else 4.
    floor_z = ground_at(centers[:, [1, 0]]) if ground_relative else route[nearest[1], 3]
    triangles = mesh.triangles[(nearest[0] <= corridor) & (centers[:, 2] < max_z) &
                               (centers[:, 2] > floor_z + .08)]
    occupied = set()
    for triangle in triangles:
        longest = np.linalg.norm(triangle - np.roll(triangle, 1, axis=0), axis=1).max()
        n = max(1, int(np.ceil(longest / (voxel / 2))))
        u, v = np.triu_indices(n + 1)
        a, b = u / n, (v - u) / n
        points = triangle[0] + a[:, None] * (triangle[1] - triangle[0]) + b[:, None] * (triangle[2] - triangle[0])
        dist, index = scene_tree.query(points[:, :2])
        floor_z = ground_at(points[:, [1, 0]]) if ground_relative else route[index, 3]
        max_z = route[index, 3] + 2.5 if ground_relative else 4.
        points = points[(dist <= corridor) & (points[:, 2] > floor_z + .10) & (points[:, 2] < max_z)]
        occupied.update(map(tuple, np.floor(points / voxel).astype(int)))
    # Merge contiguous vertical voxels only; no unobserved interior is filled.
    columns = defaultdict(list)
    for x, y, z in occupied: columns[x, y].append(z)
    boxes = []
    for (x, y), zs in sorted(columns.items()):
        zs = np.sort(zs)
        for run in np.split(zs, np.flatnonzero(np.diff(zs) != 1) + 1):
            boxes.append([(x+.5)*voxel, (y+.5)*voxel, (run[0]+run[-1]+1)*voxel/2,
                          voxel/2, voxel/2, len(run)*voxel/2])
    np.savez_compressed(output / "obstacles.npz", boxes=np.asarray(boxes, np.float64).reshape(-1, 6))
    for name in ("centerline.csv", "terrain.npz", "native_correspondence.csv", "surface_validation.npz", "route_overview.png"):
        if (base / name).is_file(): shutil.copy2(base / name, output / name)
    manifest = {**source, "status": "scene_complete_validation_pending", "scene_id": "atec_coastal_100m_" + output.name,
                "parent_terrain": str(base), "appearance": "native_DC_3DGS_highest_LOD_with_environment",
                "context_radius_m": radius, "visual_faces": len(mesh.faces),
                "collision": "measured_ground_heightfield_plus_obstacle_surface_voxels",
                "obstacle_voxel_m": voxel, "obstacle_boxes": len(boxes),
                "obstacle_corridor_radius_m": corridor, "obstacle_height_max_m": 2.5 if ground_relative else 4.,
                "obstacle_height_reference": "local_ground" if ground_relative else "scene_origin",
                "sources": provenance, "physics_validated": False,
                "limitations": (["video_matched_forest_stairs_locomotion_pending"] if ground_relative else ["selected_low_road_not_forest_stairs"]) + ["unmeasured_friction", "approximate_surface_voxel_collision", "original_capture_artifacts_and_missing_surfaces_retained"]}
    if ground_relative:
        manifest['scene_id'] = 'atec_competition_forest_100m_' + output.name
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "obstacle_boxes": len(boxes), "visual_faces": len(mesh.faces)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/pc/Datasets/atec-opendataset"))
    parser.add_argument("--base", type=Path, default=ROOT / "assets/terrains/route_100m/v007")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=35.)
    parser.add_argument("--voxel", type=float, default=.12)
    parser.add_argument("--ground-relative", action="store_true", help="Collision heights follow the measured forest terrain")
    args = parser.parse_args()
    build(args.dataset.resolve(), args.base.resolve(), args.output.resolve(), args.radius, args.voxel, args.ground_relative)
