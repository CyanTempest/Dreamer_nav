"""Read-only dataset inspection; writes diagnostics, never a navigation result."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def ply_vertices(path, max_samples=150_000):
    """Read the first vertex element of scalar binary little-endian PLY files."""
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4"}
    fields, counts = [], {}
    element = None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not PLY: {path}")
        for _ in range(100):
            tokens = stream.readline().decode("ascii").split()
            if not tokens:
                raise ValueError(f"Incomplete PLY header: {path}")
            if tokens[0] == "format" and tokens[1:] != ["binary_little_endian", "1.0"]:
                raise ValueError(f"Unsupported PLY format: {path}")
            if tokens[0] == "element":
                element = tokens[1]
                if not counts and element != "vertex":
                    raise ValueError("Vertex must be the first element")
                counts[element] = int(tokens[2])
            if tokens[0] == "property" and element == "vertex":
                if tokens[1] not in types:
                    raise ValueError(f"Unsupported vertex property: {tokens}")
                fields.append((tokens[2], types[tokens[1]]))
            if tokens[0] == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds inspection limit")
        offset = stream.tell()
    dtype = np.dtype(fields)
    count = counts["vertex"]
    if count <= 0 or offset + count * dtype.itemsize > path.stat().st_size:
        raise ValueError(f"Empty or truncated PLY vertices: {path}")
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    stride = max(1, int(np.ceil(count / max_samples)))
    sample = data[::stride]
    xyz = np.column_stack([sample[k] for k in ("x", "y", "z")])
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite sampled vertices: {path}")
    report = {"vertices": count, "faces": counts.get("face", 0),
              "sample_stride": stride, "sample_count": len(xyz),
              "sample_bounds_min": xyz.min(axis=0).tolist(),
              "sample_bounds_max": xyz.max(axis=0).tolist(),
              "properties": dtype.names, "topology_validated": False}
    if "classification" in dtype.names:
        report["sample_classification_values"] = np.unique(sample["classification"]).tolist()
        report["classification_meanings_verified"] = False
    return report, xyz


def trajectory(path):
    poses = np.loadtxt(path)
    if poses.ndim != 2 or poses.shape[1] != 8 or not np.isfinite(poses).all():
        raise ValueError(f"Expected finite timestamp x y z qw qx qy qz: {path}")
    dt = np.diff(poses[:, 0])
    xyz = poses[:, 1:4]
    report = {"rows": len(poses), "duration_s": float(poses[-1, 0] - poses[0, 0]),
              "duplicate_adjacent_timestamps": int((dt == 0).sum()),
              "backwards_timestamps": int((dt < 0).sum()),
              "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
              "raw_polyline_length_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
              "bounds_min": xyz.min(axis=0).tolist(), "bounds_max": xyz.max(axis=0).tolist(),
              "quaternion_order": "wxyz", "is_traversable_ground_centerline": False}
    return report, poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not (root / "README.md").is_file():
        parser.error("dataset root must contain README.md")
    output = args.output.resolve()
    if output == root or root in output.parents:
        parser.error("output must be outside the read-only dataset")
    if output.exists():
        parser.error("output already exists; choose a new run directory")
    output.mkdir(parents=True)
    report = {"status": "inspection_in_progress", "dataset_root": str(root),
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "coordinate_alignment_verified": False, "selected_route": None,
              "physics_scene_created": False, "navigation_executed": False}
    report_path = output / "inspection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["packages"] = {}
    for folder in ("01_LiDAR_point_cloud", "02_3DGS_map", "03_Trajectory"):
        files = [p for p in (root / folder).rglob("*") if p.is_file()]
        report["packages"][folder] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    size_checks = []
    for folder in ("01_LiDAR_point_cloud", "03_Trajectory"):
        manifest = json.loads((root / folder / "_package_info/MANIFEST.json").read_text())
        for item in manifest["files"]:
            path = root / item["path"]
            size_checks.append({"path": item["path"], "declared_bytes": item["size_bytes"],
                                "actual_bytes": path.stat().st_size if path.is_file() else None,
                                "size_matches": path.is_file() and path.stat().st_size == item["size_bytes"],
                                "declared_sha256": item["sha256"], "sha256_verified": False})
    report["manifest_size_checks"] = size_checks
    if not all(item["size_matches"] for item in size_checks):
        raise ValueError("Missing file or manifest size mismatch")
    report["pointclouds"] = {}
    for name in ("location_map.ply", "plannar_map.ply"):
        info, points = ply_vertices(root / "01_LiDAR_point_cloud/ds_map" / name)
        report["pointclouds"][name] = info
        if name == "location_map.ply":
            location_points = points
    report["trajectories"] = {}
    for name in ("poses.csv", "high_frequency_poses.csv"):
        info, poses = trajectory(root / "03_Trajectory/developer_data" / name)
        report["trajectories"][name] = info
        if name == "poses.csv":
            low_frequency = poses
    mesh_dir = root / "02_3DGS_map/render2/data/mesh"
    report["mesh_chunks"] = {}
    mesh_samples = []
    for path in sorted(mesh_dir.glob("*.ply")):
        info, points = ply_vertices(path, max_samples=2000)
        report["mesh_chunks"][path.name] = info
        mesh_samples.append(points)
    mesh_points = np.concatenate(mesh_samples)
    report["mesh_summary"] = {
        "chunks": len(mesh_samples),
        "vertices": sum(v["vertices"] for v in report["mesh_chunks"].values()),
        "faces": sum(v["faces"] for v in report["mesh_chunks"].values()),
        "sample_bounds_min": mesh_points.min(axis=0).tolist(),
        "sample_bounds_max": mesh_points.max(axis=0).tolist(),
        "collision_ready": False}
    scene = json.loads((root / "02_3DGS_map/render2/scene.lcc2").read_text())
    report["lcc2"] = {k: scene[k] for k in ("offset", "shift", "scale", "totalSplats", "totalLevels")}
    report["lcc2"]["root_bounding_box"] = scene["root"]["boundingBox"]
    report["coordinate_note"] = "Plots use native coordinates separately. Similar bounds do not establish registration."

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axs = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for ax, points, title in (
        (axs[0, 0], location_points, "Localization point cloud (sampled)"),
        (axs[0, 1], low_frequency[:, 1:4], "Capture device trajectory (not robot route)"),
        (axs[1, 0], mesh_points, "Existing mesh vertices (sampled; collision unverified)")):
        plot = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.5, cmap="terrain", rasterized=True)
        fig.colorbar(plot, ax=ax, label="Native z (m)", shrink=0.8)
        ax.set(title=title, xlabel="Native x (m)", ylabel="Native y (m)")
        ax.set_aspect("equal")
    nav = root / "01_LiDAR_point_cloud/ds_map/nav_map/global_map.pgm"
    with Image.open(nav) as im:
        report["navigation_map"] = {"width_px": im.width, "height_px": im.height,
                                    "yaml": (nav.parent / "global_map.yaml").read_text()}
        axs[1, 1].imshow(np.asarray(im), cmap="gray", origin="upper", vmin=0, vmax=255)
    axs[1, 1].set(title="Provided 2D navigation map (pixel coordinates)", xlabel="Image column", ylabel="Image row")
    fig.suptitle("ATEC dataset inspection | native frames shown separately | no route selected", fontsize=14)
    fig.savefig(output / "dataset_overview.png", dpi=150)
    plt.close(fig)
    report["status"] = "inspection_complete_alignment_pending"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "preview": str(output / "dataset_overview.png"),
                      "mesh_summary": report["mesh_summary"], "trajectories": report["trajectories"]}, indent=2))


if __name__ == "__main__":
    main()
