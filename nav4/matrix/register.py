"""Cross-validate local rigid alignment using spatially held-out stable surfaces."""
import argparse

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from nav4.mapping.common import bilinear, write_json
from nav4.matrix.detail import local_v8, xyz
from nav4.matrix.survey import OUT


MAX_WORKING_RETURNS = 6_000_000


def sample_records(source):
    """Bound the working set while retaining an even sample across acquisition runs."""
    stride = max(1, int(np.ceil(len(source) / MAX_WORKING_RETURNS)))
    return source[::stride], stride


def voxel_ids(p, step):
    grid = np.floor(p / step).astype(np.int32)
    return np.unique(grid, axis=0, return_index=True)[1]


def summary(d):
    return dict(count=len(d), median=float(np.median(d)),
                rms=float(np.sqrt(np.mean(d**2))), p95=float(np.percentile(np.abs(d), 95)))


def alignment(points, reference, normals, origin):
    # Split by disjoint 0.5 m XY blocks. Use fixed query points and normals
    # for fitting and evaluation, and record metric coverage before trimming.
    labels = np.floor((reference[:, :2]-origin[:2])/.5).astype(int)
    test = (labels[:,0] + 2*labels[:,1]) % 5 == 0
    training = np.flatnonzero(~test)
    neighbors = cKDTree(points)
    rotation = np.eye(3)
    shift = np.zeros(3)
    for _ in range(25):
        target = reference[training]-origin
        query = (target-shift) @ rotation + origin
        d, idx = neighbors.query(query, workers=4)
        x = (points[idx]-origin) @ rotation.T + shift
        n = normals[training]
        residual = np.sum((x-target)*n, axis=1)
        keep = (d < .15) & (np.abs(residual) < .06)
        jac = np.c_[np.cross(x[keep], n[keep]), n[keep]]
        r = residual[keep]
        w = np.sqrt(np.minimum(1, .015/np.maximum(abs(r), 1e-9)))
        step = np.linalg.lstsq(jac*w[:,None], -r*w, rcond=None)[0]
        update = Rotation.from_rotvec(step[:3]).as_matrix()
        rotation = update @ rotation
        shift = update @ shift + step[3:]
        if np.linalg.norm(step) < 1e-7:
            break
    p, n = reference[test], normals[test]
    output = {}
    for name, rot, t in (("identity", np.eye(3), np.zeros(3)), ("rigid", rotation, shift)):
        d, idx = neighbors.query((p-origin-t) @ rot + origin, workers=4)
        surface = (points[idx]-origin) @ rot.T + origin + t
        planar = np.sum((surface-p)*n, axis=1)
        near = d <= .10
        output[name] = dict(heldout_total=len(d), coverage_10cm=float(near.mean()),
                            all_euclidean_m=summary(d), overlapping_euclidean_m=summary(d[near]),
                            overlapping_signed_plane_m=summary(planar[near]))
    return dict(pivot=origin, rotation=rotation, translation_m=shift,
                rotation_vector_deg=Rotation.from_matrix(rotation).as_rotvec(degrees=True),
                holdout=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["removed", "map0"])
    args = parser.parse_args()
    route, origin, layers, _, _, res, _, _ = local_v8()
    ref = np.load(OUT / "ply_crop.npy")
    p = xyz(ref)
    n = np.column_stack([ref[k] for k in ("normal_x", "normal_y", "normal_z")])
    h = p[:,2] - bilinear(layers[...,0], origin, p[:,:2], res=res)
    # Near-ground hard surfaces and railing evidence carry different normal directions.
    keep = (abs(h) < .18) | ((h >= .18) & (h < 1.2) & np.isin(ref["classification"], [13,14]))
    keep &= np.linalg.norm(n, axis=1) > .8
    p, n = p[keep], n[keep]
    n /= np.linalg.norm(n, axis=1)[:, None]
    idx = voxel_ids(p, .055)
    p, n = p[idx], n[idx]
    rng = np.random.default_rng(347)
    idx = rng.choice(len(p), min(70000,len(p)), replace=False)
    p, n = p[idx], n[idx]
    src = np.load(OUT / f"{args.source}_crop.npy", mmap_mode="r")
    records, stride = sample_records(src)
    pts = xyz(records)
    labels, counts = np.unique(records["classification"], return_counts=True)
    report = alignment(pts, p, n, np.median(p, axis=0))
    report["working_points"] = len(pts)
    report["record_stride"] = stride
    report["source_classes"] = dict(zip(labels.tolist(), counts.tolist()))
    write_json(OUT / f"{args.source}_registration.json", report)


if __name__ == "__main__":
    main()
