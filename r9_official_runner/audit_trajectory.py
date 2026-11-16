"""Verify package manifest and inspect scan paths without declaring a race route."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from inspect_packages import safe_name


def path_statistics(rows):
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < 2:
        raise ValueError('Expected timestamp xyz qw qx qy qz rows')
    if not np.isfinite(rows).all():
        raise ValueError('Nonfinite trajectory values')
    dt = np.diff(rows[:, 0])
    if not np.all(dt > 0):
        raise ValueError('Timestamps are not strictly increasing')
    distance = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1)
    norms = np.linalg.norm(rows[:, 4:], axis=1)
    return dict(rows=len(rows), duration_seconds=float(np.ptp(rows[:, 0])),
                dt_quantiles_seconds=np.quantile(dt, [0, .5, .95, 1]).tolist(),
                xyz_min=rows[:, 1:4].min(0).tolist(), xyz_max=rows[:, 1:4].max(0).tolist(),
                raw_path_length_m=float(distance.sum()),
                endpoint_distance_m=float(np.linalg.norm(rows[-1, 1:4] - rows[0, 1:4])),
                speed_quantiles_mps=np.quantile(distance / dt, [.5, .95, .99, 1]).tolist(),
                quaternion_norm_range=[float(norms.min()), float(norms.max())])


def verify_manifest(archive):
    names = archive.namelist()
    if len(set(names)) != len(names) or not all(safe_name(n) for n in names):
        raise ValueError('Duplicate or unsafe archive member')
    manifests = [n for n in names if n.endswith('/_package_info/MANIFEST.json')]
    if len(manifests) != 1:
        raise ValueError('Expected exactly one package manifest')
    manifest = json.loads(archive.read(manifests[0]))
    verified = []
    for entry in manifest['files']:
        info = archive.getinfo(entry['path'])
        if info.file_size != entry['size_bytes']:
            raise ValueError('Manifest size mismatch')
        digest = hashlib.sha256()
        with archive.open(info) as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Manifest SHA256 mismatch')
        verified.append({'path': entry['path'], 'sha256': digest.hexdigest()})
    return verified


def rigid_fit(source, target):
    """Fit one proper rigid transform, no scale change; use held-out error too."""
    sc, tc = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - sc).T @ (target - tc))
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ fix @ u.T
    translation = tc - rotation @ sc
    return rotation, translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        verified = verify_manifest(archive)
        csv = np.loadtxt(io.BytesIO(archive.read('03_Trajectory/developer_data/poses.csv')))
        high = np.loadtxt(io.BytesIO(archive.read('03_Trajectory/developer_data/high_frequency_poses.csv')))
        rendered = json.loads(archive.read('03_Trajectory/render2/info/poses.json'))['poses']
    rt = np.array([float(p['ts']) for p in rendered])
    rp = np.array([p['T'] for p in rendered])
    # Match timestamps rather than trusting arrays to use the same row order.
    idx = np.searchsorted(csv[:, 0], rt)
    idx = np.minimum(idx, len(csv) - 1)
    prev = np.maximum(idx - 1, 0)
    idx = np.where(abs(csv[prev, 0] - rt) < abs(csv[idx, 0] - rt), prev, idx)
    valid = abs(csv[idx, 0] - rt) < 1e-4
    source, target = csv[idx[valid], 1:4], rp[valid]
    if len(source) < 20:
        raise ValueError('Insufficient matching render/CSV timestamps')
    train = np.arange(len(source)) % 2 == 0
    rotation, translation = rigid_fit(source[train], target[train])
    err = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    # 1 Hz samples suppress stationary scanner jitter for a useful path estimate.
    take = np.r_[True, np.diff(np.floor(csv[:, 0] - csv[0, 0])) > 0]
    sampled = csv[take]
    high_dt = np.diff(high[:, 0])
    if np.any(high_dt < 0):
        raise ValueError('High frequency timestamps reverse')
    # High-frequency export repeats correction timestamps. Keep the last
    # supplied record only for descriptive statistics; route uses 10 Hz CSV.
    high_keep = np.r_[high_dt > 0, True]
    high_stats = path_statistics(high[high_keep])
    high_stats.update(raw_rows=len(high), duplicate_timestamp_pairs=int((high_dt == 0).sum()),
                      duplicate_policy='Last record at each timestamp for statistics only')
    out = dict(archive=args.archive.name, manifest_verified=verified,
               developer_10hz=path_statistics(csv), developer_high_frequency=high_stats,
               approx_1hz=path_statistics(sampled),
               developer_to_render=dict(matched_timestamps=int(valid.sum()),
                   rotation=rotation.tolist(), translation=translation.tolist(),
                   fit_rmse_m=float(np.sqrt(np.mean(err[train] ** 2))),
                   heldout_rmse_m=float(np.sqrt(np.mean(err[~train] ** 2))),
                   max_residual_m=float(err.max()),
                   scope='Trajectory frames only; does not establish point cloud alignment'),
               route_approved=False, pointcloud_alignment_verified=False,
               note='Scanner path may include pauses, revisits, and non-race-route travel.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), developer=sampled, render=rp[valid][::10])
    print(json.dumps({k:v for k,v in out.items() if k != 'manifest_verified'}, indent=2))


if __name__ == '__main__':
    main()
