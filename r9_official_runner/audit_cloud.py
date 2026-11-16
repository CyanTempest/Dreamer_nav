"""Memory-mapped official PLY audit; reports frame evidence, not assumed alignment."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def open_ply(path):
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1',
             'int': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2'}
    properties = []
    count = None
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply': raise ValueError('Not PLY')
        if stream.readline().strip() != b'format binary_little_endian 1.0':
            raise ValueError('Only binary little endian PLY supported')
        for _ in range(256):
            line = stream.readline().decode('ascii').strip().split()
            if not line: raise ValueError('Truncated header')
            if line[0] == 'end_header': break
            if line[0] == 'element':
                if line[1] != 'vertex': raise ValueError('Unexpected nonvertex element')
                count = int(line[2])
            if line[0] == 'property':
                if len(line) != 3 or line[1] not in types: raise ValueError('Unsupported PLY property')
                properties.append((line[2], types[line[1]]))
        else: raise ValueError('Oversized PLY header')
        offset = stream.tell()
    dtype = np.dtype(properties)
    if count is None or offset + count * dtype.itemsize != Path(path).stat().st_size:
        raise ValueError('PLY record count/size mismatch')
    return np.memmap(path, dtype=dtype, offset=offset, mode='r', shape=(count,))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cloud', type=Path)
    p.add_argument('--trajectory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cloud = open_ply(args.cloud)
    sample = cloud[::max(1, len(cloud)//1000000)]
    xyz = np.stack([sample[k] for k in ('x', 'y', 'z')], -1)
    if not np.isfinite(xyz).all(): raise ValueError('Nonfinite points')
    trajectory = np.load(args.trajectory)['developer']
    distance, index = cKDTree(xyz[:, :2]).query(trajectory[:, 1:3], workers=4)
    classes, counts = np.unique(cloud['classification'], return_counts=True)
    output = dict(source=args.cloud.name, point_count=len(cloud), sampled_points=len(xyz),
                  sampled_xyz_min=xyz.min(0).tolist(), sampled_xyz_max=xyz.max(0).tolist(),
                  sampled_xyz_quantiles=np.quantile(xyz, [.01, .5, .99], axis=0).tolist(),
                  classification_counts=dict(zip(map(str, classes), map(int, counts))),
                  sampled_time_range=[float(sample['time'].min()), float(sample['time'].max())],
                  trajectory_nearest_xy_m=np.quantile(distance, [.5, .9, .99, 1]).tolist(),
                  nearest_point_minus_scanner_z_m=np.quantile(xyz[index, 2]-trajectory[:, 3], [.05,.5,.95]).tolist(),
                  alignment_verified=False,
                  note='Nearest XY distance and overlap alone cannot prove registration.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    view = xyz[::max(1, len(xyz)//150000)]
    ax[0].scatter(view[:, 0], view[:, 1], c=view[:, 2], s=.2, cmap='terrain', vmin=-3, vmax=45)
    ax[0].plot(trajectory[:, 1], trajectory[:, 2], 'r-', lw=.7, label='CSV trajectory, no transform')
    ax[0].legend(); ax[0].set_aspect('equal'); ax[0].set(title='Point cloud / trajectory frame audit', xlabel='X (m)', ylabel='Y (m)')
    elapsed=(trajectory[:, 0]-trajectory[0, 0])/60
    ax[1].plot(elapsed, distance)
    ax[1].set(title='Nearest sampled point in XY (not registration proof)', xlabel='Time (min)', ylabel='Distance (m)')
    fig.savefig(args.output.with_suffix('.png'), dpi=160)
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
