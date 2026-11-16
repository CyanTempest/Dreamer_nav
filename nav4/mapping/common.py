"""Shared paths, readers and grid helpers for the segment-A trail map.

Everything here is read-only with respect to the official dataset.  Derived
artefacts are written under ``data/trail_a`` (git-ignored, gated dataset).
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'trail_a'
CONFIG_PATH = Path(__file__).resolve().parent / 'trail_map_config.json'
PLANNING_PLY = Path('/home/hpf/wang-sm/ATEC_NAV_WSM/research/official_atec/e5a3b987/'
                    'extracted/01_LiDAR_point_cloud/ds_map/plannar_map.ply')
LOCATION_PLY = PLANNING_PLY.with_name('location_map.ply')
TRAJ_ZIP = Path('/home/hpf/atec_opendataset/03_Trajectory.zip')
LIDAR_ZIP = Path('/home/hpf/atec_opendataset/01_LiDAR_point_cloud.zip')
PLANNING_SHA256 = 'ae8e5cb2cc065529eaa3567e274a9109fab6d38caad3de253ab0b522285a3305'

# Identical to r9_official_runner/build_scan_patches.py (sensor atlas contract).
BANDS = [(.03, .2), (.2, .4), (.4, .8), (.8, 1.5), (1.5, 3.), (3., 10.)]
RES = .1
# Robot footprint samples, identical to qualify_routes.footprint_ok.
FOOT_X = np.linspace(-.515, .515, 43)
FOOT_Y = np.linspace(-.29, .29, 25)


def load_config(path=CONFIG_PATH):
    """Return the config with the human-readable justifications stripped."""
    raw = json.loads(Path(path).read_text())

    def strip(node):
        if isinstance(node, dict):
            if set(node) >= {'value', 'why'}:
                return node['value']
            return {k: strip(v) for k, v in node.items() if not k.startswith('_')}
        return node
    return strip(raw), raw


def load_poses():
    """10 Hz developer poses: timestamp x y z qw qx qy qz (whitespace separated)."""
    with zipfile.ZipFile(TRAJ_ZIP) as archive:
        rows = np.loadtxt(io.BytesIO(archive.read('03_Trajectory/developer_data/poses.csv')))
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.all(np.diff(rows[:, 0]) > 0):
        raise ValueError('Unexpected poses.csv layout')
    return rows


def load_navmap():
    """Official 2D nav map (trinary PGM, read straight from the zip)."""
    with zipfile.ZipFile(LIDAR_ZIP) as archive:
        raw = archive.read('01_LiDAR_point_cloud/ds_map/nav_map/global_map.pgm')
        meta = archive.read('01_LiDAR_point_cloud/ds_map/nav_map/global_map.yaml').decode()
    head = raw.split(b'\n', 3)
    if head[0] != b'P5':
        raise ValueError('Expected binary PGM')
    width, height = map(int, head[1].split())
    image = np.frombuffer(head[3], np.uint8)[:width * height].reshape(height, width)
    values = {}
    for line in meta.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            values[key.strip()] = value.strip()
    origin = [float(v) for v in values['origin'].strip('[]').split(',')[:2]]
    return dict(image=image, res=float(values['resolution']), origin=np.array(origin),
                free=image == 254, occupied=image == 0, unknown=image == 128, yaml=values)


def navmap_lookup(nav, mask, xy):
    """Sample a PGM mask at world XY (row 0 of the PGM is the max-y edge)."""
    height, width = mask.shape
    col = np.floor((xy[:, 0] - nav['origin'][0]) / nav['res']).astype(int)
    row = height - 1 - np.floor((xy[:, 1] - nav['origin'][1]) / nav['res']).astype(int)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    out = np.zeros(len(xy), bool)
    out[inside] = mask[row[inside], col[inside]]
    return out


def open_ply(path):
    """Memory-map a binary little-endian PLY (same reader as r9 audit_cloud.open_ply)."""
    types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'uint8': 'u1',
             'int': '<i4', 'uint': '<u4', 'short': '<i2', 'ushort': '<u2'}
    properties, count = [], None
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Not PLY')
        if stream.readline().strip() != b'format binary_little_endian 1.0':
            raise ValueError('Only binary little endian PLY supported')
        for _ in range(256):
            line = stream.readline().decode('ascii').strip().split()
            if line[0] == 'end_header':
                break
            if line[0] == 'element':
                count = int(line[2])
            if line[0] == 'property':
                properties.append((line[2], types[line[1]]))
        offset = stream.tell()
    dtype = np.dtype(properties)
    if count is None or offset + count * dtype.itemsize != Path(path).stat().st_size:
        raise ValueError('PLY record count/size mismatch')
    return np.memmap(path, dtype=dtype, offset=offset, mode='r', shape=(count,))


def signed_distance(mask, res=RES):
    """Same convention as build_scan_patches.signed_distance: <0 inside, >0 outside."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.full(mask.shape, 100., np.float32)
    if mask.all():
        return np.full(mask.shape, -100., np.float32)
    return ((ndi.distance_transform_edt(~mask) - ndi.distance_transform_edt(mask)) * res).astype(np.float32)


def obstacle_distance(mask, points_xy, origin, cfg, res=RES, chunk=128):
    """XY point clearance at grid centres; preserve the existing collision-cell mask.

    Outside: nearest body-band collision return minus configured range noise.
    Inside: negative centre-to-point distance (epsilon at an exact return). This
    is a sign convention, not a reconstruction of solid object interiors.
    `grid` selects the unchanged v6 EDT, including its empty/full conventions.
    """
    mode = cfg['obstacles']['distance_method']
    if mode == 'grid':
        return signed_distance(mask, res)
    if mode != 'points':
        raise ValueError('obstacles.distance_method must be points or grid')
    noise = cfg['obstacles']['distance_noise_m']
    if noise < 0:
        raise ValueError('obstacles.distance_noise_m must be nonnegative')
    if not np.any(mask):
        return np.full(mask.shape, 100., np.float32)
    if not len(points_xy):
        raise ValueError('colliding cells require body-band collision points')
    tree = cKDTree(points_xy)
    out = np.empty(mask.shape, np.float32)
    for a in range(0, mask.shape[0], chunk):
        ii, jj = np.meshgrid(np.arange(a, min(a + chunk, mask.shape[0])),
                             np.arange(mask.shape[1]), indexing='ij')
        xy = np.stack([ii.ravel(), jj.ravel()], -1) * res + origin
        d = tree.query(xy, workers=4)[0].reshape(ii.shape)
        out[a:a + chunk] = np.where(mask[a:a + chunk],
                                    -np.maximum(d, np.finfo(np.float32).eps), d - noise)
    return out


def bilinear(grid2d, origin, xy, fill=100., res=RES):
    """Bilinear scalar sampling at cell centres, matching TrailBank.sample.

    Each out-of-grid corner uses fill, rather than clipping to an edge cell.
    """
    p = (np.asarray(xy) - origin) / res
    ij = np.floor(p).astype(np.int64)
    u = p - ij
    vals = []
    for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        i, j = ij[..., 0] + di, ij[..., 1] + dj
        inside = (i >= 0) & (j >= 0) & (i < grid2d.shape[0]) & (j < grid2d.shape[1])
        vals.append(np.where(inside, grid2d[i.clip(0, grid2d.shape[0]-1),
                                          j.clip(0, grid2d.shape[1]-1)], fill))
    a, b, c, d = vals
    return (a*(1-u[..., 0])+b*u[..., 0])*(1-u[..., 1]) + (c*(1-u[..., 0])+d*u[..., 0])*u[..., 1]


def cell_index(xy, origin, res=RES):
    """Nearest-cell index, identical rounding to build_scan_patches / qualify_routes."""
    return np.floor((np.asarray(xy) - origin) / res + .5).astype(np.int64)


def footprint_world(xy, yaw):
    """World XY of the 43x25 footprint samples for each pose: (N, 1075, 2)."""
    x, y = np.meshgrid(FOOT_X, FOOT_Y, indexing='ij')
    local = np.stack([x, y], -1).reshape(-1, 2)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.asarray(xy)[:, None] + np.stack([c * local[:, 0] - s * local[:, 1],
                                               s * local[:, 0] + c * local[:, 1]], -1)


def footprint_ok(support_sdf, obstacle_sdf, origin, xy, yaw, chunk=512, obstacle_linear=True):
    """All samples inside, nearest support<=.025, bilinear obstacle>=0.

    obstacle_linear=False is retained for explicit historical v6 comparisons.
    """
    out = []
    shape = support_sdf.shape
    for start in range(0, len(xy), chunk):
        world = footprint_world(xy[start:start + chunk], yaw[start:start + chunk])
        ij = np.floor((world - origin) / RES + .5).astype(np.int64)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < shape[1])
        i = ij[..., 0].clip(0, shape[0] - 1)
        j = ij[..., 1].clip(0, shape[1] - 1)
        obs = bilinear(obstacle_sdf, origin, world) if obstacle_linear else obstacle_sdf[i, j]
        out.append(np.all(inside & (support_sdf[i, j] <= .025) & (obs >= 0), axis=1))
    return np.concatenate(out) if out else np.zeros(0, bool)


def swept_poses(xy, yaw):
    """Intermediate positions AND rotations, same sampling as qualify_routes.swept_footprint_ok."""
    positions, angles = [], []
    for i in range(len(xy) - 1):
        delta = (yaw[i + 1] - yaw[i] + np.pi) % (2 * np.pi) - np.pi
        count = max(1, int(np.ceil(np.linalg.norm(xy[i + 1] - xy[i]) / .025)),
                    int(np.ceil(abs(delta) / np.deg2rad(5))))
        u = np.arange(count) / count
        positions.append(xy[i] + u[:, None] * (xy[i + 1] - xy[i]))
        angles.append(yaw[i] + u * delta)
    positions.append(xy[-1:])
    angles.append(yaw[-1:])
    return np.concatenate(positions), np.concatenate(angles)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable) + '\n')


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))
