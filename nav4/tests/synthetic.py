"""Tiny synthetic trail maps in the build_scan_patches.py format, for CPU unit tests.

Route along +x from x=0 to x=length at y=0 (or reversed). Truth support uses the same local-geometry rule
as the sensor (step <= 0.2 m to an 8-neighbour, minmod grade <= 0.8), so 0.17 m stair risers are walkable.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

BANDS = [(.03, .2), (.2, .4), (.4, .8), (.8, 1.5), (1.5, 3.), (3., 10.)]
RES = .1
KINDS = ('flat', 'ramp_up', 'ramp_down', 'stairs', 'ledge', 'step_up', 'wall', 'causeway', 'endwall')


def terrain(kind, x, y, length):
    """Ground height and standing-obstacle height on world grids x, y."""
    z, h = np.zeros_like(x), np.zeros_like(x)
    ramp = np.clip(x-8, 0, 6)*np.tan(np.deg2rad(20))
    if kind == 'ramp_up':
        z = ramp
    elif kind == 'ramp_down':
        z = -ramp
    elif kind == 'stairs':                                   # ten 0.17 m risers, 0.30 m treads
        z = np.minimum(np.floor(np.clip(x-8, 0, None)/.3+1e-6), 10)*.17*(x >= 8)
    elif kind == 'ledge':                                    # 0.5 m drop at x = 10
        z = np.where(x >= 10, -.5, 0.)
    elif kind == 'step_up':                                  # 0.5 m rise at x = 10
        z = np.where(x >= 10, .5, 0.)
    elif kind == 'wall':                                     # blocks the centreline and the left side
        h = np.where((x >= 12) & (x < 12.4) & (y > 0.) & (y < 1.6), .6, 0.)
    elif kind == 'causeway':                                 # 1.6 m wide path, 0.6 m drop on both sides
        z = np.where(np.abs(y) <= .8, 0., -.6)
    elif kind == 'endwall':                                  # wall just beyond the final station
        h = np.where((x >= length-1.0+.2) & (x < length-1.0+.6) & (np.abs(y) < 1.), .6, 0.)
    elif kind != 'flat':
        raise ValueError(kind)
    return z, h


def local_support(z):
    p = np.pad(z, 1, mode='edge')
    n, m = z.shape
    nb = [p[1+i:1+i+n, 1+j:1+j+m]-z for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    step = np.max(np.abs(nb), 0)

    def minmod(a, b):
        return np.where(a*b > 0, np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.)
    grade = np.hypot(minmod(nb[6], -nb[1]), minmod(nb[4], -nb[3]))/RES
    return (step <= .2) & (grade <= .8)


def signed_distance(mask):
    return (ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask))*RES


def build(kind='flat', length=30., width=2., spacing=.2, reverse=False, dtype=np.float32, margin=8., route_y=0.):
    """Map dict with the build_scan_patches.py keys; `spacing` may be a float or an array of steps.

    route_y shifts only the route (route arrays, corridor and station layers), never the physics layers."""
    xs = np.arange(-margin, length+margin+1e-6, RES)
    ys = np.arange(-margin, margin+1e-6, RES)
    x, y = np.meshgrid(xs, ys, indexing='ij')
    z, h = terrain(kind, x, y, length)
    steps = np.full(int(round(length/spacing)), spacing) if np.isscalar(spacing) else np.asarray(spacing)
    rx = np.r_[0., np.cumsum(steps)]
    rx = rx[rx <= length+1e-6]
    if reverse:
        rx = rx[-1]-rx
    route = np.stack([rx, np.full_like(rx, route_y), np.interp(rx, xs, z[:, len(ys)//2])], -1)
    s = np.r_[0., np.cumsum(np.abs(np.diff(rx)))]
    yaw = np.full(len(rx), np.pi if reverse else 0.)
    obstacle = h > .22
    support = local_support(z) & ~np.isnan(z)
    support[[0, 1, -2, -1], :] = False
    support[:, [0, 1, -2, -1]] = False
    flat = np.stack([x, y], -1).reshape(-1, 2)
    seg_a, seg_b = route[:-1, :2], route[1:, :2]
    v = seg_b-seg_a
    u = np.clip(np.einsum('pkd,kd->pk', flat[:, None]-seg_a[None], v)/np.sum(v*v, -1), 0, 1)
    d = np.linalg.norm(flat[:, None]-seg_a[None]-u[..., None]*v[None], axis=-1)
    k = np.argmin(d, 1)
    rows = np.arange(len(flat))
    distance = d[rows, k].reshape(x.shape)
    station = (s[k]+u[rows, k]*np.diff(s)[k]).reshape(x.shape)
    layers = np.zeros(x.shape+(7,), np.float32)
    layers[..., 0] = z
    layers[..., 1] = z+h
    layers[..., 2] = signed_distance(support)
    layers[..., 3] = distance-width/2
    layers[..., 4] = station
    layers[..., 5] = signed_distance(obstacle) if obstacle.any() else 100.
    atlas = np.zeros(x.shape+(20,), np.float32)
    atlas[..., 0] = z
    atlas[..., 19] = 1.
    for b, (lo, hi) in enumerate(BANDS):
        hit = h > lo
        atlas[..., 1+3*b] = np.where(hit, lo, 0.)
        atlas[..., 2+3*b] = np.where(hit, np.minimum(h, hi), -1.)
        atlas[..., 3+3*b] = hit
    final = s[-1]-1
    return dict(layers=layers.astype(dtype), atlas=atlas.astype(dtype), origin=np.array([xs[0], ys[0]], np.float32),
                res=np.float32(RES), route=route.astype(np.float32), s=s.astype(np.float32), yaw=yaw.astype(np.float32),
                width=np.full(len(s), width, np.float32), gates=np.linspace(final/8, final, 8).astype(np.float32))


def write(root, maps):
    """Write {name: map dict} as root/<name>/map.npz plus root/manifest.json; returns the manifest path."""
    root = Path(root)
    for name, data in maps.items():
        (root/name).mkdir(parents=True, exist_ok=True)
        np.savez(root/name/'map.npz', **data)
    (root/'manifest.json').write_text(json.dumps(dict(maps=[dict(name=k) for k in maps])))
    return root/'manifest.json'
