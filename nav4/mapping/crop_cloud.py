"""Crop the planning cloud to a corridor around segment A (memory-mapped, chunked, parallel).

Output: data/trail_a/cache/crop/{x,y,z,nx,ny,nz,intensity,cls,t}.npy plus crop.json.
Corridor = union of discs of ``crop_radius_m`` around every centreline station and
every raw pose walked inside segment A (rasterised at 0.5 m, conservative).

Usage: python -m nav4.mapping.crop_cloud [--workers 12]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage as ndi

from nav4.mapping.common import DATA, PLANNING_PLY, load_config, load_poses, open_ply, write_json

FIELDS = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'intensity', 'cls', 't')
DTYPES = dict(x='f4', y='f4', z='f4', nx='f2', ny='f2', nz='f2', intensity='f2', cls='u1', t='f4')
_STATE = {}


def _init(raster, lo, res, t_ref, path, outdir):
    _STATE.update(raster=raster, lo=lo, res=res, t_ref=t_ref, cloud=open_ply(path), outdir=outdir)


def _work(args):
    part, start, stop = args
    st = _STATE
    block = np.asarray(st['cloud'][start:stop])
    i = np.floor((block['x'] - st['lo'][0]) / st['res']).astype(np.int64)
    j = np.floor((block['y'] - st['lo'][1]) / st['res']).astype(np.int64)
    r = st['raster']
    ok = (i >= 0) & (i < r.shape[0]) & (j >= 0) & (j < r.shape[1])
    ok[ok] = r[i[ok], j[ok]]
    b = block[ok]
    cols = dict(x=b['x'], y=b['y'], z=b['z'], nx=b['normal_x'], ny=b['normal_y'], nz=b['normal_z'],
                intensity=b['intensity'], cls=b['classification'], t=(b['time'] - st['t_ref']))
    out = {}
    for k in FIELDS:
        out[k] = np.ascontiguousarray(cols[k]).astype(DTYPES[k])
        np.save(os.path.join(st['outdir'], f'part{part:05d}_{k}.npy'), out[k])
    return part, int(ok.sum()), stop - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--chunk', type=int, default=2_000_000)
    a = p.parse_args()
    begin = time.time()
    cfg, _ = load_config()
    radius = cfg['grid']['crop_radius_m']
    seg = np.load(DATA / 'segment' / 'centreline.npz')
    rows = load_poses()
    pts = np.concatenate([seg['xy'], rows[seg['walked_index'], 1:3]])
    res = .5
    lo = np.floor(pts.min(0) - radius - 1)
    hi = np.ceil(pts.max(0) + radius + 1)
    shape = tuple(((hi - lo) / res).astype(int) + 1)
    seed = np.zeros(shape, bool)
    ij = np.floor((pts - lo) / res).astype(int)
    seed[ij[:, 0], ij[:, 1]] = True
    # Conservative: distance from cell centre + half diagonal.
    raster = ndi.distance_transform_edt(~seed) * res <= radius + res * 0.75
    outdir = DATA / 'cache' / 'crop'
    partdir = outdir / 'parts'
    partdir.mkdir(parents=True, exist_ok=True)
    cloud = open_ply(PLANNING_PLY)
    n = len(cloud)
    jobs = [(k, s, min(n, s + a.chunk)) for k, s in enumerate(range(0, n, a.chunk))]
    counts = {}
    with Pool(a.workers, initializer=_init, initargs=(raster, lo, res, rows[0, 0], str(PLANNING_PLY), str(partdir))) as pool:
        for part, kept, total in pool.imap_unordered(_work, jobs):
            counts[part] = kept
    order = sorted(counts)
    total = sum(counts.values())
    for k in FIELDS:
        dest = np.lib.format.open_memmap(outdir / f'{k}.npy', mode='w+', dtype=DTYPES[k], shape=(total,))
        pos = 0
        for part in order:
            f = partdir / f'part{part:05d}_{k}.npy'
            arr = np.load(f)
            dest[pos:pos + len(arr)] = arr
            pos += len(arr)
            f.unlink()
        dest.flush()
        del dest
    partdir.rmdir()
    info = dict(source=str(PLANNING_PLY), source_points=n, kept_points=total, crop_radius_m=radius,
                raster_res_m=res, raster_origin=lo.tolist(), raster_shape=list(shape),
                time_reference=float(rows[0, 0]), fields={k: DTYPES[k] for k in FIELDS},
                seconds=time.time() - begin, workers=a.workers)
    write_json(outdir / 'crop.json', info)
    print(json.dumps(info, default=str))


if __name__ == '__main__':
    main()
