"""Raster export of the accepted scene generator, without PLY or MJCF export.

Grid cells conservatively cover thin geometry; steps modify ground, not obstacles.
Axes are x, y. No training seed ever uses the mini scene.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/map'))
from scene import Scene

KINDS = {'rock': 1, 'fallen_log': 2, 'bark': 2, 'rail_post': 3,
         'rail_bar': 3, 'suspended_black_tape': 4, 'tape_stake': 4,
         'tree': 5, 'gate_post': 6, 'gate_marker': 6}
LAYERS = ['ground', 'top', 'edge', 'main_edge', 'projected_s', 'obstacle_distance', 'kind']


class TrainingScene(Scene):
    def make_props(self):
        # The accepted random generator can overlap the fixed log station with
        # a randomized 1.08 m narrow section. Such a map has no collision-free
        # passage for the 0.58 m envelope. Move the log; never widen the trail.
        original=self.features['log']
        if not self.mini:
            candidates=np.arange(10.,self.length-10,.5)
            safe=np.min(self.width_at(candidates[:,None]+np.array([-3.,0.,3.])),axis=-1)>=2.5
            safe &= np.abs(candidates-self.bend_s)>10
            if np.min(self.width_at(original+np.array([-3.,0.,3.])))<2.5:
                candidates=candidates[safe]
                self.features['log']=float(candidates[np.argmin(np.abs(candidates-original))])
        self.log_relocation=dict(original_s_m=float(original),final_s_m=float(self.features['log']))
        super().make_props()
        self.bypass_repairs=repair_bypass(self,getattr(self,'preserve_feasible',False))


def repair_bypass(sc,preserve_feasible=False):
    """Keep one envelope-wide bypass next to the log; retain every rock.

    Opposing edge rocks immediately after a transverse log can close the only
    passage. Relocate these rocks to the adjacent outer shoulder, retaining
    their geometry and height above terrain. mini_trail remains byte-identical.
    """
    repairs=[]
    if sc.mini:return repairs
    if preserve_feasible:
        from .maps_v2 import bypass_connected
        sc.bypass_validation=bypass_connected(sc)
        if sc.bypass_validation['connected']:return repairs
    for i,p in enumerate(sc.props):
        if p['kind']!='rock':continue
        xy=np.array(p['pos'][:2]);idx=np.argmin(np.sum((sc.xy-xy)**2,-1))
        s=sc.s[idx];yaw=sc.yaw[idx];normal=np.array([-np.sin(yaw),np.cos(yaw)])
        lateral=float((xy-sc.xy[idx])@normal)
        if abs(s-sc.features['log'])<3.5 and lateral<0:
            old=list(p['pos']);agl=p['pos'][2]-float(sc.elevation(*xy))
            width=float(np.interp(s,sc.s,sc.width))
            new=sc.xy[idx]-normal*(width/2+max(p['size'][:2])+.2)
            p['pos']=[float(new[0]),float(new[1]),float(sc.elevation(*new))+agl]
            repairs.append(dict(prop_index=i,old=old,new=p['pos']))
            if preserve_feasible:
                sc.bypass_validation=bypass_connected(sc)
                if sc.bypass_validation['connected']:break
    if preserve_feasible:
        assert sc.bypass_validation['connected'],'No M1 bypass after repair'
    return repairs


def rasterize(sc):
    ground = sc.ground.copy()
    top = ground.copy()
    kind = np.zeros(ground.shape, np.uint8)
    collision = np.zeros(ground.shape, bool)
    for p in sc.props:
        if p['kind'] not in KINDS and p['kind'] != 'stone_tread':
            continue
        size = p['size']
        if 'fromto' in p:
            a, b = np.array(p['fromto']).reshape(2, 3)
            lo, hi = np.minimum(a[:2], b[:2])-size[0]-.08, np.maximum(a[:2], b[:2])+size[0]+.08
        else:
            center = np.array(p['pos'])
            radius = max(size[:2]) if p['shape'] in ('box', 'ellipsoid') else size[0]
            lo, hi = center[:2]-radius-.08, center[:2]+radius+.08
        imin = np.maximum(0, np.floor((lo-sc.origin)/sc.res).astype(int))
        imax = np.minimum(ground.shape, np.ceil((hi-sc.origin)/sc.res).astype(int)+1)
        if np.any(imax <= imin):
            continue
        xx, yy = np.meshgrid(sc.origin[0]+np.arange(imin[0], imax[0])*sc.res,
                             sc.origin[1]+np.arange(imin[1], imax[1])*sc.res, indexing='ij')
        pad = sc.res * np.sqrt(2) / 2
        if 'fromto' in p:
            v = b[:2]-a[:2]
            u = np.clip(((xx-a[0])*v[0]+(yy-a[1])*v[1]) / max(v@v, 1e-9), 0, 1)
            dist = np.hypot(xx-a[0]-u*v[0], yy-a[1]-u*v[1])
            mask = dist <= size[0]+pad
            z = a[2]+u*(b[2]-a[2])+size[0]
        else:
            yaw = p.get('yaw', 0)
            dx, dy = xx-center[0], yy-center[1]
            x, y = np.cos(yaw)*dx+np.sin(yaw)*dy, -np.sin(yaw)*dx+np.cos(yaw)*dy
            shape = p['shape']
            if shape == 'box':
                mask = (abs(x) <= size[0]+pad) & (abs(y) <= size[1]+pad)
                z = np.full_like(xx, center[2]+size[2])
            elif shape == 'ellipsoid':
                r = (x/(size[0]+pad))**2+(y/(size[1]+pad))**2
                mask = r <= 1
                z = center[2]+size[2]*np.sqrt(np.maximum(0, 1-r))
            else:
                mask = np.hypot(x, y) <= size[0]+pad
                z = np.full_like(xx, center[2]+size[-1])
        sl = (slice(imin[0], imax[0]), slice(imin[1], imax[1]))
        if p['kind'] == 'stone_tread':
            ground[sl] = np.where(mask, np.maximum(ground[sl], z), ground[sl])
        else:
            higher = mask & (z > top[sl])
            kind[sl] = np.where(higher, KINDS[p['kind']], kind[sl])
            collision[sl] |= mask & (z-ground[sl] > .22)
        top[sl] = np.where(mask, np.maximum(top[sl], z), top[sl])
    top = np.maximum(top, ground)
    distance = distance_transform_edt(~collision)*sc.res - distance_transform_edt(collision)*sc.res
    return np.stack([ground, top, sc.edge, sc.main_edge, sc.projected_s, distance, kind], -1).astype(np.float32)


def generate_one(item):
    name, seed, mini = item
    directory = ROOT/'maps'/name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'map.npz').exists():
        old=json.loads((directory/'metadata.json').read_text())
        if old.get('raster_version')==3:
            return old
    start = time.monotonic()
    sc = TrainingScene(seed, mini, .1)
    layers = rasterize(sc)
    # Store continuous route every 0.2 m. Physics projects onto segments.
    selected = np.unique(np.r_[np.arange(0, len(sc.s), 4), len(sc.s)-1])
    np.savez_compressed(directory/'map.npz', layers=layers.astype(np.float16),
                        origin=sc.origin.astype(np.float32), res=np.float32(sc.res),
                        route=sc.route[selected].astype(np.float32), s=sc.s[selected].astype(np.float32),
                        width=sc.width[selected].astype(np.float32), yaw=sc.yaw[selected].astype(np.float32),
                        gates=sc.s[sc.gate_idx].astype(np.float32))
    meta = dict(name=name, seed=seed, mini=mini,raster_version=3,log_relocation=sc.log_relocation,bypass_repairs=sc.bypass_repairs,
                split='test' if mini else ('train' if seed<20000 else 'heldout'),
                features=sc.features, length=float(sc.length), shape=list(sc.ground.shape),
                props=len(sc.props), seconds=time.monotonic()-start,
                sha256=hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest())
    (directory/'metadata.json').write_text(json.dumps(meta, indent=2))
    (directory/'props.json').write_text(json.dumps(sc.props))
    print(json.dumps(meta), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=200)
    ap.add_argument('--heldout', type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    items = [(f'train_{s}', s, False) for s in range(10000,10000+args.train)]
    items += [(f'heldout_{s}', s, False) for s in range(20000,20000+args.heldout)]
    items += [('mini_trail', 871002, True)]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(generate_one, items))
    manifest = dict(generator='vendor/map/scene.py copied verbatim', layers=LAYERS, maps=records,
                    generator_sha256=hashlib.sha256((ROOT/'vendor/map/scene.py').read_bytes()).hexdigest())
    (ROOT/'maps/manifest.json').write_text(json.dumps(manifest, indent=2))
    (ROOT/'metrics/map_manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
