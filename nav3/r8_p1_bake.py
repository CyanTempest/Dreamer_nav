"""Full-route sparse 5 cm atlas from alpha-aware geometry, no semantic inputs.

Vertical plus four oblique first-return samples retain holes and height bands.
The input map supplies only tile addresses (the permitted corridor layout).
"""
import time, argparse
import numpy as np
from scipy.ndimage import grey_opening, minimum_filter, maximum_filter, uniform_filter, distance_transform_edt
from .r8_p1_common import ROOT,GEO,dump,Monitor,isolate_warp
from .r8_p0_ray import Scene

EDGES=np.array([.04,.22,.8,2.,4.,8.,16.],np.float32)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--opaque',action='store_true');args=parser.parse_args()
    suffix='opaque' if args.opaque else 'alpha'
    isolate_warp();mon=Monitor('bake_'+suffix);t0=time.perf_counter();scene=Scene()
    src=np.load(ROOT/'maps_lwy/test_862428/map.npz');inds=src['tile_indices'];origin=src['origin'];dims=src['dimensions'];res=float(src['res'])
    lookup=np.zeros(tuple(dims),np.int32);lookup[inds[:,0],inds[:,1]]=np.arange(1,len(inds)+1)
    # 6 bands, minimum + maximum observed height in each. NaN = no sampled surface.
    raw=np.full((len(inds)+1,32,32,13),np.nan,np.float32)
    ii,jj=np.meshgrid(np.arange(32),np.arange(32),indexing='ij');cells=np.stack([ii,jj],-1)
    bottom=min(r['bounds_min'][2] for r in scene.records)-1
    ceiling=max(r['bounds_max'][2] for r in scene.records)+1
    ray_s=0.;caps=0;nrays=0;stages=[]
    def cast(o,d,maximum):
        nonlocal ray_s,caps,nrays
        d=np.broadcast_to(d,o.shape).astype(np.float32).copy()
        r=np.c_[np.zeros(len(o)),np.broadcast_to(maximum,len(o))].astype(np.float32)
        out=scene.cast(o.astype(np.float32),d,r,np.zeros(4,np.float32),alpha=not args.opaque)
        ray_s+=out['seconds'];caps+=int(np.sum(out['cap']));nrays+=len(o)
        good=out['distance']>=0
        return o+d*out['distance'][:,None],good
    # No truth height / labels / surface class enters the ground estimate.
    for start in range(0,len(inds),64):
        batch=inds[start:start+64];xy=(origin+(batch[:,None,None]*32+cells)*res).reshape(-1,2)
        o=np.c_[xy,np.full(len(xy),bottom)];p,good=cast(o,[0,0,1],ceiling-bottom)
        raw[1+start:1+start+len(batch),...,0]=np.where(good,p[:,2],np.nan).reshape(-1,32,32)
        if start%640==0: print('GROUND',suffix,start,len(inds),time.perf_counter()-t0,flush=True)
    stages.append(dict(name='ground_upward',elapsed_s=time.perf_counter()-t0))
    # Sample from both ends of each height interval. Four oblique directions are
    # short: vertical displacement .10 m. Rebin actual hit coordinates, never
    # assign the entire ray to an occupied column or use a vegetation hull.
    for start in range(0,len(inds),32):
        batch=inds[start:start+32];xy=(origin+(batch[:,None,None]*32+cells)*res).reshape(-1,2)
        ground=raw[1+start:1+start+len(batch),...,0].reshape(-1)
        valid=np.isfinite(ground);xy=xy[valid];ground=ground[valid]
        for band,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:])):
            for sign in [1,-1]:
                o=np.c_[xy,ground+(lo if sign==1 else hi)]
                p,hit=cast(o,[0,0,sign],hi-lo)
                splat(raw,lookup,origin,res,p[hit],band)
            # Mid-band oblique sampling detects near-vertical foliage cards.
            # Equal 5 cm lateral and vertical offsets, not full band solid fill.
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                o=np.c_[xy,ground+(lo+hi)/2+.05]
                direction=np.array([dx,dy,-1],np.float32)/np.sqrt(2.)
                p,hit=cast(o,direction,.1*np.sqrt(2.))
                splat(raw,lookup,origin,res,p[hit],band)
        if start%320==0: print('BANDS',suffix,start,len(inds),time.perf_counter()-t0,flush=True)
    stages.append(dict(name='band_samples',elapsed_s=time.perf_counter()-t0))
    # Filter only local tiles with a 16-cell halo, no giant dense corridor array.
    # Runtime payload: sensor lower envelope, min/max surface per band, coverage.
    tiles=np.zeros((len(inds)+1,32,32,20),np.float16);tiles[0,...,0]=0
    for t,ij in enumerate(inds,1):
        grid=ij*32+np.stack(np.meshgrid(np.arange(-16,48),np.arange(-16,48),indexing='ij'),-1)
        ti=grid//32;valid=(ti[...,0]>=0)&(ti[...,1]>=0)&(ti[...,0]<dims[0])&(ti[...,1]<dims[1])
        ids=lookup[np.clip(ti[...,0],0,dims[0]-1),np.clip(ti[...,1],0,dims[1]-1)];ids=np.where(valid,ids,0)
        a=raw[ids,grid[...,0]%32,grid[...,1]%32];g=a[...,0];has=np.isfinite(g)
        if not has.any(): continue
        nearest=distance_transform_edt(~has,return_distances=False,return_indices=True)
        g=g[tuple(nearest)];g=grey_opening(g,size=13,mode='nearest')
        crop=(slice(16,48),slice(16,48));tiles[t,...,0]=g[crop]
        # 3x3 footprint approximates a 12.5 cm output cell under arbitrary yaw.
        # Store relative heights to avoid fp16 precision loss at z~20 m.
        for b in range(6):
            low=a[...,1+2*b];high=a[...,2+2*b];hit=np.isfinite(high)
            mn=minimum_filter(np.where(hit,low,np.inf),size=3)
            mx=maximum_filter(np.where(hit,high,-np.inf),size=3)
            coverage=uniform_filter(hit.astype(np.float32),size=3)
            tiles[t,...,1+3*b]=np.where(np.isfinite(mn[crop]),mn[crop]-g[crop],-1)
            tiles[t,...,2+3*b]=np.where(np.isfinite(mx[crop]),mx[crop]-g[crop],-1)
            tiles[t,...,3+3*b]=coverage[crop]
        tiles[t,...,19]=has[crop]
    path=GEO/f'r8_p1_atlas_{suffix}.npz'
    np.savez_compressed(path,tiles=tiles,lookup=lookup,origin=origin,res=np.float32(res),edges=EDGES,tile_indices=inds)
    # Raw records permit auditing reduction loss without rerunning geometry.
    rawpath=GEO/f'r8_p1_raw_{suffix}.npz';np.savez_compressed(rawpath,surfaces=raw.astype(np.float32),tile_indices=inds,origin=origin,res=res)
    elapsed=time.perf_counter()-t0
    dump(ROOT/f'metrics/r8_p1_bake_{suffix}.json',dict(status='complete',alpha=not args.opaque,seconds=elapsed,bvh_seconds=scene.build_seconds,ray_seconds=ray_s,rays=nrays,alpha_cap=caps,tiles=len(inds),cells=len(inds)*1024,area_m2=len(inds)*1024*res**2,resolution_m=res,atlas_bytes=path.stat().st_size,raw_bytes=rawpath.stat().st_size,atlas_uncompressed_bytes=tiles.nbytes+lookup.nbytes,stages=stages,band_edges_m=EDGES,semantics_used=False,map_height_used=False,global_bottom=bottom,global_ceiling=ceiling,layer_layout=['ground_absolute']+[f'{b}_{name}' for b in range(6) for name in ['surface_min_relative','surface_max_relative','coverage_3x3']]+['ground_sample_valid'],scope='All 5535 existing test_862428 tiles; full geometry retained in ray scene.'))
    print('DONE',suffix,elapsed,'atlas bytes',path.stat().st_size,flush=True);mon.close()

def splat(raw,lookup,origin,res,p,band):
    if not len(p): return
    ij=np.floor((p[:,:2]-origin)/res+.5).astype(np.int32);ti=ij//32
    valid=(ti[:,0]>=0)&(ti[:,1]>=0)&(ti[:,0]<lookup.shape[0])&(ti[:,1]<lookup.shape[1])
    ij=ij[valid];ti=ti[valid];p=p[valid]
    ids=lookup[ti[:,0],ti[:,1]];valid=ids>0;ids=ids[valid];ij=ij[valid];p=p[valid]
    # Enforce target-column band membership after oblique rebinning.
    g=raw[ids,ij[:,0]%32,ij[:,1]%32,0];height=p[:,2]-g
    valid=np.isfinite(g)&(height>=EDGES[band]-.003)&(height<=EDGES[band+1]+.003)
    idx=(ids[valid],ij[valid,0]%32,ij[valid,1]%32)
    # fmin/fmax treat NaN as absent and accumulate actual surfaces only.
    np.fmin.at(raw[...,1+2*band],idx,p[valid,2]);np.fmax.at(raw[...,2+2*band],idx,p[valid,2])

if __name__=='__main__': main()
