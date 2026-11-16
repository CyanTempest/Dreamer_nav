"""Geometric scan-patch adapter; traversability is a conservative proxy, not truth.

No semantic meanings are inferred from undocumented classification IDs.
No missing cells are made free from route/PGM information.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from audit_cloud import open_ply
from inspect_packages import file_sha256

BANDS = [(.03, .2), (.2, .4), (.4, .8), (.8, 1.5), (1.5, 3.), (3., 10.)]
RES = .1


def signed_distance(mask):
    return (ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask))*RES


def resample_route(rows):
    # Route length uses XY distance, as the current planar controller does.
    xyz = rows[:, 1:4]
    keep = np.r_[True, np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) > .03]
    xyz = xyz[keep]
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
    target = np.arange(0., s[-1], .2)
    return target, np.stack([np.interp(target, s, xyz[:, k]) for k in range(3)], -1)


def build_patch(cloud, route, station, directory, source_sha):
    begin = time.time()
    # 40 m diagnostic segment, plus map context beyond both endpoints.
    start = int(round(station/.2))
    segment = route[start:start+201].copy()
    if len(segment) != 201: raise ValueError('Incomplete segment')
    lo = np.floor((segment[:, :2].min(0)-8)/RES)*RES
    hi = np.ceil((segment[:, :2].max(0)+8)/RES)*RES
    dims = (np.ceil((hi-lo)/RES).astype(int)+1+31)//32*32
    chunks = []
    for i in range(0, len(cloud), 2000000):
        block = cloud[i:i+2000000]
        inside=(block['x']>=lo[0])&(block['x']<hi[0])&(block['y']>=lo[1])&(block['y']<hi[1])
        block=block[inside]
        if len(block): chunks.append(np.stack([block[k] for k in ('x','y','z','normal_z')], -1))
    points = np.concatenate(chunks)
    shape = tuple(dims)
    grid = np.stack(np.meshgrid(np.arange(dims[0]),np.arange(dims[1]),indexing='ij'),-1)
    world = lo+grid*RES
    nearest_distance, nearest = cKDTree(segment[:, :2]).query(world.reshape(-1,2),workers=4)
    nearest = nearest.reshape(shape); nearest_distance=nearest_distance.reshape(shape)
    scanner_z = segment[nearest, 2]
    ij=np.floor((points[:,:2]-lo)/RES+.5).astype(int)
    flat=np.ravel_multi_index(ij.T,shape)
    expected=scanner_z[ij[:,0],ij[:,1]]
    # Broad scanner-to-ground interval is a declared bootstrap assumption.
    # Surface normals reject vertical sides; robust local check rejects pits.
    candidate=(points[:,2]>=expected-2.8)&(points[:,2]<=expected-.6)&(np.abs(points[:,3])>.65)
    low=np.full(np.prod(dims),np.inf,np.float32)
    np.minimum.at(low,flat[candidate],points[candidate,2])
    low=low.reshape(shape)
    observed=np.isfinite(low)
    if not observed.any(): raise ValueError('No ground candidates')
    distance, indices=ndi.distance_transform_edt(~observed,return_indices=True)
    filled=low[tuple(indices)]
    median=ndi.median_filter(filled,size=5)
    observed &= np.abs(filled-median)<.25
    distance, indices=ndi.distance_transform_edt(~observed,return_indices=True)
    ground=low[tuple(indices)]
    # Interpolation is limited to 20 cm; gaps beyond that remain unknown.
    known=distance*RES<=.2
    relative=points[:,2]-ground[ij[:,0],ij[:,1]]
    atlas=np.zeros((*shape,20),np.float32);atlas[...,0]=ground;atlas[...,19]=known
    for b,(lower,upper) in enumerate(BANDS):
        mask=(relative>=lower)&(relative<upper)
        bottom=np.full(np.prod(dims),np.inf,np.float32)
        top=np.full(np.prod(dims),-1,np.float32)
        count=np.zeros(np.prod(dims),np.int32)
        np.minimum.at(bottom,flat[mask],relative[mask]);np.maximum.at(top,flat[mask],relative[mask])
        np.add.at(count,flat[mask],1)
        atlas[...,1+3*b]=np.where(count>0,bottom,0).reshape(shape)
        atlas[...,2+3*b]=top.reshape(shape)
        # Coverage means observed presence, not measured optical opacity.
        atlas[...,3+3*b]=(count>0).reshape(shape)
    body_hits=np.zeros(np.prod(dims),np.int32)
    body=(relative>.22)&(relative<.8)
    np.add.at(body_hits,flat[body],1)
    obstacle=(body_hits.reshape(shape)>=2)&known
    gx=(np.roll(ground,-2,axis=0)-np.roll(ground,2,axis=0))/(4*RES)
    gy=(np.roll(ground,-2,axis=1)-np.roll(ground,2,axis=1))/(4*RES)
    support=known&(np.hypot(gx,gy)<.8)
    support[[0,1,-2,-1],:]=False;support[:,[0,1,-2,-1]]=False
    support_sdf=signed_distance(support)
    obstacle_sdf=signed_distance(obstacle) if obstacle.any() else np.full(shape,100.)
    layer=np.zeros((*shape,7),np.float32)
    layer[...,0]=ground
    layer[...,1]=ground+np.maximum(0,atlas[...,2:9:3].max(-1))
    layer[...,2]=support_sdf
    layer[...,3]=nearest_distance-1. # Public route corridor prior, not free space.
    layer[...,4]=nearest*.2
    layer[...,5]=obstacle_sdf
    # Ground-relative route height is an inferred surface, never scanner Z.
    ri=np.floor((segment[:,:2]-lo)/RES+.5).astype(int)
    segment[:,2]=ground[ri[:,0],ri[:,1]]
    yaw=np.unwrap(np.arctan2(np.gradient(segment[:,1]),np.gradient(segment[:,0])))
    x,y=np.meshgrid(np.linspace(-.515,.515,43),np.linspace(-.29,.29,25),indexing='ij')
    local=np.stack([x,y],-1).reshape(-1,2)
    c,s=np.cos(yaw),np.sin(yaw)
    foot=segment[:,None,:2]+np.stack([c[:,None]*local[:,0]-s[:,None]*local[:,1],s[:,None]*local[:,0]+c[:,None]*local[:,1]],-1)
    fi=np.floor((foot-lo)/RES+.5).astype(int)
    feasible=np.all((support_sdf[fi[...,0],fi[...,1]]<=.025)&(obstacle_sdf[fi[...,0],fi[...,1]]>=0),axis=1)
    ground_known=known[ri[:,0],ri[:,1]]
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'map.npz',layers=layer.astype(np.float16),atlas=atlas.astype(np.float16),origin=lo.astype(np.float32),res=np.float32(RES),route=segment.astype(np.float32),s=np.arange(201,dtype=np.float32)*.2,yaw=yaw.astype(np.float32),width=np.full(201,3.,np.float32),gates=np.linspace(5,39,8,dtype=np.float32))
    record=dict(name=directory.name,source_sha256=source_sha,station_m=station,segment_length_m=40,
                points=len(points),ground_known_on_route=float(ground_known.mean()),
                footprint_feasible_on_route=float(feasible.mean()),start_at_2m_feasible=bool(feasible[10]),
                conservative_proxy_qualified=bool(feasible[10] and feasible.mean()>=.95),
                dimensions=dims.tolist(),seconds=time.time()-begin,
                map_sha256=hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest(),
                assumptions=dict(frame='Identity CSV-to-PLY; spatial/time evidence, no surveyed registration',
                    scanner_ground_interval_m=[.6,2.8],ground_normal_abs_z_min=.65,
                    interpolate_radius_m=.2,body_collision_height_m=[.22,.8],
                    classification='Unused: semantic IDs undocumented',
                    route='40 m excerpt of scanner path; NOT official race-route definition',
                    corridor_half_width_m=1.,wrong_route_width_m=3.))
    (directory/'metadata.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('cloud',type=Path);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stations',type=float,nargs='+',default=[20,145,270,395,520,645,770,895])
    p.add_argument('--source-sha256',required=True)
    a=p.parse_args()
    actual=file_sha256(a.cloud)
    if actual!=a.source_sha256:raise ValueError('Source checksum mismatch')
    cloud=open_ply(a.cloud)
    _,route=resample_route(np.load(a.trajectory)['developer'])
    records=[build_patch(cloud,route,s,a.output/f'segment_{s:04.0f}',actual) for s in a.stations]
    upstream=None
    sidecar=a.cloud.with_suffix('.json')
    if sidecar.exists():
        upstream=json.loads(sidecar.read_text())
        if upstream.get('crop_sha256')!=actual:raise ValueError('Crop provenance checksum mismatch')
    (a.output/'manifest.json').write_text(json.dumps(dict(source=str(a.cloud),source_sha256=actual,
        upstream_source=upstream,scope='Untrained official scan diagnostic segments; geometric proxy only',maps=records),indent=2)+'\n')


if __name__=='__main__':main()
