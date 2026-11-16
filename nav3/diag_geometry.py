"""CPU-only geometry audit, 200-map feature census and counterfactual maps."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import distance_transform_edt
from .maps import rasterize


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2))


def project(m,xy):
    xy=np.asarray(xy);r=m['route'][:,:2];j=np.argmin(np.sum((r-xy)**2,-1))
    normal=np.array([-np.sin(m['yaw'][j]),np.cos(m['yaw'][j])])
    return float(m['s'][j]),float((xy-r[j])@normal),j


def center(p):
    return np.mean(np.array(p['fromto']).reshape(2,3),0) if 'fromto' in p else np.array(p['pos'])


def load_map(name,root='maps'):return dict(np.load(Path(root)/name/'map.npz'))


def feature_row(name,root='maps'):
    m=load_map(name,root);meta=json.loads((Path(root)/name/'metadata.json').read_text());props=json.loads((Path(root)/name/'props.json').read_text())
    p=next(p for p in props if p['kind']=='fallen_log');a,b=np.array(p['fromto']).reshape(2,3)
    s,lateral,j=project(m,center(p)[:2]);yaw=m['yaw'][j]
    rocks=[]
    for q in props:
        if q['kind']!='rock':continue
        rs,rl,_=project(m,center(q)[:2])
        if abs(rs-s)<6:rocks.append(dict(ds=rs-s,lateral=rl,radius=max(q['size'][:2])))
    orientation=np.arctan2(b[1]-a[1],b[0]-a[0])-yaw
    return dict(name=name,seed=meta['seed'],s=s,width=float(m['width'][j]),width_min_3m=float(m['width'][abs(m['s']-s)<=3].min()),
        log_length=float(np.linalg.norm(b-a)),log_radius=p['size'][0],lateral=lateral,angle_deg=float(np.rad2deg(orientation)%180),
        curvature_3m=float((np.interp(s+1.5,m['s'],m['yaw'])-np.interp(s-1.5,m['s'],m['yaw']))/3),
        fork_distance=min(abs(meta['features'][k]-s) for k in ['fork_plain','fork_tape']),
        gate_distance=float(np.min(abs(m['gates']-s))),rocks=rocks,repairs=len(meta.get('bypass_repairs',[])),
        opposing_rock_3_5m=any(abs(q['ds'])<3.5 and -.95<q['lateral']<0 for q in rocks),
        opposing_rock_3_5m_any_lateral=any(abs(q['ds'])<3.5 and q['lateral']<0 and abs(q['lateral'])<1.6 for q in rocks))


def geometry():
    m=load_map('mini_trail');props=json.loads(Path('maps/mini_trail/props.json').read_text());rows=[]
    for i,p in enumerate(props):
        if p['kind'] in ['canopy','leaf']:continue
        s,lat,j=project(m,center(p)[:2])
        if 150<=s<=170:rows.append(dict(index=i,route_s=s,lateral=lat,**p))
    stations=[]
    for s in np.arange(150,170.01,.2):
        j=np.argmin(abs(m['s']-s));stations.append(dict(s=float(m['s'][j]),xyz=m['route'][j].tolist(),width=float(m['width'][j]),yaw=float(m['yaw'][j])))
    # Re-rasterizing from serialized props must preserve all local obstacle bits.
    arr=m['layers'].astype(np.float32)
    sc=SimpleNamespace(ground=arr[...,0],edge=arr[...,2],main_edge=arr[...,3],projected_s=arr[...,4],origin=m['origin'],res=.1,props=props)
    new=rasterize(sc);local=(arr[...,4]>=150)&(arr[...,4]<=170)&(arr[...,2]<4)
    occupied=arr[...,5]<0
    mismatch=int(np.sum(local&((new[...,5]<0)!=occupied)))
    bs=np.linspace(148,162,71)
    byaw=np.interp(bs,m['s'],m['yaw']);base=np.stack([np.interp(bs,m['s'],m['route'][:,k]) for k in range(2)],-1)
    from scene import smooth as scene_smooth
    smooth=scene_smooth((bs-148)/14)
    branch=base-4.6*smooth[:,None]*np.stack([-np.sin(byaw),np.cos(byaw)],-1)
    result=dict(stations=stations,props=rows,feature=feature_row('mini_trail'),
        branch_coordinates=branch.tolist(),branch_main_s=bs.tolist(),
        route_grade_percent_min=float(np.gradient(np.array([r['xyz'][2] for r in stations]),np.array([r['s'] for r in stations])).min()*100),
        route_grade_percent_max=float(np.gradient(np.array([r['xyz'][2] for r in stations]),np.array([r['s'] for r in stations])).max()*100),raster_rebuild_local_cells=int(local.sum()),
        raster_rebuild_collision_bit_mismatch=mismatch,grid_resolution_m=.1,raster_pad_m=.1*np.sqrt(2)/2,
        nearest_cell_extra_radius_m=.1*np.sqrt(2)/2,collision_height_m=.22,footprint_samples=[5,5],
        absent=['stone_tread','rail_post','rail_bar','suspended_black_tape','tape_stake'],
        fork=dict(name='fork_plain',start_s=148.,end_s=162.,side=-1,offset_end_m=4.6))
    write_json('metrics/diag_geometry.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['stations','props']}))


def census():
    manifest=json.loads(Path('maps/manifest.json').read_text());rows=[feature_row(r['name']) for r in manifest['maps'] if r['split']=='train']
    mini=feature_row('mini_trail')
    criteria=dict(log_length_within_5cm=lambda r:abs(r['log_length']-mini['log_length'])<.05,
        angle_within_5deg=lambda r:abs(r['angle_deg']-mini['angle_deg'])<5,
        width_within_10cm=lambda r:abs(r['width']-mini['width'])<.1,
        curvature_within_005=lambda r:abs(r['curvature_3m']-mini['curvature_3m'])<.005,
        opposing_rock_3_5m=lambda r:r['opposing_rock_3_5m_any_lateral'],
        gate_within_1m=lambda r:r['gate_distance']<1,
        fork_within_15m=lambda r:r['fork_distance']<15)
    counts={k:sum(bool(f(r)) for r in rows) for k,f in criteria.items()}
    write_json('metrics/diag_training_census.json',dict(n=len(rows),mini=mini,counts=counts,rows=rows));print(counts)


def variants(only=None):
    m=load_map('mini_trail');original=json.loads(Path('maps/mini_trail/props.json').read_text());a=m['layers'].astype(np.float32)
    log=next(p for p in original if p['kind']=='fallen_log');pivot=center(log);yaw=np.interp(164,m['s'],m['yaw']);normal=np.array([-np.sin(yaw),np.cos(yaw),0])
    for variant in ( [only] if only else ['remove_log','remove_opposing_rocks','shift_log_left_030','rotate_log_30','log_center_occupancy_v2'] ):
        props=copy.deepcopy(original)
        if variant=='remove_log':props=[p for p in props if p['kind'] not in ['fallen_log','bark']]
        if variant=='remove_opposing_rocks':
            props=[p for p in props if not (p['kind']=='rock' and abs(project(m,center(p)[:2])[0]-164)<3.5 and project(m,center(p)[:2])[1]<0)]
        if variant in ['shift_log_left_030','rotate_log_30','log_center_occupancy']:
            angle=np.deg2rad(30);rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            def transform(q):return (np.array(q)+normal*.3 if variant.startswith('shift') else pivot+rot@(np.array(q)-pivot)).tolist()
            for p in props:
                if p['kind'] not in ['fallen_log','bark']:continue
                if 'fromto' in p:p['fromto']=sum([transform(q) for q in np.array(p['fromto']).reshape(2,3)],[])
                else:p['pos']=transform(p['pos'])
        sc=SimpleNamespace(ground=a[...,0],edge=a[...,2],main_edge=a[...,3],projected_s=a[...,4],origin=m['origin'],res=.1,props=props)
        layers=rasterize(sc).astype(np.float16)
        # Retain the original raster outside the intervention's 5 m neighborhood.
        local=(abs(a[...,4]-164)<5)&(a[...,2]<4)
        layers=np.where(local[...,None],layers,m['layers'])
        if variant.startswith('log_center_occupancy'):
            # Sensitivity analysis only: center-sampled log occupancy without
            # pad. Keep top/BEV unchanged; this does not redefine production.
            no_log=[p for p in original if p['kind'] not in ['fallen_log','bark']]
            base=rasterize(SimpleNamespace(**{**sc.__dict__,'props':no_log}))
            occupied=base[...,5]<0
            xx,yy=np.meshgrid(m['origin'][0]+np.arange(a.shape[0])*.1,m['origin'][1]+np.arange(a.shape[1])*.1,indexing='ij')
            for p in original:
                if p['kind'] not in ['fallen_log','bark']:continue
                if 'fromto' in p:
                    aa,bb=np.array(p['fromto']).reshape(2,3);v=bb[:2]-aa[:2]
                    u=np.clip(((xx-aa[0])*v[0]+(yy-aa[1])*v[1])/(v@v),0,1)
                    dist=np.hypot(xx-aa[0]-u*v[0],yy-aa[1]-u*v[1]);z=aa[2]+u*(bb[2]-aa[2])+p['size'][0]
                else:
                    dist=np.hypot(xx-p['pos'][0],yy-p['pos'][1]);z=p['pos'][2]+p['size'][0]
                occupied|=(dist<=p['size'][0])&(z-a[...,0]>.22)
            # Preserve every non-log occupancy bit exactly, including distant
            # float16 threshold cells. Only the chosen obstacle may change.
            occupied=np.where(a[...,6]==2,occupied,m['layers'][...,5]<0)
            assert not np.any((occupied!=(m['layers'][...,5]<0))&(a[...,6]!=2))
            distance=distance_transform_edt(~occupied)*.1-distance_transform_edt(occupied)*.1
            layers=m['layers'].copy();layers[...,5]=distance.astype(np.float16)

        root=Path('maps/diag_'+variant);d=root/'mini_trail';d.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(d/'map.npz',**{**m,'layers':layers})
        digest=hashlib.sha256((d/'map.npz').read_bytes()).hexdigest()
        write_json(d/'props.json',props);write_json(root/'manifest.json',dict(maps=[dict(name='mini_trail',split='test',seed=871002,sha256=digest)]))
        write_json(root/'intervention.json',dict(variant=variant,source='mini_trail TEST ONLY',training=False,modified_cells=int(np.any(layers!=m['layers'],axis=-1).sum())))
        print(variant,flush=True)


def repro():
    import jax
    from .bank import MapBank
    from .env import FastEnv,rotate
    b=MapBank('test');e=FastEnv(b,1,False);s,_,_=e.reset(jax.random.PRNGKey(0))
    r=np.load('videos/dreamer_s0_rollout.npz')
    s.update(pos=jax.numpy.array(r['pos'][-1:]),yaw=jax.numpy.array(r['yaw'][-1:]))
    data,_=e.truth(s,s['pos'][:,None]+rotate(e.foot[None],s['yaw']))
    hits=np.asarray(data[0,:,5])<0
    print('REPLAY: collision=',bool(hits.any()),'footprint_hit_samples=',int(hits.sum()),'kind=',np.unique(np.asarray(data[0,:,6])[hits]).tolist(),'s=',float(r['s'][-1]))
    assert not hits.any(),'Reproduced Dreamer collision at 163.44 m'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['geometry','census','variants','repro']);p.add_argument('--variant');a=p.parse_args()
    if a.task=='variants':variants(a.variant)
    else:globals()[a.task]()
