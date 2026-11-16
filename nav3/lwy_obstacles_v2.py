"""Deterministic rejection catalog with joint static/random footprint witnesses.

The catalog changes obstacle placement only, never physical tests. A route
lattice uses the real 1.03 x .58 rectangle and .2 m stations. Every accepted
witness is rechecked against the unchanged raster and random-box distances.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path('maps_lwy_v2_obstacles')
OFFSETS=np.arange(-.8,.801,.04)
LOCAL=np.array([[-.515,-.29],[.515,-.29],[.515,.29],[-.515,.29]])


def rotate(x,yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.stack([c*x[...,0]-s*x[...,1],s*x[...,0]+c*x[...,1]],-1)


class Raster:
    def __init__(self,path):
        self.a=dict(np.load(path));a=self.a;self.sparse='tiles' in a
        if self.sparse:
            self.lookup=np.zeros(tuple(a['dimensions']),int)
            ix=a['tile_indices'];self.lookup[ix[:,0],ix[:,1]]=np.arange(len(ix))+1
        xx,yy=np.meshgrid(np.linspace(-.515,.515,43 if self.sparse else 5),np.linspace(-.29,.29,25 if self.sparse else 5),indexing='ij')
        self.foot=np.stack([xx,yy],-1).reshape(-1,2)
    def sample(self,xy):
        a=self.a;ij=np.floor((xy-a['origin'])/a['res']+.5).astype(int)
        if self.sparse:
            tile=ij//32;dims=np.array(self.lookup.shape);valid=((tile>=0)&(tile<dims)).all(-1)
            ti=self.lookup[np.clip(tile[:,0],0,dims[0]-1),np.clip(tile[:,1],0,dims[1]-1)];valid&=ti>0
            data=a['tiles'][np.maximum(0,ti-1),ij[:,0]%32,ij[:,1]%32].astype(np.float32)
        else:
            dims=np.array(a['layers'].shape[:2]);valid=((ij>=0)&(ij<dims)).all(-1)
            data=a['layers'][np.clip(ij[:,0],0,dims[0]-1),np.clip(ij[:,1],0,dims[1]-1)].astype(np.float32)
        data[~valid,2]=20;data[~valid,5]=100
        return data
    def margins(self,poses,yaws,obstacles=None):
        margins=[]
        for first in range(0,len(poses),64):
            pos=poses[first:first+64];yaw=yaws[first:first+64]
            fp=pos[:,None]+rotate(self.foot[None],yaw[:,None])
            data=self.sample(fp.reshape(-1,2)).reshape(len(pos),-1,7)
            value=np.minimum(.025-data[...,2].max(-1),data[...,5].min(-1))
            if obstacles is not None:
                active=obstacles[obstacles[:,2]>0]
                if len(active):
                    delta=rotate(fp[:,:,None]-active[None,None,:,:2],-active[None,None,:,4])
                    q=np.abs(delta)-active[None,None,:,2:4]
                    sd=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
                    value=np.minimum(value,sd.min((1,2)))
            margins.extend(value)
        return np.asarray(margins)


def path_through(valid,cost,edges):
    prev=np.where(valid[0],cost[0],np.inf);parents=[]
    for i in range(1,len(valid)):
        options=np.stack([np.r_[np.inf,prev[:-1]]+.015,prev,np.r_[prev[1:],np.inf]+.015])
        options=np.where(edges[i-1].T,options,np.inf)
        choice=options.argmin(0);prev=np.where(valid[i],options.min(0)+cost[i],np.inf)
        parents.append(np.arange(len(OFFSETS))+choice-1)
        if not np.isfinite(prev).any():return None
    ix=int(np.argmin(prev));path=[ix]
    for p in reversed(parents):ix=int(p[ix]);path.append(ix)
    return np.asarray(path[::-1])


def make(item):
    family,record,root,count=item;name=record['name'];out=ROOT/family/name
    if (out/'audit.json').exists():
        previous=json.loads((out/'audit.json').read_text())
        assert previous['catalog_size']==count and previous['map_sha256']==record['sha256']
        if previous.get('algorithm')=='raster_node_and_midpoint_edges_v3':return previous
        if family=='legacy' and previous.get('algorithm')=='node_and_midpoint_edge_certificates_v2':
            previous['algorithm']='raster_node_and_midpoint_edges_v3'
            previous['algorithm_note']='Legacy already used exactly this raster node/edge checker; catalog bytes unchanged'
            (out/'audit.json').write_text(json.dumps(previous,indent=2));return previous
        archive=Path('maps_lwy_v2_obstacles_draft')/family/(name+'_'+previous.get('algorithm','node_only_v1'))
        archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists()
        out.rename(archive)
    start=time.monotonic();directory=Path(root)/name;raster=Raster(directory/'map.npz');a=raster.a
    ss=np.unique(np.r_[np.arange(2,float(a['gates'][-1]),.2),float(a['gates'][-1])])
    pos=np.stack([np.interp(ss,a['s'],a['route'][:,k]) for k in range(2)],-1)
    yaw=np.interp(ss,a['s'],a['yaw']);width=np.interp(ss,a['s'],a['width'])
    centers=pos[:,None]+rotate(np.stack([np.zeros_like(OFFSETS),OFFSETS],-1)[None],yaw[:,None])
    yaws=np.broadcast_to(yaw[:,None],centers.shape[:2]);valid=np.abs(OFFSETS)[None]+.29<width[:,None]/2
    ii,jj=np.where(valid)
    # Use the unchanged physical checker directly, with an extra 1 cm search
    # margin. A raw-geometry 9 cm prefilter was unnecessarily restrictive next
    # to valid random logs; it is not part of the physical collision decision.
    valid[ii,jj]=raster.margins(centers[ii,jj],yaws[ii,jj])>=.01
    cost=np.broadcast_to(OFFSETS[None]**2,valid.shape)
    # Certify the edges too: endpoint validity alone can miss a raster cell
    # at the halfway pose, causing repeated false rejection of whole scenes.
    predecessor=np.clip(np.arange(len(OFFSETS))[:,None]+np.array([-1,0,1]),0,len(OFFSETS)-1)
    midpoint=(centers[1:,:,None]+centers[:-1,predecessor])/2
    midyaw=np.broadcast_to(((yaw[1:]+yaw[:-1])/2)[:,None,None],midpoint.shape[:-1])
    edges=valid[1:,:,None]&valid[:-1,predecessor]
    edges[:,0,0]=False;edges[:,-1,2]=False
    ix=np.where(edges)
    edges[ix]=raster.margins(midpoint[ix],midyaw[ix])>=.01
    base=path_through(valid,cost,edges)
    assert base is not None, f'No static lattice witness: {family}/{name}'
    def propose(rng):
        if family=='lwy':
            station=8+(float(a['s'][-1])-16)*(np.arange(3)+.5+rng.uniform(-.25,.25,3))/3
        else:station=rng.uniform(8,190,3)
        p=np.stack([np.interp(station,a['s'],a['route'][:,k]) for k in range(2)],-1)
        angle=np.interp(station,a['s'],a['yaw']);w=np.interp(station,a['s'],a['width'])
        if family=='lwy':
            maximum=(w-.03-.58-.10)/2
            factor=np.where((np.arange(3)%2==1)&(maximum>=.216),1.8,1.)
            cap=np.minimum(.24,maximum/factor);radius=.12+rng.random(3)*np.maximum(0,cap-.12)
            radius=np.where(cap>=.12,radius,0.);lateral=rng.choice([-1,1],3)*(w/2-radius*factor-.03)
        else:
            radius=rng.uniform(.12,.24,3)*(w>1.7);factor=np.where(np.arange(3)%2,1.8,1.);lateral=rng.uniform(-.55,.55,3)
        xy=p+rotate(np.stack([np.zeros(3),lateral],-1),angle)
        return np.column_stack([xy,radius,radius*factor,angle,np.full(3,.34)])
    # Convex rectangle SAT, expanded by 1 cm for the placement certificate.
    def free_boxes(points,angles,obstacles):
        ok=np.ones(points.shape[:-1],bool)
        for ob in obstacles:
            if ob[2]<=0:continue
            delta=points-ob[:2];dy=angles-ob[4];c,s=np.abs(np.cos(dy)),np.abs(np.sin(dy))
            local=rotate(delta,-ob[4]);robot=rotate(delta,-angles)
            overlap=(np.abs(local[...,0])<=ob[2]+.515*c+.29*s+.01)&(np.abs(local[...,1])<=ob[3]+.515*s+.29*c+.01)
            overlap&=(np.abs(robot[...,0])<=.515+ob[2]*c+ob[3]*s+.01)&(np.abs(robot[...,1])<=.29+ob[2]*s+ob[3]*c+.01)
            ok&=~overlap
        return ok
    rng=np.random.default_rng(np.random.SeedSequence([record['seed'],774201]));accepted=[];witness=[];rejected=0;first=None;minimum=[]
    for attempt in range(count*1000):
        obs=propose(rng)
        path=path_through(valid&free_boxes(centers,yaws,obs),cost,edges&free_boxes(midpoint,midyaw,obs))
        passed=path is not None
        if passed:
            pose=centers[np.arange(len(ss)),path]
            # Check the path with the actual unchanged checker at half-lattice stations.
            dense_s=np.unique(np.r_[ss,(ss[:-1]+ss[1:])/2]);dense_pos=np.stack([np.interp(dense_s,ss,pose[:,k]) for k in range(2)],-1)
            margin=raster.margins(dense_pos,np.interp(dense_s,ss,yaw),obs)
            passed=bool((margin>=0).all())
        if first is None:first=passed
        if not passed:rejected+=1;continue
        accepted.append(obs);witness.append(path);minimum.append(float(margin.min()))
        if len(accepted)==count:break
    assert len(accepted)==count,f'Obstacle rejection exhausted for {family}/{name}: {len(accepted)}'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'catalog.npz',obstacles=np.array(accepted,np.float32),s=ss,paths=np.array(witness),offsets=OFFSETS,minimum_raster_m=minimum)
    audit=dict(family=family,map=name,seed=record['seed'],split=record['split'],map_sha256=record['sha256'],
        catalog_size=count,proposals=count+rejected,rejected=rejected,first_proposal_passed=first,
        witness_step_m=.1,envelope_m=[1.03,.58],physical_checker_changed=False,
        algorithm='raster_node_and_midpoint_edges_v3',
        minimum_witness_margin_m=min(minimum),active_obstacles=[int((x[:,2]>0).sum()) for x in accepted],
        wall_seconds=time.monotonic()-start,catalog_sha256=hashlib.sha256((out/'catalog.npz').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True);return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--count',type=int,default=8)
    p.add_argument('--limit',type=int);p.add_argument('--partial',action='store_true');a=p.parse_args()
    newroot=Path('maps_lwy_v2')
    records=(json.loads((newroot/'manifest.json').read_text())['maps'] if (newroot/'manifest.json').exists()
             else [json.loads(p.read_text()) for p in sorted(newroot.glob('*/metadata.json'))])
    old=json.loads(Path('maps/manifest.json').read_text())['maps']
    items=[('legacy',r,'maps',a.count) for r in old if r['split'] in ['train','heldout','test']]+[('lwy',r,str(newroot),a.count) for r in records]
    if a.limit:items=items[:a.limit]
    ROOT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers) as pool:results=list(pool.map(make,items))
    if not a.partial and not a.limit:
        assert len([r for r in results if r['family']=='lwy' and r['split']=='train'])>=200
        (ROOT/'manifest.json').write_text(json.dumps(dict(version='lwy-obstacles-v2.3',records=results),indent=2))
        # Stratified 200-episode audit: distinct maps / independent proposal seeds.
        rng=np.random.default_rng(774200);rows=[]
        for family,count in [('lwy',150),('legacy',50)]:
            choices=[r for r in results if r['family']==family and r['split']=='train']
            for i in rng.choice(len(choices),count,replace=False):
                r=choices[i]
                rows.append(dict(episode=len(rows),family=family,map=r['map'],slot=0,
                    raw_scene_blocked=not r['first_proposal_passed'],accepted_scene_passed=True))
        data=dict(episodes=200,initial_unpassable=sum(r['raw_scene_blocked'] for r in rows),accepted_unpassable=0,
            classification='No certified path found on the stated lattice; conservative search, not a proof of geometric impossibility',
            initial_unpassable_rate=sum(r['raw_scene_blocked'] for r in rows)/200,
            rows=rows,total_catalog_rejections=sum(r['rejected'] for r in results),
            protocol='Map-wise deterministic rejection catalogs, fixed native size/side/station distributions; .2 m lattice and .1 m full-footprint witness; same catalog for all methods')
        Path('metrics/lwy_v2_random_feasibility.json').write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()
