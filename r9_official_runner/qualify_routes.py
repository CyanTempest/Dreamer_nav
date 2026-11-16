"""Find envelope-feasible local corridors without changing any physical surfaces.

This is a scan-based global route prior adaptation, not a policy training result.
Only independently rechecked paths longer than 12 m are exported for diagnostics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def footprint_ok(data,xy,yaw):
    x,y=np.meshgrid(np.linspace(-.515,.515,43),np.linspace(-.29,.29,25),indexing='ij')
    foot=np.stack([x,y],-1).reshape(-1,2)
    output=[];layers=data['layers'];origin=data['origin']
    for start in range(0,len(xy),256):
        points=xy[start:start+256];angle=yaw[start:start+256]
        c,s=np.cos(angle)[:,None],np.sin(angle)[:,None]
        world=points[:,None]+np.stack([c*foot[:,0]-s*foot[:,1],s*foot[:,0]+c*foot[:,1]],-1)
        ij=np.floor((world-origin)/.1+.5).astype(int)
        inside=(ij[...,0]>=0)&(ij[...,0]<layers.shape[0])&(ij[...,1]>=0)&(ij[...,1]<layers.shape[1])
        ij[...,0]=ij[...,0].clip(0,layers.shape[0]-1);ij[...,1]=ij[...,1].clip(0,layers.shape[1]-1)
        values=layers[ij[...,0],ij[...,1]]
        output.append(np.all(inside&(values[...,2]<=.025)&(values[...,5]>=0),axis=1))
    return np.concatenate(output)


def swept_footprint_ok(data,xy,yaw):
    """Check intermediate positions AND rotations between waypoint poses."""
    positions=[];angles=[]
    for i in range(len(xy)-1):
        delta=(yaw[i+1]-yaw[i]+np.pi)%(2*np.pi)-np.pi
        count=max(1,int(np.ceil(np.linalg.norm(xy[i+1]-xy[i])/.025)),int(np.ceil(abs(delta)/np.deg2rad(5))))
        u=np.arange(count)/count
        positions.append(xy[i]+u[:,None]*(xy[i+1]-xy[i]))
        angles.append(yaw[i]+u*delta)
    positions.append(xy[-1:]);angles.append(yaw[-1:])
    return footprint_ok(data,np.concatenate(positions),np.concatenate(angles))


def path_in_corridor(data):
    route=data['route'];yaw=data['yaw']
    offsets=np.arange(-1.5,1.5001,.05)
    lateral=np.stack([-np.sin(yaw),np.cos(yaw)],-1)
    candidates=route[:,None,:2]+lateral[:,None]*offsets[None,:,None]
    angles=np.broadcast_to(yaw[:,None],candidates.shape[:2]).reshape(-1)
    flat=candidates.reshape(-1,2)
    feasible=np.ones(len(flat),bool)
    for delta in [-np.deg2rad(15),0,np.deg2rad(15)]:feasible&=footprint_ok(data,flat,angles+delta)
    feasible=feasible.reshape(candidates.shape[:2])
    n,k=feasible.shape
    length=np.zeros((n,k),int);cost=np.full((n,k),np.inf);parent=np.full((n,k),-1,int)
    for i in range(n):
        for j in np.flatnonzero(feasible[i]):
            length[i,j]=1;cost[i,j]=abs(offsets[j])
            if i:
                prior=np.arange(max(0,j-1),min(k,j+2));prior=prior[length[i-1,prior]>0]
                if len(prior):
                    best=prior[np.lexsort((cost[i-1,prior],-length[i-1,prior]))[0]]
                    length[i,j]=length[i-1,best]+1
                    cost[i,j]=cost[i-1,best]+abs(offsets[j])*.01+abs(j-best)*.05
                    parent[i,j]=best
    best=np.lexsort((cost.ravel(),-length.ravel()))[0]
    i,j=np.unravel_index(best,length.shape);run=length[i,j]
    if run<61:return None,dict(max_connected_length_m=max(0,(int(run)-1)*.2),qualified=False)
    indices=[]
    while j>=0:
        indices.append((i,j));j=parent[i,j];i-=1
    indices.reverse()
    path=np.array([candidates[i,j] for i,j in indices])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    # Validate actual path heading at 5 cm spacing, including interpolation.
    fine_s=np.arange(0,arc[-1],.05)
    fine=np.stack([np.interp(fine_s,arc,path[:,d]) for d in range(2)],-1)
    fine_yaw=np.unwrap(np.arctan2(np.gradient(fine[:,1]),np.gradient(fine[:,0])))
    good=footprint_ok(data,fine,fine_yaw)
    swept=swept_footprint_ok(data,fine,fine_yaw)
    report=dict(max_connected_length_m=float(arc[-1]),fine_footprint_pass_fraction=float(good.mean()),
                original_station_indices=[int(indices[0][0]),int(indices[-1][0])],
                max_lateral_shift_m=float(max(abs(offsets[j]) for i,j in indices)),
                swept_footprint_pass_fraction=float(swept.mean()),qualified=bool(good.all() and swept.all()))
    return (path if report['qualified'] else None),report


def main():
    p=argparse.ArgumentParser();p.add_argument('maps',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.maps/'manifest.json').read_text());accepted=[];reports=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for record in manifest['maps']:
        with np.load(a.maps/record['name']/'map.npz') as f:data={k:f[k] for k in f.files}
        path,report=path_in_corridor(data);report['map']=record['name'];reports.append(report)
        print(json.dumps(report),flush=True)
        if path is None:continue
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        s=np.arange(0,arc[-1],.2,dtype=np.float32)
        xy=np.stack([np.interp(s,arc,path[:,d]) for d in range(2)],-1)
        ij=np.floor((xy-data['origin'])/.1+.5).astype(int)
        z=data['layers'][ij[:,0],ij[:,1],0]
        route=np.c_[xy,z].astype(np.float32)
        yaw=np.unwrap(np.arctan2(np.gradient(xy[:,1]),np.gradient(xy[:,0]))).astype(np.float32)
        if not swept_footprint_ok(data,xy,yaw).all():continue
        dims=data['layers'].shape[:2]
        grid=np.stack(np.meshgrid(np.arange(dims[0]),np.arange(dims[1]),indexing='ij'),-1)
        world=data['origin']+grid*.1
        distance,index=cKDTree(xy).query(world.reshape(-1,2),workers=4)
        layers=data['layers'].copy();layers[...,3]=(distance.reshape(dims)-1).astype(np.float16);layers[...,4]=s[index].reshape(dims).astype(np.float16)
        target=a.output/record['name'];target.mkdir(exist_ok=True)
        data.update(layers=layers,route=route,s=s,yaw=yaw,width=np.full(len(s),3.,np.float32),gates=np.linspace((s[-1]-1)/8,s[-1]-1,8,dtype=np.float32))
        np.savez_compressed(target/'map.npz',**data)
        new={**record,'segment_length_m':float(s[-1]),'route_adaptation':report,'conservative_proxy_qualified':True,
             'start_at_2m_feasible':True,'footprint_feasible_on_route':1.,
             'map_sha256':hashlib.sha256((target/'map.npz').read_bytes()).hexdigest(),
             'scope':'Feasible subsegment selection under same unvalidated scan surface proxy; not full-route evaluation'}
        (target/'metadata.json').write_text(json.dumps(new,indent=2)+'\n');accepted.append(new)
    (a.output/'manifest.json').write_text(json.dumps({**manifest,'maps':accepted,'route_adaptation':'Envelope-based lateral recentering and connected subsegment selection'},indent=2)+'\n')
    (a.output/'qualification.json').write_text(json.dumps(dict(accepted=len(accepted),examined=len(reports),reports=reports),indent=2)+'\n')


if __name__=='__main__':main()
