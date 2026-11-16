"""Independent randomized combination maps; old maps and defaults stay intact."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import label
from .maps import TrainingScene,rasterize,LAYERS

TRAIN_SEEDS=list(range(40000,40200))
BLIND_SEEDS=list(range(50000,50020))


def bypass_connected(sc,margin=0.):
    """Conservative route-aligned M1 configuration-space connectivity.

    Uses the production rasterizer (including existing inflation), nearest-cell
    queries, an 11x7 envelope sampling grid, 5 cm centers and four-connectivity.
    This is a sufficient-path screen; planner G1 remains a separate gate.
    """
    ss=sc.features['log']+np.arange(-6,6.001,.05)
    yaw=np.interp(ss,sc.s,sc.yaw);xy=sc.xy_at(ss)
    tangent=np.stack([np.cos(yaw),np.sin(yaw)],-1)
    normal=np.stack([-np.sin(yaw),np.cos(yaw)],-1)
    offsets=np.arange(-1.5,1.501,.05)
    centers=xy[:,None]+offsets[None,:,None]*normal[:,None]
    fx,fy=np.meshgrid(np.linspace(-.515,.515,11),np.linspace(-.29,.29,7),indexing='ij')
    feet=centers[:,:,None]+fx.ravel()[None,None,:,None]*tangent[:,None,None]+fy.ravel()[None,None,:,None]*normal[:,None,None]
    lo=np.maximum(0,np.floor((feet.min((0,1,2))-sc.origin)/sc.res).astype(int)-3)
    hi=np.minimum(sc.ground.shape,np.ceil((feet.max((0,1,2))-sc.origin)/sc.res).astype(int)+4)
    sl=tuple(slice(a,b) for a,b in zip(lo,hi))
    crop=SimpleNamespace(origin=sc.origin+lo*sc.res,res=sc.res,props=sc.props,
        **{k:getattr(sc,k)[sl] for k in ['ground','edge','main_edge','projected_s']})
    layers=rasterize(crop).astype(np.float16)
    ij=np.floor((feet-crop.origin)/sc.res+.5).astype(int)
    values=layers[ij[...,0],ij[...,1]]
    free=(values[...,2]<=0).all(-1)&(values[...,5]>=.099+margin).all(-1)
    labels,_=label(free)
    start=set(labels[0,abs(offsets)<.16])-{0};end=set(labels[-1,abs(offsets)<.16])-{0}
    return dict(connected=bool(start&end),min_free_centers=int(free.sum(-1).min()),
        footprint_m=[1.03,.58],samples=[11,7],center_spacing_m=.05,
        obstacle_distance_min_m=.099+margin,extra_obstacle_margin_m=margin,edge_max_m=0.,raster_inflation='unchanged')


class CombinationScene(TrainingScene):
    preserve_feasible=True

    def make_props(self):
        super().make_props()
        rng=self.rng;self.combination=dict(requested=bool(rng.random()<.8),accepted=False,attempts=0)
        if not self.combination['requested']:return
        original=copy.deepcopy(self.props);old_log=self.features['log']
        gates=self.s[self.gate_idx]
        eligible=gates[(gates>12)&(gates<self.length-12)]
        eligible=eligible[np.min(self.width_at(eligible[:,None]+np.arange(-8,8.1,1)),axis=1)>=2.5]
        eligible=eligible[np.abs(eligible-self.bend_s)>12]
        if not len(eligible):return
        for attempt in range(24):
            station=float(rng.choice(eligible)+rng.uniform(-2.5,2.5))
            side=int(rng.choice([-1,1]));length=float(rng.uniform(1.2,1.85));radius=float(rng.uniform(.10,.17))
            angle=float(rng.uniform(65,115));yaw=float(np.interp(station,self.s,self.yaw))
            tangent=np.array([np.cos(yaw),np.sin(yaw)]);normal=np.array([-np.sin(yaw),np.cos(yaw)])
            axis=np.cos(np.deg2rad(angle))*tangent+np.sin(np.deg2rad(angle))*normal
            center=self.xy_at(station)+side*normal*(length/2*np.sin(np.deg2rad(angle))+rng.uniform(.06,.22))
            ends=[center-axis*length/2,center+axis*length/2]
            ends=[[*q,float(self.elevation(*q))+radius] for q in ends]
            self.props=[copy.deepcopy(p) for p in original if p['kind'] not in ['fallen_log','bark']]
            self.props.append(dict(kind='fallen_log',shape='cylinder',pos=[0,0,0],size=[radius],rgba=[.29,.18,.08,1],fromto=ends[0]+ends[1]))
            for t in np.linspace(0,1,10):
                self.props.append(dict(kind='bark',shape='sphere',pos=((1-t)*np.array(ends[0])+t*np.array(ends[1])).tolist(),size=[radius+.005],rgba=[.25,.14,.07,1]))
            added=[]
            for k in range(int(rng.integers(1,4))):
                ds=float(rng.uniform(.7,3.4)) if k or rng.random()<.75 else float(rng.uniform(-3.4,-.7))
                s=station+ds;ryaw=float(np.interp(s,self.s,self.yaw));n=np.array([-np.sin(ryaw),np.cos(ryaw)])
                lateral=-side*float(rng.uniform(1.20,1.48));q=self.xy_at(s)+n*lateral
                h=float(rng.uniform(.18,.29))
                rock=dict(kind='rock',shape='ellipsoid',pos=[*q,float(self.elevation(*q))+.6*h],
                    size=[float(rng.uniform(.17,.31)),float(rng.uniform(.17,.29)),h],
                    yaw=float(rng.uniform(-np.pi,np.pi)),rgba=[.38,.37,.32,1])
                self.props.append(rock);added.append(dict(ds=ds,lateral=lateral,size=rock['size'],yaw=rock['yaw']))
            self.features['log']=station
            # A connected slit alone did not give the noisy planner enough
            # tracking room at G1. Require an extra 10 cm obstacle clearance
            # for newly sampled combinations, without altering the simulator.
            validation=bypass_connected(self,margin=.10)
            self.combination['attempts']=attempt+1
            if validation['connected']:
                self.bypass_validation=validation
                self.combination.update(accepted=True,side=side,log_length=length,log_radius=radius,
                    angle_deg=angle,station=station,gate_distance=float(np.min(abs(gates-station))),rocks=added)
                return
        self.props=original;self.features['log']=old_log


def generate(item):
    split,seed=item;name=f'{split}/{seed}';directory=Path('maps_v2')/name
    if (directory/'metadata.json').exists():return json.loads((directory/'metadata.json').read_text())
    directory.mkdir(parents=True,exist_ok=True)
    sc=CombinationScene(seed,False,.1);selected=np.unique(np.r_[np.arange(0,len(sc.s),4),len(sc.s)-1])
    validation=bypass_connected(sc);assert validation['connected'],seed
    np.savez_compressed(directory/'map.npz',layers=rasterize(sc).astype(np.float16),origin=sc.origin.astype(np.float32),res=np.float32(.1),
        route=sc.route[selected].astype(np.float32),s=sc.s[selected].astype(np.float32),width=sc.width[selected].astype(np.float32),
        yaw=sc.yaw[selected].astype(np.float32),gates=sc.s[sc.gate_idx].astype(np.float32))
    meta=dict(name=name,seed=seed,mini=False,split=split,features=sc.features,combination=sc.combination,
        bypass_repairs=sc.bypass_repairs,bypass_validation=validation,log_relocation=sc.log_relocation,
        combination_validation=getattr(sc,'bypass_validation',validation),
        sha256=hashlib.sha256((directory/'map.npz').read_bytes()).hexdigest())
    (directory/'props.json').write_text(json.dumps(sc.props))
    (directory/'metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(name=name,combination=sc.combination,validation=validation)),flush=True)
    return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    root=Path('maps_v2');root.mkdir(exist_ok=True)
    assert not (root/'manifest.json').exists(),'Refuse to replace a published map set'
    with ProcessPoolExecutor(a.workers) as pool:
        records=list(pool.map(generate,[('train',s) for s in TRAIN_SEEDS]+[('blind',s) for s in BLIND_SEEDS]))
    manifest=dict(maps=records,layers=LAYERS,train_seeds=TRAIN_SEEDS,blind_seeds=BLIND_SEEDS,
        combination_probability=.8,mini_source_used=False,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    Path('metrics/fix_map_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
