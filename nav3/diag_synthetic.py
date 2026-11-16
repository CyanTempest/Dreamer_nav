"""Generate independent random-map diagnostic holdouts; never train on these.

Pairs share a fresh random scene. The intervention adds two independently sampled
right-shoulder rocks after its generated log, not geometry copied from mini-trail.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .maps import TrainingScene,rasterize
from .diag_geometry import write_json


def main():
    records={kind:[] for kind in ['base','rocks']}
    for seed in range(30000,30004):
        sc=TrainingScene(seed,False,.1);base=copy.deepcopy(sc.props)
        for kind in ['base','rocks']:
            sc.props=copy.deepcopy(base);added=[]
            if kind=='rocks':
                rng=np.random.default_rng(seed+88000)
                for ds in [rng.uniform(1.1,1.7),rng.uniform(2.5,3.1)]:
                    s=sc.features['log']+ds;yaw=np.interp(s,sc.s,sc.yaw);normal=np.array([-np.sin(yaw),np.cos(yaw)])
                    xy=sc.xy_at(s)-normal*rng.uniform(1.20,1.40);h=rng.uniform(.18,.24)
                    p=dict(kind='rock',shape='ellipsoid',pos=[*xy,float(sc.elevation(*xy))+h*.6],
                        size=[rng.uniform(.20,.29),rng.uniform(.19,.28),h],yaw=float(rng.uniform(-3,3)),rgba=[.38,.37,.32,1])
                    sc.props.append(p);added.append(p)
            root=Path('maps/diag_synthetic_'+kind);name=f'random_{seed}';d=root/name;d.mkdir(parents=True,exist_ok=True)
            selected=np.unique(np.r_[np.arange(0,len(sc.s),4),len(sc.s)-1])
            np.savez_compressed(d/'map.npz',layers=rasterize(sc).astype(np.float16),origin=sc.origin.astype(np.float32),res=np.float32(.1),
                route=sc.route[selected].astype(np.float32),s=sc.s[selected].astype(np.float32),width=sc.width[selected].astype(np.float32),
                yaw=sc.yaw[selected].astype(np.float32),gates=sc.s[sc.gate_idx].astype(np.float32))
            meta=dict(name=name,seed=seed,split='heldout',mini=False,features=sc.features,added_props=added,
                sha256=hashlib.sha256((d/'map.npz').read_bytes()).hexdigest(),diagnostic_only=True)
            write_json(d/'metadata.json',meta);write_json(d/'props.json',sc.props);records[kind].append(meta)
            print(seed,kind,flush=True)
    for kind,rows in records.items():write_json(Path('maps/diag_synthetic_'+kind)/'manifest.json',dict(maps=rows))
    write_json('metrics/diag_synthetic_generation.json',dict(seeds=list(range(30000,30004)),mini_source_used=False,training=False,records=records))

if __name__=='__main__':main()
