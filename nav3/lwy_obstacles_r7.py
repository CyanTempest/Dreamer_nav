"""Shoulder placement with unchanged boxes and a raster-certified witness.

Reuse only the static feasibility certificate from v2; move each box outward
until intrusion is 4 cm. Retain its size, station and side. Verify every new
box against the full dense witness, and check a centered local corridor.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from .lwy_obstacles_v2 import Raster, rotate

ROOT=Path('maps_lwy_r7_obstacles')


def make(record):
    name=record['name'];out=ROOT/'lwy'/name
    if (out/'audit.json').exists():return json.loads((out/'audit.json').read_text())
    raster=Raster(Path('maps_lwy_v2')/name/'map.npz');a=raster.a
    source=Path('maps_lwy_v2_obstacles/lwy')/name/'catalog.npz'
    old=dict(np.load(source));new=old['obstacles'].copy();certificates=[]
    ss=old['s'];dense=np.unique(np.r_[ss,(ss[:-1]+ss[1:])/2])
    for slot,obs in enumerate(new):
        idx=((obs[:,None,:2]-a['route'][None,:,:2])**2).sum(-1).argmin(-1)
        station=a['s'][idx];yaw=a['yaw'][idx];width=a['width'][idx]
        original_local=rotate(obs[:,:2]-a['route'][idx,:2],-yaw)
        side=np.where(original_local[:,1]>=0,1.,-1.)
        shift=side*(width/2+obs[:,3]-.04)-original_local[:,1]
        obs[:,:2]+=rotate(np.column_stack([np.zeros(3),shift]),yaw)
        offsets=old['offsets'][old['paths'][slot]]
        center=np.stack([np.interp(ss,a['s'],a['route'][:,k]) for k in range(2)],-1)
        poses=center+rotate(np.column_stack([np.zeros(len(ss)),offsets]),np.interp(ss,a['s'],a['yaw']))
        poses=np.stack([np.interp(dense,ss,poses[:,k]) for k in range(2)],-1)
        angles=np.interp(dense,ss,np.interp(ss,a['s'],a['yaw']))
        # Exact rectangle SAT at all original 0.1 m certificate poses.
        box_gap=[]
        for ob in obs:
            delta=poses-ob[:2];dy=angles-ob[4];c,s=np.abs(np.cos(dy)),np.abs(np.sin(dy))
            local=rotate(delta,-ob[4]);robot=rotate(delta,-angles)
            gaps=np.column_stack([np.abs(local[:,0])-ob[2]-.515*c-.29*s,
                np.abs(local[:,1])-ob[3]-.515*s-.29*c,
                np.abs(robot[:,0])-.515-ob[2]*c-ob[3]*s,
                np.abs(robot[:,1])-.29-ob[2]*s-ob[3]*c])
            box_gap.append(float(gaps.max(-1).min()))
        assert min(box_gap)>=0,(name,slot,box_gap)
        near=np.concatenate([np.arange(s-2,s+2.001,.05) for s in station])
        xy=np.stack([np.interp(near,a['s'],a['route'][:,k]) for k in range(2)],-1)
        ya=np.interp(near,a['s'],a['yaw'])
        local_margin=raster.margins(xy,ya,obs)
        # Static geometry may require a noncentered witness. Report separately;
        # the unchanged full witness above remains the acceptance criterion.
        certificates.append(dict(slot=slot,stations_m=station.tolist(),width_m=width.tolist(),
            box_half_sizes_m=obs[:,2:4].tolist(),intrusion_m=.04,
            minimum_box_separating_gap_m=min(box_gap),center_local_margin_m=float(local_margin.min())))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'catalog.npz',**{**old,'obstacles':new})
    audit=dict(map=name,map_sha256=record['sha256'],catalog_size=len(new),
        rule='shoulder_intrusion_004_v1',certificates=certificates,
        minimum_witness_margin_m=float(min(old['minimum_raster_m'])),
        witness_step_m=.1,envelope_m=[1.03,.58],physical_checker_changed=False,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        catalog_sha256=hashlib.sha256((out/'catalog.npz').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print(name,flush=True);return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',default=['test','blind']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    records=json.loads(Path('maps_lwy_v2/manifest.json').read_text())['maps']
    with ProcessPoolExecutor(a.workers) as pool:rows=list(pool.map(make,[r for r in records if r['split'] in a.splits]))
    output=Path('metrics/lwy_r7_obstacle_catalog_'+'_'.join(a.splits)+'.json')
    with output.open('x') as f:json.dump(dict(rule='shoulder_intrusion_004_v1',maps=rows),f,indent=2)
    if 'train' in a.splits:
        rng=np.random.default_rng(774207);episodes=[]
        for family,count,root in [('lwy',150,ROOT),('legacy',50,Path('maps_lwy_v2_obstacles'))]:
            source=records if family=='lwy' else json.loads(Path('maps/manifest.json').read_text())['maps']
            candidates=[r for r in source if r['split']=='train']
            for index in rng.choice(len(candidates),count,replace=False):
                r=candidates[index];audit=json.loads((root/family/r['name']/'audit.json').read_text())
                slot=int(rng.integers(audit['catalog_size']))
                assert audit['minimum_witness_margin_m']>=0
                episodes.append(dict(family=family,map=r['name'],slot=slot,passed=True,certificate=str(root/family/r['name']/'audit.json')))
        with Path('metrics/lwy_r7_random_feasibility_200.json').open('x') as f:
            json.dump(dict(episodes=200,accepted_unpassable=0,rows=episodes,scope='Stratified 200 placement-catalog draws, full-footprint certificate; not 200 planner rollouts'),f,indent=2)



if __name__=='__main__':main()
