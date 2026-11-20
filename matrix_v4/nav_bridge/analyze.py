"""Compare observations at five physical positions and summarize the measured run."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def observations(out):
    paths=sorted(out.glob('leg-*/steps-*.npz'))
    required=['pose','station','vel','epoch','sim','action','state','assist','lidar_age','lateral','clearance','bev']
    chunks={k:[] for k in required}
    by_leg=[]
    for p in paths:
        with np.load(p) as z:
            for k in required: chunks[k].append(z[k])
            by_leg.extend([p.parent.name]*len(z['epoch']))
    return {k:np.concatenate(v) for k,v in chunks.items()}, np.array(by_leg)

def image(bev):
    height=bev[...,0]/255.
    obstacle=bev[...,1]/255.
    rgb=np.stack([.16+.55*height+.65*obstacle,.17+.47*height+.31*bev[...,2]/255,.19+.5*height],-1)
    rgb[bev[...,3]>127]=[.1,.13,.18]
    return np.clip(rgb[::-1,::-1],0,1)

def main(out,baseline,assembly):
    data,leg=observations(out)
    ref=np.load(baseline)
    refpos=ref['pos']; refbev=ref['bev']; refs=ref['s']; refyaw=ref['yaw']
    shift=np.asarray(json.loads(assembly.read_text())['shared_shift_m'])
    world=data['pose'][:,:2]+shift[:2]
    s=data['station']
    q=np.linspace(max(2.,min(s)),max(s),5)
    fig,axes=plt.subplots(5,6,figsize=(12,10),layout='constrained')
    pairs=[]
    for row,target in enumerate(q):
        i=int(np.argmin(abs(s-target)))
        j=int(np.argmin(np.linalg.norm(refpos-world[i],axis=1)))
        mx=data['bev'][i]
        sim=refbev[j]
        common=(mx[...,3]==0)&(sim[...,3]==0)
        x,y,z,w=data['pose'][i,3:]
        heading=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        diff=heading-refyaw[j]
        pairs.append(dict(heading_gap_deg=float(np.rad2deg(np.arctan2(np.sin(diff),np.cos(diff)))),
            jointly_known=float(common.mean()),
            height_mae_m=float(np.abs(mx[...,0].astype(float)-sim[...,0])[common].mean()*4/255) if common.any() else None,
            obstacle_mae_m=float(np.abs(mx[...,1].astype(float)-sim[...,1])[common].mean()*2/255) if common.any() else None,
            matrix_s=float(s[i]),nav4_s=float(refs[j]),xy_gap_m=float(np.linalg.norm(refpos[j]-world[i])),
            matrix_unknown=float(mx[...,3].mean()/255),nav4_unknown=float(sim[...,3].mean()/255),
            matrix_support=float(mx[...,2].mean()/255),nav4_support=float(sim[...,2].mean()/255),nav4_step=j))
        for k,(b,t) in enumerate([(mx,'MATRiX'),(sim,'nav4')]):
            for col,(vis,title) in enumerate([(image(b),'terrain'),(b[::-1,::-1,2],'support'),(b[::-1,::-1,3],'unknown')]):
                ax=axes[row,3*k+col]
                ax.imshow(vis,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                if row==0: ax.set_title(t+' '+title,fontsize=10)
                if col==0: ax.set_ylabel(f"{(s[i] if k==0 else refs[j]):.1f} m")
    fig.suptitle('A trail   |   Onboard BEV at five matched positions',fontsize=15)
    fig.savefig(out/'bev_comparison.png',dpi=145);plt.close(fig)
    with (out/'bev_matches.csv').open('w') as f:
        d=csv.DictWriter(f,fieldnames=pairs[0]);d.writeheader();d.writerows(pairs)
    np.savez_compressed(out/'trajectory.npz',**data,leg=leg)
    events=json.loads((out/'events.json').read_text()) if (out/'events.json').exists() else []
    event_rows=[]
    for n,e in enumerate(events):
        name=f'leg-{n:04}'
        inds=np.flatnonzero(leg==name)
        if not len(inds): continue
        i=inds[-1]
        x,y,z,w=data['pose'][i,3:]
        roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        detail=e['reason']
        if detail=='overturned_or_low':
            detail='+'.join(k for k,b in [('low_body',data['clearance'][i]<.26),('tilt',max(abs(pitch),abs(roll))>1.1)] if b)
        nxt=np.flatnonzero(leg==f'leg-{n+1:04}')
        event_rows.append(dict(number=n+1,reason=e['reason'],detail=detail,failure_m=e['failure_s'],placement_m=e['placement_s'],
            actual_advance_m=e['placement_s']-e['failure_s'],roll_deg=float(np.rad2deg(roll)),pitch_deg=float(np.rad2deg(pitch)),
            clearance_m=data['clearance'][i],lateral_m=data['lateral'][i],restart_wall_s=float(data['epoch'][nxt[0]]-data['epoch'][i]) if len(nxt) else None))
    with (out/'assists.csv').open('w') as f:
        if event_rows:
            writer=csv.DictWriter(f,fieldnames=event_rows[0]);writer.writeheader();writer.writerows(event_rows)
    speed=np.linalg.norm(data['vel'][:,:2],axis=1)
    dist=simtime=0.
    gaps=[]; last=None
    for name in sorted(set(leg)):
        mask=leg==name
        if last is not None: gaps.append(float(data['epoch'][mask][0]-last))
        last=data['epoch'][mask][-1]
        dist+=np.linalg.norm(np.diff(world[mask],axis=0),axis=1).sum()
        simtime+=np.ptp(data['sim'][mask])
    dt=np.diff(data['sim'])[leg[1:]==leg[:-1]]
    summary=dict(steps=len(s),max_station=float(max(s)),end_station=float(s[-1]),
        decision_hz=len(dt)/float(dt.sum()) if len(dt) else None,wall_seconds=float(np.ptp(data['epoch'])),
        driving_sim_seconds=float(simtime),distance_excluding_carries=float(dist),
        reset_wall_seconds=gaps,total_reset_wall_seconds=sum(gaps),
        median_reset_wall_seconds=float(np.median(gaps)) if gaps else 0.,
        mean_horizontal_speed=float(speed.mean()),net_station_speed=float((s[-1]-s[0])/max(simtime,.001)),
        assists=len(events),assist_rate_per_100m=len(events)*100/max(1,max(s)-.75),pairs=pairs,
        median_lidar_age=float(np.nanmedian(data['lidar_age'])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='pairs'},indent=2))

if __name__=='__main__': main(*map(Path,sys.argv[1:]))
