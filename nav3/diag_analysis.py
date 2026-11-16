"""Reproduce collision attribution, figures and counters from diag recordings."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from .diag_geometry import project,load_map,write_json,center


def rotation(yaw):return np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])


def footprint(pos,yaw,length=1.03,width=.58):
    return np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])@rotation(yaw).T+pos


def target(a):
    a=np.clip(a,-1,1);radius=.5+(a[0]+1)*1.25;angle=a[1]*np.pi/3
    return radius*np.array([np.cos(angle),np.sin(angle)])


def sample(m,xy):
    ij=np.floor((xy-m['origin'])/.1+.5).astype(int)
    ij=np.clip(ij,0,np.array(m['layers'].shape[:2])-1)
    return m['layers'][ij[...,0],ij[...,1]].astype(float)


def raw_gap(pos,yaw,props,length=1.03,width=.58):
    """Distance to the physical log (cylinder axis plus spherical bark).

    Axis minimization has <= 0.5 mm discretization error; endpoint bark is exact.
    Positive means separated even from the full XY projection, not just z>.22.
    """
    gaps=[]
    for p in props:
        if p['kind'] not in ['fallen_log','bark']:continue
        pts=np.linspace(*np.array(p['fromto']).reshape(2,3)[:,:2],1601) if 'fromto' in p else np.array(p['pos'][:2])[None]
        local=(pts-pos)@rotation(yaw)
        q=np.abs(local)-[length/2,width/2]
        dist=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
        gaps.append(float(dist.min()-p['size'][0]))
    return min(gaps)


def scene_plot(m,props,collisions):
    a=m['layers'];gx=m['origin'][0]+np.arange(a.shape[0])*.1;gy=m['origin'][1]+np.arange(a.shape[1])*.1
    route=m['route'][(m['s']>=150)&(m['s']<=170),:2]
    lo=route.min(0)-[2,5];hi=route.max(0)+[2,4]
    ix=np.flatnonzero((gx>=lo[0])&(gx<=hi[0]));iy=np.flatnonzero((gy>=lo[1])&(gy<=hi[1]));sub=a[np.ix_(ix,iy)]
    rgb=np.ones((*sub.shape[:2],3))*.94;rgb[sub[...,2]<=0]=[.83,.9,.80];rgb[sub[...,5]<0]=[.35,.29,.22]
    fig,axes=plt.subplots(2,1,figsize=(15,10),layout='constrained')
    for ax in axes:
        ax.imshow(rgb.swapaxes(0,1),extent=[gx[ix[0]]-.05,gx[ix[-1]]+.05,gy[iy[0]]-.05,gy[iy[-1]]+.05],origin='lower')
        ax.plot(route[:,0],route[:,1],'k--',lw=1,label='Route center')
        for ss in [150,155,160,164,165,170]:
            j=np.argmin(abs(m['s']-ss));ax.text(*m['route'][j,:2],f' s={ss}',fontsize=8,clip_on=True)
        for p in props:
            if p['kind']=='fallen_log':
                aa,bb=np.array(p['fromto']).reshape(2,3)[:,:2];v=bb-aa;normal=np.array([-v[1],v[0]])/np.linalg.norm(v)*.14;ax.add_patch(Polygon([aa+normal,bb+normal,bb-normal,aa-normal],color='#b26e28',label='Raw log projection'))
        for p in props:
            if p['kind']=='bark':ax.add_patch(Circle(p['pos'][:2],p['size'][0],color='#b26e28'))
        for name,color in [('full_dreamer','#d33b3b'),('full_ppo','#3387bb'),('full_planner','#35a05c')]:
            d=json.loads(Path(f'metrics/diag_{name}.json').read_text())
            for k,r in enumerate(d['rows']):
                tr=np.array(r['trajectory']);sel=(tr[:,0]>=150)&(tr[:,0]<=171)
                ax.plot(tr[sel,1],tr[sel,2],color=color,alpha=.2,lw=.8,label=name.replace('full_','') if k==0 else None)
        for k,c in enumerate(collisions):
            xy=np.array(c['first_hit_pos']);ax.plot(*xy,'rx',ms=4,label='29 first-hit centers' if k==0 else None)
            ax.add_patch(Polygon(footprint(xy,c['first_hit_yaw']),fill=False,edgecolor='#e23b3b',alpha=.35,lw=.7))
            points=np.array(c['hit_points']);ax.plot(points[:,0],points[:,1],'.',color='#8a00b0',ms=3,label='Foot samples in occupied cells' if k==0 else None)
        ax.set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal');ax.grid(alpha=.15)
    axes[0].set_title('Mini-trail s=150–170 m: support, obstacles, routes and collision envelopes')
    axes[0].legend(loc='upper right',fontsize=8,ncol=3)
    axes[1].set(xlim=(133.8,137.8),ylim=(23.6,27.4),title='Log endpoint: all 29 first collision envelopes (1.03 × 0.58 m)')
    fig.savefig('figures/diag_scene.png',dpi=180);plt.close(fig)


def evidence(all_episodes=False):
    m=load_map('mini_trail');props=json.loads(Path('maps/mini_trail/props.json').read_text())
    data=json.loads(Path('metrics/diag_full_dreamer.json').read_text());r=dict(np.load('metrics/diag_full_dreamer.npz'))
    old=json.loads(Path('metrics/eval_dreamer_s0_test_ob0_sf0.json').read_text())['rows']
    local=np.stack(np.meshgrid(-2+(np.arange(64)+.5)*.125,-4+(np.arange(64)+.5)*.125,indexing='ij'),-1)
    collisions=[];samples=[];last_ids={}
    for row in data['rows']:
        if row['failure']!='collision':continue
        ep=row['episode'];ids=np.flatnonzero(r['episode']==ep);last_ids[ep]=ids[-26:];last=ids[-1]
        sub=np.flatnonzero(r['sub_hit'][last]);k=int(sub[0]);pos=r['sub_pos'][last,k];yaw=r['sub_yaw'][last,k]
        xx,yy=np.meshgrid(np.linspace(-.515,.515,5),np.linspace(-.29,.29,5),indexing='ij')
        fp=np.stack([xx,yy],-1).reshape(-1,2)@rotation(yaw).T+pos;fv=sample(m,fp);hit=fv[:,5]<0
        collisions.append(dict(episode=ep,s=row['progress_m'],first_hit_substep=k,first_hit_pos=pos.tolist(),first_hit_yaw=float(yaw),
            hit_samples=int(hit.sum()),kinds=np.unique(fv[hit,6]).tolist(),hit_points=fp[hit].tolist(),
            envelope_raw_gap_first_m=raw_gap(pos,yaw,props),body_raw_gap_first_m=raw_gap(pos,yaw,props,.93,.48),
            envelope_raw_gap_terminal_m=raw_gap(np.array(row['pos']),row['yaw'],props)))
        for i in ids[-26:]:
            world=r['pos'][i]+local@rotation(r['yaw'][i]).T;d=sample(m,world)
            log=(d[...,6]==2)&(d[...,5]<0);bev=r['bev'][i];seen=(bev[...,1]>28)&(bev[...,3]<128)
            truth=r['priv_trav'][i,...,0]>127;pred=r['pred'][i,...,0]>.5
            # Obstacle-induced exclusion region near the endpoint, inside support.
            aa=np.array(props[618]['fromto'][:2]);hazard=(d[...,5]<.6)&(d[...,2]<-.4)&(np.linalg.norm(world-aa,axis=-1)<2)&(~truth)
            metrics=dict(episode=ep,tick=int(r['tick'][i]),seconds_before_collision=float((r['tick'][last]-r['tick'][i]+1)*.2),
                s=float(r['s'][i]),log_cells=int(log.sum()),log_seen=int((seen&log).sum()),log_pred_free=int((pred&log).sum()),log_seen_dilated=int((maximum_filter(seen,size=3)&log).sum()),
                hazard_cells=int(hazard.sum()),hazard_false_free=int((pred&hazard).sum()),
                iou_inter=int((truth&pred).sum()),iou_union=int((truth|pred).sum()))
            for field,label in [('action','dreamer'),('planner_action','planner'),('ppo_action','ppo_shadow')]:
                xy=target(r[field][i]);ij=np.clip(np.floor((xy+[2,4])/.125).astype(int),0,63);ix=tuple(ij)
                metrics[label+'_target_truth_free']=bool(truth[ix]);metrics[label+'_target_pred_free']=bool(pred[ix])
                metrics[label+'_speed']=float((np.clip(r[field][i,2],-1,1)+1)*.4)
            samples.append(metrics)
    totals={k:sum(x[k] for x in samples) for k in ['log_cells','log_seen','log_pred_free','log_seen_dilated','hazard_cells','hazard_false_free','iou_inter','iou_union']}
    totals.update(log_pred_false_free_rate=totals['log_pred_free']/totals['log_cells'],log_recall=totals['log_seen']/totals['log_cells'],log_recall_with_one_pixel_tolerance=totals['log_seen_dilated']/totals['log_cells'],
        hazard_false_free_rate=totals['hazard_false_free']/totals['hazard_cells'],overall_iou=totals['iou_inter']/totals['iou_union'],frames=len(samples))
    for label in ['dreamer','planner','ppo_shadow']:
        totals[label+'_unsafe_target_rate']=np.mean([not x[label+'_target_truth_free'] for x in samples]).item()
        totals[label+'_known_unsafe_target_rate']=np.mean([not x[label+'_target_truth_free'] and not x[label+'_target_pred_free'] for x in samples]).item()
        totals[label+'_speed_mean']=np.mean([x[label+'_speed'] for x in samples]).item()
    result=dict(window_s=5.2,frames=samples,aggregate=totals,collisions=collisions,
        old_collision_progress_max_difference_m=max(abs(a['progress_m']-b['progress_m']) for a,b in zip(old,data['rows']) if a['failure']=='collision'),
        physical_projection_overlap_first=sum(x['envelope_raw_gap_first_m']<=0 for x in collisions),
        physical_projection_overlap_terminal=sum(x['envelope_raw_gap_terminal_m']<=0 for x in collisions))
    write_json('metrics/diag_attribution.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['frames','collisions']},indent=2))
    scene_plot(m,props,collisions)
    for ep in (list(last_ids) if all_episodes else [0]):
        # Exact policy inputs, each panel has physical local coordinates.
        ids=last_ids[ep];chosen=ids[[0,10,16,21,25]]
        fig,axes=plt.subplots(5,8,figsize=(22,14),layout='constrained')
        labels=['Relative ground','Obstacle height','Observed support','Unknown','Route corridor','Age / 8 s','priv_trav','Decoded trav']
        for row,i in enumerate(chosen):
            b=r['bev'][i].astype(float)/255
            arrays=[b[...,0]*4-2,b[...,1]*2,b[...,2],b[...,3],b[...,4],b[...,5],r['priv_trav'][i,...,0]/255,r['pred'][i,...,0]]
            for col,(arr,label) in enumerate(zip(arrays,labels)):
                ax=axes[row,col];vmin=-.5 if col==0 else 0;vmax=.5 if col<2 else 1
                ax.imshow(arr.T,origin='lower',extent=(-2,6,-4,4),vmin=vmin,vmax=vmax,cmap='viridis')
                ax.add_patch(Polygon(footprint([0,0],0),fill=False,edgecolor='white',lw=.6));tar=target(r['action'][i]);ax.plot(*tar,'r+',ms=8)
                ax.set_title(label if row==0 else '',fontsize=10)
                if col==0:ax.set_ylabel(f't=-{(r["tick"][ids[-1]]-r["tick"][i]+1)*.2:.1f}s\ns={r["s"][i]:.2f} m\ny left (m)')
                if row==4:ax.set_xlabel('x forward (m)')
        fig.suptitle(f'Episode {ep}: all six delayed BEV inputs, privileged truth, decoder; red + = executed target')
        fig.savefig(f'figures/diag_bev_sequence_ep{ep:02d}.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained');ids=last_ids[ep];t=(r['tick'][ids]-r['tick'][ids[-1]]-1)*.2
        tar=np.array([target(a) for a in r['action'][ids]])
        axes[0].plot(t,tar[:,0],label='target forward');axes[0].plot(t,tar[:,1],label='target left');axes[0].set(ylabel='Target (m)')
        subt=(t[:,None]+np.arange(1,5)[None]*.05).ravel()
        for k,name in enumerate(['vx','vy','wz']):
            axes[1].plot(subt,r['sub_cmd'][ids,:,k].ravel(),label='command '+name)
            axes[2].plot(subt,r['sub_vel'][ids,:,k].ravel(),label='actual '+name)
        for ax in axes:ax.grid(alpha=.2);ax.legend(ncol=3);ax.set(xlim=(-5.2,0))
        axes[1].set(ylabel='Tracker m/s, rad/s');axes[2].set(xlabel='Seconds before terminal collision',ylabel='Body m/s, rad/s')
        fig.savefig(f'figures/diag_controls_ep{ep:02d}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    vals=[c['envelope_raw_gap_first_m'] for c in collisions]
    axes[0].hist(np.array(vals)*100,bins=10);axes[0].set(xlabel='First-hit envelope to raw log gap (cm)',ylabel='Episodes',title='Raster conservatism (positive = raw geometry separated)')
    for label in ['dreamer','planner','ppo_shadow']:
        axes[1].bar(label,totals[label+'_unsafe_target_rate']*100)
    axes[1].set(ylabel='Target on non-traversable truth (%)',title='Same input history, last 5.2 s (754 frames)')
    fig.savefig('figures/diag_clearance_actions.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-episodes',action='store_true');args=p.parse_args();evidence(args.all_episodes)
