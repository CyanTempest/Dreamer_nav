"""Reference metrics, noise refit, fixed ablations, and three-way figures."""
import argparse,time,json
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p1_common import ROOT,GEO,dump,Monitor
from .r8_p1_online import SimpleEnv,Atlas,state_at,DEFAULT
from .lwy_bank import LwyBank
from .env import decode

def dec(x):
    return (x.astype(np.float32)/255-np.array([.5,0,0,0,0,0]))*np.array([4,2,1,1,1,1])

def iou(a,b):
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else None

def metrics(a,b):
    a,b=dec(a),dec(b);ka=a[...,3]<.5;kb=b[...,3]<.5;common=ka&kb
    sa=(a[...,2]>.5)&ka;sb=(b[...,2]>.5)&kb
    oa=(a[...,1]>=.22)&ka;ob=(b[...,1]>=.22)&kb
    return dict(support_iou=iou(sa,sb),obstacle_iou=iou(oa,ob),unknown_iou=iou(~ka,~kb),known_iou=iou(ka,kb),
        support_iou_common=iou(sa&common,sb&common),obstacle_iou_common=iou(oa&common,ob&common),
        height_mae_m=float(np.abs(a[...,0]-b[...,0])[common].mean()) if common.any() else None,
        channel_mae_all=np.abs(a-b).mean((0,1)),channel_mae_common=np.abs(a-b)[common].mean(0) if common.any() else np.full(6,np.nan),
        unknown_fraction=float((~ka).mean()),reference_unknown_fraction=float((~kb).mean()),common_cells=int(common.sum()),
        support_intersection=int((sa&sb).sum()),support_union=int((sa|sb).sum()),obstacle_intersection=int((oa&ob).sum()),obstacle_union=int((oa|ob).sum()),unknown_intersection=int((~ka&~kb).sum()),unknown_union=int((~ka|~kb).sum()))

def summary(rows):
    out={}
    for k in ['support_iou','obstacle_iou','unknown_iou','known_iou','height_mae_m','unknown_fraction','reference_unknown_fraction','channel_mae_all','channel_mae_common']:
        out[k]=np.nanmean([r[k] if r[k] is not None else np.nan for r in rows],axis=0)
    for key in ['support','obstacle','unknown']:
        out[key+'_iou_micro']=sum(r[key+'_intersection'] for r in rows)/max(1,sum(r[key+'_union'] for r in rows))
    out['height_mae_weighted_m']=sum(r['height_mae_m']*r['common_cells'] for r in rows)/sum(r['common_cells'] for r in rows)
    return out

def run(bank,atlas,refs,config):
    env=SimpleEnv(bank,len(refs['poses']),atlas=atlas,config={'position_walk_std':0.,'yaw_walk_std':0.,'max_observation_delay_steps':0,**config})
    state=state_at(env,refs['poses'],refs['stations'])
    _,obs,_=env.observe(state);jax.block_until_ready(obs)
    return np.asarray(obs['bev'])

def first_frame(measured,seen,poses,local,prior):
    """Exact P0 first-frame world scatter, used for fast noise calibration only."""
    out=[]
    for m,s,p,pr in zip(measured,seen,poses,prior):
        cy,sy=np.cos(p[3]),np.sin(p[3]);R=np.array([[cy,-sy],[sy,cy]])
        world=local@R.T+p[:2];origin=np.floor(p[:2]/.125)*.125-8
        idx=np.clip(np.floor((world-origin)/.125).astype(int),0,127);idx=idx[...,0]*128+idx[...,1]
        mem=np.full((128*128,3),-np.inf);np.maximum.at(mem,idx[s],m[s]);m=mem[idx];known=np.isfinite(m[...,0]);m[~known]=0
        bev=np.concatenate([m,(~known)[...,None],pr[...,None],(~known)[...,None]],-1)
        out.append((np.clip(bev*np.array([.25,.5,1,1,1,1])+[.5,0,0,0,0,0],0,1)*255).astype(np.uint8))
    return np.array(out)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    mon=Monitor('validation');t0=time.perf_counter();refs=np.load(GEO/'r8_p1_references.npz');bank=LwyBank('test',limit=1,root='maps_lwy');atlas=Atlas()
    arrays={};arrays['simple_unfit']=run(bank,atlas,refs,{})
    print('INITIAL',summary([metrics(a,b) for a,b in zip(arrays['simple_unfit'][:10],refs['reference'][:10])]),flush=True)
    if args.quick:
        np.savez_compressed(GEO/'r8_p1_quick.npz',bev=arrays['simple_unfit'])
        dump(ROOT/'metrics/r8_p1_quick.json',dict(rows=[metrics(a,b) for a,b in zip(arrays['simple_unfit'],refs['reference'])]));mon.close();return
    # Noise calibration uses the 27 probe records only; the six intermediate
    # poses are untouched. Ideal geometry has no independently identifiable
    # sensor height noise; optimize synthetic noise against deterministic refs.
    fitids=np.flatnonzero(refs['split']=='probe')
    env=SimpleEnv(bank,len(fitids),atlas=atlas,config={'position_walk_std':0.,'yaw_walk_std':0.,'max_observation_delay_steps':0})
    st=state_at(env,refs['poses'][fitids],refs['stations'][fitids])
    measured,seen,_=jax.jit(SimpleEnv.sensor_frame)(env,st,jax.random.PRNGKey(991))
    measured,seen=np.asarray(measured),np.asarray(seen)
    rng=np.random.default_rng(991);uniform=rng.random(seen.shape);normal=rng.normal(size=seen.shape)
    distance=np.linalg.norm(refs['local'],axis=-1);target=refs['reference'][fitids]
    prior=refs['clean'][fitids,...,4]/255.;candidates=[]
    for b in [0.,.005,.01,.02,.03]:
        for d in [0.,.00025,.0005,.001,.002,.004,.008,.015]:
            keep=seen&(uniform>=b+d*distance**2)
            pred=first_frame(measured,keep,refs['poses'][fitids],refs['local'],prior)
            m=summary([metrics(a,r) for a,r in zip(pred,target)])
            candidates.append(dict(base=b,distance=d,unknown_iou=m['unknown_iou'],unknown_fraction=m['unknown_fraction'],reference_unknown_fraction=m['reference_unknown_fraction']))
    # Match fraction, then spatial IoU: noise must not be used as an arbitrary
    # per-pose mask fitted to reference labels.
    best=min(candidates,key=lambda r:(abs(r['unknown_fraction']-r['reference_unknown_fraction']),-r['unknown_iou']))
    noise=[]
    for sigma in [0.,.005,.01,.015,.025,.04,.06]:
        m=measured.copy();m[...,0]+=normal*sigma
        pred=first_frame(m,seen,refs['poses'][fitids],refs['local'],prior)
        score=summary([metrics(a,r) for a,r in zip(pred,target)])
        noise.append(dict(std_m=sigma,height_mae_m=score['height_mae_m']))
    std=min(noise,key=lambda r:r['height_mae_m'])['std_m']
    config=dict(p1_noise_base=best['base'],p1_noise_distance=best['distance'],p1_height_std=std)
    dump(ROOT/'metrics/r8_p1_noise_fit.json',dict(fit_indices=fitids,holdout_used=False,selected_config=config,distance_dropout_candidates=candidates,height_noise_candidates=noise,objective='unknown fraction absolute error, tie-break unknown IoU; height MAE separately',caveat='Ideal noiseless P0 does not identify real sensor stochastic error. Geometric residual is not calibrated measurement noise.'))
    arrays['simple']=run(bank,atlas,refs,config)
    arrays['no_visibility']=run(bank,atlas,refs,{**config,'p1_visibility':False})
    arrays['old_noise']=run(bank,atlas,refs,dict(p1_noise_base=.03,p1_noise_distance=.015,p1_height_std=.025))
    arrays['ground_only']=run(bank,atlas,refs,{**config,'p1_max_band':0})
    opaque=Atlas(opaque=True);arrays['no_alpha']=run(bank,opaque,refs,config)
    arrays['training_clean']=refs['clean'];arrays['training_noisy']=refs['noisy']
    rows=[]
    for i in range(len(refs['poses'])):
        row=dict(index=i,station_m=refs['stations'][i],split=str(refs['split'][i]),offset_m=refs['offsets'][i],yaw_offset_deg=refs['yaw_offsets'][i],methods={})
        for name,a in arrays.items():row['methods'][name]=metrics(a[i],refs['reference'][i])
        rows.append(row)
    summaries={split:{name:summary([r['methods'][name] for r in rows if split=='all' or r['split']==split]) for name in arrays} for split in ['station','probe','holdout','all']}
    dump(ROOT/'metrics/r8_p1_validation.json',dict(rows=rows,summary=summaries,seconds=time.perf_counter()-t0,jax=jax.__version__,backend=jax.default_backend(),thresholds=dict(obstacle_m=.22,support=.5),height='decoded uint8, common known cells',alpha_reference=True,camera_pitch_deg=0,fit=config))
    np.savez_compressed(GEO/'r8_p1_predictions.npz',**arrays)
    for i,row in enumerate(rows):
        fig,axs=plt.subplots(6,3,figsize=(11,17),layout='constrained')
        for col,(title,a) in enumerate([('P0 rays / horizontal Gemini',refs['reference'][i]),('P1 static atlas + visibility',arrays['simple'][i]),('Current training BEV / noise on',refs['noisy'][i])]):
            d=dec(a);known=d[...,3]<.5
            for ch in range(6):
                im=d[...,ch] if ch>=3 else np.where(known,d[...,ch],np.nan)
                vmin,vmax=(-1,1) if ch==0 else (0,2) if ch==1 else (0,1)
                artist=axs[ch,col].imshow(im,origin='lower',extent=(-4,4,-2,6),vmin=vmin,vmax=vmax,cmap=['terrain','magma','viridis','gray_r','Blues','plasma'][ch])
                axs[ch,col].set_title(title if ch==0 else ['ground [m]','obstacle [m]','support','unknown','route prior','age / 8s'][ch]);axs[ch,col].set_xlabel('left [m]');axs[ch,col].set_ylabel('forward [m]')
                if col==2:fig.colorbar(artist,ax=axs[ch,:],shrink=.7)
        fig.suptitle(f'{row["split"]} #{i} | s={row["station_m"]:g}m | offset={row["offset_m"]:+.2f}m yaw={row["yaw_offset_deg"]:+.1f}deg')
        fig.savefig(ROOT/f'figures/r8_p1_compare_{i:02d}_{row["split"]}_{int(row["station_m"]):04d}.png',dpi=100);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    for ax,key,target in zip(axs,['support_iou','obstacle_iou','unknown_iou','height_mae_m'],[.7,.7,.7,.06]):
        for method in ['simple','training_clean','training_noisy']:
            ax.plot(refs['stations'][:10],[r['methods'][method][key] for r in rows[:10]],'o-',label=method)
        ax.axhline(target,color='red',ls='--');ax.set_title(key);ax.set_xlabel('station [m]');ax.grid(alpha=.2)
    axs[0].legend();fig.savefig(ROOT/'figures/r8_p1_summary.png',dpi=140);plt.close(fig)
    print('FINAL',summaries['station']['simple'],flush=True);mon.close()

if __name__=='__main__':main()
