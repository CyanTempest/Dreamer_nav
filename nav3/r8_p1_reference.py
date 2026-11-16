"""43 horizontal-camera reference views, same immutable P0 raster and thresholds."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from .r8_p1_common import ROOT,GEO,dump,Monitor,isolate_warp
from .r8_p0_ray import Scene,airy,camera
from .r8_p0_measure import raster,points
from .r8_p0_common import rot
from .lwy_bank import LwyBank
from .lwy_env import LwyEnv

def main():
    isolate_warp(); mon=Monitor('reference'); t0=time.perf_counter()
    base=np.load(GEO/'r8_baseline.npz'); probe=np.load(GEO/'r8_bake_probe.npz')
    bank=LwyBank('test',limit=1,root='maps_lwy')
    poses=list(base['poses']); stations=list(base['stations']); split=['station']*10
    offsets=[0.]*10; yaw_offsets=[0.]*10
    for p,key in zip(probe['poses'],probe['keys']):
        poses.append(p); stations.append(base['stations'][int(key[0])]);split.append('probe')
        offsets.append(float(key[1]));yaw_offsets.append(float(key[2]))
    for k in [2,3,4]:
        for off,yoff in [(-.15,-7.5),(.15,7.5)]:
            p=base['poses'][k].copy();p[:2]+=np.array([-np.sin(p[3]),np.cos(p[3])])*off
            p[3]+=np.deg2rad(yoff);p[2]=float(np.asarray(bank.sample(jnp.array([0]),jnp.array(p[None,:2]),0,linear=True))[0])
            poses.append(p);stations.append(base['stations'][k]);split.append('holdout');offsets.append(off);yaw_offsets.append(yoff)
    poses=np.array(poses,np.float32); n=len(poses); local=base['local']
    arrays=dict(poses=poses,stations=np.array(stations),split=np.array(split),offsets=np.array(offsets),yaw_offsets=np.array(yaw_offsets),local=local)
    for label,options in [('clean',dict(dropout_base=0.,dropout_distance=0.,height_std=0.,position_walk_std=0.,yaw_walk_std=0.,max_observation_delay_steps=0)),('noisy',{})]:
        env=LwyEnv(bank,n,config={'base_height_jitter':0.,**options})
        state=env._reset(jax.random.PRNGKey(88001),jnp.zeros(n,jnp.int32),jnp.array(stations),observe=False)
        state.update(pos=jnp.array(poses[:,:2]),yaw=jnp.array(poses[:,3]),mempos=jnp.floor(jnp.array(poses[:,:2])/.125)*.125-8,height=jnp.full(n,.54),memyaw=jnp.array(poses[:,3]))
        _,obs,_=env.observe(state);jax.block_until_ready(obs);arrays[label]=np.array(obs['bev'])
        # The ten station baselines remain exactly the already audited P0 realizations.
        arrays[label][:10]=base[label+'_bev']
    prior=arrays['clean'][...,4]/255.
    scene=Scene();oa,da,ra=airy();oc,dc,rc,cosz,shape=camera(0)
    refs=[];rawlow=[];rawhigh=[];rawknown=[];grounds=[];rows=[]
    p0=np.load(GEO/'r8_sensor_bevs.npz')
    for i,p in enumerate(poses):
        start=time.perf_counter();caps=0
        if i<10:
            s=int(stations[i]);a=np.load(GEO/f'r8_airy_{s:04d}.npz');ap=a['points_body'];al=a['semantic']
            c=np.load(GEO/f'r8_gemini_{s:04d}_pitch0.npz');depth=c['depth_m'].ravel();valid=np.isfinite(depth)
            cp=oc[valid]+dc[valid]*(depth[valid]/cosz[valid])[:,None];cl=c['semantic'].ravel()[valid]
        else:
            a=scene.cast(oa,da,ra,p);c=scene.cast(oc,dc,rc,p);ap,al=points(oa,da,a);cp,cl=points(oc,dc,c)
            caps=int(np.sum(a['cap'])+np.sum(c['cap']))
        bev,m=raster(np.r_[ap,cp],np.r_[al,cl],prior[i],p,local)
        # Float32 depth reconstruction can straddle a cell edge; use P0's saved exact raster.
        reused_delta=None
        if i<10:
            original=p0[f'bev_{int(stations[i])}_0'];reused_delta=int(np.any(bev!=original,axis=-1).sum());bev=original
        refs.append(bev);rawlow.append(m['zmin']);rawhigh.append(m['zmax']);rawknown.append(m['count']>0);grounds.append(m['ground_estimate'])
        rows.append(dict(index=i,station_m=stations[i],split=split[i],offset=offsets[i],yaw_offset=yaw_offsets[i],seconds=time.perf_counter()-start,alpha_cap=caps,reuse=i<10,depth_reconstruction_changed_cells=reused_delta))
        print('REFERENCE',i,split[i],stations[i],rows[-1]['seconds'],flush=True)
    arrays.update(reference=np.array(refs),raw_low=np.array(rawlow),raw_high=np.array(rawhigh),raw_known=np.array(rawknown),ground_estimate=np.array(grounds))
    np.savez_compressed(GEO/'r8_p1_references.npz',**arrays)
    dump(ROOT/'metrics/r8_p1_reference.json',dict(camera_pitch=0,camera_xyz=[.3,0,.64],airy_height_mean=.495,poses=43,distinct_poses=40,full_resolution_rays_per_pose=1196800,seconds=time.perf_counter()-t0,bvh_seconds=scene.build_seconds,rows=rows,holdout_used_for_fitting=False,station_reference='Exact P0 saved pitch0 BEV reused; depth reconstruction only supplies diagnostics.'))
    mon.close()

if __name__=='__main__': main()
