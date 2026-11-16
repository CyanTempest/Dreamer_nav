"""Small real bake/lookup feasibility probe; not the proposed whole-route bake."""
import time,json
import numpy as np
import warp as wp
import jax.numpy as jnp
from .lwy_bank import LwyBank
from .r8_p0_ray import Scene,airy,camera
from .r8_p0_measure import points,raster,decode,iou
from .r8_p0_common import ROOT,GEO,dump,GPUMonitor,rot,json_ready

@wp.kernel
def lookup(table:wp.array(dtype=wp.uint8),indices:wp.array(dtype=int),out:wp.array(dtype=wp.uint8),seed:int):
 i=wp.tid();env=i//24576;within=i%24576;cell=within//6;ch=within%6
 value=table[indices[env]*24576+within]
 x=-2.0+(float(cell//64)+.5)*.125;y=-4.0+(float(cell%64)+.5)*.125
 state=wp.rand_init(seed,env*4096+cell);drop=wp.randf(state)<.03+.015*(x*x+y*y)
 if drop:
  if ch==0:value=wp.uint8(127)
  elif ch==1 or ch==2:value=wp.uint8(0)
  elif ch==3 or ch==5:value=wp.uint8(255)
 out[i]=value

def main():
 monitor=GPUMonitor('bake');scene=Scene();bank=LwyBank('test',limit=1,root='maps_lwy');base=np.load(GEO/'r8_baseline.npz');oa,da,ra=airy();oc,dc,rc,_,_=camera(-15);o=np.r_[oa,oc];d=np.r_[da,dc];ranges=np.r_[ra,rc]
 local=base['local'];samples=[];poses=[];times=[];raytimes=[];keys=[]
 def make(k,offset,yawoff):
  pose=base['poses'][k].copy();pose[:2]+=np.array([-np.sin(pose[3]),np.cos(pose[3])])*offset;pose[3]+=np.deg2rad(yawoff)
  pose[2]=float(np.asarray(bank.sample(jnp.array([0]),jnp.array(pose[None,:2]),0,linear=True))[0]);world=local@rot(pose[3])[:2,:2].T+pose[:2]
  prior=np.asarray(bank.sample(jnp.array([0]),jnp.array(world[None]),3,linear=True))[0]<0
  result=scene.cast(o,d,ranges,pose);p,l=points(o,d,result);bev,m=raster(p,l,prior,pose,local)
  return pose,bev,result['seconds']
 # Warm first; time full ray + download + NumPy BEV pipeline separately from compile.
 make(2,0,0)
 for k in [2,3,4]:
  for offset in [-.3,0,.3]:
   for yaw in [-15,0,15]:
    start=time.perf_counter();pose,b,rt=make(k,offset,yaw);times.append(time.perf_counter()-start);raytimes.append(rt);samples.append(b);poses.append(pose);keys.append([k,offset,yaw])
  print('BAKE_STATION',float(base['stations'][k]),'complete',flush=True)
 holdout=[]
 for k in [2,3,4]:
  for offset,yaw in [(-.15,-7.5),(.15,7.5)]:
   pose,truth,rt=make(k,offset,yaw);choice=int(np.argmin([1e6*(key[0]!=k)+(key[1]-offset)**2/.3**2+(key[2]-yaw)**2/15**2 for key in keys]));est=decode(samples[choice]);target=decode(truth);common=(est[...,3]<.5)&(target[...,3]<.5)
   holdout.append(dict(station_m=float(base['stations'][k]),offset_m=offset,yaw_offset_deg=yaw,selected_key=keys[choice],known_iou=iou(est[...,3]<.5,target[...,3]<.5),support_iou=iou(est[...,2]>.5,target[...,2]>.5),height_common_mae_m=float(np.abs(est[...,0]-target[...,0])[common].mean()),warning='nearest cached body-frame view, no reprojection; establishes error of naive lookup'))
 table=np.stack(samples);np.savez_compressed(GEO/'r8_bake_probe.npz',bev=table,poses=np.array(poses),keys=np.array(keys))
 tab=wp.array(table.ravel(),dtype=wp.uint8,device='cuda');idx=wp.array(np.arange(2048,dtype=np.int32)%len(table),dtype=int,device='cuda');output=wp.empty(2048*24576,dtype=wp.uint8,device='cuda');wp.launch(lookup,len(output),inputs=[tab,idx,output,88008],device='cuda');wp.synchronize()
 trials=[]
 for rep in range(5):
  t=time.perf_counter()
  for step in range(20):wp.launch(lookup,len(output),inputs=[tab,idx,output,88008+step],device='cuda')
  wp.synchronize();trials.append((time.perf_counter()-t)/20)
 result=dict(status='complete',poses=27,stations_m=[190,320,480],lateral_offsets_m=[-.3,0,.3],yaw_offsets_deg=[-15,0,15],camera_pitch_deg=-15,full_resolution=True,bake_pipeline_seconds=times,bake_ray_seconds=raytimes,bake_mean_seconds_per_pose=float(np.mean(times)),bake_ray_mean_seconds_per_pose=float(np.mean(raytimes)),table_bytes=int(table.nbytes),lookup_envs=2048,lookup_trials_seconds=trials,lookup_median_seconds=float(np.median(trials)),lookup_output_bytes=len(output),holdout=holdout,scope='GPU gather plus distance dropout; resident 27-view table only, hot-cache microbenchmark; excludes large-table misses, pose nearest search, reprojection, rolling accumulation, JAX bridge and policy',proposed_full_bake=dict(station_step_m=.5,station_count=2001,lateral_count=5,yaw_count=7,pose_count=2001*5*7,uint8_bev_bytes=2001*5*7*24576,full_serial_pipeline_seconds_estimate=float(np.mean(times)*2001*5*7),is_estimate=True))
 dump(ROOT/'metrics/r8_p0_bake_probe.json',json_ready(result));print(json.dumps(result,indent=2),flush=True);monitor.close()
if __name__=='__main__':main()
