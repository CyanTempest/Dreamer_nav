"""Measured 2048-pose complete passes with bounded chunks; ray casting only."""
import time,json,gc,subprocess
import numpy as np
import warp as wp
from .r8_p0_ray import Scene,airy,camera
from .r8_p0_common import ROOT,GEO,dump,GPUMonitor,rot

def main():
 mon=GPUMonitor('throughput');scene=Scene();base=np.load(GEO/'r8_baseline.npz');rng=np.random.default_rng(88004);poses=base['poses'][np.arange(2048)%10].copy()
 # Distinct poses; preserve ground z as station estimate. Synthetic throughput workload only.
 lateral=rng.uniform(-.25,.25,2048);poses[:,:2]+=np.c_[-np.sin(poses[:,3]),np.cos(poses[:,3])]*lateral[:,None];poses[:,3]+=rng.uniform(-.3,.3,2048)
 rows=[]
 for name,bs,az,cs,cam_on in [('full',1,1,1,True),('half',2,2,2,True),('quarter',4,4,4,True),('eighth',8,8,8,True),('airy_full_only',1,1,1,False)]:
  o,d,r=airy(bs,az)
  if cam_on:
   oc,dc,rc,_,shape=camera(-15,cs);o=np.r_[o,oc];d=np.r_[d,dc];r=np.r_[r,rc]
  else:shape=None
  chunk=16 if name=='full' else 64
  nr=len(d);args,outs=scene.prepare(o,d,r,poses[:chunk]);scene.launch(args,outs);wp.synchronize()
  elapsed=[];chunks=[];caps=0;checksum=0.
  for repeat in range(3):
   start=time.perf_counter();ct=[]
   for begin in range(0,2048,chunk):
    wp.copy(args[2],wp.array(poses[begin:begin+chunk,:3],dtype=wp.vec3,device='cuda'));wp.copy(args[3],wp.array(poses[begin:begin+chunk,3],dtype=float,device='cuda'))
    t=time.perf_counter();scene.launch(args,outs);wp.synchronize();ct.append(time.perf_counter()-t)
    if begin%512==0:print('BENCH_PROGRESS',name,repeat,begin,'elapsed',time.perf_counter()-start,flush=True)
   elapsed.append(time.perf_counter()-start);chunks.append(ct)
  # Copy only once after timing; outputs are explicitly materialized on GPU each batch.
  dist=outs[0].numpy();caps=int(outs[3].numpy().sum());checksum=float(dist.sum())
  rec=dict(name=name,envs=2048,airy_channels=int(len(np.arange(0,96,bs))),airy_azimuths=int(len(np.arange(0,900,az))),camera_shape=shape,rays_per_env=nr,total_rays=nr*2048,chunk_envs=chunk,repeats_seconds=elapsed,median_seconds=float(np.median(elapsed)),median_env_steps_per_second=2048/float(np.median(elapsed)),rays_per_second=nr*2048/float(np.median(elapsed)),chunk_kernel_seconds=chunks,gpu_output_bytes=chunk*nr*16,final_chunk_alpha_cap_count=caps,final_chunk_output_checksum=checksum)
  rows.append(rec);dump(ROOT/'metrics/r8_p0_throughput.json',dict(status='in_progress',rows=rows));print('BENCH_DONE',name,elapsed,flush=True)
  del args,outs;gc.collect()
 result=dict(status='complete',rows=rows,bvh_build_seconds=scene.build_seconds,device=str(wp.get_device('cuda')),workload='2048 distinct poses distributed evenly over ten stations; lateral +-0.25m,yaw +-0.3rad; fixed station ground height; no dynamics',timing='wall time including pose uploads, per-chunk sync and output stores; excludes BVH construction, RGB stereo matching, point cloud download, BEV reduction and policy training; alpha cutoff enabled',shared_gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],text=True))
 dump(ROOT/'metrics/r8_p0_throughput.json',result);mon.close()
if __name__=='__main__':main()
