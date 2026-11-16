"""Exact instanced Warp Mesh BLAS + Bvh TLAS; alpha-cutout first-return rays."""
import os,time,json,sys
from pathlib import Path
import numpy as np
import warp as wp
from .r8_p0_common import ROOT,GEO,dump,rot
from .calibration import mounts
wp.config.kernel_cache_dir=str(GEO/'r8_cache/warp')
wp.init()

@wp.func
def alpha_at(mid:int,uv:wp.vec2,meta:wp.array(dtype=wp.vec3i),tex:wp.array(dtype=float)):
 info=meta[mid];w=info[1];h=info[2]
 a=float(1.0)
 if w>0:
  x=(uv[0]-wp.floor(uv[0]))*float(w)-0.5
  y=(1.0-(uv[1]-wp.floor(uv[1])))*float(h)-0.5
  ix=int(wp.floor(x));iy=int(wp.floor(y));fx=x-wp.floor(x);fy=y-wp.floor(y)
  x0=(ix+w)%w;x1=(ix+1+w)%w;y0=(iy+h)%h;y1=(iy+1+h)%h
  a=((1.0-fx)*(1.0-fy)*float(tex[info[0]+y0*w+x0])+fx*(1.0-fy)*float(tex[info[0]+y0*w+x1])+(1.0-fx)*fy*float(tex[info[0]+y1*w+x0])+fx*fy*float(tex[info[0]+y1*w+x1]))
 return a

@wp.kernel
def trace(tlas:wp.uint64,mesh_ids:wp.array(dtype=wp.uint64),inverse:wp.array(dtype=wp.mat44),offset:wp.array(dtype=wp.int32),
 sem:wp.array(dtype=wp.int32),mat:wp.array(dtype=wp.int32),uvs:wp.array(dtype=wp.vec2),ameta:wp.array(dtype=wp.vec3i),tex:wp.array(dtype=float),
 origins:wp.array(dtype=wp.vec3),directions:wp.array(dtype=wp.vec3),pose_xyz:wp.array(dtype=wp.vec3),pose_yaw:wp.array(dtype=float),nr:int,
 ranges:wp.array(dtype=wp.vec2),alpha_enabled:int,remove_soft:int,
 out_t:wp.array(dtype=float),out_face:wp.array(dtype=int),out_inst:wp.array(dtype=int),out_cap:wp.array(dtype=int)):
 tid=wp.tid();e=tid//nr;r=tid%nr;cy=wp.cos(pose_yaw[e]);sy=wp.sin(pose_yaw[e]);o0=origins[r];d0=directions[r]
 o=wp.vec3(cy*o0[0]-sy*o0[1],sy*o0[0]+cy*o0[1],o0[2])+pose_xyz[e]
 d=wp.vec3(cy*d0[0]-sy*d0[1],sy*d0[0]+cy*d0[1],d0[2])
 best=ranges[r][1];face=int(-1);inst=int(-1);cap=int(0)
 query=wp.bvh_query_ray(tlas,o,d);idx=int(0)
 while wp.bvh_query_next(query,idx):
  inv=inverse[idx];lo=wp.transform_point(inv,o);ld0=wp.transform_vector(inv,d);scale=wp.length(ld0);ld=ld0/scale
  advance=float(0.0);attempt=int(0)
  while advance<best*scale and attempt<128:
   hit=wp.mesh_query_ray(mesh_ids[idx],lo+advance*ld,ld,best*scale-advance)
   if not hit.result:break
   fi=hit.face+offset[idx];cl=sem[fi]
   visible=True
   if remove_soft==1 and (cl==8 or cl==9 or cl==10 or cl==11):visible=False
   if alpha_enabled==1:
    uv=uvs[fi*3]*hit.u+uvs[fi*3+1]*hit.v+uvs[fi*3+2]*(1.0-hit.u-hit.v)
    if alpha_at(mat[fi],uv,ameta,tex)<0.5:visible=False
   if visible:
    best=(advance+hit.t)/scale;face=fi;inst=idx;break
   advance=advance+hit.t+0.00002*scale;attempt+=1
  if attempt==128:cap+=1
 # Below minimum range still occludes; never see through a near object.
 if best<ranges[r][0]:face=-1;inst=-1
 out_t[tid]=best if face>=0 else -1.0;out_face[tid]=face;out_inst[tid]=inst;out_cap[tid]=cap

class Scene:
 def __init__(self):
  t=time.perf_counter();records=json.loads((GEO/'r8_instances.json').read_text());self.records=records
  self.meshes=[];self.data=[];so=[];ma=[];uv=[];offsets=[];n=0
  for k in range(max(r['template'] for r in records)+1):
   z=np.load(GEO/f'r8_template_{k:03d}.npz');v=z['vertices'];f=z['faces'];s=z['semantic'];self.data.append({x:z[x] for x in z.files})
   self.meshes.append(wp.Mesh(wp.array(v,dtype=wp.vec3,device='cuda'),wp.array(f.ravel(),dtype=wp.int32,device='cuda'),support_winding_number=False))
   offsets.append(n);n+=len(f);so.append(s);ma.append(z['material']);uv.append(z['uv'].reshape(-1,2))
  self.semantic=np.concatenate(so);self.sem=wp.array(self.semantic.astype(np.int32),dtype=int,device='cuda');self.mat=wp.array(np.concatenate(ma),dtype=int,device='cuda');self.uv=wp.array(np.concatenate(uv),dtype=wp.vec2,device='cuda')
  self.ids=wp.array(np.array([self.meshes[r['template']].id for r in records],np.uint64),dtype=wp.uint64,device='cuda')
  self.inv=wp.array(np.linalg.inv(np.array([r['matrix_world'] for r in records],np.float64)).astype(np.float32),dtype=wp.mat44,device='cuda')
  self.off=wp.array(np.array([offsets[r['template']] for r in records],np.int32),dtype=int,device='cuda')
  self.lower=wp.array(np.array([r['bounds_min'] for r in records],np.float32)-.0001,dtype=wp.vec3,device='cuda');self.upper=wp.array(np.array([r['bounds_max'] for r in records],np.float32)+.0001,dtype=wp.vec3,device='cuda');self.tlas=wp.Bvh(self.lower,self.upper)
  a=json.loads((GEO/'r8_materials.json').read_text());self.ameta=wp.array(np.array([[m['offset'],m['width'],m['height']] for m in a],np.int32),dtype=wp.vec3i,device='cuda');self.tex=wp.array(np.load(GEO/'r8_alpha_linear.npy'),dtype=float,device='cuda')
  wp.synchronize();self.build_seconds=time.perf_counter()-t;print('SCENE_BUILT',self.build_seconds,'s',flush=True)
 def prepare(self,o,d,ranges,poses):
  nr=len(d);n=len(poses)*nr
  return [wp.array(o,dtype=wp.vec3,device='cuda'),wp.array(d,dtype=wp.vec3,device='cuda'),wp.array(poses[:,:3],dtype=wp.vec3,device='cuda'),wp.array(poses[:,3],dtype=float,device='cuda'),nr,wp.array(ranges,dtype=wp.vec2,device='cuda')], [wp.empty(n,dtype=x,device='cuda') for x in [float,int,int,int]]
 def launch(self,args,outs,alpha=True,remove_soft=False):
  wp.launch(trace,len(outs[0]),inputs=[self.tlas.id,self.ids,self.inv,self.off,self.sem,self.mat,self.uv,self.ameta,self.tex,*args,int(alpha),int(remove_soft),*outs],device='cuda')
 def cast(self,o,d,ranges,pose,alpha=True,remove_soft=False):
  args,outs=self.prepare(o,d,ranges,np.asarray(pose,np.float32).reshape(-1,4));t=time.perf_counter();self.launch(args,outs,alpha,remove_soft);wp.synchronize();elapsed=time.perf_counter()-t
  dist,face,instance,cap=[a.numpy() for a in outs];sem=np.where(face>=0,self.semantic[np.maximum(face,0)],0)
  return dict(distance=dist,face=face,instance=instance,semantic=sem,cap=cap,seconds=elapsed)

def airy(stride=1,az_stride=None,legacy=False):
 # Import the exact generator used by raycheck, including nominal angle table.
 sys.path.insert(0,str(ROOT/'vendor/map'))
 from sensors import rays
 native,_,_,_=rays(stride,az_stride or stride,0)
 ts=mounts().copy();ts[:,2,3]+=.54 if legacy else .495-float(ts[:,2,3].mean())
 origins=np.concatenate([np.repeat(t[:3,3][None],len(native),0) for t in ts]);directions=np.concatenate([native@t[:3,:3].T for t in ts]);directions/=np.linalg.norm(directions,axis=1)[:,None]
 return origins.astype(np.float32),directions.astype(np.float32),np.tile([.1,60.],(len(directions),1)).astype(np.float32)

def camera(pitch=-15,stride=1):
 w,h=1280,800;fx=w/(2*np.tan(np.deg2rad(90/2)));fy=h/(2*np.tan(np.deg2rad(65/2)));cx=(w-1)/2;cy=(h-1)/2
 u,v=np.meshgrid(np.arange(0,w,stride),np.arange(0,h,stride));q=np.stack([np.ones_like(u),-(u-cx)/fx,-(v-cy)/fy],-1).reshape(-1,3).astype(np.float32);q/=np.linalg.norm(q,axis=-1)[:,None];cosz=q[:,0].copy()
 p=np.deg2rad(pitch);R=np.array([[np.cos(p),0,-np.sin(p)],[0,1,0],[np.sin(p),0,np.cos(p)]],np.float32)
 d=q@R.T;o=np.tile([.30,0,.64],(len(d),1)).astype(np.float32)
 # Provisional disparity range=128: min optical Z=.5 at 1280x800 (datasheet 4.7).
 ranges=np.c_[.5/cosz,20./cosz].astype(np.float32)
 return o,d,ranges,cosz,(len(np.arange(0,h,stride)),len(np.arange(0,w,stride)))

def extrinsics():
 old=mounts();ts=old.copy();shift=.495-float(old[:,2,3].mean());ts[:,2,3]+=shift
 result=dict(coordinate='ground-centered x forward, y left, z up; yaw-only body in LWY',airy=dict(count=2,channels=96,azimuths=900,azimuth_step_deg=.4,range_m=[.1,60],native_generator='vendor/map/sensors.py:rays(1,1,0), exactly as nav3/raycheck.py',relative_chain_source='nav3/calibration.py + configs/calibration.yaml, measured',height_mean_m=.495,height_source='user approximate standing estimate 2026-11-13',common_z_translation_m=shift,raw_base_transforms=old.tolist(),ground_transforms=ts.tolist(),legacy_ground_transforms=(old+np.pad(np.array([[[0,0,0,.54]],[[0,0,0,.54]]]),((0,0),(2,1),(0,0)))).tolist()),gemini=dict(model='Gemini 336L',resolution=[1280,800],depth_fov_degrees=[90,65],fx=640.,fy=float(800/(2*np.tan(np.deg2rad(32.5)))),cx=639.5,cy=399.5,distortion=[0,0,0,0,0],optical_z_range_m=[.5,20],ideal_range_m=[.25,6],hardware_range_m=[.17,20],baseline_m=.095,pitches_deg=[0,-15,-30],translation_ground_m=[.3,0,.64],height_source='user approximate standing estimate',xy_source='provisional front center; related elf3 head model incompatible with this chassis',pitch_source='provisional sweep',intrinsics_source='provisional pinhole from official FOV; no device calibration found',min_z_source='V1.6 section 4.7; chosen provisional disparity 128 at 1280x800',spec_url='https://www.orbbec.com/wp-content/uploads/2025/06/Gemini-330-series-Datasheet-V1.6.pdf',search_roots=['exp_nav3','/home/hpf/visualnav','/home/hpf/wyg_hiking'],found_configs=['visualnav-transformer/deployment/config/camera_front.yaml: generic USB 160x120','legged_lab/sensors/camera/camera_cfg.py: generic 480x270, 0.2..15m','legged_lab/assets/bx_29dof/mjcf/elf3.xml: torso z=1.1, head fromto x=-.01..+.02,z=.30; no Gemini mounting chain'],simulation='ideal first optical surface; not stereo matching, reflectance, sunlight, motion, rolling acquisition or chassis self-occlusion'))
 g=result['gemini'];g.update(nominal_depth_fps=30,chosen_disparity_search_range=128,disparity_status='provisional; datasheet also supports 256 -> 0.25 m min Z',yaw_deg=0,roll_deg=0,optical_frame='x right, y down, z forward',distortion_status='provisional zero distortion; no SDK K/D available')
 g['ground_from_optical_transforms']={};A=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
 for pitch in g['pitches_deg']:
  p=np.deg2rad(pitch);R=np.array([[np.cos(p),0,-np.sin(p)],[0,1,0],[np.sin(p),0,np.cos(p)]]);T=np.eye(4);T[:3,:3]=R@A;T[:3,3]=g['translation_ground_m'];g['ground_from_optical_transforms'][str(pitch)]=T.tolist()
 result['airy'].update(nominal_vertical_angles_deg=json.loads((ROOT/'vendor/map/sources/airy_specs.json').read_text())['nominal_vertical_angles_deg'],beam_angles_status='inherited nominal reference pattern, not per-device factory calibration',frame_index=0,rotation_status='measured extrinsic chain',xy_translation_status='measured extrinsic chain')
 dump(ROOT/'metrics/r8_p0_sensor_extrinsics.json',result)
 return result
if __name__=='__main__':
 from .r8_p0_common import GPUMonitor
 mon=GPUMonitor('smoke');extrinsics();scene=Scene();base=np.load(GEO/'r8_baseline.npz');o,d,r=airy(8,10);result=scene.cast(o,d,r,base['poses'][2]);print('SMOKE',len(d),np.sum(result['distance']>0),result['seconds'],np.sum(result['cap']),flush=True);mon.close()
