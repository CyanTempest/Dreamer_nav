"""Ten fixed LWY stations, three camera pitches; no policies, no training."""
import time,json
import numpy as np
from scipy.ndimage import distance_transform_edt,grey_opening
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p0_common import ROOT,GEO,dump,json_ready,GPUMonitor,rot
from .r8_p0_ray import Scene,airy,camera
SOFT=[8,10,11];MIXED=[9];HARD=[3,5,7,12,13,15]

def encode(x):return (np.clip(x*np.array([.25,.5,1,1,1,1])+[.5,0,0,0,0,0],0,1)*255).astype(np.uint8)
def decode(x):return (x.astype(float)/255-[.5,0,0,0,0,0])*[4,2,1,1,1,1]

def points(o,d,result):
 good=result['distance']>0
 return (o[good]+d[good]*result['distance'][good,None]),result['semantic'][good]

def raster(p,labels,prior,pose,local):
 ij=np.floor((p[:,:2]+[2,4])/.125).astype(int);keep=np.all((ij>=0)&(ij<64),1)
 p=p[keep];labels=labels[keep];ij=ij[keep];flat=ij[:,0]*64+ij[:,1]
 count=np.bincount(flat,minlength=4096);low=np.full(4096,np.inf);high=np.full(4096,-np.inf)
 np.minimum.at(low,flat,p[:,2]);np.maximum.at(high,flat,p[:,2]);known=(count>0).reshape(64,64)
 low=low.reshape(64,64);high=high.reshape(64,64)
 if known.any():
  nearest=distance_transform_edt(~known,return_distances=False,return_indices=True)
  filled=low[tuple(nearest)]
 else:filled=np.zeros((64,64))
 # Sensor-only lower envelope; holes remain unknown. Opening removes small above-ground objects.
 ground=grey_opening(filled,size=(5,5),mode='nearest');height=np.maximum(0,high-ground)
 dx=(np.roll(ground,-2,0)-np.roll(ground,2,0))/.5;dy=(np.roll(ground,-2,1)-np.roll(ground,2,1))/.5
 support=(np.hypot(dx,dy)<.8)&(height<.22)&(ground>-.55)
 # Match training chassis blind footprint and first-frame world-memory max scatter.
 known&=~((abs(local[...,0])<.49)&(abs(local[...,1])<.29))
 measured=np.stack([ground,height,support],-1)
 world=local@rot(pose[3])[:2,:2].T+pose[:2];origin=np.floor(pose[:2]/.125)*.125-8
 idx=np.clip(np.floor((world-origin)/.125).astype(int),0,127);idx=idx[...,0]*128+idx[...,1]
 mem=np.full((128*128,3),-np.inf);np.maximum.at(mem,idx[known],measured[known]);measured=mem[idx];known=np.isfinite(measured[...,0]);measured[~known]=0
 bev=np.concatenate([measured,(~known)[...,None],prior[...,None],(~known)[...,None]],-1)
 masks={}
 for name,classes in [('soft',SOFT),('shrub',MIXED),('hard',HARD),('terrain',[1,2,4]),('root',[6]),('branch',[15])]:
  mask=np.zeros(4096,bool);mask[flat[np.isin(labels,classes)]]=1;masks[name]=mask.reshape(64,64)
 # Audit raw terrain-only heights; never used to build sensor observation.
 g=np.full(4096,np.inf);hit=np.isin(labels,[1,2,4]);np.minimum.at(g,flat[hit],p[hit,2]);masks['terrain_height']=g.reshape(64,64)
 masks.update(count=count.reshape(64,64),zmin=low,zmax=high,ground_estimate=ground)
 return encode(bev),masks

def iou(a,b):
 u=(a|b).sum();return float((a&b).sum()/u) if u else None

def comparison(ref,ray,mask):
 a=decode(ref);b=decode(ray);known=(a[...,3]<.5)&(b[...,3]<.5);ka=a[...,3]<.5;kb=b[...,3]<.5
 obsa=(a[...,1]>=.22)&ka;obsb=(b[...,1]>=.22)&kb;ta=(a[...,2]>.5)&ka;tb=(b[...,2]>.5)&kb
 out=dict(common_cells=int(known.sum()),ray_known_cells=int(kb.sum()),reference_known_cells=int(ka.sum()),obstacle_iou_all=iou(obsa,obsb),support_iou_all=iou(ta,tb),obstacle_iou_common=iou(obsa&known,obsb&known),support_iou_common=iou(ta&known,tb&known),height_mae_m=float(np.abs(a[...,0]-b[...,0])[known].mean()) if known.any() else None,channel_mae_all=np.abs(a-b).mean((0,1)).tolist(),channel_mae_common=np.abs(a-b)[known].mean(0).tolist() if known.any() else [None]*6,missing_hard_cells=int((mask['hard']&obsb&~obsa).sum()),soft_obstacle_cells=int((mask['soft']&obsb).sum()),shrub_obstacle_cells=int((mask['shrub']&obsb).sum()),hard_obstacle_cells=int((mask['hard']&obsb).sum()),soft_obstacle_without_hard=int((mask['soft']&obsb&~mask['hard']).sum()))
 return out

def main():
 mon=GPUMonitor('stations');scene=Scene();base=np.load(GEO/'r8_baseline.npz');rows=[];oa,da,ra=airy()
 # Warm compile outside station timings.
 scene.cast(oa[:1],da[:1],ra[:1],base['poses'][0]);t0=time.perf_counter();all_arrays={}
 for k,(station,pose) in enumerate(zip(base['stations'],base['poses'])):
  t=time.perf_counter();a=scene.cast(oa,da,ra,pose);ap,al=points(oa,da,a);astore=dict(points_body=ap,points_world=ap@rot(pose[3]).T+pose[:3],semantic=al,distance=a['distance'].reshape(2,96,900),face=a['face'].reshape(2,96,900),instance=a['instance'].reshape(2,96,900),pose=pose)
  np.savez_compressed(GEO/f'r8_airy_{int(station):04d}.npz',**astore)
  prior=base['clean_decoded'][k,...,4];ab,am=raster(ap,al,prior,pose,base['local'])
  cams=[];bevs=[];masks=[];records=[]
  for pitch in [0,-15,-30]:
   oc,dc,rc,cosz,shape=camera(pitch);c=scene.cast(oc,dc,rc,pose);cp,cl=points(oc,dc,c);depth=np.where(c['distance']>0,c['distance']*cosz,np.nan).reshape(shape)
   np.savez_compressed(GEO/f'r8_gemini_{int(station):04d}_pitch{pitch}.npz',depth_m=depth,semantic=c['semantic'].reshape(shape),pose=pose,pitch_deg=pitch)
   p=np.concatenate([ap,cp]);labels=np.concatenate([al,cl]);bev,m=raster(p,labels,prior,pose,base['local']);cb,cm=raster(cp,cl,prior,pose,base['local'])
   rec=dict(station_m=float(station),pitch_deg=pitch,airy_returns=[int((a['distance'].reshape(2,-1)[i]>0).sum()) for i in range(2)],camera_returns=int(len(cp)),airy_seconds=a['seconds'],camera_seconds=c['seconds'],alpha_cap_rays=int((a['cap']>0).sum()+(c['cap']>0).sum()),camera_known_cells=int((decode(cb)[...,3]<.5).sum()),camera_corridor_known_cells=int(((decode(cb)[...,3]<.5)&(prior>.5)).sum()),fused_corridor_known_cells=int(((decode(bev)[...,3]<.5)&(prior>.5)).sum()),airy_known_cells=int((decode(ab)[...,3]<.5).sum()),camera_added_cells=int(((decode(bev)[...,3]<.5)&(decode(ab)[...,3]>.5)).sum()),clean=comparison(base['clean_bev'][k],bev,m),noisy=comparison(base['noisy_bev'][k],bev,m))
   common=np.isfinite(m['terrain_height'])&(base['clean_decoded'][k,...,3]<.5)
   rec['terrain_truth_height_mae_m']=float(np.abs(m['terrain_height']-base['truth'][k,...,0]+pose[2])[common].mean()) if common.any() else None
   rec['terrain_common_cells']=int(common.sum())
   cmask=cm['count']>0;xx=base['local'][...,0];rec['camera_bev_forward_extent_m']=[float(xx[cmask].min()),float(xx[cmask].max())] if cmask.any() else None
   records.append(rec);rows.append(rec);cams.append(depth);bevs.append(bev);masks.append(m)
   all_arrays[f'bev_{int(station)}_{pitch}']=bev
   all_arrays[f'masks_{int(station)}_{pitch}']=np.stack([m[n] for n in ['soft','shrub','hard','terrain','root','branch']],-1)
  # Counterfactual at recommended sweep middle: remove all foliage + shrubs, keep wood branches.
  ac=scene.cast(oa,da,ra,pose,remove_soft=True);oc,dc,rc,_,_=camera(-15);cc=scene.cast(oc,dc,rc,pose,remove_soft=True)
  p1,l1=points(oa,da,ac);p2,l2=points(oc,dc,cc);b0,m0=raster(np.concatenate([p1,p2]),np.concatenate([l1,l2]),prior,pose,base['local'])
  all_arrays[f'counterfactual_bev_{int(station)}']=b0
  cf=dict(changed_cells=int(np.any(b0!=bevs[1],axis=-1).sum()),known_gain_cells=int(((decode(b0)[...,3]<.5)&(decode(bevs[1])[...,3]>.5)).sum()),obstacle_disappeared_cells=int(((decode(bevs[1])[...,1]>=.22)&(decode(b0)[...,1]<.22)).sum()),alpha_cap_rays=int((ac['cap']>0).sum()+(cc['cap']>0).sum()))
  records[1]['remove_foliage_and_shrub_counterfactual']=cf
  # Legacy sensor height separates new mounting estimate from vegetation geometry.
  ol,dl,rl=airy(legacy=True);legacy=scene.cast(ol,dl,rl,pose);lp,ll=points(ol,dl,legacy);lb,lm=raster(lp,ll,prior,pose,base['local'])
  records[1]['airy_height_only_changed_cells']=int(np.any(lb!=ab,axis=-1).sum())
  # Opaque cutout-card ablation, captures alpha impact without changing geometry.
  ao=scene.cast(oa,da,ra,pose,alpha=False);op,olabel=points(oa,da,ao);ob,om=raster(op,olabel,prior,pose,base['local'])
  records[1]['airy_opaque_vs_alpha_changed_cells']=int(np.any(ob!=ab,axis=-1).sum())
  fig,axs=plt.subplots(3,5,figsize=(17,10),layout='constrained');imgs=[decode(base['clean_bev'][k]),decode(base['noisy_bev'][k]),*[decode(x) for x in bevs]];names=['Training clean','Training noisy','Rays pitch 0','Rays pitch -15','Rays pitch -30']
  for j,(im,title) in enumerate(zip(imgs,names)):
   known=im[...,3]<.5
   for row,ch,vmin,vmax,cmap in [(0,0,-1,1,'terrain'),(1,1,0,1,'magma'),(2,2,0,1,'viridis')]:
    axs[row,j].imshow(np.where(known,im[...,ch],np.nan),origin='lower',extent=(-4,4,-2,6),vmin=vmin,vmax=vmax,cmap=cmap);axs[row,j].set_title(f'{title}: '+['ground [m]','obstacle [m]','support'][row]);axs[row,j].set_xlabel('Left [m]');axs[row,j].set_ylabel('Forward [m]')
  fig.suptitle(f'LWY s={station:g} m | gray/white = unknown | 0.125 m cells');fig.savefig(ROOT/f'figures/r8_p0_bev_{int(station):04d}.png',dpi=140);plt.close(fig)
  fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
  for ax,im,pitch in zip(axs,cams,[0,-15,-30]):
   artist=ax.imshow(im,vmin=.5,vmax=10,cmap='turbo');ax.set_title(f'Gemini 1280x800, pitch {pitch} deg');ax.set_xlabel('pixel u');ax.set_ylabel('pixel v')
  fig.colorbar(artist,ax=axs,label='Optical Z [m]');fig.suptitle(f'LWY s={station:g} m');fig.savefig(ROOT/f'figures/r8_p0_depth_{int(station):04d}.png',dpi=140);plt.close(fig)
  fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');m=masks[1]
  for ax,key in zip(axs,['soft','shrub','hard']):ax.imshow(m[key],origin='lower',extent=(-4,4,-2,6));ax.set_title(key+' returns');ax.set_xlabel('Left [m]');ax.set_ylabel('Forward [m]')
  fig.suptitle(f's={station:g}m, fused pitch -15, semantic audit only');fig.savefig(ROOT/f'figures/r8_p0_semantic_{int(station):04d}.png',dpi=120);plt.close(fig)
  print('STATION',station,'seconds',time.perf_counter()-t,'caps',sum(r['alpha_cap_rays'] for r in records),'known',[r['clean']['ray_known_cells'] for r in records],flush=True)
  dump(ROOT/'metrics/r8_p0_stations.json',json_ready(dict(status='in_progress',rows=rows)))
 np.savez_compressed(GEO/'r8_sensor_bevs.npz',**all_arrays)
 dump(ROOT/'metrics/r8_p0_stations.json',json_ready(dict(status='complete',seconds=time.perf_counter()-t0,bvh_build_seconds=scene.build_seconds,rows=rows,ground_estimator='per-cell minimum; nearest-fill for estimation only; 5x5 morphological opening; pointless cells remain unknown',support='same thresholds and finite differences as FastEnv._frame; no semantic oracle',memory='same first-frame world-aligned max scatter; no temporal rollout',obstacle_threshold_m=.22)))
 mon.close()
if __name__=='__main__':main()
