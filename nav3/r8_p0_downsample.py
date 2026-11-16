"""Exact ray subsets from stored full-resolution returns; no new ray workload."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p0_common import ROOT,GEO,dump,json_ready
from .r8_p0_ray import airy,camera
from .r8_p0_measure import raster,comparison

def main():
 base=np.load(GEO/'r8_baseline.npz');full=np.load(GEO/'r8_sensor_bevs.npz');rows=[]
 for stride in [2,4,8]:
  oa,da,ra=airy(stride,stride);oc,dc,rc,cosz,shape=camera(-15,stride)
  for k,s in enumerate(base['stations']):
   a=np.load(GEO/f'r8_airy_{int(s):04d}.npz');c=np.load(GEO/f'r8_gemini_{int(s):04d}_pitch-15.npz');labels=np.zeros((2,96,900),np.int32);labels[a['distance']>0]=a['semantic'];dist=a['distance'][:,::stride,::stride].ravel();sem=labels[:,::stride,::stride].ravel();good=dist>0;p=oa[good]+da[good]*dist[good,None];ls=sem[good]
   z=c['depth_m'][::stride,::stride].ravel();good=np.isfinite(z);cp=oc[good]+dc[good]*(z[good]/cosz[good])[:,None];cl=c['semantic'][::stride,::stride].ravel()[good]
   bev,m=raster(np.r_[p,cp],np.r_[ls,cl],base['clean_decoded'][k,...,4],base['poses'][k],base['local']);rec=comparison(full[f'bev_{int(s)}_-15'],bev,m);rows.append(dict(station_m=float(s),stride=stride,**rec))
 aggregates=[]
 for stride in [2,4,8]:
  rs=[r for r in rows if r['stride']==stride];aggregates.append(dict(stride=stride,**{key:float(np.mean([r[key] for r in rs])) for key in ['ray_known_cells','reference_known_cells','obstacle_iou_all','support_iou_all','height_mae_m']}))
 dump(ROOT/'metrics/r8_p0_downsample.json',json_ready(dict(method='Decimate exact full-return arrays at stride in both beam/azimuth and image u/v; same FOV, same alpha and range rules. No LOD.',rows=rows,aggregate=aggregates)))
 fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,key in zip(axs,['ray_known_cells','support_iou_all','height_mae_m']):ax.plot([r['stride'] for r in aggregates],[r[key] for r in aggregates],'-o');ax.set(xlabel='Per-axis ray stride',ylabel=key);ax.grid(alpha=.2)
 fig.suptitle('Exact ray-subset degradation vs full resolution, 10 stations');fig.savefig(ROOT/'figures/r8_p0_downsample.png',dpi=160);plt.close(fig);print(json.dumps(aggregates,indent=2),flush=True)
if __name__=='__main__':main()
