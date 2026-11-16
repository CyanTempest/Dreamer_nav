"""Audit semantic/clearance attribution separately from the sensor-only BEV."""
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .r8_p0_common import ROOT,GEO,dump,json_ready
from .r8_p0_ray import camera
from .r8_p0_measure import raster,decode,SOFT,HARD

def main():
 base=np.load(GEO/'r8_baseline.npz');sensors=np.load(GEO/'r8_sensor_bevs.npz');stations=json.loads((ROOT/'metrics/r8_p0_stations.json').read_text())['rows'];rows=[]
 for k,station in enumerate(base['stations']):
  pose=base['poses'][k];a=np.load(GEO/f'r8_airy_{int(station):04d}.npz');c=np.load(GEO/f'r8_gemini_{int(station):04d}_pitch-15.npz');o,d,r,cosz,shape=camera(-15)
  depth=c['depth_m'].ravel();good=np.isfinite(depth);cp=o[good]+d[good]*(depth[good]/cosz[good])[:,None];p=np.r_[a['points_body'],cp];labels=np.r_[a['semantic'],c['semantic'].ravel()[good]]
  bev,m=raster(p,labels,base['clean_decoded'][k,...,4],pose,base['local'])
  assert np.array_equal(bev,sensors[f'bev_{int(station)}_-15']), 'Reconstructed optical-Z must reproduce BEV'
  ij=np.floor((p[:,:2]+[2,4])/.125).astype(int);keep=np.all((ij>=0)&(ij<64),1);ij=ij[keep];p=p[keep];labels=labels[keep];flat=ij[:,0]*64+ij[:,1]
  rel=p[:,2]-m['ground_estimate'][ij[:,0],ij[:,1]];bands={}
  for band,cond in [('low',(rel>=.22)&(rel<=.80)),('overhead',rel>.80),('near_ground',(rel>=0)&(rel<.22))]:
   for label,classes in [('soft',SOFT),('shrub',[9]),('hard',HARD),('branch',[15]),('rock',[5]),('rail',[3]),('trunk',[7]),('root',[6])]:
    mk=np.zeros(4096,bool);mk[flat[cond&np.isin(labels,classes)]]=1;bands[band+'_'+label]=mk.reshape(64,64)
  truth_obstacle=base['truth'][k,...,1]-base['truth'][k,...,0];known=base['clean_decoded'][k,...,3]<.5
  rec=dict(station_m=float(station),**{name+'_cells':int(mask.sum()) for name,mask in bands.items()},low_hard_missing_from_truth_cells=int((bands['low_hard']&(truth_obstacle<.22)).sum()),low_hard_missing_due_unknown_cells=int((bands['low_hard']&(truth_obstacle>=.22)&~known).sum()),low_hard_missing_from_clean_bev_cells=int((bands['low_hard']&((base['clean_decoded'][k,...,1]<.22)|~known)).sum()),soft_only_low_cells=int((bands['low_soft']&~bands['low_hard']).sum()),shrub_only_low_cells=int((bands['low_shrub']&~bands['low_hard']).sum()),overhead_hard_without_low_hard_cells=int((bands['overhead_hard']&~bands['low_hard']).sum()))
  rec['low_hard_missing_by_class']={cl:int((bands['low_'+cl]&(truth_obstacle<.22)).sum()) for cl in ['branch','rock','rail','trunk']}
  rows.append(rec)
  fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
  views=[bands['low_soft'],bands['low_shrub'],bands['low_hard']&(truth_obstacle<.22),bands['overhead_hard']&~bands['low_hard']];titles=['Flexible leaves/grass/fern: 0.22-0.80 m','Shrubs (mixed): 0.22-0.80 m','Hard low returns absent in map envelope','Hard overhead only (>0.80 m)']
  for ax,img,title in zip(axs,views,titles):ax.imshow(img,origin='lower',extent=(-4,4,-2,6),vmin=0,vmax=1,cmap='magma');ax.set(title=title,xlabel='Left [m]',ylabel='Forward [m]')
  fig.suptitle(f'LWY {station:g} m | height above sensor-estimated ground; audit, not collision truth');fig.savefig(ROOT/f'figures/r8_p0_clearance_{int(station):04d}.png',dpi=140);plt.close(fig)
 pitch=[]
 for p in [0,-15,-30]:
  rs=[r for r in stations if r['pitch_deg']==p];pitch.append(dict(pitch_deg=p,mean_camera_known_cells=float(np.mean([r['camera_known_cells'] for r in rs])),mean_camera_corridor_known_cells=float(np.mean([r['camera_corridor_known_cells'] for r in rs])),mean_fused_known_cells=float(np.mean([r['clean']['ray_known_cells'] for r in rs])),mean_fused_corridor_known_cells=float(np.mean([r['fused_corridor_known_cells'] for r in rs])),mean_added_cells=float(np.mean([r['camera_added_cells'] for r in rs])),mean_support_iou_clean=float(np.mean([r['clean']['support_iou_all'] for r in rs])),mean_height_mae_m=float(np.mean([r['clean']['height_mae_m'] for r in rs]))))
 selected=[r for r in stations if r['pitch_deg']==-15];aggregate={}
 for version in ['clean','noisy']:
  aggregate[version]={key:float(np.mean([r[version][key] for r in selected])) for key in ['reference_known_cells','ray_known_cells','obstacle_iou_all','support_iou_all','obstacle_iou_common','support_iou_common','height_mae_m']}
  aggregate[version]['channel_mae_all']=np.mean([r[version]['channel_mae_all'] for r in selected],0).tolist();aggregate[version]['channel_mae_common']=np.mean([r[version]['channel_mae_common'] for r in selected],0).tolist()
 aggregate['height_cell_weighted_mae_m']=sum(r['clean']['height_mae_m']*r['clean']['common_cells'] for r in selected)/sum(r['clean']['common_cells'] for r in selected)
 aggregate['terrain_height_cell_weighted_mae_m']=sum(r['terrain_truth_height_mae_m']*r['terrain_common_cells'] for r in selected)/sum(r['terrain_common_cells'] for r in selected)
 aggregate['clearance_counts']={key:sum(r[key] for r in rows) for key in rows[0] if key.endswith('_cells')}
 aggregate['hard_missing_class_counts']={key:sum(r['low_hard_missing_by_class'][key] for r in rows) for key in rows[0]['low_hard_missing_by_class']}
 aggregate['counterfactual']={key:sum(r['remove_foliage_and_shrub_counterfactual'][key] for r in selected) for key in selected[0]['remove_foliage_and_shrub_counterfactual']}
 aggregate['mean_legacy_height_changed_cells']=float(np.mean([r['airy_height_only_changed_cells'] for r in selected]));aggregate['mean_alpha_changed_cells']=float(np.mean([r['airy_opaque_vs_alpha_changed_cells'] for r in selected]))
 dump(ROOT/'metrics/r8_p0_analysis.json',json_ready(dict(rows=rows,pitch=pitch,aggregate=aggregate,clearance_band_m=[.22,.80],clearance_band_status='provisional 0.80 m upper bound; not measured swept body volume',semantic_limitations='Leaf, grass and fern classes are presumptively flexible; shrub contains wood and foliage and must not be declared passable; actual mechanical traversability unmeasured',counts_overlap='Classes can share cells; missing map envelope is not proof of physically absent collider')))
 fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained');x=base['stations']
 for label in ['clean','noisy']:
  axs[0,0].plot(x,[1-r[label]['reference_known_cells']/4096 for r in selected],'-o',label='Training '+label)
 axs[0,0].plot(x,[1-r['clean']['ray_known_cells']/4096 for r in selected],'-o',label='Full geometry');axs[0,0].set_ylabel('Unknown fraction');axs[0,0].legend()
 axs[0,1].plot(x,[r['clean']['support_iou_all'] for r in selected],'-o',label='Support IoU');axs[0,1].plot(x,[r['clean']['obstacle_iou_all'] for r in selected],'-o',label='Obstacle IoU');axs[0,1].legend();axs[0,1].set_ylim(0,1)
 axs[1,0].plot(x,[r['clean']['height_mae_m'] for r in selected],'-o',label='Sensor ground estimator');axs[1,0].plot(x,[r['terrain_truth_height_mae_m'] for r in selected],'-o',label='Terrain-hit geometry audit');axs[1,0].set_ylabel('Height MAE [m]');axs[1,0].legend()
 for key in ['low_soft_cells','low_shrub_cells','low_hard_missing_from_truth_cells']:axs[1,1].plot(x,[r[key] for r in rows],'-o',label=key)
 axs[1,1].legend(fontsize=8);axs[1,1].set_ylabel('BEV cells')
 for ax in axs.ravel():ax.set_xlabel('Station [m]');ax.grid(alpha=.2)
 fig.suptitle('Full resolution dual Airy + Gemini (-15 deg), 10 LWY zones');fig.savefig(ROOT/'figures/r8_p0_summary.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,key,title in zip(axs,['mean_camera_known_cells','mean_camera_corridor_known_cells','mean_fused_corridor_known_cells'],['Camera coverage','Camera route-prior coverage','Fused route-prior coverage']):ax.bar([str(r['pitch_deg']) for r in pitch],[r[key] for r in pitch]);ax.set(title=title,xlabel='Pitch [deg]',ylabel='Mean known cells')
 fig.savefig(ROOT/'figures/r8_p0_pitch.png',dpi=160);plt.close(fig);print(json.dumps(dict(pitch=pitch,aggregate=aggregate),indent=2),flush=True)
if __name__=='__main__':main()
