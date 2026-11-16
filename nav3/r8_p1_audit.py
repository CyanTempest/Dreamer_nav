"""Quantify additional reduction costs, alpha holes and reference residuals."""
import json,time
import numpy as np
from .r8_p1_common import ROOT,GEO,dump,Monitor
from .r8_p1_online import Atlas,fitted_config
from .r8_p1_validate import run,metrics,summary,dec
from .lwy_bank import LwyBank

def main():
    monitor=Monitor('audit');t=time.perf_counter();a=np.load(GEO/'r8_p1_atlas_alpha.npz');tiles=a['tiles'].copy()
    raw=np.load(GEO/'r8_p1_raw_alpha.npz')['surfaces'];opaque=np.load(GEO/'r8_p1_raw_opaque.npz')['surfaces']
    audit=[]
    for b in range(6):
        hit=np.isfinite(raw[...,2+b*2]);ohit=np.isfinite(opaque[...,2+b*2]);both=hit&ohit
        audit.append(dict(band=b,alpha_occupied_columns=int(hit.sum()),opaque_occupied_columns=int(ohit.sum()),opaque_added=int((ohit&~hit).sum()),alpha_only=int((hit&~ohit).sum()),
            common_surface_max_mae_m=float(np.abs(raw[...,2+b*2]-opaque[...,2+b*2])[both].mean()),
            dilated_footprint_columns=int((tiles[...,3+b*3]>.05).sum())))
        tiles[...,1+b*3]=np.where(hit,raw[...,1+b*2]-tiles[...,0],-1)
        tiles[...,2+b*3]=np.where(hit,raw[...,2+b*2]-tiles[...,0],-1)
        tiles[...,3+b*3]=hit
    path=GEO/'r8_p1_atlas_point.npz'
    np.savez_compressed(path,tiles=tiles,lookup=a['lookup'],origin=a['origin'],res=a['res'],edges=a['edges'],tile_indices=a['tile_indices'])
    del raw,opaque,tiles
    refs=np.load(GEO/'r8_p1_references.npz');preds=np.load(GEO/'r8_p1_predictions.npz');bank=LwyBank('test',limit=1,root='maps_lwy')
    atlas=Atlas();config=fitted_config();arrays={}
    for name,cfg in {'no_beam_sampling':{'p1_beam_sampling':False},'static_ground':{'p1_visible_ground':False},'no_online_opening':{'p1_ground_open':False},'no_canopy_above_2m':{'p1_max_band':3},'dense_low_horizon':{'p1_block_coverage':.7}}.items():
        arrays[name]=run(bank,atlas,refs,{**config,**cfg});print('ABLATION',name,flush=True)
    arrays['point_crop']=run(bank,Atlas(variant='point'),refs,config)
    sums={split:{name:summary([metrics(p,gt) for p,gt,s in zip(arr,refs['reference'],refs['split']) if split=='all' or s==split]) for name,arr in arrays.items()} for split in ['station','probe','holdout','all']}
    residual=[]
    for i,(p,r) in enumerate(zip(preds['simple'],refs['reference'])):
        p,r=dec(p),dec(r);pk=p[...,3]<.5;rk=r[...,3]<.5;common=pk&rk
        support=r[...,2]>.5;err=p[...,0]-r[...,0]
        residual.append(dict(index=i,station=float(refs['stations'][i]),split=str(refs['split'][i]),unknown_false_positive=int((~pk&rk).sum()),unknown_false_negative=int((pk&~rk).sum()),
            reference_support_height_mae_m=float(np.abs(err[common&support]).mean()) if (common&support).any() else None,
            reference_nonsupport_height_mae_m=float(np.abs(err[common&~support]).mean()),
            signed_ground_error_m=float(err[common].mean()),ground_error_p90_m=float(np.percentile(np.abs(err[common]),90))))
    # Numerical costs of reductions independent of semantic labels.
    arr=np.load(GEO/'r8_p1_atlas_alpha.npz')['tiles'];vals=arr[...,1:19:3];tops=arr[...,2:19:3]
    spans=[(tops[...,b]-vals[...,b])[vals[...,b]>=0] for b in range(6)]
    spanstats=[dict(band=b,mean_span_m=float(x.mean()),p95_span_m=float(np.percentile(x,95)),max_span_m=float(x.max())) for b,x in enumerate(spans)]
    dump(ROOT/'metrics/r8_p1_reduction_audit.json',dict(alpha_raw=audit,span_stats=spanstats,extra_ablations=sums,residual=residual,seconds=time.perf_counter()-t,point_atlas_bytes=path.stat().st_size,
        warning='Min/max only records a height interval, not every interior hole. Main visibility uses terrain horizon; foliage self-occlusion is unresolved.'))
    np.savez_compressed(GEO/'r8_p1_extra_predictions.npz',**arrays)
    print('AUDIT DONE',flush=True);monitor.close()

if __name__=='__main__':main()
