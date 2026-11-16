"""Full 96x900 dual-Air y ray reference with the measured extrinsic chain.

Only the copied ray generator/runtime are imported. The original scene is read
from exp_nav3/ray_scene; mounts are patched in memory, never in reference dirs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import minimum_filter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/map'))
from .calibration import mounts

_model = _data = _scene = None


def init_ray():
    global _model,_data,_scene
    import mujoco
    from runtime import load_scene
    _scene=load_scene(ROOT/'ray_scene')
    _model=mujoco.MjModel.from_xml_path(str(ROOT/'ray_scene/scene.xml'))
    _data=mujoco.MjData(_model)
    # Existing body origin is ground, whereas calibration base origin is 0.54 m AGL.
    for k,name in enumerate(['front','rear']):
        _model.site_pos[_model.site(f'lidar_{name}').id]=mounts()[k,:3,3]+[0,0,.54]
        _model.geom_pos[_model.geom(f'airy_{name}').id]=mounts()[k,:3,3]+[0,0,.54]


def ray_one(item):
    import mujoco
    from runtime import place_body
    from sensors import rays
    index,feature,s=item
    out=ROOT/'metrics/rays'
    out.mkdir(exist_ok=True)
    path=out/f'pose_{index:03d}.npz'
    if path.exists():
        return json.loads(path.with_suffix('.json').read_text())
    before=time.monotonic()
    pose=place_body(_model,_data,_scene,s)
    body=_model.body('m1').id
    rot=_data.xmat[body].reshape(3,3)
    native,ch,az,gap=rays(1,1,index)
    clouds=[];geomids=[];counts=[]
    groups=np.array([1,1,0,0,0,0],np.uint8)
    for k,name in enumerate(['front','rear']):
        # Native +z is the hemisphere axis. Full measured rotation is retained.
        directions=np.ascontiguousarray(native@mounts()[k,:3,:3].T@rot.T)
        origin=_data.site_xpos[_model.site(f'lidar_{name}').id].copy()
        ids=np.empty(len(native),np.int32);ranges=np.empty(len(native),np.float64)
        mujoco.mj_multiRay(_model,_data,origin,directions.ravel(),groups,True,-1,ids,ranges,None,len(native),60.)
        self_hit=(ids>=0)&(_model.geom_bodyid[np.maximum(ids,0)]==body)
        valid=(ranges>=.1)&(ranges<=60)&(~gap)&(~self_hit)
        clouds.append(origin+directions[valid]*ranges[valid,None])
        geomids.append(ids[valid]);counts.append(int(valid.sum()))
    points=np.concatenate(clouds);ids=np.concatenate(geomids)
    # Bird's-eye navigation frame is gravity aligned, yaw only (not camera pitch).
    cy,sy=np.cos(pose[3]),np.sin(pose[3])
    r=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    local=(points-pose[:3])@r
    ij=np.floor((local[:,:2]+[2,4])/.125).astype(int)
    keep=np.all((ij>=0)&(ij<64),axis=-1)
    local,ij,ids=local[keep],ij[keep],ids[keep]
    flat=ij[:,0]*64+ij[:,1]
    count=np.bincount(flat,minlength=4096).reshape(64,64)
    zmin=np.full(4096,np.inf);zmax=np.full(4096,-np.inf)
    np.minimum.at(zmin,flat,local[:,2]);np.maximum.at(zmax,flat,local[:,2])
    zmin=zmin.reshape(64,64);zmax=zmax.reshape(64,64)
    # Keep raw return height; height error is against terrain only in terrain-hit cells.
    names=np.array([_model.geom(int(i)).name for i in ids])
    masks={}
    for label,prefix in [('rail',('rail_post','rail_bar')),('tape',('suspended_black_tape',)),('log',('fallen_log','bark'))]:
        hits=np.array([n.startswith(prefix) for n in names])
        mask=np.zeros(4096,bool);mask[flat[hits]]=True;masks[label]=mask.reshape(64,64)
    terrain=np.array([n=='terrain' or n.startswith('stone_tread') for n in names])
    ground=np.full(4096,np.inf)
    np.minimum.at(ground,flat[terrain],local[terrain,2])
    ground=ground.reshape(64,64)
    np.savez_compressed(path,pose=pose,s=np.float32(s),count=count,zmin=zmin,zmax=zmax,ground=ground,**masks)
    rec=dict(index=index,feature=feature,s_m=float(s),rays_per_sensor=len(native),returns=counts,
             seconds=time.monotonic()-before,pose=pose.tolist())
    path.with_suffix('.json').write_text(json.dumps(rec,indent=2))
    print(json.dumps(rec),flush=True)
    return rec


def reference(workers):
    import yaml
    meta=yaml.safe_load((ROOT/'ray_scene/route.yaml').read_text())
    feats=meta['features']
    centers=[('narrow',feats['narrow']),('stairs',feats['stairs_up']),('slope',48.),
             ('fork_tape',feats['tape_view']),('log',feats['log']),('bend',feats['bend'])]
    items=[(i*10+j,name,float(s+off)) for i,(name,s) in enumerate(centers) for j,off in enumerate(np.linspace(-3.8,2.,10))]
    with ProcessPoolExecutor(workers,initializer=init_ray) as pool:
        rows=list(pool.map(ray_one,items))
    (ROOT/'metrics/ray_reference.json').write_text(json.dumps(dict(poses=60,mounts='calibrated',rows=rows),indent=2))


def compare():
    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .bank import MapBank
    from .env import FastEnv,decode,rotate
    reference=json.loads((ROOT/'metrics/ray_reference.json').read_text())
    bank=MapBank('test');n=len(reference['rows'])
    env=FastEnv(bank,n,config={'base_height_jitter':0.,'position_walk_std':0.,'yaw_walk_std':0.,'max_observation_delay_steps':0})
    s=jnp.array([r['s_m'] for r in reference['rows']])
    state,obs,extra=env.reset(jax.random.PRNGKey(4096),jnp.zeros(n,jnp.int32),s)
    poses=np.array([r['pose'] for r in reference['rows']])
    state['pos']=jnp.array(poses[:,:2],jnp.float32);state['yaw']=jnp.array(poses[:,3],jnp.float32)
    state['height']=jnp.full(n,.54)
    measured,seen,visible,trav,rd=jax.jit(env._frame)(state,jax.random.PRNGKey(44))
    fast=np.array(measured);seen=np.array(seen);visible=np.array(visible)
    xy=state['pos'][:,None,None]+rotate(env.local[None],state['yaw'])
    truth,_=env.truth(state,xy)
    truth=np.array(truth)
    rows=[]
    for i,r in enumerate(reference['rows']):
        ray=np.load(ROOT/'metrics/rays'/f'pose_{i:03d}.npz')
        known=ray['count']>0
        common=np.isfinite(ray['ground'])&seen[i]
        height_error=np.abs(ray['ground']-fast[i,...,0])
        union=(~known)|(~seen[i]);inter=(~known)&(~seen[i])
        rec=dict(index=i,feature=r['feature'],s_m=r['s_m'],ray_unknown=float((~known).mean()),
            fast_unknown=float((~seen[i]).mean()),shadow_iou=float(inter.sum()/max(1,union.sum())),
            height_common_cells=int(common.sum()),height_abs_error_sum=float(height_error[common].sum()),
            height_mae_m=float(height_error[common].mean()) if common.any() else None)
        for name,code in [('rail',3),('tape',4),('log',2)]:
            target=truth[i,...,6]==code
            rec[name]=dict(target_cells=int(target.sum()),ray_detected_cells=int((ray[name]&target).sum()),
                fast_detected_cells=int((seen[i]&target&(fast[i,...,1]>.12)).sum()),
                ray_any=bool(ray[name].any()),fast_any=bool((seen[i]&target).any()))
        rows.append(rec)
        if i%10==4:
            fig,axes=plt.subplots(1,4,figsize=(13,3.5),layout='constrained')
            images=[np.where(known,ray['zmin'],np.nan),np.where(seen[i],fast[i,...,0],np.nan),
                    (~known).astype(float)-(~seen[i]).astype(float),np.array(trav[i],float)]
            titles=['Full dual Airy return height','Fast noisy height','Unknown: ray - fast','Privileged traversability']
            for ax,img,title in zip(axes,images,titles):
                ax.imshow(img,origin='lower',extent=(-4,4,-2,6),cmap='viridis' if 'Unknown' not in title else 'coolwarm')
                ax.set(title=title,xlabel='Left (m)',ylabel='Forward (m)')
            fig.suptitle(f'{r["feature"]}, s={r["s_m"]:.2f} m')
            fig.savefig(ROOT/'figures'/f'noise_{r["feature"]}.png',dpi=150);plt.close(fig)
    aggregate=dict(poses=n,ray_unknown=float(np.mean([r['ray_unknown'] for r in rows])),fast_unknown=float(np.mean([r['fast_unknown'] for r in rows])),
        shadow_iou=float(np.mean([r['shadow_iou'] for r in rows])),
        height_mae_m=sum(r['height_abs_error_sum'] for r in rows)/max(1,sum(r['height_common_cells'] for r in rows)))
    for name in ['rail','tape','log']:
        count=sum(r[name]['target_cells'] for r in rows)
        aggregate[name]=dict(target_cells=count,
            ray_cell_recall=sum(r[name]['ray_detected_cells'] for r in rows)/max(1,count),
            fast_cell_recall=sum(r[name]['fast_detected_cells'] for r in rows)/max(1,count))
    result=dict(aggregate=aggregate,rows=rows,
        limitations=['Single-frame comparison; moving accumulation not validated by this static pose set.',
                     'Heightfield top envelope makes suspended ribbon and rail opaque below their true height.',
                     'Fast model uses yaw-only body attitude; MuJoCo reference pitches with terrain.',
                     'Raster class includes stakes with tape; tape reference counts ribbon only.'])
    (ROOT/'metrics/noise_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(aggregate,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',action='store_true');ap.add_argument('--compare',action='store_true');ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args()
    if a.reference:reference(a.workers)
    if a.compare:compare()


if __name__=='__main__':main()
