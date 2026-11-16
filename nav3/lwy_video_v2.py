"""Same-seed policy inference in the original scene; kinematic MuJoCo rendering."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .lwy_comparison_runtime import setup,suffix,level
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/lwy'))
from trail_scene import Route


def record(method,checkpoint,reference_path=None):
    import jax
    import jax.numpy as jnp
    from .lwy_bank import LwyBank
    from .lwy_safe_obstacles import SafeLwyEnv
    from .lwy_policy_v2 import EpisodePolicy
    from .lwy_control_v2 import planning_result
    bank=LwyBank('test',root='maps_lwy_v2');env=SafeLwyEnv(bank,1,False,config=json.loads(Path('configs/lwy_v2_tracker.json').read_text()))
    policy=EpisodePolicy(method,[940000],checkpoint)
    reset=jax.jit(jax.vmap(lambda e,k,m:e._reset(k,m[None]),in_axes=(None,0,0)))
    step=jax.jit(jax.vmap(lambda e,s,a:e._step(s,a[None]),in_axes=(None,0,0)))
    state,obs,_=reset(env,jnp.array([jax.random.PRNGKey(940000)]),jnp.array([0]))
    frames=[];digest=hashlib.sha256()
    for tick in range(18002):
        action=policy(obs,lambda o:planning_result(o)[0])
        state,obs,_=step(env,state,action)
        pose=np.r_[np.asarray(state['pos'][0,0]),float(state['yaw'][0,0]),np.asarray(action[0])]
        digest.update(pose.astype(np.float32).tobytes())
        terminal=bool(obs['is_last'][0,0])
        if tick%3==0 or terminal:
            frames.append(dict(pos=pose[:2],yaw=pose[2],action=pose[3:],s=float(state['s'][0,0]),
                gates=int(state['gates'][0,0]),time=(tick+1)*.2,bev=np.asarray(obs['bev'][0,0]),terminal=terminal))
        if terminal:break
    assert terminal
    reference=json.loads(Path(reference_path or f'metrics/lwy_v2_final_{method}_original_ob0_sf0{suffix()}.json').read_text())['rows'][0]
    assert digest.hexdigest()==reference['action_pose_sha256'],'Video must reproduce the final-evaluation episode exactly'
    out=Path('videos');out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'lwy_v2_{method}{suffix()}.npz',**{k:np.array([r[k] for r in frames]) for k in frames[0]})
    metadata=dict(method=method,checkpoint=checkpoint,noise_seed=940000,policy_seed=1000940000,
        action_pose_sha256=digest.hexdigest(),failure=reference['failure'],gates=reference['gates'],
        progress_m=reference['progress_m'],duration_s=frames[-1]['time'],frames=len(frames),
        random_obstacles=False,safety_filter=False,rendering_only_no_leg_dynamics=True)
    (out/f'lwy_v2_{method}{suffix()}.json').write_text(json.dumps(metadata,indent=2));return frames,metadata


def render(rollouts,metas,tag='lwy_v2_dreamer_vs_ppo'):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    package=Path('../../ATEC_Env');xml=package/'scene/atec_park_trail_1km_mujoco.xml'
    model=mujoco.MjModel.from_xml_path(str(xml.resolve()))
    model.vis.quality.shadowsize=1024;model.vis.quality.offsamples=2
    model.stat.extent=10;model.vis.map.znear=.01;model.vis.map.zfar=200
    model.vis.global_.offwidth=720;model.vis.global_.offheight=420
    if 'validation_marker' in [model.body(i).name for i in range(model.nbody)]:
        marker=model.body('validation_marker').id;model.geom_rgba[model.geom_bodyid==marker,3]=0
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,height=420,width=720,max_geom=max(10000,model.ngeom+20))
    route=Route(json.loads((package/'config/track_spec.json').read_text()))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera);camera.distance=4.5;camera.elevation=-35
    count=max(map(len,rollouts));indices=np.unique(np.linspace(0,count-1,min(count,900)).astype(int))
    output=Path('videos')/f'{tag}.mp4';output.parent.mkdir(exist_ok=True)
    with imageio.get_writer(output,fps=15,codec='libx264',quality=7,macro_block_size=None) as writer:
        for index in indices:
            panels=[]
            for j,frames in enumerate(rollouts):
                f=frames[min(index,len(frames)-1)];xy=f['pos'];yaw=f['yaw'];z=route.height(f['s'])
                camera.lookat[:]=[*xy,z+.35];camera.azimuth=np.degrees(yaw)+145
                renderer.update_scene(data,camera=camera)
                g=renderer.scene.geoms[renderer.scene.ngeom];c,s=np.cos(yaw),np.sin(yaw)
                matrix=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_BOX,np.array([.515,.29,.15]),np.array([*xy,z+.3]),matrix.ravel(),np.array([.1,.65,.9,1],np.float32))
                renderer.scene.ngeom+=1
                panel=Image.new('RGB',(720,510),'#14222b');panel.paste(Image.fromarray(renderer.render()),(0,70));d=ImageDraw.Draw(panel)
                d.text((16,8),metas[j]['method']+' | filter OFF | same seed',font=font,fill='white')
                d.text((16,38),f"t={f['time']:.1f}s  s={f['s']:.2f}m  gates={f['gates']}/8"+('  TERMINATED' if f['terminal'] else ''),font=small,fill='#c6d8e2')
                d.text((16,490),'Original LWY XML; fast-env pose replay; cyan = 1.03 x 0.58 m envelope',font=small,fill='#c6d8e2')
                panels.append(np.asarray(panel))
            writer.append_data(np.concatenate(panels,axis=1))
    renderer.close()
    Path(f'metrics/{tag}.json').write_text(json.dumps(dict(path=str(output),rendered_frames=len(indices),fps=15,
        original_xml=str(xml),source_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),rollouts=metas,
        timeline='Every third 5 Hz policy step; uniformly subsampled to at most 900 frames; shorter rollout freezes at termination'),indent=2))
    print(str(output),flush=True)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--protocol-probe',action='store_true');p.add_argument('--selection',default='metrics/lwy_v2_model_selection.json');p.add_argument('--result-prefix',default='lwy_v2');a=p.parse_args()
    setup()
    if a.probe:
        route=Route(json.loads(Path('../../ATEC_Env/config/track_spec.json').read_text()))
        frames=[dict(pos=route.position(s)[:2],yaw=np.arctan2(*route.frame(s)[0][1::-1]),s=s,time=i*.2,gates=0,terminal=False) for i,s in enumerate([2.,2.1])]
        render([frames,frames],[{'method':'renderer probe'},{'method':'renderer probe'}],tag='lwy_v2_renderer_probe')
        return
    if a.protocol_probe:
        rows=[]
        for method in ['dreamer_s0','ppo_s0']:
            reference=f'metrics/lwy_v2_runtime_candidate_{method}_serial.json' if level()==1 else None
            _,meta=record(method,f'checkpoints/{method}/latest.pkl',reference);rows.append(meta)
        Path(f'metrics/lwy_v2_video_protocol{suffix()}.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
        print('Video recording reproduces both original final-evaluation episodes exactly',flush=True)
        return
    selected=json.loads(Path(a.selection).read_text())['selected']
    runs=[];metas=[]
    for method in ['dreamer_lwy','ppo_lwy']:
        frames,meta=record(method,selected[method]['checkpoint'],f'metrics/{a.result_prefix}_final_{method}_original_ob0_sf0{suffix()}.json');runs.append(frames);metas.append(meta)
    render(runs,metas,tag=a.result_prefix+'_dreamer_vs_ppo'+suffix())


if __name__=='__main__':main()
