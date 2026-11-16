"""Record real checkpoint inference, then visualize fast-env poses in MuJoCo.

Rendering is kinematic only: this is not an A2 locomotion validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path('videos')

def record(name,batch=30,seed=9137,fixed=False):
    if fixed:
        from .fix_eval import setup
        setup()
    import pickle
    import jax
    import jax.numpy as jnp
    from .bank import MapBank
    from .env import FastEnv
    from .evaluate import FAILURES
    cfg=json.loads(Path(f'metrics/config_{name}.json').read_text())
    checkpoint=Path(f'checkpoints/{name}/latest.pkl')
    with checkpoint.open('rb') as f:data=pickle.load(f)
    if fixed:
        from .diag_rollout import load_policy
        policy=load_policy(name)
    elif name.startswith('dreamer'):
        from .dreamer_adapter import DreamerPolicy
        policy=DreamerPolicy(cfg['seed'],f'checkpoints/{name}',not cfg['no_head'],cfg['amnesic'])
        policy.agent.load(data['agent'])
    else:
        from .ppo import PPOPolicy
        policy=PPOPolicy(cfg['seed'])
        policy.params=jax.tree.map(jnp.asarray,data['policy'][0])
    del data
    policy.carry=None;policy.key=jax.random.PRNGKey(seed)
    bank=MapBank('test')
    env=FastEnv(bank,batch,False)
    state,obs,_=env.reset(jax.random.PRNGKey(seed),jnp.zeros(batch,jnp.int32))
    frames=[]
    def capture(action,t,terminal=False):
        frames.append(dict(pos=np.asarray(state['pos'][0]),yaw=float(state['yaw'][0]),
            s=float(state['s'][0]),gates=int(state['gates'][0]),time=t,
            action=np.asarray(action),bev=np.asarray(obs['bev'][0]),terminal=terminal))
    for tick in range(env.cfg['max_steps']+2):
        action=policy(obs)
        if tick%3==0:capture(action[0],tick*env.cfg['dt'])
        state,obs,_=env.step(state,action)
        if bool(obs['is_last'][0]):
            capture(np.zeros(3),(tick+1)*env.cfg['dt'],True)
            break
    assert frames[-1]['terminal'], 'Incomplete rollout'
    np.savez_compressed(OUT/f'{name}_rollout.npz',**{k:np.asarray([r[k] for r in frames]) for k in frames[0]})
    result=dict(method=name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        seed=seed,batch=batch,episode=0,random_obstacles=False,safety_filter=False,
        gates=int(state['gates'][0]),failure=FAILURES[int(state['failure'][0])],
        duration_s=float(frames[-1]['time']),progress_m=float(state['s'][0]),
        frames=len(frames),scope='Fast-env inference; MuJoCo rendering only, no leg physics')
    (OUT/f'{name}_rollout.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

def render(names=None):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import map_coordinates
    sys.path.insert(0,str(Path('vendor/map').resolve()))
    from runtime import load_scene, local_lighting
    sc=load_scene(Path('ray_scene').resolve())
    model=mujoco.MjModel.from_xml_path(str(Path('ray_scene/scene.xml').resolve()))
    model.vis.quality.shadowsize=2048
    model.vis.quality.offsamples=2
    body=model.body('m1').id
    model.geom_rgba[model.geom_bodyid==body,3]=0
    proxy=model.geom('m1_chassis').id
    model.geom_size[proxy]=[.515,.29,.15]
    model.geom_rgba[proxy]=[.12,.65,.92,1]
    model.geom_rgba[model.geom('m1_front').id]=[1,.65,.05,1]
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=500,width=800)
    grid=np.load('maps/mini_trail/map.npz')
    ground=grid['layers'][...,0].astype(np.float32);origin=grid['origin']
    def z_at(pos):
        ij=(pos-origin)/.1
        return float(map_coordinates(ground,ij[:,None],order=1,mode='nearest')[0])
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,22);small=ImageFont.truetype(fontpath,16)
    names=names or ['dreamer_s0','ppo_s0']
    rolls=[np.load(OUT/f'{n}_rollout.npz') for n in names]
    metas=[json.loads((OUT/f'{n}_rollout.json').read_text()) for n in names]
    count=max(len(r['time']) for r in rolls)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.distance=5.5;camera.elevation=-38
    def panel(j,index):
        r=rolls[j];i=min(index,len(r['time'])-1)
        pos=r['pos'][i];yaw=float(r['yaw'][i]);s=float(r['s'][i]);z=z_at(pos)
        data.mocap_pos[0]=[*pos,z+.025]
        data.mocap_quat[0]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        mujoco.mj_forward(model,data)
        local_lighting(model,sc,s)
        camera.lookat[:]=[*pos,z+.35]
        camera.azimuth=np.rad2deg(yaw)+145
        renderer.update_scene(data,camera=camera)
        # A cyan marker shows the local target commanded by the learned policy.
        if not r['terminal'][i]:
            a=r['action'][i];radius=.5+1.25*(a[0]+1);theta=yaw+a[1]*np.pi/3
            target=pos+radius*np.array([np.cos(theta),np.sin(theta)])
            g=renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.10,.10,.10]),
                np.array([*target,z_at(target)+.18]),np.eye(3).ravel(),np.array([0.,1.,.7,1.],np.float32))
            renderer.scene.ngeom+=1
        frame=Image.new('RGB',(800,640),'#101820');frame.paste(Image.fromarray(renderer.render()),(0,78))
        d=ImageDraw.Draw(frame)
        title='DreamerV3' if j==0 else 'PPO + GRU'
        d.text((18,10),title+'  |  checkpoint inference',font=font,fill='white')
        d.text((18,43),f"t={r['time'][i]:.1f}s   route={s:.1f}m   gates={r['gates'][i]}/8   filter OFF",font=small,fill='#b9d7e8')
        b=r['bev'][i][::-1,::-1]
        rgb=np.zeros((64,64,3),np.uint8)+35
        known=b[...,3]<128
        h=b[...,0].astype(float)/255
        rgb[known]=np.stack([70+80*h,95+70*h,80+70*h],-1)[known]
        rgb[(b[...,4]>127)&known]=[100,160,105]
        rgb[(b[...,1]>28)&known]=[240,130,60]
        inset=Image.fromarray(rgb).resize((160,160),Image.Resampling.NEAREST)
        frame.paste(inset,(626,390));d.rectangle((625,389,787,551),outline='white',width=2)
        d.text((627,365),'Policy BEV input',font=small,fill='white')
        d.ellipse((702,506,710,514),fill='#25baf3')
        status='RUNNING'
        if r['terminal'][i]:status=f"END: {metas[j]['failure'].upper()} | {metas[j]['gates']}/8 gates"
        d.text((18,584),status,font=font,fill='#ffba78' if r['terminal'][i] else '#87dfb4')
        d.text((18,617),'Fast-env proxy / no leg physics  |  MuJoCo display  |  7.2x',font=small,fill='#c6d2dc')
        return np.asarray(frame)
    opts=dict(fps=12,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart'])
    with imageio.get_writer(OUT/'mini_trail_comparison.mp4',**opts) as combined, imageio.get_writer(OUT/'dreamer_inference.mp4',**opts) as w0,imageio.get_writer(OUT/'ppo_inference.mp4',**opts) as w1:
        for i in range(count+36):
            a,b=panel(0,i),panel(1,i)
            combined.append_data(np.concatenate([a,b],axis=1))
            w0.append_data(a);w1.append_data(b)
            if i in (0,count//2,count-1):Image.fromarray(np.concatenate([a,b],1)).save(OUT/f'preview_{i}.jpg')
            if i%100==0:print(json.dumps(dict(frame=i,total=count+36)),flush=True)
    renderer.close()
    print('Video rendering complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',choices=['dreamer_s0','ppo_s0','dreamer_s0_ft','ppo_s0_ft']);p.add_argument('--render',action='store_true')
    p.add_argument('--output',default='videos');p.add_argument('--batch',type=int,default=30);p.add_argument('--seed',type=int,default=9137)
    p.add_argument('--fixed',action='store_true');p.add_argument('--finetuned',action='store_true')
    args=p.parse_args();OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
    if args.record:record(args.record,args.batch,args.seed,args.fixed)
    if args.render:render(['dreamer_s0_ft','ppo_s0_ft'] if args.finetuned else None)
