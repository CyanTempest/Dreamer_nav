"""RoboSense Airy 96 geometry sampled by MuJoCo mj_multiRay (not height marching)."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image,ImageDraw
import mujoco
from scene import ROOT,write_ply
from runtime import load_scene,place_body,local_lighting
from render import caption

SPEC=json.loads((ROOT/'sources/airy_specs.json').read_text())
COLORS={'front':np.array([31,205,231],np.uint8),'rear':np.array([255,117,61],np.uint8)}


def rays(beam_stride=1,azimuth_stride=1,frame=0):
    channels=np.arange(0,96,beam_stride)
    azimuth=np.arange(0,900,azimuth_stride)*.4
    ch,az=np.meshgrid(channels,azimuth,indexing='ij')
    el=np.deg2rad(np.array(SPEC['nominal_vertical_angles_deg'])[ch]);a=np.deg2rad(az)
    native=np.stack([np.cos(el)*np.cos(a),np.cos(el)*np.sin(a),np.sin(el)],-1).reshape(-1,3)
    gap=(frame%10==9)&(az.ravel()>=180)&(az.ravel()<212)
    return native,ch.ravel(),az.ravel(),gap


def scan_pair(model,data,beam_stride=1,azimuth_stride=1,frame=0):
    native,ch,az,gap=rays(beam_stride,azimuth_stride,frame)
    rot=data.xmat[model.body('m1').id].reshape(3,3)
    # Include chassis self-occlusion, exclude only the emitting sensor housings.
    groups=np.array([1,1,0,0,0,0],np.uint8)
    output={}
    for name,side in [('front',1),('rear',-1)]:
        local=np.c_[side*native[:,2],side*native[:,1],-native[:,0]]
        directions=np.ascontiguousarray(local@rot.T)
        origin=data.site_xpos[model.site(f'lidar_{name}').id].copy()
        ids=np.empty(len(directions),np.int32);ranges=np.empty(len(directions),np.float64)
        before=time.time()
        mujoco.mj_multiRay(model,data,origin,directions.ravel(),groups,True,-1,ids,ranges,None,len(directions),60.)
        self_hit=(ids>=0)&(model.geom_bodyid[np.maximum(ids,0)]==model.body('m1').id)
        valid=(ranges>=.1)&(ranges<=60)&(~gap)&(~self_hit)
        points=origin+directions[valid]*ranges[valid,None]
        names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(i)) or '' for i in ids[valid]]
        tape=sum(n.startswith('suspended_black_tape') for n in names)
        output[name]=dict(points=points,origin=origin,directions=directions,range=ranges,geomid=ids,valid=valid,
                          channel=ch,azimuth_deg=az,gap=gap,self_hits=int(self_hit.sum()),tape_hits=tape,ray_seconds=time.time()-before)
        print(f'{name}: {len(directions)} rays, {len(points)} returns, {tape} ribbon hits, {time.time()-before:.2f}s',flush=True)
    return output


def overlay(model,data,rgb,depth,clouds,camera_name):
    camera=model.camera(camera_name).id
    R=data.cam_xmat[camera].reshape(3,3);origin=data.cam_xpos[camera]
    h,w=rgb.shape[:2];f=h/(2*np.tan(np.deg2rad(model.cam_fovy[camera]/2)))
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    counts={}
    for name,cloud in clouds.items():
        pc=(cloud['points']-origin)@R;z=-pc[:,2]
        good=z>.1
        pc=pc[good];z=z[good]
        uv=np.c_[w/2+f*pc[:,0]/z,h/2-f*pc[:,1]/z]
        ij=np.rint(uv).astype(int)
        inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
        ij,z=ij[inside],z[inside]
        visible=np.abs(depth[ij[:,1],ij[:,0]]-z)<.10
        ij=ij[visible];counts[name]=len(ij)
        for x,y in ij:draw.rectangle((int(x),int(y),int(x)+1,int(y)+1),fill=tuple(COLORS[name]))
    return np.asarray(im),counts


def cloud_comparison(full,small,pose,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('ZYX',[pose[3],-pose[4],0]).as_matrix()
    fig,axes=plt.subplots(1,2,figsize=(14,7),sharex=True,sharey=True,facecolor='#101b19')
    for ax,clouds,title in zip(axes,[small,full],['Preview: 24 channels x 180 azimuths / sensor','Full: 96 channels x 900 azimuths / sensor']):
        ax.set_facecolor('#101b19')
        for name,v in clouds.items():
            p=(v['points']-pose[:3])@R
            keep=(np.abs(p[:,0])<13)&(np.abs(p[:,1])<9)
            ax.scatter(p[keep,0],p[keep,1],s=.6,alpha=.65,color=COLORS[name]/255,label=f'{name}: {len(v["points"]):,} returns')
        ax.add_patch(Rectangle((-.465,-.24),.93,.48,facecolor='#f6ead2'))
        ax.arrow(.7,0,1,0,width=.025,color='#f6ead2')
        ax.set(xlim=(-13,13),ylim=(-9,9),aspect='equal',xlabel='body x / forward (m)',ylabel='body y / left (m)',title=title)
        ax.tick_params(colors='#d5e1d9');ax.xaxis.label.set_color('#d5e1d9');ax.yaxis.label.set_color('#d5e1d9');ax.title.set_color('#e7e4ce')
        ax.grid(alpha=.12);ax.legend(loc='upper right',markerscale=5,fontsize=8)
    fig.suptitle('Same pose | MuJoCo ray intersections | cyan: front Airy / orange: rear Airy',color='#e7e4ce',fontsize=13)
    fig.tight_layout();fig.savefig(out/'19_airy_full_vs_preview.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--map',type=Path,default=ROOT/'maps/mini_trail');ap.add_argument('--s',type=float);ap.add_argument('--frame',type=int,default=0)
    args=ap.parse_args();out=ROOT/'renders';pointdir=args.map/'pointclouds';pointdir.mkdir(exist_ok=True)
    scene=load_scene(args.map);m=mujoco.MjModel.from_xml_path(str(args.map/'scene.xml'));d=mujoco.MjData(m)
    s=args.s if args.s is not None else scene.features['fork_tape']
    local_lighting(m,scene,s)
    pose=place_body(m,d,scene,s)
    small=scan_pair(m,d,4,5,args.frame);full=scan_pair(m,d,1,1,args.frame)
    report=dict(source=SPEC['source'],pose_s_m=s,pose_xyz_yaw_pitch=pose.tolist(),frame=args.frame,
                nominal_full_rays_per_sensor=86400,preview_rays_per_sensor=4320,
                geometry='mj_multiRay nearest geometric hit, 0.1-60 m, ideal single return; no noise / reflectance / weather / motion distortion',
                mounts='Assumed horizontal forward/rear hemispheres, x=+/-0.465 m, z=0.54 m; not factory calibration',scans={})
    for label,clouds in [('preview',small),('full',full)]:
        p=[];colors=[]
        for name,v in clouds.items():
            write_ply(pointdir/f'airy_{name}_{label}.ply',v['points'],np.tile(COLORS[name],(len(v['points']),1)))
            np.savez_compressed(pointdir/f'airy_{name}_{label}.npz',**v)
            p.append(v['points']);colors.append(np.tile(COLORS[name],(len(v['points']),1)))
            report['scans'][f'{name}_{label}']={k:v[k] for k in ['self_hits','tape_hits','ray_seconds']}
            report['scans'][f'{name}_{label}'].update(rays=len(v['range']),returns=len(v['points']))
        write_ply(pointdir/f'airy_pair_{label}.ply',np.concatenate(p),np.concatenate(colors))
    with mujoco.Renderer(m,height=720,width=1280) as r:
        opt=mujoco.MjvOption();opt.sitegroup[:]=0
        for camera,prefix in [('front_rgb','16_front'),('rear_rgb','18_rear')]:
            r.update_scene(d,camera=camera,scene_option=opt)
            rgb=r.render().copy()
            Image.fromarray(caption(rgb,camera.upper(),f'Same pose s = {s:.1f} m | camera height 0.555 m | RGB')).save(out/f'{prefix}_rgb.png')
            r.enable_depth_rendering();depth=r.render().copy();r.disable_depth_rendering()
            np.savez_compressed(pointdir/f'{camera}_calibration.npz',position=d.cam_xpos[m.camera(camera).id],rotation=d.cam_xmat[m.camera(camera).id].reshape(3,3),fovy=m.cam_fovy[m.camera(camera).id],width=1280,height=720,depth_m=depth)
            for label,clouds in [('preview',small),('full',full)]:
                over,counts=overlay(m,d,rgb,depth,clouds,camera)
                title=f'{camera.upper()} + AIRY / {label.upper()}'
                detail=f'Same pose s = {s:.1f} m | cyan FRONT / orange REAR | visible projected hits {counts}'
                Image.fromarray(caption(over,title,detail)).save(out/f'{prefix}_overlay_{label}.png')
                report[f'{camera}_{label}_projected']=counts
    cloud_comparison(full,small,pose,out)
    # Independent single-ray agreement check: rules out a bad vector layout.
    errors=[];ids=np.zeros(1,np.int32);groups=np.array([1,1,0,0,0,0],np.uint8)
    for v in full.values():
        for j in np.linspace(0,len(v['range'])-1,128,dtype=int):
            hit=mujoco.mj_ray(m,d,v['origin'],v['directions'][j],groups,True,-1,ids)
            errors.append(abs(hit-v['range'][j]))
    report['mj_ray_vs_multiRay_max_abs_m']=float(max(errors))
    (ROOT/'metrics/sensors.json').write_text(json.dumps(report,indent=2))
    assert max(errors)<1e-8
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
