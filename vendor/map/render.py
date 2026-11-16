"""Actual MuJoCo EGL RGB stills and a kinematic, constant-arc-speed fly-through."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse
import json
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import mujoco
import imageio.v2 as imageio
from scene import ROOT
from runtime import load_scene,place_body,follow_camera,local_lighting


def caption(rgb,title,detail=''):
    im=Image.fromarray(rgb)
    draw=ImageDraw.Draw(im,'RGBA')
    draw.rectangle((0,0,im.width,83),fill=(15,22,18,224))
    fp='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fp,24); small=ImageFont.truetype(fp,16)
    draw.text((25,12),title,font=font,fill=(238,232,209,255))
    draw.text((26,49),detail,font=small,fill=(192,207,192,255))
    return np.asarray(im)


def gate_annotations(rgb,renderer,scene):
    """Project the actual 3D gate positions into an overview RGB frame."""
    cameras=renderer.scene.camera
    origin=(cameras[0].pos+cameras[1].pos)/2
    forward=cameras[0].forward;up=cameras[0].up;right=np.cross(forward,up)
    h,w=rgb.shape[:2];f=h*cameras[0].frustum_near/(cameras[0].frustum_top-cameras[0].frustum_bottom)
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for gate in scene.meta['gates']:
        p=np.array(gate['center'])+[0,0,1.6]-origin;depth=np.dot(p,forward)
        if depth<=0:continue
        u=w/2+f*np.dot(p,right)/depth;v=h/2-f*np.dot(p,up)/depth
        draw.ellipse((u-15,v-15,u+15,v+15),fill=(28,51,37),outline=(232,213,166),width=2)
        draw.text((u,v),str(gate['id']),anchor='mm',font=font,fill=(247,236,197))
    return np.asarray(im)


def render_stills(model,data,scene,out):
    manifest={}
    opt=mujoco.MjvOption();opt.sitegroup[:]=0
    def save(name,title,s,camera,detail=''):
        local_lighting(model,scene,s,overview=name.startswith(('01','02')))
        place_body(model,data,scene,s)
        with mujoco.Renderer(model,height=900,width=1440) as renderer:
            renderer.update_scene(data,camera=camera,scene_option=opt)
            rgb=renderer.render().copy()
            if name.startswith(('01','02')):rgb=gate_annotations(rgb,renderer,scene)
        Image.fromarray(caption(rgb,title,detail)).save(out/f'{name}.png')
        manifest[name]=dict(s_m=float(s),file=f'{name}.png')
        print(f'saved {name}',flush=True)
    mid=np.array([(scene.x[0]+scene.x[-1])/2,(scene.y[0]+scene.y[-1])/2,float(np.median(scene.ground))])
    cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    cam.lookat[:]=mid;cam.azimuth=90;cam.elevation=-90;cam.distance=155
    save('01_overview_top','MINI TRAIL / PLAN VIEW',3,cam,'208 m | 8 ordered gates | two forks | 1.08 m minimum width | MuJoCo EGL')
    cam.azimuth=105;cam.elevation=-49;cam.distance=155
    save('02_overview_bird','MINI TRAIL / MOUNTAIN PATH',3,cam,'Natural earth, leaf shoulders, hillside and green safety rail')
    f=scene.features
    views=[
        ('03_width_transition','SMOOTH WIDTH TRANSITION',f['narrow']-8,9,-37,-22,3.0),
        ('04_narrow_drop_rail','NARROW TRAIL / DROP / GREEN RAIL',f['narrow']-.8,6.4,-25,24,1.8),
        ('05_sharp_bend','SHARP BEND / 4.5 m MINIMUM RADIUS',f['bend']-3.5,11,-42,-22,3.0),
        ('06_stairs_up','STONE STEPS / 4 x 0.15 m UP',f['stairs_up']-3.4,5.9,-22,-17,2.3),
        ('07_stairs_down','STONE STEPS / 4 x 0.15 m DOWN',f['stairs_down']-3.2,6,-29,-20,2.2),
        ('08_fork_tape','FORK 1 / SUSPENDED BLACK RIBBON',f['fork_tape']+1.3,8.8,-32,22,3),
        ('09_fork_plain','FORK 2 / CONTINUOUS TERRAIN JOIN',f['fork_plain']-.5,10,-35,-28,3.3),
        ('10_uphill','CONTINUOUS 15 DEGREE ASCENT',42.,8,-20,-24,3),
        ('11_downhill','CONTINUOUS 15 DEGREE DESCENT',125.,8,-25,-20,3),
        ('12_leaf_shoulder','FALLEN LEAVES / TRAIL SHOULDER',f['leaves']-1.3,5.0,-25,29,1.5),
        ('13_log_rocks','FALLEN LOG AND ROCKS',f['log']-2,5.5,-25,23,2),
        ('14_m1_lidar_mounts','M1 ENVELOPE / FRONT + REAR AIRY',f['narrow']-12,2.7,-18,70,.1),
        ('15_tape_close','BLACK RIBBON / 0.40 m AGL / 45 mm',f['fork_tape']+3.8,5.2,-22,34,1.8),
    ]
    for name,title,s,distance,elevation,side,ahead in views:
        cam=follow_camera(scene,s,distance,elevation,side,ahead)
        save(name,title,s,cam,f's = {s:.1f} m  |  road width = {float(scene.width_at(s)):.2f} m  |  kinematic M1')
    (out/'stills_manifest.json').write_text(json.dumps(manifest,indent=2))


def render_video(model,data,scene,out,duration=45.,fps=20):
    # Constant speed in full 3D arc length, including stair rise, resampled at FPS.
    arc=np.linspace(.5,scene.arc3d[-1]-.5,round(duration*fps))
    positions=np.interp(arc,scene.arc3d,scene.s)
    opt=mujoco.MjvOption();opt.sitegroup[:]=0
    path=out/'mini_trail_follow_45s.mp4'
    started=time.time()
    local_lighting(model,scene,float(positions[0]))
    with mujoco.Renderer(model,height=720,width=1280) as renderer, imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=16,ffmpeg_log_level='error') as writer:
        for i,s in enumerate(positions):
            local_lighting(model,scene,s)
            place_body(model,data,scene,s)
            cam=follow_camera(scene,s,distance=6.5,elevation=-27,side=-18,ahead=1.8)
            renderer.update_scene(data,camera=cam,scene_option=opt)
            rgb=renderer.render().copy()
            detail=f'{i/fps:05.1f} / {duration:.0f} s  |  route {s:06.1f} / {scene.length:.0f} m  |  constant arc speed {(arc[-1]-arc[0])/duration:.2f} m/s'
            writer.append_data(caption(rgb,'MINI TRAIL / KINEMATIC SCENE TOUR',detail))
            if i%100==0:print(f'video {i}/{len(positions)}, elapsed {time.time()-started:.1f}s',flush=True)
    report=dict(file=str(path),duration_s=duration,fps=fps,frames=len(positions),resolution=[1280,720],motion='kinematic, constant 3D arc speed; no controller or dynamics',render_seconds=time.time()-started)
    (out/'video_manifest.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--map',type=Path,default=ROOT/'maps/mini_trail');ap.add_argument('--stills',action='store_true');ap.add_argument('--video',action='store_true');ap.add_argument('--duration',type=float,default=45);ap.add_argument('--fps',type=int,default=20)
    a=ap.parse_args();out=ROOT/'renders';out.mkdir(exist_ok=True)
    scene=load_scene(a.map);model=mujoco.MjModel.from_xml_path(str(a.map/'scene.xml'));data=mujoco.MjData(model)
    print(f'MuJoCo {mujoco.__version__}: {model.ngeom} geoms, EGL={os.environ["MUJOCO_GL"]}',flush=True)
    if a.stills or not a.video:render_stills(model,data,scene,out)
    if a.video:render_video(model,data,scene,out,a.duration,a.fps)


if __name__=='__main__':main()
