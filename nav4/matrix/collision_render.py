"""Render matched views of the serialized colliders with OSMesa."""
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nav4.mapping.common import write_json
from nav4.matrix.collision import FULL,V2
from nav4.matrix.collision_checks import local,model_of
from nav4.matrix.full_scan import route
from nav4.matrix.full_visual import camera
from nav4.matrix.stairs import path_at,heading

WIDTH,HEIGHT=768,432
GROUND=".22 .59 .68 1"
STAIR=".96 .70 .27 1"
HARD=".79 .36 .31 1"


def render(version,frame,xy):
    folder=FULL if version=="v1" else V2
    root=local(folder/"scene_full.xml",xy,radius=12,visual_hard=version=="v1")
    world=root.find("worldbody")
    for node in list(world):
        if node.tag=="light":
            continue
        flex=node.tag=="flexcomp"
        if not flex and node.get("contype")=="0":
            world.remove(node)
            continue
        name=node.get("name","")
        if flex:
            node.set("file",str((folder/node.get("file")).resolve()))
        node.attrib.pop("material",None)
        rgba=GROUND if flex or node.get("type")=="hfield" else STAIR if "stair_" in name else HARD
        if "lower_layer" in name:
            rgba=".32 .77 .48 1"
        node.set("rgba",rgba)
    camera(frame,0,world,frame["fov"])
    m=model_of(root)
    d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    opt=mujoco.MjvOption()
    opt.geomgroup[:]=1
    opt.flexgroup[:]=1
    opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE]=1
    opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT]=0
    opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE]=0
    with mujoco.Renderer(m,height=HEIGHT,width=WIDTH,max_geom=max(20000,m.ngeom+500)) as renderer:
        renderer.update_scene(d,camera="view_00",scene_option=opt)
        return Image.fromarray(renderer.render().copy())


def main():
    assert os.environ["MUJOCO_GL"]=="osmesa"
    _,path=route()
    folder=V2/"figures"
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,17)
    label=ImageFont.truetype(font_path,22)
    with (FULL/"route_vs_v8.csv").open() as f:
        import csv
        z={round(float(r["station_m"]),1):float(r["scene_z_m"]) for r in csv.DictReader(f)}
    frames={}
    for station,name,distance,side,elev in [(6.2,"entry",5.,3.6,2.6),(55.,"corridor",4.,7.,9.5),
            (617.9,"descent",4.4,3.5,3.1),(715.3,"bridge",4.2,3.4,2.8)]:
        xy=path_at(path,station)
        forward=heading(path,station)
        normal=np.array([-forward[1],forward[0]])
        frame=dict(target=np.r_[xy,z[station]+.30],eye=np.r_[xy-distance*forward+side*normal,z[station]+elev],fov=53)
        frames[name]=dict(station_m=station,**frame)
        a=render("v1",frame,xy)
        b=render("v2",frame,xy)
        canvas=Image.new("RGB",(2*WIDTH+2,HEIGHT+87),(27,34,39))
        draw=ImageDraw.Draw(canvas)
        for x,im,caption in [(0,a,"full_v1   Rigid triangle flex"),(WIDTH+2,b,"full_v2   Tiled height fields")]:
            draw.text((x+16,10),caption,font=label,fill="white")
            canvas.paste(im,(x,40))
        draw.text((16,HEIGHT+52),f"{station:.1f} m   /   {name}",font=font,fill="white")
        for x,color,text in [(370,"#4099ad","Ground"),(550,"#e5b159","Treads"),(730,"#bd6357","Hard obstacles"),(980,"#62bd86","Lower layer")]:
            draw.rectangle((x,HEIGHT+55,x+14,HEIGHT+69),fill=color)
            draw.text((x+23,HEIGHT+51),text,font=font,fill="white")
        canvas.save(folder/f"collision_{name}.png")
        print(name,flush=True)
    write_json(folder/"cameras.json",frames)


if __name__=="__main__":
    main()
