"""Load exported maps and place the visual M1; never advances dynamics."""
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from scene import Scene, ROOT


def load_scene(directory=ROOT/'maps/mini_trail'):
    directory=Path(directory)
    d=np.load(directory/'heightfield.npz')
    sc=Scene.__new__(Scene)
    for name in ['ground','support','leaf','origin','route','s','width','main_edge','edge','projected_s','drop_weight']:
        setattr(sc,name,d[name])
    sc.res=float(d['res']);sc.xy=sc.route[:,:2];sc.z=sc.route[:,2]
    v=np.gradient(sc.xy,axis=0);sc.yaw=np.unwrap(np.arctan2(v[:,1],v[:,0]))
    sc.x=sc.origin[0]+np.arange(sc.ground.shape[0])*sc.res
    sc.y=sc.origin[1]+np.arange(sc.ground.shape[1])*sc.res
    meta=yaml.safe_load((directory/'route.yaml').read_text())
    sc.features=meta['features'];sc.length=meta['horizontal_length_m']
    sc.arc3d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(sc.route,axis=0),axis=1))]
    sc.meta=meta;sc.directory=directory
    return sc


def place_body(model,data,scene,s):
    import mujoco
    pose=scene.pose_at(float(s))
    # Rotation about body y is negative for a nose-up slope (x-forward, z-up).
    q=Rotation.from_euler('ZYX',[pose[3],-pose[4],0]).as_quat()
    data.mocap_pos[0]=pose[:3]+[0,0,.008]
    data.mocap_quat[0]=q[[3,0,1,2]]
    mujoco.mj_forward(model,data)
    return pose


def follow_camera(scene,s,distance=6.8,elevation=-26,side=-18,ahead=1.7):
    import mujoco
    p=scene.pose_at(s)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.lookat[:]=scene.pose_at(min(scene.length,s+ahead))[:3]+[0,0,.35]
    camera.distance=distance
    camera.azimuth=np.rad2deg(p[3])+side
    camera.elevation=elevation
    return camera


def local_lighting(model,scene,s,overview=False):
    """Fit the shadow map to the viewed area instead of a 400 m global square."""
    p=scene.pose_at(s)[:3]
    model.stat.extent=110. if overview else 18.
    model.vis.map.znear=.0005 if overview else .003
    model.vis.map.zfar=6. if overview else 32.
    model.vis.map.fogstart=3. if overview else 12.
    model.vis.map.fogend=6. if overview else 25.
    if overview:
        center=np.array([(scene.x[0]+scene.x[-1])/2,(scene.y[0]+scene.y[-1])/2,float(np.median(scene.ground))])
        model.light_pos[0]=center+[35,-50,110]
    else:model.light_pos[0]=p+[10,-14,30]
