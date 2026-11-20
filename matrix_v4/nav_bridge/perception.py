"""Assemble nav4 observations from dual Airy, base Odom and IMU."""
from collections import deque
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from scipy.spatial.transform import Rotation
import zenoh

from sensors import Reader, motion
from control import attitude
from nav4.common import bev_xy, encode
from nav4.sensor import support_mask


def cloud(raw):
    r = Reader(raw)
    start = r.header()
    h, w, n = r.take('III', 4)
    fields = {}
    for _ in range(n):
        name = r.string()
        fields[name] = (r.take('I', 4)[0], r.take('B', 1)[0], r.take('I', 4)[0])
    order = '>' if r.take('B', 1)[0] else '<'
    point, row, size = r.take('III', 4)
    if size != row*h or row < w*point or len(raw) < r.pos+size:
        raise ValueError('invalid PointCloud2 strides')
    def column(name, fmt, kind):
        offset, actual, count = fields[name]
        assert actual == kind and count == 1
        return np.ndarray((h,w), order+fmt, raw, r.pos+offset, (row,point))
    return (np.stack([column(k,'f4',7) for k in 'xyz'], -1),
            column('timestamp','f8',8), start)


class Stream:
    def __init__(self):
        cfg = zenoh.Config()
        cfg.insert_json5('mode','"client"')
        cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
        self.session = zenoh.open(cfg)
        self.odom, self.imu = deque(maxlen=500), deque(maxlen=1800)
        self.latest = {}
        self.epoch_offset = time.time()-time.monotonic()
        self.subs = [self.session.declare_subscriber('rt/'+name, self.receive)
                     for name in ('front_lidar','rear_lidar','imu','odom')]

    def receive(self, sample):
        received = time.monotonic()
        name = str(sample.key_expr).split('/')[-1]
        raw = sample.payload.to_bytes()
        if name.endswith('lidar'):
            self.latest[name] = received, raw
        else:
            row = (received, *motion(raw, name == 'odom'))
            getattr(self,name).append(row)

    def close(self):
        for sub in self.subs:
            sub.undeclare()
        self.session.close()


def sensor_pose(sensor):
    angles = sensor['rotation']
    rotation = Rotation.from_euler('xyz', [-angles['roll'], -angles['pitch'], angles['yaw']], degrees=True)
    return rotation.as_matrix(), np.array([sensor['position'][k] for k in 'xyz'])


class Memory:
    def __init__(self, cfg, sensors):
        self.cfg = cfg
        self.mount = {name: sensor_pose(sensors[key]) for name,key in
                      [('front_lidar','lidar'), ('rear_lidar','rear_lidar')]}
        self.res, self.size = cfg['bev']['res'], cfg['memory']['size']
        self.local = bev_xy(cfg)
        self.process = jax.jit(lambda h,k,o,a: encode(jnp.stack([
            h, o, support_mask(h[None],k[None],o[None],cfg)[0], ~k,
            jnp.zeros_like(h), a], -1), cfg))
        self.clear()

    def clear(self):
        shape = (self.size,self.size)
        self.low, self.high = np.zeros(shape,'f'), np.zeros(shape,'f')
        self.tag = np.full(shape+(2,), -2147483648, 'i')
        self.seen = np.full(shape,-np.inf)
        self.previous = None
        self.last_frame = {}

    def update(self, stream, pose, stamp, yaw):
        hist = list(stream.odom)
        wall = np.array([a[0] for a in hist])
        stamps = np.array([a[1] for a in hist])
        poses = np.array([a[2] for a in hist])
        clouds, capture = [], []
        ages = []
        for name,(recv,raw) in list(stream.latest.items()):
            if self.last_frame.get(name) == recv:
                continue
            xyz, ts, start = cloud(raw)
            self.last_frame[name] = recv
            mat, trans = self.mount[name]
            # Deskew in 10 firing intervals against measured Odom receipt anchors.
            # Airy timestamps are epoch seconds, Odom headers run on the physics clock.
            for chunk in np.array_split(np.arange(len(xyz)),10):
                t = float(np.median(ts[chunk]))-stream.epoch_offset
                i = int(np.searchsorted(wall,t))
                if not 0 < i < len(wall) or wall[i]-wall[i-1] > .2:
                    continue
                u = (t-wall[i-1])/(wall[i]-wall[i-1])
                p0,p1 = poses[i-1:i+1]
                quat = (1-u)*p0[3:]+u*p1[3:]*np.sign(p0[3:]@p1[3:])
                loc = (1-u)*p0[:3]+u*p1[:3]
                acq = (1-u)*stamps[i-1]+u*stamps[i]
                b = xyz[chunk].reshape(-1,3)
                good = np.isfinite(b).all(1) & (np.max(abs(b),axis=1)<12)
                b = b[good] @ mat.T + trans
                # Body and leg returns are inside the swept robot envelope.
                b = b[(abs(b[:,0])>.55) | (abs(b[:,1])>.34)]
                world = b @ Rotation.from_quat(quat).as_matrix().T + loc
                keep = np.max(abs(world[:,:2]-pose[:2]),axis=1) < (self.size/2-1)*self.res
                world = world[keep]
                clouds.append(world)
                capture.append(np.full(len(world),acq))
            ages.append(stamp-np.interp(np.median(ts)-stream.epoch_offset,wall,stamps))
        if clouds:
            pts, times = np.concatenate(clouds), np.concatenate(capture)
            ij = np.floor(pts[:,:2]/self.res).astype('i')
            idx = (ij[:,0]%self.size)*self.size + ij[:,1]%self.size
            lo, hi = np.full(self.size**2,np.inf), np.full(self.size**2,-np.inf)
            acq = np.full(self.size**2,-np.inf)
            np.minimum.at(lo,idx,pts[:,2])
            np.maximum.at(hi,idx,pts[:,2])
            np.maximum.at(acq,idx,times)
            ids = np.flatnonzero(np.isfinite(lo))
            self.low.flat[ids], self.high.flat[ids] = lo[ids], hi[ids]
            self.seen.flat[ids] = acq[ids]
            self.tag.reshape(-1,2)[idx] = ij
        c,s = np.cos(yaw),np.sin(yaw)
        world = self.local @ np.array([[c,s],[-s,c]]) + pose[:2]
        ij = np.floor(world/self.res).astype('i')
        a,b = ij[...,0]%self.size, ij[...,1]%self.size
        age = np.clip((stamp-self.seen[a,b])/self.cfg['memory']['seconds'],0,1)
        known = (self.tag[a,b]==ij).all(-1) & (age<1)
        lo,hi = self.low[a,b],self.high[a,b]
        size = self.cfg['sensor']['ground_opening']
        opened = minimum_filter(np.where(known,lo,np.inf),size=size,mode='nearest')
        opened = maximum_filter(np.where(np.isfinite(opened),opened,-np.inf),size=size,mode='nearest')
        # Estimate the terrain beneath the base from measured neighbouring returns.
        ring = known & (np.linalg.norm(self.local,axis=-1)<.95)
        z0 = np.median(opened[ring]) if ring.any() else pose[2]-.50
        h = np.where(known,opened-z0,0).astype('f')
        o = np.where(known,np.maximum(hi-opened,0),0).astype('f')
        image = np.asarray(self.process(h,known,o,np.where(known,age,1).astype('f')))
        delayed = image if self.previous is None else self.previous
        self.previous = image
        return delayed, ages
