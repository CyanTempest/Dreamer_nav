"""Planar nav4 velocity from the same Odom poses used for spatial registration."""
import math
import numpy as np
from control import attitude

def planar_velocity(history, stamp, pose, yaw, interval=.10):
    # Published ZGWS linear twist places forward motion on axis 2. Pose
    # differences avoid guessing sensor/link bases and yield planar yaw rate.
    rows=list(history)
    stamps=np.array([r[1] for r in rows])
    i=max(0,int(np.searchsorted(stamps,stamp-interval,side='right'))-1)
    dt=stamp-stamps[i]
    if not .03 <= dt <= .25:
        raise ValueError(f'Unusable odometry derivative interval {dt}')
    past=rows[i][2]
    delta=(pose[:2]-past[:2])/dt
    old_yaw=attitude(past[3:])[2]
    turn=math.remainder(yaw-old_yaw,2*math.pi)
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1],turn/dt],np.float32)
