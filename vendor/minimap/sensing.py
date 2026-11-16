"""Airy forward model and deployable pointcloud-only rolling BEV builder.

No Scene reference exists in BEVBuilder. Axes: x forward, y left, z up;
BEV array axis 0=x, axis 1=y. Pose = estimated map [x,y,z,yaw].
"""
import time
import numpy as np
from numba import njit
from scipy.ndimage import minimum_filter, maximum_filter

# Appendix D of the manufacturer's Airy user guide, 96 actual nominal angles.
ANGLES=np.array([-.07,.88,1.81,2.76,3.69,4.62,5.54,6.48,7.41,8.34,9.27,10.21,11.15,12.09,13.03,13.98,14.92,15.87,16.82,17.77,18.72,19.67,20.62,21.57,22.51,23.45,24.4,25.33,26.28,27.21,28.15,29.08,30.02,30.95,31.88,32.82,33.74,34.68,35.62,36.55,37.5,38.43,39.37,40.31,41.25,42.21,43.16,44.09,45.05,46,46.95,47.9,48.85,49.8,50.73,51.69,52.62,53.56,54.5,55.45,56.37,57.3,58.24,59.18,60.12,61.05,61.99,62.93,63.86,64.81,65.76,66.69,67.65,68.6,69.56,70.51,71.46,72.42,73.37,74.33,75.29,76.24,77.19,78.14,79.07,80.02,80.96,81.9,82.84,83.78,84.7,85.64,86.57,87.52,88.46,89.4])
BODY=np.array([.93,.48,.585])
ENVELOPE=np.array([1.03,.58])

@njit(cache=True)
def box_hit(o,d,b):
    c,s=np.cos(b[6]),np.sin(b[6]);dx=o[0]-b[0];dy=o[1]-b[1]
    oo=np.array([c*dx+s*dy,-s*dx+c*dy,o[2]-b[2]])
    dd=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1],d[2]])
    lo=0.;hi=1e9
    for j in range(3):
        if abs(dd[j])<1e-8:
            if abs(oo[j])>b[3+j]:return 1e9
        else:
            a=(-b[3+j]-oo[j])/dd[j];z=(b[3+j]-oo[j])/dd[j]
            lo=max(lo,min(a,z));hi=min(hi,max(a,z))
    return lo if hi>=lo else 1e9

@njit(cache=True)
def cast(o,dirs,ground,origin,res,objects,body,maxrange):
    out=np.full(len(dirs),np.nan,np.float32)
    for k in range(len(dirs)):
        d=dirs[k];best=maxrange
        self_hit=box_hit(o,d,body)
        for b in objects:
            best=min(best,box_hit(o,d,b))
        # Heightfield ray marching, 6 cm step + bisection, exact slabs for thin solids.
        t=.1
        while t<=best:
            p=o+t*d;i=int(np.floor((p[0]-origin[0])/res+.5));j=int(np.floor((p[1]-origin[1])/res+.5))
            if 0<=i<ground.shape[0] and 0<=j<ground.shape[1] and p[2]<=ground[i,j]:
                low=max(.1,t-.06);high=t
                for _ in range(5):
                    m=(low+high)/2;q=o+m*d;ii=int(np.floor((q[0]-origin[0])/res+.5));jj=int(np.floor((q[1]-origin[1])/res+.5))
                    if 0<=ii<ground.shape[0] and 0<=jj<ground.shape[1] and q[2]<=ground[ii,jj]:high=m
                    else:low=m
                best=high;break
            t+=.06
        if .1<=best<maxrange and best<self_hit:out[k]=best
    return out

class AiryPair:
    def __init__(self,beam_stride=6,azimuth_stride=10):
        self.channels=np.arange(0,96,beam_stride)
        self.az=np.arange(0,900,azimuth_stride)*.4*np.pi/180
        el=ANGLES[self.channels]*np.pi/180
        el,az=np.meshgrid(el,self.az,indexing='ij')
        native=np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)],-1).reshape(-1,3)
        # ASSUMED horizontal optical axes, front points +x and rear -x.
        # Native positive z hemisphere is rotated into forward/rear half spaces.
        self.front=np.stack([native[:,2],native[:,1],-native[:,0]],-1)
        self.back=np.stack([-native[:,2],-native[:,1],-native[:,0]],-1)
        self.az_per_ray=np.tile(self.az,len(self.channels))
    def scan(self,scene,pose,rng,frame=0):
        x,y,z,yaw=pose;c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        local=scene.objects[np.linalg.norm(scene.objects[:,:2]-[x,y],axis=1)<14]
        body=np.array([x,y,z+.585/2,.93/2,.48/2,.585/2,yaw,0])
        points=[]
        for side,dirs in [(1,self.front),(-1,self.back)]:
            origin=np.array([side*.49,0,.56]);o=origin@R.T+[x,y,z]
            d=dirs@R.T
            ranges=cast(o,d,scene.ground,scene.origin,scene.res,local,body,12.)
            valid=np.isfinite(ranges)&(rng.random(len(ranges))>.025)
            if frame%10==9:valid&=~((self.az_per_ray>=np.pi)&(self.az_per_ray<np.pi+32*np.pi/180))
            rr=ranges[valid]+rng.normal(0,.015,valid.sum())
            # Output already calibrated into body frame, as a real ROS adapter would.
            points.append(origin+dirs[valid]*rr[:,None])
        return np.concatenate(points).astype(np.float32)

class BEVBuilder:
    def __init__(self,res=.125,extent=8.,history_s=.8):
        self.res,self.extent,self.history_s=res,extent,history_s
        self.n=round(extent/res);self.frames=[]
    def update(self,points_body,pose_est,timestamp):
        x,y,z,a=pose_est;c,s=np.cos(a),np.sin(a)
        R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        p=points_body@R.T+[x,y,z]
        self.frames.append((timestamp,p))
        self.frames=[(t,p) for t,p in self.frames if timestamp-t<=self.history_s+1e-6]
    def render(self,pose_est,timestamp):
        x,y,z,a=pose_est;c,s=np.cos(a),np.sin(a);R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        p=np.concatenate([p for t,p in self.frames]);ages=np.concatenate([np.full(len(p),timestamp-t) for t,p in self.frames])
        p=(p-[x,y,z])@R
        ij=np.floor((p[:,:2]+self.extent/2)/self.res).astype(int)
        good=(ij>=0).all(1)&(ij<self.n).all(1)&(p[:,2]>-3)&(p[:,2]<3)
        ij,p,ages=ij[good],p[good],ages[good]
        low=np.full((self.n,self.n),np.inf,np.float32);high=np.full_like(low,-np.inf);age=np.full_like(low,1.)
        np.minimum.at(low,tuple(ij.T),p[:,2]);np.maximum.at(high,tuple(ij.T),p[:,2]);np.minimum.at(age,tuple(ij.T),ages)
        known=np.isfinite(low)
        # At most 1 cell interpolation, while preserving explicit unknown channel.
        floor=minimum_filter(low,3,mode='constant',cval=np.inf)
        # Positive returns above nearby floor become volume occupancy; no truth labels.
        ground=np.minimum(low,floor+.23)
        height=np.where(known,np.maximum(0,high-ground),0.)
        localspan=maximum_filter(np.where(known,ground,-np.inf),3)-minimum_filter(np.where(known,ground,np.inf),3)
        support=known&(localspan<.32)&(height<.28)&(ground>-.5)&(ground<.5)
        elev=np.where(known,np.clip(ground,-2,2),0.)
        bev=np.stack([elev,np.minimum(height,2),support.astype(float),(~known).astype(float),np.minimum(age/self.history_s,1)],-1).astype(np.float32)
        return bev

def encode_bev(bev):
    lo=np.array([-2,0,0,0,0]);scale=np.array([4,2,1,1,1])
    return np.round(np.clip((bev-lo)/scale,0,1)*255).astype(np.uint8)

def decode_bev(image):
    return image.astype(np.float32)/255*np.array([4,2,1,1,1],np.float32)+np.array([-2,0,0,0,0],np.float32)
