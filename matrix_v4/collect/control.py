from dataclasses import dataclass
import math
from typing import Protocol
import numpy as np


class Driver(Protocol):
    def command(self, observation, route_state): ...


def attitude(q):
    x,y,z,w = q
    return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            math.asin(np.clip(2*(w*y-z*x),-1,1)),
            math.atan2(2*(w*z+x*y),1-2*(z*z+y*y)))


@dataclass
class RouteState:
    s: float
    lateral: float
    clearance: float
    roll: float
    pitch: float
    error: float
    outside: bool


class Route:
    def __init__(self, station, xy, ground, shift_z, bounds, cfg):
        self.s, self.xy, self.ground, self.shift_z = station, xy, ground, shift_z
        self.bounds, self.cfg = bounds,cfg

    def locate(self, pose, previous):
        # Stay on the local segment of hairpins. Permit projections past either endpoint.
        lo, hi = self.s.searchsorted([previous-5, previous+5])
        seg = self.xy[lo:min(len(self.s),hi+1)]
        dif = np.diff(seg,axis=0)
        p = np.asarray(pose[:2])
        t = np.clip(((p-seg[:-1])*dif).sum(1)/np.maximum((dif*dif).sum(1),1e-12),0,1)
        delta = p-(seg[:-1]+t[:,None]*dif)
        idx = int(np.argmin((delta*delta).sum(1)))
        along = self.s[lo+idx] + t[idx]*(self.s[lo+idx+1]-self.s[lo+idx])
        unit = dif[idx]/max(np.linalg.norm(dif[idx]),1e-12)
        lateral = float(delta[idx] @ [-unit[1],unit[0]])
        # Continue projection beyond the two ends of the entire frozen route.
        if along == self.s[0] or along == self.s[-1]:
            along += delta[idx] @ unit
        ground = self.ground
        k = np.clip(np.searchsorted(ground[:,0,0],along),0,len(ground)-1)
        if k and abs(ground[k-1,0,0]-along)<abs(ground[k,0,0]-along):
            k -= 1
        j = np.argmin(abs(ground[k,:,1]-lateral))
        height = ground[k,j,-1]-self.shift_z
        roll,pitch,yaw = attitude(pose[3:])
        goal = [np.interp(along+self.cfg['lookahead_m'], self.s, axis) for axis in self.xy.T]
        heading = math.atan2(goal[1]-p[1],goal[0]-p[0])
        return RouteState(along, lateral, pose[2]-height, roll,pitch,
                          math.remainder(heading-yaw, 2*math.pi),
                          along<self.bounds[0] or along>=self.bounds[1])


def upright(st, cfg, initial=False):
    tilt = cfg['stand_max_tilt_rad' if initial else 'max_tilt_rad']
    height = cfg['stand_min_clearance_m' if initial else 'min_clearance_m']
    return st.clearance>=height and max(abs(st.roll),abs(st.pitch))<tilt


class Pursuit:
    def __init__(self,cfg,rng):
        self.cfg,self.rng = cfg,rng
        u = rng.random()
        self.mode = ('gaussian' if u<cfg['gaussian_fraction'] else 'burst'
                     if u<cfg['gaussian_fraction']+cfg['burst_fraction'] else 'clean')
        self.next = cfg['burst_interval_s']*rng.random()
        self.until = 0
        self.burst = np.zeros(3)

    def command(self, observation, route_state):
        c,e,t = self.cfg,route_state.error,observation['elapsed_sim_s']
        action = np.array([c['speed']*max(0,math.cos(e)) if abs(e)<.95 else 0,
                           0,np.clip(c['yaw_gain']*e,-c['max_turn'],c['max_turn'])])
        if self.mode=='gaussian':
            action += self.rng.normal(0,c['noise_sigma'])
        if self.mode=='burst':
            if t>=self.next:
                self.until=t+self.rng.uniform(1,3)
                self.next=self.until+c['burst_interval_s']
                self.burst=self.rng.uniform(-1,1,size=3)
            if t<self.until:
                action=self.burst
        return np.clip(action,-1,1)


class Monitor:
    def __init__(self,cfg,station,stamp):
        self.cfg,self.best,self.last = cfg,station,stamp

    def reason(self,st,stamp):
        if st.outside:
            return 'window_exit'
        if abs(st.lateral)>self.cfg['cross_track_m']:
            return 'off_route'
        if not np.isfinite(st.clearance):
            return 'missing_ground'
        if not upright(st,self.cfg):
            return 'overturned'
        if st.s >= self.best+self.cfg['stall_progress_m']:
            self.best,self.last=st.s,stamp
        if stamp-self.last>=self.cfg['stall_s']:
            return 'stuck'
        return ''
