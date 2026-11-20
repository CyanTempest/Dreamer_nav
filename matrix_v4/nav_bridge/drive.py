"""Run the saved actor, record each decision, and handle physical cold resets."""
from collections import defaultdict
from pathlib import Path
import json
import math
import os
import subprocess
import sys
import threading
import time

import jax
import numpy as np
from actuator import Actuator
from control import Route, attitude
from matrix_mc_udp_test import UdpRobotController
from nav4.common import action_target, load_config
from nav4.policy import DreamerPolicy, read_checkpoint
from nav4.tracker import command
from perception import Memory, Stream
from odometry import planar_velocity
from runtime import BRIDGE, ROOT, Instance, set_pose

def save_json(path, item):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(item,indent=2))
    tmp.replace(path)

class Heartbeat:
    def __init__(self,stream,scale,tracker):
        self.motor=Actuator()
        self.stream,self.scale,self.tracker=stream,scale,tracker
        self.target=None
        self.last=np.zeros(3)
        self.active=True
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def loop(self):
        while self.active:
            t=time.monotonic()
            row=self.stream.odom[-1]
            p=np.array(row[2])
            yaw=attitude(p[3:])[2]
            target=self.target
            if target and t-target[2]<.45 and t-row[0]<.25:
                delta=target[0]-p[:2]
                c,s=np.cos(yaw),np.sin(yaw)
                local=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]],'f')
                si=np.asarray(self.tracker(local,np.float32(target[1])))
                self.last=np.array([np.clip(si[0]/self.scale[0],-1,1),0,np.clip(si[2]/self.scale[1],-1,1)])
            else:
                self.last=np.zeros(3)
            self.motor.send(self.last[0],self.last[2],yaw,stair=self.motor.index==0)
            time.sleep(max(0,.05-time.monotonic()+t))

    def stop(self):
        self.active=False
        self.thread.join()
        self.motor.send(0,0,0)

class Evaluation:
    def __init__(self):
        payload=read_checkpoint('/content/best.pkl')
        meta=payload['nav4']
        self.cfg=load_config('env',meta['env_overrides'])
        assert self.cfg['odometry_inputs'] and not self.cfg['route_inputs']
        self.actor=DreamerPolicy(0,'/content/policy-log',deterministic=True,
             obs=meta['obs_space'],jax_overrides=dict(autotune=1,deterministic=True)).load(payload)
        del payload
        assert jax.default_backend()=='gpu'
        self.obs={k:np.zeros((1,*v[1]),v[0]) for k,v in meta['obs_space'].items()}
        self.obs['is_first'][:]=True
        np.asarray(self.actor(self.obs))
        self.actor.key=jax.random.PRNGKey(1234)
        self.target=jax.jit(lambda a:action_target(a,self.cfg))
        self.tracker=jax.jit(lambda xy,speed:command(xy,speed,self.cfg))
        self.target(np.zeros(3,'f'))
        self.tracker(np.ones(2,'f'),np.float32(.6)).block_until_ready()
        root=Path('/content/matrix-v4/inputs/data/trail_a_mujoco/full_v2')
        self.shift=np.array(json.loads((root/'assembly.json').read_text())['shared_shift_m'])
        z=np.load('/content/reference/route.npz')
        self.width=z['width']
        self.finish=float(z['gates'][-1])
        self.route=Route(z['s'],z['route'][:,:2]-self.shift[:2],np.load('/content/reference/ground.npy'),
                         self.shift[2],(0,self.finish),dict(lookahead_m=.6))
        self.distance=0.
        self.origin_yaw=None
        self.previous_xy=None
        self.memory=None
        self.events=[]
        save_json(ROOT/'config.json',dict(env=self.cfg,obs=meta['obs_space'],gpu=str(jax.devices()),finish=self.finish))

    def episode(self,out,station,assist):
        stream=Stream()
        udp=UdpRobotController('127.0.0.1',7447,20,False)
        beat=None
        rec=defaultdict(list)
        chunks=0
        def flush():
            nonlocal chunks
            if rec:
                path=out/f'steps-{chunks:04}.npz'
                tmp=path.with_suffix('.partial')
                with tmp.open('wb') as file:
                    np.savez_compressed(file,**{k:np.asarray(v) for k,v in rec.items()})
                tmp.replace(path)
                chunks+=1
                rec.clear()
        try:
            udp.stand(4)
            if not stream.odom or not stream.imu:
                raise RuntimeError('no odometry or IMU after stand')
            sensor_cfg=json.loads((out/'config.json').read_text())['robot']['sensors']
            if self.memory is None:
                self.memory=Memory(self.cfg,sensor_cfg)
            self.memory.clear()
            deadline=time.monotonic()+30
            while not all(side in stream.latest for side in ('front_lidar','rear_lidar')):
                if time.monotonic()>deadline:
                    raise RuntimeError('missing Airy messages after stand: '+repr(list(stream.latest)))
                time.sleep(.05)
            print('Airy present',[(key,len(row[1])) for key,row in stream.latest.items()],flush=True)
            self.actor.carry=None
            prev=np.zeros(3,'f')
            first=True
            t0=stream.odom[-1][1]
            tick=t0
            beat=Heartbeat(stream,self.scale,self.tracker)
            best=station
            progress_t=t0
            current=station
            while True:
                arrival,stamp,xyzq,twist=stream.odom[-1]
                if time.monotonic()-arrival>2:
                    raise TimeoutError('odom stopped')
                if stamp<tick-.003:
                    time.sleep(.004)
                    continue
                tick+=.2*max(1, int((stamp-tick)/.2)+1)
                pose=np.array(xyzq)
                roll,pitch,yaw=attitude(pose[3:])
                if self.previous_xy is not None:
                    self.distance+=np.linalg.norm(pose[:2]-self.previous_xy)
                self.previous_xy=pose[:2].copy()
                if self.origin_yaw is None: self.origin_yaw=yaw
                imu=min(list(stream.imu)[-45:],key=lambda x:abs(x[1]-stamp))
                ir,ip,_=attitude(imu[2])
                vel=planar_velocity(stream.odom,stamp,pose,yaw)
                state=np.array([self.distance/1000,math.cos(yaw-self.origin_yaw),
                    math.sin(yaw-self.origin_yaw),ip,ir,0,0,*vel,*prev],'f')
                bev,ages=self.memory.update(stream,pose,stamp,yaw)
                if first and not np.any(bev[...,3]==0):
                    if time.monotonic()>deadline:
                        raise RuntimeError(f'No registered LiDAR cells before policy start, age={ages}')
                    time.sleep(.02)
                    continue
                self.obs.update(bev=bev[None],state=state[None],is_first=np.array([first]))
                first=False
                act=np.asarray(self.actor(self.obs))[0]
                local,speed=self.target(act)
                c,s=np.cos(yaw),np.sin(yaw)
                offset=np.array([[c,-s],[s,c]])@np.asarray(local)
                prev=act.copy()
                # Route data is consulted only after actor inference.
                st=self.route.locate(pose,current)
                current=float(st.s)
                width=np.interp(current,self.route.s,self.width)
                reason=''
                if max(abs(roll),abs(pitch))>1.1 or st.clearance<.26: reason='overturned_or_low'
                elif abs(st.lateral)>width/2+self.cfg['task']['off_route_margin']: reason='off_route'
                elif current<best-self.cfg['task']['max_reverse']: reason='reverse'
                elif stamp-progress_t>=35: reason='stuck'
                if current>best+.015: progress_t=stamp
                best=max(best,current)
                if not reason and current>=self.finish: reason='finish'
                if (ROOT/'wrap').exists(): reason='budget'
                beat.target=None if reason else (pose[:2]+offset,float(speed),time.monotonic())
                row=dict(epoch=arrival+stream.epoch_offset,sim=stamp,pose=pose,vel=vel,action=act,state=state,
                         bev=bev,station=current,lateral=st.lateral,clearance=st.clearance,
                         raw_twist=np.asarray(twist),imu_quat=np.asarray(imu[2]),
                         norm_command=beat.last.copy(),lidar_age=np.mean(ages) if ages else np.nan,
                         imu_delta=imu[1]-stamp,assist=assist)
                for k,v in row.items(): rec[k].append(v)
                if len(rec['epoch'])>=50 or reason:
                    flush()
                    save_json(ROOT/'status.json',dict(s=current,best=best,assist=assist,sim=stamp-t0,
                             reason=reason,episode=out.name,epoch=time.time()))
                if reason: return reason,current,float(stamp-t0)
        finally:
            if beat: beat.stop()
            stream.close()
            udp.close()
            flush()

    def run(self):
        calpath=ROOT/'calibration/calibration.json'
        while not calpath.exists():
            if (ROOT/'wrap').exists(): return
            time.sleep(1)
        cal=json.loads(calpath.read_text())
        self.scale=[cal['forward_scale'],cal['yaw_scale']]
        if not (.1<self.scale[0]<4 and .15<self.scale[1]<4):
            raise ValueError(f'Invalid calibration {self.scale}')
        station=.75
        placement='entry-0.75'
        start=time.time()
        total_sim=0.
        for num in range(2000):
            if (ROOT/'wrap').exists(): break
            out=ROOT/f'leg-{num:04}'
            if out.exists(): raise FileExistsError(out)
            instance=Instance(out,placement)
            try:
                instance.ready()
                reason,s,elapsed=self.episode(out,station,num)
            finally:
                instance.close()
            total_sim+=elapsed
            end=dict(reason=reason,s=s,sim=elapsed,start=station,assists=num,epoch=time.time())
            save_json(out/'end.json',end)
            subprocess.Popen([sys.executable,str(BRIDGE/'video.py'),str(out)],
                             stdout=(out/'video.log').open('w'),stderr=subprocess.STDOUT)
            if reason in ('finish','budget'): break
            event=dict(failure_s=s,reason=reason,requested_s=s+1,epoch=time.time())
            if s+1>=self.finish-self.cfg['assist']['end_clearance']:
                event.update(placement_s=self.finish,terminal=True)
                self.events.append(event)
                save_json(ROOT/'events.json',self.events)
                reason='finish_assisted'
                break
            req=ROOT/'request.json'
            answer=ROOT/f'pose-{num:04}.json'
            save_json(req,dict(leg=num,station=max(0.,s+1)))
            while not answer.exists() and not (ROOT/'wrap').exists(): time.sleep(.5)
            if not answer.exists(): reason='budget'; break
            pose=json.loads(answer.read_text())
            station=pose['station_m']
            event.update(placement_s=station)
            self.events.append(event)
            save_json(ROOT/'events.json',self.events)
            placement='assist'+str(num)
            set_pose(placement,pose)
        save_json(ROOT/'outcome.json',dict(reason=reason,assists=len(self.events),final_station=self.finish if reason=='finish_assisted' else s,
                 wall=time.time()-start,sim=total_sim,odometry_distance=self.distance))

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    Evaluation().run()
