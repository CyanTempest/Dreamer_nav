"""One cold-start episode in its own net, IPC and mount namespaces (VM only)."""
import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import zenoh
from matrix_mc_udp_test import UdpRobotController
from control import Route, Monitor, Pursuit, upright
from sensors import motion, decode, scan_age, sim_at_wall
from shard import Episode


class Topics:
    def __init__(self):
        self.latest={}
        self.count=Counter()
        self.period={}
        self.error=None
        self.anchors=deque(maxlen=1024)
        self.imu=deque(maxlen=1024)
        self.subs=[]
        cfg=zenoh.Config()
        cfg.insert_json5('mode','"client"')
        cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
        self.session=zenoh.open(cfg)
        for key in ('odom','imu','front_lidar','rear_lidar'):
            self.subs.append(self.session.declare_subscriber('rt/'+key,self.receive))

    def receive(self,sample):
        received=time.monotonic()
        k=str(sample.key_expr).split('/')[-1]
        self.count[k]+=1
        self.period[k]=(self.period.get(k,(received,))[0],received)
        try:
            b=sample.payload.to_bytes()
            if 'lidar' in k:
                # Decode only decision-selected frames, outside the Zenoh callback.
                self.latest[k]=(received,b)
            else:
                values=motion(b,k=='odom')
                if k=='odom':
                    self.anchors.append((received,values[0]))
                row=(received,*values)
                if k=='imu':
                    self.imu.append(row)
                self.latest[k]=row
        except Exception as e:
            self.error=repr(e)

    def close(self):
        for sub in self.subs:
            sub.undeclare()
        self.session.close()


class Commands:
    def __init__(self,path):
        self.path,self.idx=path,0

    def send(self,v,stair=False):
        target=self.path/f'col_cmd_{self.idx}.js'
        temp=target.with_suffix('.tmp')
        temp.write_text('module.exports='+json.dumps(dict(v=list(v),stair=stair,
                       expires=int(time.time()*1000)+500))+';')
        temp.replace(target)
        self.idx+=1


def run(args):
    base=Path('/content/matrix-v4/inputs/data/trail_a_mujoco/full_v2')
    inputs=Path('/content/col/inputs')
    cfg=json.loads(args.config.read_text())
    pose_spec=json.loads((args.run/'spawn.json').read_text())
    window=pose_spec['window']
    meta=dict(window=window,seed=pose_spec['seed'],spawn=pose_spec,gpu=args.gpu,
              instances=args.n,driver='pure_pursuit',config=cfg)
    assembly=json.loads((base/'assembly.json').read_text())
    bounds=next(r['station_m'] for r in assembly['statics'] if r['window']==window+'.xml')
    shift=assembly['shared_shift_m']
    with np.load(base/'route.npz') as data:
        route=Route(data['station_m'],data['xy_m']-shift[:2],np.load(inputs/'ground.npy'),shift[2],bounds,cfg)
    driver=Pursuit(cfg,np.random.default_rng(meta['seed']))
    meta['noise']=driver.mode
    meta['epoch_minus_mono']=offset=time.time()-time.monotonic()
    meta['command_units']='normalized SetWalkCommand forward,left,yaw; odom twist in SI'
    stream=Episode(args.output,args.run.name,meta)
    topics=Topics()
    commands=Commands(args.js)
    zero=np.zeros(3)
    reason='worker_error'
    counts=Counter()
    accepted=False
    odom_range=[]
    wall_range=[]
    caches={}
    try:
        udp=UdpRobotController('127.0.0.1',7447,20,False)
        udp.stand(cfg['stand_s'])
        udp.close()
        commands.send(zero,True)
        until=time.monotonic()+8
        while 'COL_STAIR_ACK true' not in (args.run/'console.log').read_text(errors='replace') and time.monotonic()<until:
            time.sleep(.1)
        if 'COL_STAIR_ACK true' not in (args.run/'console.log').read_text(errors='replace'):
            reason='stair_rejected'
            return
        previous=pose_spec['station_m']
        next_tick=None
        missing_imu_since=None
        meta['imu_pauses']=[]
        reason='benchmark_end'
        while time.time()<args.end:
            now=time.monotonic()
            samples=topics.latest.copy()
            if topics.error:
                raise ValueError(topics.error)
            odom=samples.get('odom')
            if odom is None or now-odom[0]>cfg['odom_timeout_s']:
                reason='odom_timeout'
                break
            recv,stamp,pose,twist=odom
            if next_tick is not None and stamp<next_tick-1e-6:
                if stamp<last_stamp-1e-6:
                    reason='clock_reset'
                    break
                time.sleep(.005)
                continue
            imu=next((v for v in reversed(tuple(topics.imu)) if v[0]<=now and v[1]<=stamp),None)
            if imu is None or stamp-imu[1]>.1:
                if missing_imu_since is None:
                    missing_imu_since=now
                    meta['imu_pauses'].append(dict(mono=now,odom=stamp,action=[0,0,0]))
                    commands.send(zero)
                if now-missing_imu_since>=cfg['odom_timeout_s']:
                    reason='imu_timeout'
                    break
                time.sleep(.005)
                continue
            missing_imu_since=None
            st=route.locate(pose,previous)
            if next_tick is None:
                if not upright(st,cfg,initial=True):
                    reason='stand_failed'
                    break
                accepted=True
                monitor=Monitor(cfg,st.s,stamp)
                first=next_tick=stamp
                previous=st.s
            last_stamp=stamp
            next_tick += (1+max(0,int((stamp-next_tick+1e-6)/cfg['decision_s'])))*cfg['decision_s']
            terminal=monitor.reason(st,stamp)
            scans=[]
            anchors=tuple(topics.anchors)
            ages=[]
            for side in ('front','rear'):
                value=samples.get(side+'_lidar')
                if value is not None and (side not in caches or value[0]!=caches[side].received):
                    caches[side]=decode(value[1],value[0])
                scan=caches.get(side)
                scans.append(scan)
                ages.append(scan_age(scan,now,offset))
            obs=dict(front=scans[0],rear=scans[1],imu=imu[2:],elapsed_sim_s=stamp-first)
            act=zero if terminal else driver.command(obs,st)
            commands.send(act)
            qualified=all(age<=cfg['age_s'] for age in ages) and upright(st,cfg)
            failure=terminal in ('stuck','off_route','overturned')
            reward=cfg['progress_weight']*np.clip(st.s-previous,-cfg['progress_clip_m'],cfg['progress_clip_m'])-cfg['time_penalty']-cfg['failure_penalty']*failure
            row=dict(mono=now,odom_header=stamp,odom_received=recv,imu_header=imu[1],imu_received=imu[0],
                     age_s=ages,scan_sim_end=[sim_at_wall(anchors,v.end-offset) if v else np.nan for v in scans],
                     imu_quat=imu[2],imu_omega=imu[3],action=act,actual_twist=twist,
                     priv_pose=pose,priv_progress=st.s,priv_cross_track=st.lateral,priv_clearance=st.clearance,
                     reward=reward,qualified=bool(qualified),terminal=terminal,
                     frame_index=[stream.scan(k,v) for k,v in zip(('front','rear'),scans)])
            stream.rows.append(row)
            if len(stream.rows)==1 or len(stream.rows)%50==0:
                print('decision',len(stream.rows),'sim',round(stamp-first,3),'s',round(st.s,3),
                      'age',np.round(ages,3).tolist(),'qualified',counts['qualified']+int(qualified),flush=True)
            counts['qualified']+=bool(qualified)
            if not odom_range:
                odom_range.append(stamp)
                wall_range.append(now)
            previous=st.s
            if terminal:
                reason=terminal
                break
        if odom_range:
            odom_range.append(last_stamp)
            wall_range.append(now)
    except Exception:
        reason='worker_error'
        traceback.print_exc()
    finally:
        commands.send(zero)
        topics.close()
        if stream.rows and not stream.rows[-1]['terminal']:
            stream.rows[-1]['terminal']=reason
        meta.update(stand_accepted=accepted,topic_counts=dict(topics.count),topic_period=topics.period,
                    odom_range=odom_range,mono_range=wall_range)
        stream.finish(reason)
        (args.run/'episode.json').write_text(json.dumps(dict(meta,reason=reason,steps=len(stream.rows),
                     qualified=counts['qualified'],closed=time.time())))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('run','js','output','config'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--end',type=float,required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--gpu',required=True)
    run(p.parse_args())
