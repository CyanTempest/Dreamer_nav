import json
import time
from pathlib import Path
import numpy as np
import zenoh
from sensors import motion
from control import attitude
from actuator import Actuator
from matrix_mc_udp_test import UdpRobotController

def measure(out):
    cfg = zenoh.Config()
    cfg.insert_json5('mode','"client"')
    cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
    session = zenoh.open(cfg)
    rows=[]
    def receive(sample):
        t,p,v=motion(sample.payload.to_bytes(),True)
        rows.append([time.monotonic(),t,*p,*v])
    sub=session.declare_subscriber('rt/odom',receive)
    udp=UdpRobotController('127.0.0.1',7447,20,False)
    udp.stand(4)
    motor=Actuator()
    intervals=[]
    for f,w,seconds in [(0,0,1),(.25,0,5),(.45,0,5),(0,0,1),(0,.4,4),(0,-.4,4),(0,0,1)]:
        start=time.monotonic()
        while time.monotonic()-start<seconds:
            motor.send(f,w,attitude(rows[-1][5:9])[2],stair=motor.index==0)
            time.sleep(.18)
        intervals.append([start,time.monotonic(),f,w])
    sub.undeclare()
    session.close()
    udp.close()
    a=np.array(rows)
    samples=[]
    for start,end,f,w in intervals:
        b=a[(a[:,0]>=start+1)&(a[:,0]<end)]
        if len(b)<10 or not (f or w): continue
        heading=np.unwrap([attitude(q)[2] for q in b[:,5:9]])
        forward=np.mean(np.sum(np.diff(b[:,2:4],axis=0)*np.stack([np.cos(heading[:-1]),np.sin(heading[:-1])],1),1)/np.diff(b[:,1]))
        yaw=(heading[-1]-heading[0])/(b[-1,1]-b[0,1])
        samples.append(dict(f=f,w=w,forward=forward,yaw=yaw,n=len(b),pitch=float(np.max(abs(np.array([attitude(q)[1] for q in b[:,5:9]]))))))
    with (out/'calibration.json').open('w') as fd:
        json.dump(dict(samples=samples,forward_scale=np.median([x['forward']/x['f'] for x in samples if x['f']]),yaw_scale=np.median([x['yaw']/x['w'] for x in samples if x['w']])),fd,indent=2)
    np.savez_compressed(out/'cal_trace.npz',odom=a,intervals=intervals)

if __name__=='__main__':
    import sys
    measure(Path(sys.argv[1]))
