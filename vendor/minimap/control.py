"""Shared noisy-BEV sampling planner and independent geometry filter, no scene access."""
import numpy as np
from numba import njit
from sensing import decode_bev

SCALE=np.array([.9,.45,.9],np.float32)

@njit(cache=True)
def assess(bev,action):
    vx,vy,w=action*SCALE
    danger=0.;unknown=0.;samples=0
    for t in (.25,.5,.8,1.1):
        a=w*t;c=np.cos(a);s=np.sin(a)
        if abs(w)>.001:
            px=(vx*np.sin(a)+vy*(np.cos(a)-1))/w;py=(vx*(1-np.cos(a))+vy*np.sin(a))/w
        else:px=vx*t;py=vy*t
        for xx in (-.515,-.25,0.,.25,.515):
            for yy in (-.29,0.,.29):
                x=px+c*xx-s*yy;y=py+s*xx+c*yy
                if abs(x)<.52 and abs(y)<.30:continue
                i=int((x+4)/.125);j=int((y+4)/.125)
                samples+=1
                if i<1 or i>62 or j<1 or j>62:danger+=1;continue
                # A one-cell measured neighborhood tolerates angular subsampling;
                # unknown is retained separately, never re-labelled as free.
                seen=0.;sup=0.;obstacle=0.;drop=0.
                for di in (-1,0,1):
                    for dj in (-1,0,1):
                        if bev[i+di,j+dj,3]<.5:
                            seen+=1;sup+=bev[i+di,j+dj,2]
                            obstacle=max(obstacle,bev[i+di,j+dj,1]);drop=min(drop,bev[i+di,j+dj,0])
                if seen==0:unknown+=1
                elif obstacle>.30 or drop<-.50 or sup/seen<.25:danger+=1
    return danger/max(1,samples),unknown/max(1,samples)

@njit(cache=True)
def choose(bev,goals,prev):
    # Goals are the same 3 noisy body-frame points supplied to Dreamer.
    gx,gy=goals[0,0],goals[0,1]
    heading=np.arctan2(goals[1,1]-gy,goals[1,0]-gx)
    desired_w=min(.7,max(-.7,1.4*heading))
    desired_y=min(.4,max(-.4,1.0*gy))
    best=np.zeros(3,np.float32);bestcost=1e9
    for vx in (-.18,0.,.35,.65,.85):
        for vy in (-.35,-.17,0.,.17,.35,desired_y):
            for w in (desired_w-.25,desired_w,desired_w+.25):
                act=np.array([vx/.9,vy/.45,min(.9,max(-.9,w))/.9],np.float32)
                risk,unknown=assess(bev,act)
                score=120*risk+3*unknown+1.6*(vy-desired_y)**2+.6*(w-desired_w)**2-1.6*vx*np.cos(heading)-.6*vy*np.sin(heading)+.05*np.sum((act-prev)**2)
                if score<bestcost:bestcost=score;best=act
    return best

def planner(image,state):
    bev=decode_bev(image)
    return choose(bev,state[:6].reshape(3,2)*4,state[10:13])

def safety_filter(image,state,proposed):
    bev=decode_bev(image);risk,unknown=assess(bev,np.asarray(proposed,np.float32))
    # Route-conditioned bounding is also available to the baseline through state.
    goals=state[:6].reshape(3,2)*4
    speed=proposed*SCALE
    predicted_y=speed[1]*1.1+.5*speed[0]*speed[2]*1.1**2
    route_y=goals[0,1]*max(.2,speed[0]) / max(.5,goals[0,0])
    corridor=abs(predicted_y-route_y)>.32 or (speed[0]<-.07)
    if risk>.015 or unknown>.6 or corridor:
        candidate=choose(bev,goals,state[10:13]);r,u=assess(bev,candidate)
        if r>.025 or u>.7:candidate=np.zeros(3,np.float32)
        return candidate,True,('route' if corridor else 'geometry' if risk>.015 else 'unknown')
    return np.clip(proposed,-1,1),False,'accepted'
