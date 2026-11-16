"""Noisy-observation sampling planner and independent geometric action filter.

Adapted from exp_minimap/control.py: score swept footprints, penalize unknowns,
and route-condition candidates. The returned action is a local target, not velocity.
"""
import jax
import jax.numpy as jnp
from .env import decode,rotate


def target_action(target,speed):
    radius = jnp.linalg.norm(target,axis=-1)
    angle = jnp.arctan2(target[...,1],target[...,0])
    return jnp.clip(jnp.stack([(radius-.5)/1.25-1,angle/(jnp.pi/3),speed/.4-1],-1),-1,1)


def assess(bev,target):
    """Assess observed rectangles along a 2 m path, with separate unknown risk."""
    n,k = target.shape[:2]
    fractions = jnp.array([.15,.3,.5,.7,1.])
    path = target[:,:,None,:]*fractions[None,None,:,None]
    theta = jnp.arctan2(target[...,1],target[...,0])
    xx,yy = jnp.meshgrid(jnp.array([-.515,-.25,0,.25,.515]),jnp.array([-.29,0,.29]),indexing='ij')
    foot = jnp.stack([xx,yy],-1).reshape(-1,2)
    points = path[:,:,:,None,:]+rotate(foot[None,None,None],theta[:,:,None]*fractions[None,None,:])
    ij = jnp.floor((points+jnp.array([2,4]))/.125).astype(jnp.int32)
    inside = jnp.all((ij>=0)&(ij<64),-1)
    ij = jnp.clip(ij,0,63)
    v = bev[jnp.arange(n)[:,None,None,None],ij[...,0],ij[...,1]]
    known = v[...,3]<.5
    # A maximum of observed obstacle heights, not averaging thin poles away.
    obstacle = known&(v[...,1]>.21)
    drop = known&(v[...,0]<-.6)
    body = (jnp.abs(points[...,0])<.54)&(jnp.abs(points[...,1])<.32)
    unknown = (~known)&(~body)
    risk = jnp.mean((obstacle|drop|(~inside)).astype(jnp.float32),(-1,-2))
    corridor = jnp.mean((v[...,4]<.5).astype(jnp.float32),(-1,-2))
    return risk,jnp.mean(unknown.astype(jnp.float32),(-1,-2)),corridor


@jax.jit
def lane_planner(obs):
    bev = decode(obs['bev'])
    goals = obs['state'][:,:6].reshape(-1,3,2)*4
    direction = goals[:,2]-goals[:,0]
    direction = direction/jnp.maximum(jnp.linalg.norm(direction,axis=-1,keepdims=True),.01)
    normal = jnp.stack([-direction[:,1],direction[:,0]],-1)
    offsets = jnp.linspace(-.84,.84,15)
    # Evaluate whole offset lanes, so the target does not recenter while the
    # front/side envelope is still alongside a log. The old straight-target
    # score allowed precisely that early return to the centerline.
    stations = jnp.array([.25,.5,.75,1.,1.25,1.5,1.75,2.,2.5,3.])
    origin = 2*goals[:,0]-goals[:,1]
    idx = jnp.minimum(stations.astype(jnp.int32),2)
    points = jnp.concatenate([origin[:,None],goals],1)
    u = stations-idx
    route = points[:,idx]*(1-u[None,:,None])+points[:,idx+1]*u[None,:,None]
    lane = route[:,None]+offsets[None,:,None,None]*normal[:,None,None]
    lane -= (origin[:,None,None]+offsets[None,:,None,None]*normal[:,None,None])*jnp.exp(-stations[None,None,:,None]/.55)
    # Conservative measured obstacle dilation covers 10 cm map rasterization,
    # 12.5 cm observation bins and the controller's finite reaction distance.
    obstacle = jnp.where(bev[...,3]<.5,bev[...,1],0.)
    obstacle = jax.lax.reduce_window(obstacle,0.,jax.lax.max,(1,3,3),(1,1,1),'SAME')
    angle = jnp.arctan2(direction[:,1],direction[:,0])
    xx,yy = jnp.meshgrid(jnp.linspace(-.515,.515,7),jnp.array([-.29,0,.29]),indexing='ij')
    foot = jnp.stack([xx,yy],-1).reshape(-1,2)
    samples = lane[:,:,:,None]+rotate(foot[None,None,None],angle[:,None,None])
    ij = jnp.clip(jnp.floor((samples+jnp.array([2.,4.]))/.125).astype(jnp.int32),0,63)
    n = len(goals);batch = jnp.arange(n)[:,None,None,None]
    heights = obstacle[batch,ij[...,0],ij[...,1]]
    values = bev[batch,ij[...,0],ij[...,1]]
    risk = jnp.mean((heights>.21).astype(jnp.float32),(-1,-2))
    corridor = jnp.mean((values[...,4]<.5).astype(jnp.float32),(-1,-2))
    unknown = jnp.mean(values[...,3],(-1,-2))
    score = 400*risk+80*corridor+.05*unknown+.7*offsets[None]**2
    best = jnp.argmin(score,-1)
    target = lane[jnp.arange(n),best,3]
    chosen_risk = risk[jnp.arange(n),best]
    nearby = jnp.any((obstacle>.21)&(jnp.abs(env_y())<1.4)&(env_x()>.3)&(env_x()<3.5),axis=(1,2))
    speed = jnp.where(chosen_risk>.005,.12,jnp.where(nearby,.32,.55))
    return target_action(target,speed)


def env_x():
    return (-2+(jnp.arange(64)+.5)*.125)[:,None]


def env_y():
    return (-4+(jnp.arange(64)+.5)*.125)[None,:]


@jax.jit
def grid_planner(obs):
    """Forward grid search over measured free rectangles, following a whole path.

    A 3 m horizon keeps the robot in a bypass until its rear envelope has passed.
    Unknown sensor cells carry a small cost; route-prior cells do not become
    obstacle-free labels. No privileged map or simulator state is read.
    """
    bev=decode(obs['bev']);n=len(bev)
    obstacle=(bev[...,1]>.20)&(bev[...,3]<.5)
    # Max pooling retains poles/logs. Footprint includes a 9 cm lateral margin.
    occupied=jax.lax.reduce_window(obstacle.astype(jnp.float32),0.,jax.lax.max,(1,11,7),(1,1,1),'SAME')
    outside=jax.lax.reduce_window((bev[...,4]<.5).astype(jnp.float32),0.,jax.lax.max,(1,9,7),(1,1,1),'SAME')
    corridor=bev[...,4]
    yy=env_y()
    center=jnp.sum(corridor*yy,axis=-1)/jnp.maximum(corridor.sum(-1),1)
    centercost=(yy-center[...,None])**2
    costs=1000*occupied+1000*outside+.035*centercost+.003*bev[...,3]+.01
    # Backward dynamic program, x rows 16..39 (0..3 m), 8-connected moves.
    goals=obs['state'][:,:6].reshape(n,3,2)*4
    terminal=.3*(yy[0][None]-goals[:,2,1,None])**2
    def backup(next_cost,row):
        neighbors=jnp.stack([jnp.pad(next_cost,((0,0),(1,0)),constant_values=1e6)[:,:64]+.012,
            next_cost,jnp.pad(next_cost,((0,0),(0,1)),constant_values=1e6)[:,1:]+.012],-1)
        current=row+jnp.min(neighbors,-1)
        return current,current
    _,reverse=jax.lax.scan(backup,terminal,costs[:,16:40].swapaxes(0,1)[::-1])
    values=reverse[::-1].swapaxes(0,1)
    yidx=jnp.where(values[:,0,31]<values[:,0,32],31,32)
    def forward(y,row):
        options=jnp.clip(y[:,None]+jnp.arange(-1,2),0,63)
        val=row[jnp.arange(n)[:,None],options]+jnp.array([.012,0.,.012])
        new=options[jnp.arange(n),jnp.argmin(val,-1)]
        return new,new
    _,path=jax.lax.scan(forward,yidx,values[:,1:].swapaxes(0,1))
    # Look ahead 0.94 m, and keep an explicit residual velocity safety margin.
    target=jnp.stack([jnp.full(n,.9375),-4+(path[6]+.5)*.125],-1)
    # Route heading improves tracking when a raster row has equal left/right cost.
    target=target.at[:,1].add(jnp.clip(goals[:,0,1]*.05,-.015,.015))
    blocked=values[jnp.arange(n),0,yidx]>500
    nearby=jnp.any(obstacle&(jnp.abs(yy)<1.5)&(env_x()>.3)&(env_x()<3.5),axis=(1,2))
    speed=jnp.where(blocked,0.,jnp.where(nearby,.26,.5))
    return target_action(target,speed)


@jax.jit
def planning_result(obs):
    """Sample offset paths using the same omnidirectional tracking dynamics.

    The score includes heading-dependent rectangular footprints and command
    inertia, rather than treating a target point as an instantly reachable path.
    """
    bev=decode(obs['bev']);n=len(bev)
    goals=obs['state'][:,:6].reshape(n,3,2)*4
    direction=goals[:,2]-goals[:,0]
    direction=direction/jnp.maximum(jnp.linalg.norm(direction,axis=-1,keepdims=True),.01)
    normal=jnp.stack([-direction[:,1],direction[:,0]],-1)
    origin=2*goals[:,0]-goals[:,1]
    offsets=jnp.linspace(-.84,.84,15);k=len(offsets)
    measured=jnp.where(bev[...,3]<.5,bev[...,1],0.)
    obstacle=jax.lax.reduce_window(measured,0.,jax.lax.max,(1,3,3),(1,1,1),'SAME')
    nearby=jnp.any((measured>.20)&(jnp.abs(env_y())<1.5)&(env_x()>.1)&(env_x()<4),axis=(1,2))
    speed=jnp.where(nearby,.28,.5)
    pos=jnp.zeros((n,k,2));yaw=jnp.zeros((n,k))
    vel=jnp.broadcast_to(obs['state'][:,7:10,None].swapaxes(1,2),(n,k,3))
    queue=jnp.repeat(vel[:,:,None],3,axis=2)
    xx,yy=jnp.meshgrid(jnp.linspace(-.555,.555,7),jnp.linspace(-.33,.33,5),indexing='ij')
    footprint=jnp.stack([xx,yy],-1).reshape(-1,2)
    def target_at(pos):
        along=jnp.clip(jnp.sum((pos-origin[:,None])*direction[:,None],-1)+1.2,0.,5.)
        idx=jnp.minimum(along.astype(jnp.int32),2);u=along-idx
        points=jnp.concatenate([origin[:,None],goals],1)
        a=points[jnp.arange(n)[:,None],idx];b=points[jnp.arange(n)[:,None],idx+1]
        return a*(1-u[...,None])+b*u[...,None]+offsets[None,:,None]*normal[:,None]
    targets=target_at(pos)
    def simulate(carry,_):
        pos,yaw,vel,queue,risk,border,unknown=carry
        local=rotate(target_at(pos)-pos,-yaw)
        norm=jnp.maximum(jnp.linalg.norm(local,axis=-1),.1)
        cmdxy=local/norm[...,None]*speed[:,None,None]
        cmdw=jnp.clip(2*speed[:,None]*local[...,1]/norm**2,-.75,.75)
        cmd=jnp.concatenate([cmdxy,cmdw[...,None]],-1)
        queue=jnp.concatenate([cmd[:,:,None],queue[:,:,:-1]],2)
        dv=(queue[:,:,-1]-vel)*(.1/.18)
        vel=vel+jnp.clip(dv,-jnp.array([.065,.065,.12]),jnp.array([.065,.065,.12]))
        pos=pos+rotate(vel[...,:2],yaw)*.1;yaw=yaw+vel[...,2]*.1
        sample=pos[:,:,None]+rotate(footprint[None,None],yaw)
        ij=jnp.clip(jnp.floor((sample+jnp.array([2,4]))/.125).astype(jnp.int32),0,63)
        bi=jnp.arange(n)[:,None,None]
        ob=obstacle[bi,ij[...,0],ij[...,1]]
        value=bev[bi,ij[...,0],ij[...,1]]
        risk+=jnp.mean((ob>.20).astype(jnp.float32),-1)
        border+=jnp.mean((value[...,4]<.5).astype(jnp.float32),-1)
        unknown+=jnp.mean(value[...,3],-1)
        return (pos,yaw,vel,queue,risk,border,unknown),None
    zeros=jnp.zeros((n,k))
    (pos,yaw,vel,queue,risk,border,unknown),_=jax.lax.scan(simulate,(pos,yaw,vel,queue,zeros,zeros,zeros),None,length=70)
    score=500*risk+300*border+.002*unknown+offsets[None]**2
    best=jnp.argmin(score,-1)
    target=targets[jnp.arange(n),best]
    # Brake if every sampled dynamic trajectory has observed collision risk.
    selected=risk[jnp.arange(n),best]+border[jnp.arange(n),best]
    speed=jnp.where(selected>.2,jnp.minimum(speed,.12),speed)
    return target_action(target,speed),dict(selected_risk=risk[jnp.arange(n),best],selected_border=border[jnp.arange(n),best],
        offset=offsets[best],min_risk=jnp.min(risk,-1),scores=score)


@jax.jit
def planner(obs):
    return planning_result(obs)[0]


@jax.jit
def safety_filter(obs,action):
    distance = .5+(action[:,0]+1)*1.25
    angle = action[:,1]*jnp.pi/3
    target = jnp.stack([distance*jnp.cos(angle),distance*jnp.sin(angle)],-1)
    risk,unknown,corridor = assess(decode(obs['bev']),target[:,None])
    intervene = (risk[:,0]>.01)|(corridor[:,0]>.15)|(unknown[:,0]>.65)
    replacement = planner(obs)
    return jnp.where(intervene[:,None],replacement,action),intervene


@jax.jit
def direct_safety_filter(obs,action):
    velocity=action*jnp.array([.8,.6,.75])
    t=1.5
    # Average-heading integration is conservative at these bounded yaw rates.
    xy=rotate(velocity[:,:2]*t,velocity[:,2]*t/2)
    risk,unknown,corridor=assess(decode(obs['bev']),xy[:,None])
    intervene=(risk[:,0]>.01)|(corridor[:,0]>.15)|(unknown[:,0]>.65)|(velocity[:,0]<-.05)
    target=planner(obs);angle=target[:,1]*jnp.pi/3;speed=(target[:,2]+1)*.4
    radius=.5+(target[:,0]+1)*1.25
    replacement=jnp.clip(jnp.stack([speed*jnp.cos(angle)/.8,speed*jnp.sin(angle)/.6,2*speed*jnp.sin(angle)/radius/.75],-1),-1,1)
    return jnp.where(intervene[:,None],replacement,action),intervene
