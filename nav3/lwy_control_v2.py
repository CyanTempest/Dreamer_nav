"""Parameter-only LWY planner candidates for controlled regressions: legacy dynamic search with only redundant obstacle pooling removed.

Physical collision/support decisions and the planner's larger prediction footprint
are unchanged. Keep this explicit so the legacy controller fingerprint survives.
"""
from functools import partial
import jax
import jax.numpy as jnp
from .env import decode, rotate
from .control import target_action, env_x, env_y


@partial(jax.jit, static_argnames=('candidates','horizon'))
def planning_result(obs, candidates=15, offset_weight=1., lookahead=1.2, prediction_speed=.5, prediction_margin=.04, horizon=70, border_weight=300.):
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
    offsets=jnp.linspace(-.84,.84,candidates);k=len(offsets)
    measured=jnp.where(bev[...,3]<.5,bev[...,1],0.)
    obstacle=measured  # LWY already covers geometry; avoid a second 12.5 cm dilation.
    nearby=jnp.any((measured>.20)&(jnp.abs(env_y())<1.5)&(env_x()>.1)&(env_x()<4),axis=(1,2))
    speed=jnp.where(nearby,.28,prediction_speed)
    pos=jnp.zeros((n,k,2));yaw=jnp.zeros((n,k))
    vel=jnp.broadcast_to(obs['state'][:,7:10,None].swapaxes(1,2),(n,k,3))
    queue=jnp.repeat(vel[:,:,None],3,axis=2)
    xx,yy=jnp.meshgrid(jnp.linspace(-.515-prediction_margin,.515+prediction_margin,7),jnp.linspace(-.29-prediction_margin,.29+prediction_margin,5),indexing='ij')
    footprint=jnp.stack([xx,yy],-1).reshape(-1,2)
    def target_at(pos):
        along=jnp.clip(jnp.sum((pos-origin[:,None])*direction[:,None],-1)+lookahead,0.,5.)
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
    (pos,yaw,vel,queue,risk,border,unknown),_=jax.lax.scan(simulate,(pos,yaw,vel,queue,zeros,zeros,zeros),None,length=horizon)
    score=500*risk+border_weight*border+.002*unknown+offset_weight*offsets[None]**2
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
