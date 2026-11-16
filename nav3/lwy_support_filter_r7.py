"""Optional observed support margin, no privileged inputs or training changes."""
import jax
import jax.numpy as jnp
from .env import decode,rotate


@jax.jit
def support_risk(obs,action,margin=.08):
    bev=decode(obs['bev']);n=len(action);action=jnp.clip(action,-1,1)
    radius=.5+(action[:,0]+1)*1.25;angle=action[:,1]*jnp.pi/3
    target=jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1)
    speed=(action[:,2]+1)*.4
    xx,yy=jnp.meshgrid(jnp.linspace(-.515,.515,7),jnp.linspace(-.29,.29,5),indexing='ij')
    foot=jnp.stack([xx,yy],-1).reshape(-1,2)
    # Dilation only in the optional filter; physical envelope is untouched.
    offsets=jnp.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])*margin
    foot=(foot[:,None]+offsets[None]).reshape(-1,2)
    vel=obs['state'][:,7:10]
    def advance(carry,_):
        pos,yaw,vel,risk=carry
        delta=rotate(target-pos,-yaw);norm=jnp.maximum(jnp.linalg.norm(delta,axis=-1),.15)
        cmd=jnp.concatenate([delta/norm[:,None]*jnp.minimum(speed,norm)[:,None],
            jnp.clip(4*speed*delta[:,1]/norm**2,-.75,.75)[:,None]],-1)
        vel+=jnp.clip((cmd-vel)*(.1/.18),-jnp.array([.065,.065,.12]),jnp.array([.065,.065,.12]))
        pos+=rotate(vel[:,:2],yaw)*.1;yaw+=vel[:,2]*.1
        xy=pos[:,None]+rotate(foot[None],yaw)
        ij=jnp.clip(jnp.floor((xy+jnp.array([2,4]))/.125).astype(jnp.int32),0,63)
        value=bev[jnp.arange(n)[:,None],ij[...,0],ij[...,1]]
        unsupported=(value[...,2]<.5)&(value[...,3]<.5)
        outside_route=value[...,4]<.5
        risk|=jnp.any(unsupported|outside_route,axis=-1)
        return (pos,yaw,vel,risk),None
    (_,_,_,risk),_=jax.lax.scan(advance,(jnp.zeros((n,2)),jnp.zeros(n),vel,jnp.zeros(n,bool)),None,length=8)
    return risk
