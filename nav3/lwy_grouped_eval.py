"""Single-environment vmap in fixed groups, sharing atlas and temporary buffers."""
from functools import partial
import jax
import jax.numpy as jnp

GROUP=20


def pad(tree,n):
    size=((n+GROUP-1)//GROUP)*GROUP
    return jax.tree.map(lambda x:jnp.pad(x,[(0,size-n)]+[(0,0)]*(x.ndim-1)),tree)


@jax.jit
def reset(env,keys,mids):
    n=len(keys);keys,mids=pad((keys,mids),n)
    keys=keys.reshape((-1,GROUP,*keys.shape[1:]));mids=mids.reshape((-1,GROUP))
    def group(args):
        k,m=args
        return jax.vmap(lambda e,k,m:e._reset(k,m[None]),in_axes=(None,0,0))(env,k,m)
    values=jax.lax.map(group,(keys,mids))
    return jax.tree.map(lambda x:x.reshape((-1,*x.shape[2:]))[:n],values)


@jax.jit
def step(env,state,obs,extra,action,done):
    n=len(action)
    state,obs,extra,action=pad((state,obs,extra,action),n)
    done=jnp.pad(done,[(0,len(action)-n)],constant_values=True)
    arrays=jax.tree.map(lambda x:x.reshape((-1,GROUP,*x.shape[1:])),(state,obs,extra,action,done))
    def group(args):
        s,o,e,a,d=args
        return jax.lax.cond(jnp.all(d),lambda _:(s,o,e),
            lambda _:jax.vmap(lambda env,s,a:env._step(s,a[None]),in_axes=(None,0,0))(env,s,a),None)
    values=jax.lax.map(group,arrays)
    return jax.tree.map(lambda x:x.reshape((-1,*x.shape[2:]))[:n],values)


def main():
    import json,time
    import numpy as np
    from pathlib import Path
    from .lwy_bank import LwyBank
    from .lwy_safe_obstacles import SafeLwyEnv
    env=SafeLwyEnv(LwyBank('test',root='maps_lwy_v2'),1,False,config={'yaw_tracking_gain':4.})
    old_reset=jax.jit(jax.vmap(lambda e,k,m:e._reset(k,m[None]),in_axes=(None,0,0)))
    old_step=jax.jit(jax.vmap(lambda e,s,a:e._step(s,a[None]),in_axes=(None,0,0)))
    keys=jnp.stack([jax.random.PRNGKey(940000+i) for i in range(100)]);mids=jnp.zeros(100,jnp.int32)
    old=old_reset(env,keys,mids);new=reset(env,keys,mids)
    equal=jax.jit(lambda a,b:jnp.all(jnp.array([jnp.array_equal(x,y) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b))])))
    assert bool(equal(old,new)),'reset differs'
    action=jax.random.uniform(jax.random.PRNGKey(239),(100,3),minval=-1,maxval=1)
    for i in range(100):
        old=old_step(env,old[0],action)
        new=step(env,*new,action,jnp.zeros(100,bool))
        assert bool(equal(old,new)),f'step differs at {i}'
    measurements={}
    for mode in ['direct','grouped']:
        values=old_reset(env,keys,mids);jax.block_until_ready(values);begin=time.monotonic()
        for _ in range(200):
            values=old_step(env,values[0],action) if mode=='direct' else step(env,*values,action,jnp.zeros(100,bool))
        jax.block_until_ready(values);measurements[mode]=time.monotonic()-begin
    result=dict(passed=True,exact_state_observation_extra_steps=100,episodes=100,group=GROUP,seconds_for_200_steps=measurements,
        speedup=measurements['direct']/measurements['grouped'])
    Path('metrics/lwy_v2_grouped_eval_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    from .lwy_eval_v2 import setup
    setup();main()
