"""Opt-in round-five training adapters, leaving the production physics frozen."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from .env import FastEnv,rotate
from .provenance import fingerprint


class ExposureMonitor:
    """Count actual collector frames near combinations, separately from prefill."""
    def __init__(self,bank,name):
        import numpy as np
        self.name=name;self.records=bank.records
        self.logs=np.array([r['features']['log'] for r in bank.records])
        self.counts=np.zeros((2,bank.count),np.int64)
        self.near=np.zeros_like(self.counts)

    def observe(self,state,prefill=False):
        import numpy as np
        mids=np.asarray(state['mids']);valid=~np.asarray(state['done'])
        near=valid&(np.asarray(state['s'])>=self.logs[mids]-6)&(np.asarray(state['s'])<=self.logs[mids]+7)
        i=0 if prefill else 1
        self.counts[i]+=np.bincount(mids[valid],minlength=len(self.records))
        self.near[i]+=np.bincount(mids[near],minlength=len(self.records))

    def restore(self,data):
        import numpy as np
        if data:
            assert data['map_seeds']==[r['seed'] for r in self.records]
            self.counts=np.asarray(data['collector_frames'],np.int64)
            self.near=np.asarray(data['near_log_frames'],np.int64)

    def snapshot(self):
        from .memory import atomic_json
        data=dict(map_seeds=[r['seed'] for r in self.records],phases=['prefill','online'],
            new_combination=[r.get('source_root')=='maps_v2' and r.get('combination',{}).get('accepted',False) for r in self.records],
            collector_frames=self.counts.tolist(),near_log_frames=self.near.tolist(),window_m=[-6,7])
        atomic_json(f'metrics/fix_exposure_{self.name}.json',data)
        return data


def require_fix_g1():
    gate=json.loads(Path('metrics/fix_g1.json').read_text())
    assert gate['passed'] and gate['episodes']>=100
    assert gate['environment_fingerprint']==fingerprint()
    import hashlib
    assert gate['manifest_sha256']==hashlib.sha256(Path('maps_v2/manifest.json').read_bytes()).hexdigest()


def add_arguments(p):
    p.add_argument('--map-root',default='maps')
    p.add_argument('--reset-environment',action='store_true')
    p.add_argument('--margin-weight',type=float,default=0.)
    p.add_argument('--margin-distance',type=float,default=.20)
    p.add_argument('--fix-round',action='store_true')


def training_env(bank,n,direct=False,weight=0.,margin=.20):
    env=FastEnv(bank,n,True,direct)
    if not weight:return env
    assert weight>0 and margin>0
    # The native step/termination/observation remains unchanged. Only reward
    # includes a distance-to-occupied-raster penalty over the actual footprint.
    native_step=env.step
    def reward(base,resetting,nxt,obs,extra):
        feet=nxt['pos'][:,None]+rotate(base.foot[None],nxt['yaw'])
        truth,_=base.truth(nxt,feet)
        clearance=jnp.min(truth[...,5],axis=-1)
        cost=weight*jnp.clip((margin-clearance)/margin,0,1)*(~resetting)
        nxt={**nxt,'reward':nxt['reward']-cost,'ep_return':nxt['ep_return']-cost}
        return nxt,{**obs,'reward':nxt['reward']},extra
    compiled=jax.jit(reward)
    def step(state,action):
        nxt,obs,extra=native_step(state,action)
        return compiled(env,state['done'],nxt,obs,extra)
    env.step=step
    return env
