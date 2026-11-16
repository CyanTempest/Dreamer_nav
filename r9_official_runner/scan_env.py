"""Current Dreamer observation/action contract on immutable official scan patches."""
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nav3.bank import MapBank
from nav3.r8_p1_online import SimpleEnv, fitted_config


def gather(grid, origins, res, mids, xy):
    mids=mids.reshape((len(mids),)+(1,)*(xy.ndim-2))
    ij=jnp.floor((xy-origins[mids])/res+.5).astype(jnp.int32)
    valid=(ij[...,0]>=0)&(ij[...,1]>=0)&(ij[...,0]<grid.shape[1])&(ij[...,1]<grid.shape[2])
    value=grid[mids,jnp.clip(ij[...,0],0,grid.shape[1]-1),jnp.clip(ij[...,1],0,grid.shape[2]-1)]
    return value.astype(jnp.float32),valid


@jax.tree_util.register_pytree_node_class
class ScanBank(MapBank):
    ARRAY_KEYS=('layers','origins','routes','arcs','widths','yaws','gates','lengths')
    def __init__(self,root):
        root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
        self.records=manifest['maps'];self.count=len(self.records);self.names=[r['name'] for r in self.records]
        arrays=[np.load(root/n/'map.npz') for n in self.names]
        if not arrays:raise ValueError('No scan patches passed route qualification')
        max_route=max(len(a['s']) for a in arrays)
        def pad(value):
            return np.pad(value,[(0,max_route-len(value))]+[(0,0)]*(value.ndim-1),mode='edge')
        shape=np.max([a['layers'].shape[:2] for a in arrays],axis=0)
        layers=np.zeros((self.count,*shape,7),np.float16);layers[...,2:4]=20;layers[...,5]=100
        atlases=np.zeros((self.count,*shape,20),np.float16)
        atlases[...,2:19:3]=-1
        for i,a in enumerate(arrays):
            x,y=a['layers'].shape[:2];layers[i,:x,:y]=a['layers'];atlases[i,:x,:y]=a['atlas']
            assert float(a['res']) == float(np.float32(.1))
        values=dict(layers=layers,origins=np.stack([a['origin'] for a in arrays]),
                    routes=np.stack([pad(a['route']) for a in arrays]),arcs=np.stack([pad(a['s']) for a in arrays]),
                    widths=np.stack([pad(a['width']) for a in arrays]),yaws=np.stack([pad(a['yaw']) for a in arrays]),
                    gates=np.stack([a['gates'] for a in arrays]),lengths=np.array([len(a['s']) for a in arrays],np.int32))
        for k,v in values.items():setattr(self,k,jnp.asarray(v))
        self.initial_atlases=jnp.asarray(atlases)
        for a in arrays:a.close()

    def sample(self,mids,xy,layer=None,linear=False):
        # Nearest cell for occupancy; bilinear ground is used by base dynamics.
        def at(points):
            val,valid=gather(self.layers,self.origins,.1,mids,points)
            empty=jnp.array([0,0,20,20,0,100,0],jnp.float32)
            val=jnp.where(valid[...,None],val,empty)
            return val if layer is None else val[...,layer]
        if not linear:return at(xy)
        idx=mids.reshape((len(mids),)+(1,)*(xy.ndim-2))
        p=(xy-self.origins[idx])/.1;ij=jnp.floor(p);u=p-ij
        lo=self.origins[idx]+ij*.1
        if layer is None:u=u[...,None,:]
        a,b,c,d=[at(lo+jnp.array(o)*.1) for o in ((0,0),(1,0),(0,1),(1,1))]
        return (a*(1-u[...,0])+b*u[...,0])*(1-u[...,1])+(c*(1-u[...,0])+d*u[...,0])*u[...,1]


@jax.tree_util.register_pytree_node_class
class ScanAtlas:
    def __init__(self,bank,n):
        self.layers=bank.initial_atlases;self.origins=bank.origins;self.mids=jnp.arange(n)%bank.count
    def tree_flatten(self):return (self.layers,self.origins,self.mids),None
    @classmethod
    def tree_unflatten(cls,aux,arrays):
        obj=cls.__new__(cls);obj.layers,obj.origins,obj.mids=arrays;return obj
    def sample(self,xy):
        values,valid=gather(self.layers,self.origins,.1,self.mids,xy)
        empty=jnp.zeros(20).at[2:19:3].set(-1)
        return jnp.where(valid[...,None],values,empty)


@jax.tree_util.register_pytree_node_class
class ScanEnv(SimpleEnv):
    def __init__(self,bank,n,body_bands=False):
        config={**fitted_config(),**json.loads(Path('configs/lwy_v2_tracker.json').read_text()),
                'max_steps':1500,'p1_max_band':3 if body_bands else 6}
        super().__init__(bank,n,atlas=ScanAtlas(bank,n),config=config)
        self.reset=partial(jax.jit(ScanEnv._reset),self)
        self.step=partial(jax.jit(ScanEnv._step),self)
        self.observe=partial(jax.jit(ScanEnv._observe),self)
    def _reset(self,key,mids=None,start_s=None,observe=True):
        # Atlas and physics map IDs are fixed together for each vector slot.
        return super()._reset(key,self.atlas.mids,start_s,observe)


def make_env(root,repeats=2,body_bands=False):
    bank=ScanBank(root)
    return ScanEnv(bank,bank.count*repeats,body_bands)
