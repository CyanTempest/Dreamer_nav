"""Sparse tiled GPU map bank. Same map seed set is shared across algorithms."""
import json
from pathlib import Path
import numpy as np
import jax


@jax.tree_util.register_pytree_node_class
class MapBank:
    ARRAY_KEYS=('tiles','lookup','origins','routes','arcs','widths','yaws','gates','lengths')

    def tree_flatten(self):
        return tuple(getattr(self,k) for k in self.ARRAY_KEYS),self.count

    @classmethod
    def tree_unflatten(cls,count,arrays):
        obj=cls.__new__(cls);obj.count=count
        for k,v in zip(cls.ARRAY_KEYS,arrays):setattr(obj,k,v)
        return obj

    def __init__(self, split='train', limit=None, root='maps'):
        import jax.numpy as jnp
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_text())
        self.records = [r for r in manifest['maps'] if r['split'] == split]
        if limit:
            self.records = self.records[:limit]
        self.names = [r['name'] for r in self.records]
        self.count = len(self.names)
        assert self.count
        cache = root/f'atlas_{split}_{self.count}'
        cache.mkdir(exist_ok=True)
        keys = self.ARRAY_KEYS
        signature=[r['sha256'] for r in self.records]
        existing=json.loads((cache/'done.json').read_text()) if (cache/'done.json').exists() else {}
        if existing.get('signature')!=signature:
            block = 32
            arrays = [np.load(root/name/'map.npz') for name in self.names]
            dims = np.array([a['layers'].shape[:2] for a in arrays])
            shape = (dims.max(0)+block-1)//block
            lookup = np.zeros((self.count, *shape), np.int32)
            empty = np.zeros((block, block, 7), np.float16)
            empty[..., 2:4] = 20
            empty[..., 5] = 100
            tiles = [empty]
            for m, a in enumerate(arrays):
                arr = a['layers']
                for i in range((arr.shape[0]+block-1)//block):
                    for k in range((arr.shape[1]+block-1)//block):
                        part = arr[i*block:(i+1)*block, k*block:(k+1)*block]
                        if np.min(part[..., 2]) > 8:
                            continue
                        tile = empty.copy()
                        tile[:part.shape[0], :part.shape[1]] = part
                        lookup[m,i,k] = len(tiles)
                        tiles.append(tile)
            maxr = max(len(a['s']) for a in arrays)
            pad = lambda a: np.pad(a, [(0,maxr-len(a))]+[(0,0)]*(a.ndim-1), mode='edge')
            values = dict(tiles=np.stack(tiles), lookup=lookup,
                origins=np.stack([a['origin'] for a in arrays]),
                routes=np.stack([pad(a['route']) for a in arrays]),
                arcs=np.stack([pad(a['s']) for a in arrays]),
                widths=np.stack([pad(a['width']) for a in arrays]),
                yaws=np.stack([pad(a['yaw']) for a in arrays]),
                gates=np.stack([a['gates'] for a in arrays]),
                lengths=np.array([len(a['s']) for a in arrays], np.int32))
            for k,v in values.items():
                np.save(cache/f'{k}.npy', v)
            (cache/'done.json').write_text(json.dumps(dict(names=self.names,signature=signature,bytes=int(values['tiles'].nbytes))))
        for k in keys:
            setattr(self, k, jnp.asarray(np.load(cache/f'{k}.npy', mmap_mode='r')))

    def sample(self, mids, xy, layer=None, linear=False):
        import jax.numpy as jnp
        # mids [B], xy [B,...,2]
        mids = mids.reshape((len(mids),)+(1,)*(xy.ndim-2))
        p = (xy-self.origins[mids])/.1
        def get(ij):
            tile = ij//32
            valid = (tile[...,0]>=0)&(tile[...,1]>=0)&(tile[...,0]<self.lookup.shape[1])&(tile[...,1]<self.lookup.shape[2])
            ti = self.lookup[mids, jnp.clip(tile[...,0],0,self.lookup.shape[1]-1), jnp.clip(tile[...,1],0,self.lookup.shape[2]-1)]
            ti = jnp.where(valid, ti, 0)
            if layer is None:
                return self.tiles[ti, ij[...,0]%32, ij[...,1]%32].astype(jnp.float32)
            return self.tiles[ti, ij[...,0]%32, ij[...,1]%32, layer].astype(jnp.float32)
        if not linear:
            return get(jnp.floor(p+.5).astype(jnp.int32))
        ij = jnp.floor(p).astype(jnp.int32)
        u = p-ij
        if layer is None:
            u = u[...,None,:]
        a,b,c,d = [get(ij+jnp.array(o)) for o in ((0,0),(1,0),(0,1),(1,1))]
        return (a*(1-u[...,0])+b*u[...,0])*(1-u[...,1])+(c*(1-u[...,0])+d*u[...,0])*u[...,1]

    def route_at(self, mids, s):
        import jax.numpy as jnp
        mids = mids.reshape((len(mids),)+(1,)*(s.ndim-1))
        idx = jnp.clip((s/.2).astype(jnp.int32),0,self.lengths[mids]-2)
        lo,hi = self.arcs[mids,idx],self.arcs[mids,idx+1]
        u = jnp.clip((s-lo)/jnp.maximum(hi-lo,1e-6),0,1)
        p = self.routes[mids,idx]*(1-u[...,None])+self.routes[mids,idx+1]*u[...,None]
        yaw = self.yaws[mids,idx]*(1-u)+self.yaws[mids,idx+1]*u
        width = self.widths[mids,idx]*(1-u)+self.widths[mids,idx+1]*u
        return p, yaw, width

    def project(self, mids, xy):
        import jax.numpy as jnp
        rough = self.sample(mids,xy,4)
        idx = jnp.clip((rough[:,None]/.2).astype(jnp.int32)+jnp.arange(-2,3),0,self.lengths[mids,None]-2)
        a,b = self.routes[mids[:,None],idx,:2],self.routes[mids[:,None],idx+1,:2]
        v = b-a
        u = jnp.clip(jnp.sum((xy[:,None]-a)*v,-1)/jnp.maximum(jnp.sum(v*v,-1),1e-8),0,1)
        d = xy[:,None]-a-u[...,None]*v
        best = jnp.argmin(jnp.sum(d*d,-1),-1)
        idx,u,d,v = [x[jnp.arange(len(mids)),best] for x in (idx,u,d,v)]
        s = self.arcs[mids,idx]+u*(self.arcs[mids,idx+1]-self.arcs[mids,idx])
        signed = (v[:,0]*d[:,1]-v[:,1]*d[:,0])/jnp.maximum(jnp.linalg.norm(v,axis=-1),1e-6)
        return s,signed
