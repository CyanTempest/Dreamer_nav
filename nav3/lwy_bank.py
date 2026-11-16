"""LWY sparse world atlas; legacy MapBank and its cache trees are untouched."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .bank import MapBank


@jax.tree_util.register_pytree_node_class
class LwyBank(MapBank):
    ARRAY_KEYS = MapBank.ARRAY_KEYS+('station_tiles', 'resolutions')

    def __init__(self, split='train', limit=None, root='maps_lwy'):
        root = Path(root)
        self.records = [r for r in json.loads((root/'manifest.json').read_text())['maps'] if r['split'] == split]
        if limit: self.records = self.records[:limit]
        self.names = [r['name'] for r in self.records]; self.count = len(self.records)
        assert self.count
        arrays = [np.load(root/name/'map.npz') for name in self.names]
        dims = np.array([a['dimensions'] for a in arrays]).max(0)
        lookup = np.zeros((self.count, *dims), np.int32)
        empty = np.zeros((32, 32, 7), np.float16)
        empty[..., 2:4] = 20; empty[..., 5] = 100
        tiles = [empty[None]]; stations = [np.zeros((1, 32, 32), np.float32)]
        cursor = 1
        for i, a in enumerate(arrays):
            indices = a['tile_indices']; t = a['tiles']
            lookup[i, indices[:, 0], indices[:, 1]] = np.arange(cursor, cursor+len(t))
            cursor += len(t); tiles.append(t); stations.append(a['station_tiles'])
        maxr = max(len(a['s']) for a in arrays)
        def pad(a):
            return np.pad(a, [(0, maxr-len(a))]+[(0, 0)]*(a.ndim-1), mode='edge')
        values = dict(tiles=np.concatenate(tiles), lookup=lookup, station_tiles=np.concatenate(stations),
            origins=np.stack([a['origin'] for a in arrays]), resolutions=np.array([a['res'] for a in arrays]),
            routes=np.stack([pad(a['route']) for a in arrays]), arcs=np.stack([pad(a['s']) for a in arrays]),
            widths=np.stack([pad(a['width']) for a in arrays]), yaws=np.stack([pad(a['yaw']) for a in arrays]),
            gates=np.stack([a['gates'] for a in arrays]), lengths=np.array([len(a['s']) for a in arrays], np.int32))
        self.host_bytes = sum(a.nbytes for a in values.values())
        for k, v in values.items(): setattr(self, k, jnp.asarray(v))
        for a in arrays: a.close()

    def sample(self, mids, xy, layer=None, linear=False):
        mids = mids.reshape((len(mids),)+(1,)*(xy.ndim-2))
        p = (xy-self.origins[mids])/self.resolutions[mids][..., None]
        def get(ij):
            tile = ij//32
            valid = (tile[..., 0] >= 0)&(tile[..., 1] >= 0)&(tile[..., 0] < self.lookup.shape[1])&(tile[..., 1] < self.lookup.shape[2])
            ti = self.lookup[mids, jnp.clip(tile[..., 0], 0, self.lookup.shape[1]-1), jnp.clip(tile[..., 1], 0, self.lookup.shape[2]-1)]
            ti = jnp.where(valid, ti, 0)
            ii, jj = ij[..., 0]%32, ij[..., 1]%32
            if layer == 4: return self.station_tiles[ti, ii, jj]
            if layer is None:
                return self.tiles[ti, ii, jj].astype(jnp.float32).at[..., 4].set(self.station_tiles[ti, ii, jj])
            return self.tiles[ti, ii, jj, layer].astype(jnp.float32)
        if not linear: return get(jnp.floor(p+.5).astype(jnp.int32))
        ij = jnp.floor(p).astype(jnp.int32); u = p-ij
        if layer is None: u = u[..., None, :]
        a, b, c, d = [get(ij+jnp.array(o)) for o in ((0, 0), (1, 0), (0, 1), (1, 1))]
        return (a*(1-u[..., 0])+b*u[..., 0])*(1-u[..., 1])+(c*(1-u[..., 0])+d*u[..., 0])*u[..., 1]
