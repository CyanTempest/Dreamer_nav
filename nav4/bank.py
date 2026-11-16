"""TrailBank: dense device arrays for the maps listed in a manifest (build_scan_patches.py format).

layers[M,X,Y,7] float32: 0 ground z, 1 top z, 2 support signed distance (<= support_limit is supported),
    3 corridor distance, 4 route station, 5 obstacle signed distance (<0 collision), 6 obstacle type.
ground[M,X,Y] float32 = atlas channel 0; bands[M,X,Y,19] float16 = atlas channels 1..19
    (per band b: bottom 3b, top 3b+1 (-1 = empty), coverage 3b+2; channel 18 = ground known).
Cells outside a map read as EMPTY: unsupported, outside the corridor, unknown to the sensor.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

EMPTY = np.array([0., 0., 20., 20., 0., 100., 0.], np.float32)


def manifest_path(path):
    path = Path(path)
    return path/'manifest.json' if path.is_dir() else path


@jax.tree_util.register_pytree_node_class
class TrailBank:
    KEYS = ('layers', 'ground', 'bands', 'origins', 'routes', 'arcs', 'yaws', 'widths', 'gates', 'lengths')

    def __init__(self, manifest, names=None):
        path = manifest_path(manifest)
        records = [r for r in json.loads(path.read_text())['maps'] if names is None or r['name'] in names]
        if not records:
            raise ValueError(f'no maps selected from {path}')
        maps = []
        for r in records:
            with np.load(path.parent/r.get('path', f"{r['name']}/map.npz")) as z:
                maps.append({k: z[k] for k in ('layers', 'atlas', 'origin', 'res', 'route', 's', 'yaw', 'width', 'gates')})
        res = {float(m['res']) for m in maps}
        if len(res) != 1:
            raise ValueError(f'maps disagree on resolution: {res}')
        self.res, self.names, self.count = res.pop(), tuple(r['name'] for r in records), len(maps)
        shape = (self.count, *np.max([m['layers'].shape[:2] for m in maps], 0))
        layers = np.broadcast_to(EMPTY, shape+(7,)).copy()
        ground = np.zeros(shape, np.float32)
        bands = np.zeros(shape+(19,), np.float16)
        bands[..., 1:18:3] = -1
        for i, m in enumerate(maps):
            x, y = m['layers'].shape[:2]
            layers[i, :x, :y] = m['layers']
            ground[i, :x, :y] = m['atlas'][..., 0]
            bands[i, :x, :y] = m['atlas'][..., 1:]
        longest = max(len(m['s']) for m in maps)

        def stack(key):
            return np.stack([np.pad(m[key], [(0, longest-len(m[key]))]+[(0, 0)]*(m[key].ndim-1), mode='edge')
                             for m in maps]).astype(np.float32)
        for key, value in dict(layers=layers, ground=ground, bands=bands,
                               origins=np.stack([m['origin'] for m in maps]).astype(np.float32),
                               routes=stack('route'), arcs=stack('s'), yaws=stack('yaw'), widths=stack('width'),
                               gates=np.stack([m['gates'] for m in maps]).astype(np.float32),
                               lengths=np.array([len(m['s']) for m in maps], np.int32)).items():
            setattr(self, key, jnp.asarray(value))
        spacing = [float(np.diff(m['s']).min()) for m in maps]
        if min(spacing) <= 0:
            raise ValueError('route arc length must be strictly increasing')
        self.min_spacing = min(spacing)

    def tree_flatten(self):
        return tuple(getattr(self, k) for k in self.KEYS), (self.res, self.names, self.count, self.min_spacing)

    @classmethod
    def tree_unflatten(cls, aux, arrays):
        obj = cls.__new__(cls)
        obj.res, obj.names, obj.count, obj.min_spacing = aux
        for k, v in zip(cls.KEYS, arrays):
            setattr(obj, k, v)
        return obj

    def _cells(self, mids, xy):
        mids = mids.reshape(mids.shape+(1,)*(xy.ndim-1-mids.ndim))
        return mids, (xy-self.origins[mids])/self.res

    def _gather(self, array, mids, ij, fill):
        valid = (ij[..., 0] >= 0) & (ij[..., 1] >= 0) & (ij[..., 0] < array.shape[1]) & (ij[..., 1] < array.shape[2])
        value = array[mids, jnp.clip(ij[..., 0], 0, array.shape[1]-1), jnp.clip(ij[..., 1], 0, array.shape[2]-1)]
        valid = valid.reshape(valid.shape+(1,)*(value.ndim-valid.ndim))
        return jnp.where(valid, value.astype(jnp.float32), fill)

    def sample(self, mids, xy, layer=None, linear=False):
        """Physical layers at xy [B,...,2] for map ids [B]; nearest cell, or bilinear when linear."""
        mids, p = self._cells(mids, xy)
        layers = self.layers if layer is None else self.layers[..., layer]
        fill = jnp.asarray(EMPTY if layer is None else EMPTY[layer])
        if not linear:
            return self._gather(layers, mids, jnp.floor(p+.5).astype(jnp.int32), fill)
        ij = jnp.floor(p).astype(jnp.int32)
        u = p-ij
        if layer is None:
            u = u[..., None, :]
        a, b, c, d = [self._gather(layers, mids, ij+jnp.array(o), fill) for o in ((0, 0), (1, 0), (0, 1), (1, 1))]
        return (a*(1-u[..., 0])+b*u[..., 0])*(1-u[..., 1])+(c*(1-u[..., 0])+d*u[..., 0])*u[..., 1]

    def atlas(self, mids, xy, bands):
        """Sensor atlas: ground [B,...], band (bottom, top, coverage) [B,...,bands] each, known [B,...]."""
        mids, p = self._cells(mids, xy)
        ij = jnp.floor(p+.5).astype(jnp.int32)
        ground = self._gather(self.ground, mids, ij, 0.)
        empty = jnp.zeros(19).at[1:18:3].set(-1)
        a = self._gather(self.bands, mids, ij, empty)
        return ground, a[..., 0:3*bands:3], a[..., 1:3*bands:3], a[..., 2:3*bands:3], a[..., 18]

    def atlas_ground(self, mids, xy):
        mids, p = self._cells(mids, xy)
        return self._gather(self.ground, mids, jnp.floor(p+.5).astype(jnp.int32), 0.)

    def final(self, mids):
        """Final station = last gate."""
        return self.gates[mids, -1]

    def index(self, mids, s):
        """Segment index i with arcs[i] <= s < arcs[i+1], clipped to [0, length-2] (binary search on real arcs)."""
        mids = mids.reshape(mids.shape+(1,)*(s.ndim-mids.ndim))
        last = jnp.broadcast_to(self.lengths[mids]-1, s.shape)
        lo, hi = jnp.zeros_like(last), last
        for _ in range(int(np.ceil(np.log2(self.arcs.shape[1])))+1):
            mid = (lo+hi+1)//2
            go = self.arcs[mids, mid] <= s
            lo, hi = jnp.where(go, mid, lo), jnp.where(go, hi, mid-1)
        return jnp.clip(lo, 0, last-1)

    def route_at(self, mids, s):
        """Centreline point [...,3], yaw and width at station s [B,...]."""
        idx = self.index(mids, s)
        mids = mids.reshape(mids.shape+(1,)*(s.ndim-mids.ndim))
        lo, hi = self.arcs[mids, idx], self.arcs[mids, idx+1]
        u = jnp.clip((s-lo)/jnp.maximum(hi-lo, 1e-6), 0, 1)
        p = self.routes[mids, idx]*(1-u[..., None])+self.routes[mids, idx+1]*u[..., None]
        yaw = self.yaws[mids, idx]*(1-u)+self.yaws[mids, idx+1]*u
        width = self.widths[mids, idx]*(1-u)+self.widths[mids, idx+1]*u
        return p, yaw, width

    def project(self, mids, xy, hint, window):
        """Station and signed lateral offset of xy [B,2], searching +-window segments around station hint [B]."""
        idx = self.index(mids, hint)[:, None]+jnp.arange(-window, window+1)
        idx = jnp.clip(idx, 0, self.lengths[mids, None]-2)
        a, b = self.routes[mids[:, None], idx, :2], self.routes[mids[:, None], idx+1, :2]
        v = b-a
        u = jnp.clip(jnp.sum((xy[:, None]-a)*v, -1)/jnp.maximum(jnp.sum(v*v, -1), 1e-8), 0, 1)
        d = xy[:, None]-a-u[..., None]*v
        best = jnp.argmin(jnp.sum(d*d, -1), -1)
        rows = jnp.arange(len(mids))
        idx, u, d, v = idx[rows, best], u[rows, best], d[rows, best], v[rows, best]
        s = self.arcs[mids, idx]+u*(self.arcs[mids, idx+1]-self.arcs[mids, idx])
        lateral = (v[:, 0]*d[:, 1]-v[:, 1]*d[:, 0])/jnp.maximum(jnp.linalg.norm(v, axis=-1), 1e-6)
        return s, lateral
