import numpy as np, jax, jax.numpy as jnp
from nav4.bank import TrailBank
b = TrailBank('r9_official_runner/maps_scan_holdout', ['segment_1010'])
r = np.asarray(b.routes[0]); s = np.asarray(b.arcs[0]); L = int(b.lengths[0])
print('len', L, 'route[80:90]', np.c_[s[78:92], r[78:92]].round(3))
xy = np.array([-21.69, 55.61])
d = np.linalg.norm(r[:L, :2]-xy, axis=1); k = np.argsort(d)[:5]; print('nearest pts', k, d[k].round(3), s[k])
for hint in (16.0, 16.6, 16.8):
    st, lat = b.project(jnp.zeros(1, jnp.int32), jnp.asarray(xy[None], jnp.float32), jnp.array([hint]), 6)
    print('hint', hint, 'proj', st, lat, 'index', b.index(jnp.zeros(1, jnp.int32), jnp.array([hint])))
z = np.load('r9_official_runner/maps_scan_holdout/segment_1010/map.npz')
print('yaw', z['yaw'][78:92].round(3), 'width', z['width'][80:86])
