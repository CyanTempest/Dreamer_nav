"""Config loading, planar geometry and the BEV/action encodings shared by env, planner and policy."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

CONFIG = Path(__file__).resolve().parent/'config'


def load_config(name, overrides=None):
    """Read nav4/config/<name>.json and apply {'section.key': value} overrides (keys must exist)."""
    cfg = json.loads((CONFIG/f'{name}.json').read_text())
    for key, value in (overrides or {}).items():
        *path, last = key.split('.')
        node = cfg
        for part in path:
            node = node[part]
        if last not in node:
            raise KeyError(f'unknown {name} config key {key}')
        node[last] = value
    return cfg


def rotate(points, yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    while c.ndim < points.ndim-1:
        c, s = c[..., None], s[..., None]
    return jnp.stack([c*points[..., 0]-s*points[..., 1], s*points[..., 0]+c*points[..., 1]], -1)


def wrap(x):
    return (x+jnp.pi) % (2*jnp.pi)-jnp.pi


def control_dt(cfg):
    return cfg['timing']['dt']/cfg['timing']['substeps']


def encode(bev, cfg):
    b = cfg['bev']
    return (jnp.clip(bev*jnp.array(b['scale'])+jnp.array(b['offset']), 0, 1)*255).astype(jnp.uint8)


def decode(bev, cfg):
    b = cfg['bev']
    return (bev.astype(jnp.float32)/255-jnp.array(b['offset']))/jnp.array(b['scale'])


def action_target(action, cfg):
    """Policy action in [-1,1]^3 -> (local target xy, speed)."""
    a = cfg['action']
    u = (jnp.clip(action, -1, 1)+1)/2
    radius = a['radius'][0]+u[..., 0]*(a['radius'][1]-a['radius'][0])
    bearing = action[..., 1].clip(-1, 1)*np.deg2rad(a['bearing_deg'])
    speed = a['speed'][0]+u[..., 2]*(a['speed'][1]-a['speed'][0])
    return jnp.stack([radius*jnp.cos(bearing), radius*jnp.sin(bearing)], -1), speed


def target_action(target, speed, cfg):
    """Inverse of action_target (clipped to the action box)."""
    a = cfg['action']
    radius = jnp.linalg.norm(target, axis=-1)
    bearing = jnp.arctan2(target[..., 1], target[..., 0])
    return jnp.clip(jnp.stack([2*(radius-a['radius'][0])/(a['radius'][1]-a['radius'][0])-1,
                               bearing/np.deg2rad(a['bearing_deg']),
                               2*(speed-a['speed'][0])/(a['speed'][1]-a['speed'][0])-1], -1), -1, 1)


def bev_xy(cfg, pad=(0, 0)):
    """Robot-frame cell centres [size+2*pad0, size+2*pad1, 2]; x forward from -rear, y left, centred."""
    b = cfg['bev']
    i = np.arange(-pad[0], b['size']+pad[0])
    j = np.arange(-pad[1], b['size']+pad[1])
    x, y = np.meshgrid(-b['rear']+(i+.5)*b['res'], -b['size']*b['res']/2+(j+.5)*b['res'], indexing='ij')
    return np.stack([x, y], -1).astype(np.float32)


def bev_index(xy, cfg):
    """Robot-frame xy -> integer BEV cell index (unclipped)."""
    b = cfg['bev']
    return jnp.floor((xy+jnp.array([b['rear'], b['size']*b['res']/2]))/b['res']).astype(jnp.int32)


def grid(length, width, shape):
    """Rectangle sample points [nx*ny, 2] spanning +-length/2, +-width/2."""
    x, y = np.meshgrid(np.linspace(-length/2, length/2, shape[0]), np.linspace(-width/2, width/2, shape[1]),
                       indexing='ij')
    return np.stack([x, y], -1).reshape(-1, 2).astype(np.float32)
