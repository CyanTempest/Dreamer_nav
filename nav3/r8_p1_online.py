"""Opt-in JAX static atlas sensor. Existing LwyEnv files remain untouched.

Only _frame changes. FastEnv._observe supplies the original world-aligned
128x128 memory, 8 s expiry, public route prior, and the 6-channel encoding.
"""
from functools import partial
import json
import numpy as np
import jax
import jax.numpy as jnp
from .lwy_env import LwyEnv
from .env import rotate
from .r8_p1_common import GEO,ROOT

@jax.tree_util.register_pytree_node_class
class Atlas:
    def __init__(self,opaque=False,variant=None):
        a=np.load(GEO/f'r8_p1_atlas_{variant or ("opaque" if opaque else "alpha")}.npz')
        self.tiles=jnp.asarray(a['tiles']);self.lookup=jnp.asarray(a['lookup']);self.origin=jnp.asarray(a['origin'])
        self.bytes=self.tiles.nbytes+self.lookup.nbytes
    def tree_flatten(self): return (self.tiles,self.lookup,self.origin),None
    @classmethod
    def tree_unflatten(cls,aux,arrays):
        obj=cls.__new__(cls);obj.tiles,obj.lookup,obj.origin=arrays;return obj
    def sample(self,xy):
        ij=jnp.floor((xy-self.origin)/.05+.5).astype(jnp.int32);ti=ij//32
        valid=(ti[...,0]>=0)&(ti[...,1]>=0)&(ti[...,0]<self.lookup.shape[0])&(ti[...,1]<self.lookup.shape[1])
        ids=self.lookup[jnp.clip(ti[...,0],0,self.lookup.shape[0]-1),jnp.clip(ti[...,1],0,self.lookup.shape[1]-1)]
        ids=jnp.where(valid,ids,0)
        return self.tiles[ids,ij[...,0]%32,ij[...,1]%32].astype(jnp.float32)

DEFAULT=dict(p1_visibility=True,p1_coverage=.05,p1_block_coverage=1.1,
             p1_max_band=6,p1_horizon_tolerance=.025,p1_ground_open=True,
             p1_noise_base=0.,p1_noise_distance=0.,p1_height_std=0.,p1_visible_ground=True,
             p1_beam_sampling=True,p1_sensor_enabled=True)

def opening(x,size=5):
    p=size//2
    x=jnp.pad(x,((0,0),(p,p),(p,p)),mode='edge')
    x=jax.lax.reduce_window(x,jnp.inf,jax.lax.min,(1,size,size),(1,1,1),'VALID')
    x=jnp.pad(x,((0,0),(p,p),(p,p)),mode='edge')
    return jax.lax.reduce_window(x,-jnp.inf,jax.lax.max,(1,size,size),(1,1,1),'VALID')

@jax.tree_util.register_pytree_node_class
class SimpleEnv(LwyEnv):
    ARRAY_KEYS=LwyEnv.ARRAY_KEYS+('atlas','next_beam',)
    def __init__(self,bank,num_envs=1024,atlas=None,config=None):
        options={**DEFAULT,'base_height_jitter':0.,**(config or {})}
        super().__init__(bank,num_envs,random_obstacles=False,config=options)
        self.atlas=atlas if atlas is not None else Atlas()
        # Preserve measured rotation/XY; apply the P0 common height shift.
        self.mount=self.mount.at[:,2,3].add(.495-jnp.mean(self.mount[:,2,3]))
        angles=np.array(json.loads((ROOT/'vendor/map/sources/airy_specs.json').read_text())['nominal_vertical_angles_deg'])
        bins=np.arange(-100,9101)*.01
        self.next_beam=jnp.asarray(np.r_[angles,1000.][np.searchsorted(angles,bins)])
        self.observe=partial(jax.jit(SimpleEnv._observe),self)
        self.reset=partial(jax.jit(SimpleEnv._reset),self)
        self.step=partial(jax.jit(SimpleEnv._step),self)

    def sensor_frame(self,state,key):
        """Batched, environment-independent operations; usable under jit/vmap."""
        c=self.cfg;pos,yaw=state['pos'],state['yaw'];n=pos.shape[0]
        world=pos[:,None,None]+rotate(self.local[None],yaw)
        a=self.atlas.sample(world);g=a[...,0]
        z0=self.bank.sample(state['mids'],pos,0,linear=True)
        low=a[...,1:19:3];high=a[...,2:19:3];cov=a[...,3:19:3]
        exists=(high>=0)&(cov>=c['p1_coverage'])&(jnp.arange(6)<c['p1_max_band'])
        # Seven candidate surfaces per cell: ground and six sampled height bands.
        target=jnp.concatenate([g[...,None],g[...,None]+high],-1)
        lower=jnp.concatenate([g[...,None],g[...,None]+low],-1)
        gx=(jnp.roll(g,-1,axis=1)-jnp.roll(g,1,axis=1))/.25
        gy=(jnp.roll(g,-1,axis=2)-jnp.roll(g,1,axis=2))/.25
        present=jnp.concatenate([(a[...,19]>.5)[...,None],exists],-1)
        vis=jnp.zeros_like(present)
        for k in range(3):
            origin_local=self.mount[k,:2,3] if k<2 else jnp.array([.3,0.])
            zsensor=z0+(self.mount[k,2,3] if k<2 else .64)
            origin=pos+rotate(jnp.broadcast_to(origin_local,pos.shape),yaw)
            rel=self.local-origin_local
            rr=jnp.linalg.norm(rel,axis=-1);delta=target-zsensor[:,None,None,None]
            if k<2:
                axis=self.mount[k,:3,2]
                fov=(rel[None,...,0,None]*axis[0]+rel[None,...,1,None]*axis[1]+delta*axis[2]>=0)
                fov&=(jnp.sqrt(rr[None,...,None]**2+delta**2)>=.1)&(jnp.sqrt(rr[None,...,None]**2+delta**2)<=60.)
                if c['p1_beam_sampling']:
                    # Raster-plane angular footprint vs the actual nominal 96
                    # Airy elevation angles. This costs a LUT gather, not rays.
                    dz=(target+lower)*.5-zsensor[:,None,None,None]
                    rx=rel[None,...,0,None];ry=rel[None,...,1,None]
                    proj=rx*axis[0]+ry*axis[1]+dz*axis[2]
                    r2=rr[None,...,None]**2+dz**2
                    denom=jnp.maximum(r2*jnp.sqrt(jnp.maximum(r2-proj**2,1e-8)),1e-8)
                    ax=(axis[0]*r2-proj*rx)/denom
                    ay=(axis[1]*r2-proj*ry)/denom
                    az=(axis[2]*r2-proj*dz)/denom
                    width=.0625*(jnp.abs(ax+az*gx[...,None])+jnp.abs(ay+az*gy[...,None]))+(target-lower)*.5*jnp.abs(az)
                    angle=jnp.arcsin(jnp.clip(proj/jnp.sqrt(jnp.maximum(r2,1e-8)),-1,1))
                    minimum=(angle-width)*180/jnp.pi;maximum=(angle+width)*180/jnp.pi
                    bin_index=jnp.clip(jnp.ceil(minimum*100).astype(jnp.int32)+100,0,len(self.next_beam)-1)
                    fov&=self.next_beam[bin_index]<=maximum
            else:
                forward=rel[None,...,0,None]
                fov=(forward>=.5)&(forward<=20)&(jnp.abs(rel[None,...,1,None])<=forward)&(jnp.abs(delta)<=forward*np.tan(np.deg2rad(32.5)))
            if c['p1_visibility']:
                pxy=origin[:,None,None]+rotate(self.polar[None],yaw)
                pa=self.atlas.sample(pxy)
                # Lower heightfield plus optically dense low-band surface. A
                # canopy is never extruded into a solid column down to terrain.
                block=jnp.max(jnp.where((pa[...,3:10:3]>=c['p1_block_coverage'])&(pa[...,2:9:3]<.8),jnp.maximum(pa[...,2:9:3],0),0),axis=-1)
                top=pa[...,0]+block
                slope=(top-zsensor[:,None,None])/jnp.maximum(self.radii[None,None],.08)
                horizon=jax.lax.associative_scan(jnp.maximum,slope,axis=-1)
                horizon=jnp.concatenate([jnp.full_like(horizon[...,:1],-100),horizon[...,:-1]],-1)
                angle=jnp.arctan2(rel[...,1],rel[...,0])
                ai=jnp.floor((angle+jnp.pi)/(2*jnp.pi)*c['polar_angles']).astype(jnp.int32)%c['polar_angles']
                ri=jnp.clip(jnp.floor(rr/c['polar_res']).astype(jnp.int32),0,c['polar_bins']-1)
                h=horizon[:,ai,ri]
                fov&=(delta/jnp.maximum(rr[None,...,None],.08)>=h[...,None]-c['p1_horizon_tolerance'])
            vis|=fov
        blind=(jnp.abs(self.local[...,0])<.49)&(jnp.abs(self.local[...,1])<.29)
        vis&=present&~blind[None,...,None]
        seen=jnp.any(vis,axis=-1)
        # Approximate P0's view-dependent lowest return before opening. Hidden
        # terrain must not force an exposed shrub/branch return down to zero.
        if c['p1_visible_ground']:
            ground=jnp.min(jnp.where(vis,lower,jnp.inf),axis=-1)
            ground=jnp.where(seen,ground,g)
        else:
            ground=g
        ground=opening(ground) if c['p1_ground_open'] else ground
        obstacle=jnp.maximum(0,jnp.max(jnp.where(vis,target,-jnp.inf),axis=-1)-ground)
        ground=ground-z0[:,None,None]
        dx=(jnp.roll(ground,-2,axis=1)-jnp.roll(ground,2,axis=1))/.5
        dy=(jnp.roll(ground,-2,axis=2)-jnp.roll(ground,2,axis=2))/.5
        support=(jnp.hypot(dx,dy)<.8)&(obstacle<.22)&(ground>-.55)
        k1,k2=jax.random.split(key)
        distance=jnp.linalg.norm(self.local,axis=-1)
        dropout=jnp.clip(c['p1_noise_base']+c['p1_noise_distance']*distance**2,0,1)
        keep=seen&(jax.random.uniform(k1,seen.shape)>=dropout)
        keep&=c['p1_sensor_enabled']
        noisy=ground+jax.random.normal(k2,ground.shape)*c['p1_height_std']
        measured=jnp.stack([noisy,obstacle,support.astype(jnp.float32)],-1)
        return measured,keep,seen

    def _frame(self,state,key):
        measured,seen,visible=self.sensor_frame(state,key)
        # Original independent decoder target, excluded from the six actor channels.
        extxy=state['pos'][:,None,None]+rotate(self.extended[None],state['yaw'])
        ext,rd=self.truth(state,extxy)
        free=(ext[...,2]<-.045)&(ext[...,5]>.045)
        trav=jax.lax.reduce_window(free.astype(jnp.int32),1,jax.lax.min,(1,9,5),(1,1,1),'VALID')>0
        random_dist=jnp.full_like(seen,100.,dtype=jnp.float32)
        return measured,seen,visible,trav,random_dist

def state_at(env,poses,stations,key=88001):
    n=len(poses)
    state=env._reset(jax.random.PRNGKey(key),jnp.zeros(n,jnp.int32),jnp.asarray(stations),observe=False)
    pos=jnp.asarray(poses[:,:2]);yaw=jnp.asarray(poses[:,3])
    state.update(pos=pos,yaw=yaw,mempos=jnp.floor(pos/.125)*.125-8,memyaw=yaw,height=jnp.full(n,.54))
    return state

def fitted_config():
    p=ROOT/'metrics/r8_p1_noise_fit.json'
    return json.loads(p.read_text())['selected_config'] if p.exists() else {}
