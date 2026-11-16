"""JAX vector environment; 5 Hz target actions, 20 Hz controller and body.

No ray per laser beam. Occlusion is a polar terrain horizon per calibrated sensor.
The route prior is public; collision/support/privileged maps are never actor inputs.
"""
import json
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .calibration import mounts


def rotate(points, yaw):
    c,s = jnp.cos(yaw),jnp.sin(yaw)
    while c.ndim < points.ndim-1:
        c,s = c[...,None],s[...,None]
    return jnp.stack([c*points[...,0]-s*points[...,1],s*points[...,0]+c*points[...,1]],-1)


def wrap(x):
    return (x+jnp.pi)%(2*jnp.pi)-jnp.pi


def encode(bev):
    scale = jnp.array([1/4,1/2,1,1,1,1])
    offset = jnp.array([.5,0,0,0,0,0])
    return (jnp.clip(bev*scale+offset,0,1)*255).astype(jnp.uint8)


def decode(bev):
    return (bev.astype(jnp.float32)/255-jnp.array([.5,0,0,0,0,0]))*jnp.array([4,2,1,1,1,1])


@jax.tree_util.register_pytree_node_class
class FastEnv:
    ARRAY_KEYS=('bank','mount','local','extended','polar','radii','memory_ij','foot')

    def tree_flatten(self):
        return tuple(getattr(self,k) for k in self.ARRAY_KEYS),(self.n,self.random_obstacles,self.direct,tuple(self.cfg.items()))

    @classmethod
    def tree_unflatten(cls,aux,arrays):
        obj=cls.__new__(cls)
        obj.n,obj.random_obstacles,obj.direct,config=aux;obj.cfg=dict(config)
        for k,v in zip(cls.ARRAY_KEYS,arrays):setattr(obj,k,v)
        return obj

    def __init__(self, bank, num_envs=1024, random_obstacles=False, direct=False, config=None):
        self.bank, self.n, self.random_obstacles, self.direct = bank,num_envs,random_obstacles,direct
        self.cfg = json.loads(Path('configs/env.json').read_text())
        if config:
            self.cfg.update(config)
        self.mount = jnp.asarray(mounts())
        c = self.cfg
        x,y = jnp.meshgrid(-2+(jnp.arange(64)+.5)*.125,-4+(jnp.arange(64)+.5)*.125,indexing='ij')
        self.local = jnp.stack([x,y],-1)
        # Extended truth crop supports orientation-aligned rectangular erosion.
        x,y = jnp.meshgrid(-2+(jnp.arange(-4,68)+.5)*.125,-4+(jnp.arange(-2,66)+.5)*.125,indexing='ij')
        self.extended = jnp.stack([x,y],-1)
        a = (jnp.arange(c['polar_angles'])+.5)*2*jnp.pi/c['polar_angles']-jnp.pi
        r = (jnp.arange(c['polar_bins'])+.5)*c['polar_res']
        self.polar = jnp.stack([jnp.cos(a[:,None])*r,jnp.sin(a[:,None])*r],-1)
        self.radii = r
        mi,mj=jnp.meshgrid(jnp.arange(128),jnp.arange(128),indexing='ij')
        self.memory_ij=jnp.stack([mi,mj],-1)
        x,y = jnp.meshgrid(jnp.linspace(-c['envelope_length']/2,c['envelope_length']/2,5),
                          jnp.linspace(-c['envelope_width']/2,c['envelope_width']/2,5), indexing='ij')
        self.foot = jnp.stack([x,y],-1).reshape(-1,2)
        # Map arrays are dynamic device arguments, not gigabytes of XLA constants.
        self.reset = partial(jax.jit(FastEnv._reset),self)
        self.step = partial(jax.jit(FastEnv._step),self)
        self.observe = partial(jax.jit(FastEnv._observe),self)

    def _reset(self, key, mids=None, start_s=None, observe=True):
        c,n = self.cfg,self.n
        keys = jax.random.split(key,12)
        if mids is None:
            mids = jax.random.randint(keys[0],(n,),0,self.bank.count)
        if start_s is None:
            start_s = jnp.full(n,2.,jnp.float32)
        p,yaw,width = self.bank.route_at(mids,start_s)
        offset = jax.random.uniform(keys[1],(n,),minval=-.06,maxval=.06)
        pos = p[:,:2]+rotate(jnp.stack([jnp.zeros(n),offset],-1),yaw)
        yaw = yaw+jax.random.normal(keys[2],(n,))*.02
        # Episodic obstacles lie in wider trail sections and are absent from route prior.
        count = c['random_obstacles']
        os = jax.random.uniform(keys[3],(n,count),minval=8.,maxval=190.)
        op,oy,ow = self.bank.route_at(mids,os)
        offset = jax.random.uniform(keys[4],(n,count),minval=-.55,maxval=.55)
        op = op[...,:2]+rotate(jnp.stack([jnp.zeros_like(offset),offset],-1),oy)
        radii = jax.random.uniform(keys[5],(n,count),minval=.12,maxval=.24)
        radii = jnp.where((ow>1.7)&self.random_obstacles,radii,0.)
        # Alternating rocks and short fallen logs, oriented across the route.
        sizes = jnp.stack([radii,jnp.where(jnp.arange(count)%2==0,radii,radii*1.8)],-1)
        obstacles = jnp.concatenate([op, sizes,oy[...,None],jnp.full((n,count,1),.34)],-1)
        state = dict(key=keys[6],mids=mids,pos=pos,yaw=yaw,vel=jnp.zeros((n,3)),prev=jnp.zeros((n,3)),
            queue=jnp.zeros((n,6,3)),delay=jax.random.randint(keys[7],(n,),1,6),
            gain=jax.random.uniform(keys[8],(n,1),minval=c['gain_min'],maxval=c['gain_max']),
            height=c['base_height']+jax.random.uniform(keys[9],(n,),minval=-c['base_height_jitter'],maxval=c['base_height_jitter']),
            drift=jnp.zeros((n,3)),s=start_s,best_s=start_s,stuck=jnp.zeros(n,jnp.int32),steps=jnp.zeros(n,jnp.int32),
            gates=jnp.sum(start_s[:,None]>=self.bank.gates[mids],-1).astype(jnp.int32),
            ep_return=jnp.zeros(n),distance=jnp.zeros(n),obstacles=obstacles,
            memory=jnp.zeros((n,128,128,3)),age=jnp.full((n,128,128),100.),
            mempos=jnp.floor(pos/.125)*.125-8,memyaw=yaw,
            memz=self.bank.sample(mids,pos,0,linear=True),
            obs_queue=jnp.zeros((n,2,64,64,6),jnp.uint8),first=jnp.ones(n,bool),
            done=jnp.zeros(n,bool),reward=jnp.zeros(n),failure=jnp.zeros(n,jnp.int32))
        if not observe:
            return state
        state,obs,extra = self._observe(state)
        state['obs_queue'] = jnp.repeat(obs['bev'][:,None],2,axis=1)
        return state,obs,extra

    def _random_geometry(self,state,xy):
        n = len(xy)
        obs = state['obstacles'].reshape((n,)+(1,)*(xy.ndim-2)+(-1,6))
        d = xy[...,None,:]-obs[...,:2]
        d = rotate(d,-obs[...,4])
        q = jnp.abs(d)-obs[...,2:4]
        sd = jnp.linalg.norm(jnp.maximum(q,0),axis=-1)+jnp.minimum(jnp.maximum(q[...,0],q[...,1]),0)
        enabled = obs[...,2]>0
        distance = jnp.min(jnp.where(enabled,sd,100.),-1)
        height = jnp.max(jnp.where(enabled&(sd<.06),obs[...,5],0.),-1)
        return distance,height

    def truth(self,state,xy):
        data = self.bank.sample(state['mids'],xy)
        rd,rh = self._random_geometry(state,xy)
        data = data.at[...,1].set(jnp.maximum(data[...,1],data[...,0]+rh))
        data = data.at[...,5].set(jnp.minimum(data[...,5],rd))
        return data,rd

    def _frame(self,state,key):
        """Single noisy return image, occlusion mask, complete target. No support truth in image."""
        c = self.cfg
        pos,yaw = state['pos'],state['yaw']
        xy = pos[:,None,None]+rotate(self.local[None],yaw)
        data,random_dist = self.truth(state,xy)
        z0 = self.bank.sample(state['mids'],pos,0,linear=True)
        visible = jnp.zeros((self.n,64,64),bool)
        for k in range(2):
            origin_local = self.mount[k,:2,3]
            origin = pos+rotate(jnp.broadcast_to(origin_local,pos.shape),yaw)
            zsensor = z0+state['height']+self.mount[k,2,3]
            pxy = origin[:,None,None]+rotate(self.polar[None],yaw)
            pd,_ = self.truth(state,pxy)
            slope = (pd[...,1]-zsensor[:,None,None])/jnp.maximum(self.radii[None,None],.08)
            horizon = jax.lax.associative_scan(jnp.maximum,slope,axis=-1)
            horizon = jnp.concatenate([jnp.full_like(horizon[...,:1],-100),horizon[...,:-1]],-1)
            rel = self.local-origin_local
            angle = jnp.arctan2(rel[...,1],rel[...,0])
            ai = jnp.floor((angle+jnp.pi)/(2*jnp.pi)*c['polar_angles']).astype(jnp.int32)%c['polar_angles']
            rr = jnp.linalg.norm(rel,axis=-1)
            ri = jnp.clip(jnp.floor(rr/c['polar_res']).astype(jnp.int32),0,c['polar_bins']-1)
            h = horizon[:,ai,ri]
            target_slope = (data[...,1]-zsensor[:,None,None])/jnp.maximum(rr,.08)
            axis = self.mount[k,:3,2]
            hemisphere = rel[...,0]*axis[0]+rel[...,1]*axis[1]+(data[...,1]-zsensor[:,None,None])*axis[2] >= -.01
            # Geometric chassis shadow (upper deck ends at +/-0.28 m, main deck 0.40 m).
            # Native hemisphere keeps most cross-body rays out; near chassis cells stay unknown.
            blind = (jnp.abs(self.local[...,0])<.49)&(jnp.abs(self.local[...,1])<.29)
            visible |= (target_slope>=h-c['horizon_tolerance'])&hemisphere&(~blind)&(rr>.15)
        key1,key2 = jax.random.split(key)
        distance = jnp.linalg.norm(self.local,axis=-1)
        dropout = c['dropout_base']+c['dropout_distance']*distance**2
        seen = visible&(jax.random.uniform(key1,visible.shape)>dropout)
        ground = data[...,0]-z0[:,None,None]
        obstacle = jnp.maximum(0,data[...,1]-data[...,0])
        noisy = ground+jax.random.normal(key2,ground.shape)*c['height_std']
        # Geometry-derived support likelihood, not semantic ground-truth path support.
        dx = (jnp.roll(ground,-2,axis=1)-jnp.roll(ground,2,axis=1))/.5
        dy = (jnp.roll(ground,-2,axis=2)-jnp.roll(ground,2,axis=2))/.5
        support = (jnp.hypot(dx,dy)<.8)&(obstacle<c['collision_height'])&(ground>-.55)
        measured = jnp.stack([noisy,obstacle,support.astype(jnp.float32)],-1)
        extxy = pos[:,None,None]+rotate(self.extended[None],yaw)
        ext,_ = self.truth(state,extxy)
        # 9x5 center samples span 1.0x0.5 m; signed boundary margin covers remaining 0.03x0.08 m.
        free = (ext[...,2]<-.045)&(ext[...,5]>.045)
        trav = jax.lax.reduce_window(free.astype(jnp.int32),1,jax.lax.min,(1,9,5),(1,1,1),'VALID')>0
        return measured,seen,visible,trav,random_dist

    def _observe(self,state):
        c = self.cfg
        key,framekey,driftkey,speedkey = jax.random.split(state['key'],4)
        state = {**state,'key':key}
        drift = (1-c['localization_reversion'])*state['drift']+jax.random.normal(driftkey,(self.n,3))*jnp.array([c['position_walk_std'],c['position_walk_std'],c['yaw_walk_std']])
        state['drift'] = drift
        estpos,estyaw = state['pos']+drift[:,:2],state['yaw']+drift[:,2]
        measured,seen,visible,trav,rd = self._frame(state,framekey)
        # Accumulate in a world-aligned rolling grid. Shifts are integer cells;
        # repeated subpixel nearest-neighbor warps would pin old objects to the
        # body at low speed and erase/move them as new free returns arrive.
        world = estpos[:,None,None]+rotate(self.local[None],estyaw)
        origin=jnp.floor(estpos/.125)*.125-8
        shift=jnp.rint((origin-state['mempos'])/.125).astype(jnp.int32)
        old=self.memory_ij[None]+shift[:,None,None]
        inside=jnp.all((old>=0)&(old<128),axis=-1)
        old=jnp.clip(old,0,127)
        batch=jnp.arange(self.n)[:,None,None]
        memory=state['memory'][batch,old[...,0],old[...,1]]
        age=state['age'][batch,old[...,0],old[...,1]]+c['dt']
        age=jnp.where(inside,age,100.)
        znow = self.bank.sample(state['mids'],state['pos'],0,linear=True)
        ij=jnp.clip(jnp.floor((world-origin[:,None,None])/.125).astype(jnp.int32),0,127)
        fresh=jnp.zeros((self.n,128,128),bool).at[batch,ij[...,0],ij[...,1]].max(seen)
        absolute=measured.at[...,0].add(znow[:,None,None])
        updates=jnp.full_like(memory,-jnp.inf).at[batch,ij[...,0],ij[...,1]].max(jnp.where(seen[...,None],absolute,-jnp.inf))
        memory=jnp.where(fresh[...,None],updates,memory)
        age=jnp.where(fresh,0.,age)
        local_memory=memory[batch,ij[...,0],ij[...,1]]
        local_age=age[batch,ij[...,0],ij[...,1]]
        known=local_age<c['memory_seconds']
        local_memory=local_memory.at[...,0].add(-znow[:,None,None])
        local_memory=jnp.where(known[...,None],local_memory,0.)
        # Only the correct main branch appears in this prior. It uses estimated localization.
        corridor = self.bank.sample(state['mids'],world,3,linear=True)<0
        image = jnp.concatenate([local_memory,(~known)[...,None],corridor[...,None],jnp.clip(local_age/c['memory_seconds'],0,1)[...,None]],-1)
        image = encode(image)
        queue = jnp.concatenate([image[:,None],state['obs_queue'][:,:1]],1)
        delay = c['max_observation_delay_steps']
        delayed = jnp.where(state['first'][:,None,None,None],image,queue[:,delay])
        target_s = state['s'][:,None]+jnp.array([1.,2.,3.])
        route,_,_ = self.bank.route_at(state['mids'],target_s)
        goals = rotate(route[...,:2]-estpos[:,None],-estyaw)/4
        speed = state['vel']+jax.random.normal(speedkey,(self.n,3))*.01
        vec = jnp.concatenate([goals.reshape(self.n,6),state['gates'][:,None]/8,speed,state['prev']],-1).astype(jnp.float32)
        obs = dict(bev=delayed,state=vec,priv_trav=(trav[...,None]*255).astype(jnp.uint8),
                   reward=state['reward'],is_first=state['first'],is_last=state['done'],
                   is_terminal=state['done']&(state['failure']!=5))
        state.update(memory=memory,age=age,mempos=origin,memyaw=estyaw,memz=znow,obs_queue=queue)
        extra = dict(occluded=~visible,random_near=rd<.6,clean_trav=trav)
        return state,obs,extra

    def _step(self,state,action):
        c = self.cfg
        # Terminal observations are emitted once, then the next call returns a reset.
        resetting = state['done']
        key,slipkey,resetkey = jax.random.split(state['key'],3)
        action = jnp.clip(action,-1,1)
        distance = .5+(action[:,0]+1)*1.25
        angle = action[:,1]*jnp.pi/3
        target = state['pos']+rotate(jnp.stack([distance*jnp.cos(angle),distance*jnp.sin(angle)],-1),state['yaw'])
        speed = (action[:,2]+1)*.4
        terrain = self.bank.sample(state['mids'],state['pos'][:,None]+rotate(jnp.array([[[-.42,0],[.42,0]]]),state['yaw']),0,linear=True)
        steep = jnp.abs(terrain[:,1]-terrain[:,0])>.12
        speed = jnp.minimum(speed,jnp.where(steep,c['terrain_speed'],c['max_speed']))
        slip = jax.random.normal(slipkey,(4,self.n,2))*c['slip_std']
        def substep(carry,noise):
            pos,yaw,vel,queue,fall,hit = carry
            local = rotate(target-pos,-yaw)
            norm = jnp.maximum(jnp.linalg.norm(local,axis=-1),.15)
            cmdxy = local/norm[:,None]*jnp.minimum(speed,norm)[:,None]
            cmdw = jnp.clip(2*speed*local[:,1]/norm**2,-c['max_yaw_rate'],c['max_yaw_rate'])
            cmd = jnp.concatenate([cmdxy,cmdw[:,None]],-1)
            if self.direct:
                cmd = action*jnp.array([.8,.6,.75])
                limit = jnp.where(steep,c['terrain_speed'],c['max_speed'])
                cmd = cmd.at[:,:2].set(cmd[:,:2]*jnp.minimum(1,limit/jnp.maximum(jnp.linalg.norm(cmd[:,:2],axis=-1),.01))[:,None])
            queue = jnp.concatenate([cmd[:,None],queue[:,:-1]],axis=1)
            delayed = queue[jnp.arange(self.n),state['delay']]
            dv = (delayed*state['gain']-vel)*(c['control_dt']/c['motor_tau'])
            lim = jnp.array([c['max_accel'],c['max_accel'],c['max_yaw_accel']])*c['control_dt']
            vel = vel+jnp.clip(dv,-lim,lim)
            slip_scale=jnp.minimum(1.,jnp.linalg.norm(vel[:,:2],axis=-1)/.5)
            pos = pos+rotate(vel[:,:2]+noise*slip_scale[:,None],yaw)*c['control_dt']
            yaw = wrap(yaw+vel[:,2]*c['control_dt'])
            fp = pos[:,None]+rotate(self.foot[None],yaw)
            truth,_ = self.truth(state,fp)
            fall |= jnp.any(truth[...,2]>.025,axis=-1)
            hit |= jnp.any(truth[...,5]<0,axis=-1)
            return (pos,yaw,vel,queue,fall,hit),None
        init = (state['pos'],state['yaw'],state['vel'],state['queue'],jnp.zeros(self.n,bool),jnp.zeros(self.n,bool))
        (pos,yaw,vel,queue,fall,hit),_ = jax.lax.scan(substep,init,slip)
        s,lateral = self.bank.project(state['mids'],pos)
        progress = jnp.clip(s-state['s'],-.5,.5)
        # Ordered crossing requires center inside gate span and forward progress.
        gate_s = self.bank.gates[state['mids'],jnp.minimum(state['gates'],7)]
        _,_,width = self.bank.route_at(state['mids'],s)
        crossing = (state['s']<gate_s)&(s>=gate_s)&(jnp.abs(lateral)<width/2)&(state['gates']<8)
        gates = state['gates']+crossing.astype(jnp.int32)
        best = jnp.maximum(s,state['best_s'])
        stuck = jnp.where(s>state['best_s']+.015,0,state['stuck']+1)
        steps = state['steps']+1
        wrong = (jnp.abs(lateral)>width/2+.35)|(s<state['best_s']-2)
        timeout = (steps>=c['max_steps'])|(stuck*c['dt']>=c['stuck_seconds'])
        failure = jnp.where(fall,1,jnp.where(hit,2,jnp.where(wrong,3,jnp.where(timeout,5,0)))).astype(jnp.int32)
        done = (failure>0)|(gates==8)
        center,_ = self.truth(state,pos)
        near = jnp.maximum(0,.5-jnp.minimum(-center[:,2],center[:,5]))
        smooth = jnp.mean((action-state['prev'])**2,-1)
        reward = c['progress_reward']*progress+c['gate_reward']*crossing-c['time_penalty']-.05*smooth-.04*near-.03*(lateral/jnp.maximum(width/2,.1))**2-c['failure_penalty']*(failure>0)
        state = {**state,'key':key,'pos':pos,'yaw':yaw,'vel':vel,'queue':queue,'prev':action,
            's':s,'best_s':best,'stuck':stuck,'steps':steps,'gates':gates,'first':jnp.zeros(self.n,bool),
            'reward':reward,'done':done,'failure':failure,'ep_return':state['ep_return']+reward,
            'distance':state['distance']+jnp.linalg.norm(pos-state['pos'],axis=-1)}
        def reset_some(current):
            fresh = self._reset(resetkey,observe=False)
            choose = lambda a,b:jnp.where(resetting.reshape((self.n,)+(1,)*(a.ndim-1)),b,a)
            return {k:(fresh[k] if k=='key' else choose(current[k],fresh[k])) for k in current}
        state = jax.lax.cond(jnp.any(resetting),reset_some,lambda x:x,state)
        return self._observe(state)
