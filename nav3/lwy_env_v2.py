"""Optional tracker yaw gain; frozen support/collision code copied verbatim.

Default gain 2 reproduces LwyEnv. Only cmdw gain is configurable; envelope,
terrain speed, translational dynamics, noise and collision tests are unchanged.
"""
from functools import partial
import jax
import jax.numpy as jnp
from .env import rotate, wrap
from .lwy_env import LwyEnv

@jax.tree_util.register_pytree_node_class
class LwyEnvV2(LwyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = partial(jax.jit(LwyEnvV2._step), self)

    def _step(self,state,action):
        look = state['pos'][:, None]+rotate(jnp.array([[[0., 0.], [.4, 0.]]]), state['yaw'])
        root = jnp.any(self.bank.sample(state['mids'], look, 6) == 2, -1)
        action = action.at[:, 2].set(jnp.where(root, jnp.minimum(action[:, 2], self.cfg['terrain_speed']/.4-1), action[:, 2]))
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
            cmdw = jnp.clip(c.get('yaw_tracking_gain',2.)*speed*local[:,1]/norm**2,-c['max_yaw_rate'],c['max_yaw_rate'])
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


def physics_trace_v2(env,state,action):
    """Replay the four actual 50 ms substeps, including the identical slip key.

    Isolated diagnostic mirror of FastEnv._step; endpoint equality is checked.
    """
    c=env.cfg;action=jnp.clip(action,-1,1)
    _,slipkey,_=jax.random.split(state['key'],3)
    radius=.5+(action[:,0]+1)*1.25;angle=action[:,1]*jnp.pi/3
    target=state['pos']+rotate(jnp.stack([radius*jnp.cos(angle),radius*jnp.sin(angle)],-1),state['yaw'])
    terrain=env.bank.sample(state['mids'],state['pos'][:,None]+rotate(jnp.array([[[-.42,0],[.42,0]]]),state['yaw']),0,linear=True)
    speed=jnp.minimum((action[:,2]+1)*.4,jnp.where(jnp.abs(terrain[:,1]-terrain[:,0])>.12,c['terrain_speed'],c['max_speed']))
    slip=jax.random.normal(slipkey,(4,env.n,2))*c['slip_std']
    def step(carry,noise):
        pos,yaw,vel,queue=carry
        local=rotate(target-pos,-yaw);norm=jnp.maximum(jnp.linalg.norm(local,axis=-1),.15)
        cmd=jnp.concatenate([local/norm[:,None]*jnp.minimum(speed,norm)[:,None],
            jnp.clip(c.get('yaw_tracking_gain',2.)*speed*local[:,1]/norm**2,-c['max_yaw_rate'],c['max_yaw_rate'])[:,None]],-1)
        queue=jnp.concatenate([cmd[:,None],queue[:,:-1]],1)
        delayed=queue[jnp.arange(env.n),state['delay']]
        lim=jnp.array([c['max_accel'],c['max_accel'],c['max_yaw_accel']])*c['control_dt']
        vel=vel+jnp.clip((delayed*state['gain']-vel)*(c['control_dt']/c['motor_tau']),-lim,lim)
        scale=jnp.minimum(1.,jnp.linalg.norm(vel[:,:2],axis=-1)/.5)
        pos=pos+rotate(vel[:,:2]+noise*scale[:,None],yaw)*c['control_dt'];yaw=wrap(yaw+vel[:,2]*c['control_dt'])
        data,_=env.truth(state,pos[:,None]+rotate(env.foot[None],yaw))
        hit=jnp.any(data[...,5]<0,-1)
        return (pos,yaw,vel,queue),(pos,yaw,vel,cmd,hit)
    _,trace=jax.lax.scan(step,(state['pos'],state['yaw'],state['vel'],state['queue']),slip)
    return trace

