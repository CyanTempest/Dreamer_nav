"""Recurrent clipped PPO in JAX. Actor and critic receive only public observations."""
import jax
import jax.numpy as jnp
import numpy as np
import optax


def init_params(seed=0):
    rng=np.random.default_rng(seed)
    params={}
    def weight(name,shape,scale=1.):
        fan=np.prod(shape[:-1]);params[name]=jnp.array(rng.normal(0,scale/np.sqrt(fan),shape),jnp.float32)
    channels=[6,32,48,64,64]
    for i in range(4):
        weight(f'conv{i}',(3,3,channels[i],channels[i+1]),np.sqrt(2))
        params[f'bias{i}']=jnp.zeros(channels[i+1])
    weight('embed',(1024+13,256),np.sqrt(2));params['embed_bias']=jnp.zeros(256)
    weight('gru_x',(256,768));weight('gru_h',(256,768));params['gru_bias']=jnp.zeros(768)
    weight('actor',(256,3),.01);params['actor_bias']=jnp.zeros(3);params['logstd']=jnp.full(3,-.7)
    weight('critic',(256,1));params['critic_bias']=jnp.zeros(1)
    return params


def encode(params,bev,state):
    shape=bev.shape[:-3]
    x=bev.astype(jnp.float32).reshape((-1,64,64,6))/255-.5
    for i in range(4):
        x=jax.lax.conv_general_dilated(x,params[f'conv{i}'],(2,2),'SAME',dimension_numbers=('NHWC','HWIO','NHWC'))+params[f'bias{i}']
        x=jax.nn.silu(x)
    x=jnp.concatenate([x.reshape((x.shape[0],-1)),state.reshape((-1,13))],-1)
    return jax.nn.tanh(x@params['embed']+params['embed_bias']).reshape((*shape,256))


def recurrent(params,hidden,x,first):
    hidden=jnp.where(first[...,None],0.,hidden)
    ix=x@params['gru_x']+params['gru_bias'];hx=hidden@params['gru_h']
    ir,iz,ic=jnp.split(ix,3,-1);hr,hz,hc=jnp.split(hx,3,-1)
    reset=jax.nn.sigmoid(ir+hr);update=jax.nn.sigmoid(iz+hz)
    candidate=jnp.tanh(ic+reset*hc)
    hidden=update*hidden+(1-update)*candidate
    mean=hidden@params['actor']+params['actor_bias']
    value=(hidden@params['critic']+params['critic_bias'])[...,0]
    return hidden,mean,value


def logprob(params,mean,raw):
    logstd=jnp.clip(params['logstd'],-3,1)
    lp=-.5*((raw-mean)/jnp.exp(logstd))**2-logstd-.5*np.log(2*np.pi)
    correction=2*(np.log(2)-raw-jax.nn.softplus(-2*raw))
    return jnp.sum(lp-correction,-1)


@jax.jit
def act(params,hidden,obs,key):
    x=encode(params,obs['bev'],obs['state'])
    hidden,mean,value=recurrent(params,hidden,x,obs['is_first'])
    raw=mean+jnp.exp(jnp.clip(params['logstd'],-3,1))*jax.random.normal(key,mean.shape)
    return hidden,jnp.tanh(raw),raw,logprob(params,mean,raw),value


def sequence(params,hidden,bev,state,first):
    x=encode(params,bev,state)
    def step(h,args):
        nh,mu,v=recurrent(params,h,*args)
        return nh,(mu,v)
    _,(mu,value)=jax.lax.scan(step,hidden,(x,first))
    return mu,value


def loss(params,batch):
    mean,value=sequence(params,batch['hidden'][0],batch['bev'],batch['state'],batch['first'])
    lp=logprob(params,mean,batch['raw'])
    ratio=jnp.exp(lp-batch['logprob'])
    valid=batch['valid'].astype(jnp.float32)
    avg=lambda v:jnp.sum(v*valid)/jnp.maximum(valid.sum(),1)
    adv=batch['advantage']
    adv=(adv-avg(adv))/jnp.sqrt(avg((adv-avg(adv))**2)+1e-8)
    ploss=avg(jnp.maximum(-adv*ratio,-adv*jnp.clip(ratio,.8,1.2)))
    vclip=batch['value']+jnp.clip(value-batch['value'],-.2,.2)
    vloss=.5*avg(jnp.maximum((value-batch['return'])**2,(vclip-batch['return'])**2))
    entropy=jnp.sum(jnp.clip(params['logstd'],-3,1)+.5*np.log(2*np.pi*np.e))
    total=ploss+.5*vloss-.003*entropy
    return total,dict(policy_loss=ploss,value_loss=vloss,entropy=entropy,approx_kl=avg(batch['logprob']-lp),clip_fraction=avg(jnp.abs(ratio-1)>.2))


OPT=optax.chain(optax.clip_by_global_norm(.5),optax.adam(3e-4,eps=1e-5))


@jax.jit
def update(params,optimizer,batch):
    (_,metrics),grad=jax.value_and_grad(loss,has_aux=True)(params,batch)
    delta,optimizer=OPT.update(grad,optimizer,params)
    return optax.apply_updates(params,delta),optimizer,metrics


@jax.jit
def advantages(reward,value,next_value,last,terminal,valid):
    delta=reward+.997*(1-terminal)*next_value-value
    def step(carry,x):
        d,end,keep=x
        advantage=(d+.997*.95*(1-end)*carry)*keep
        return advantage,advantage
    _,adv=jax.lax.scan(step,jnp.zeros_like(reward[0]),(delta,last,valid),reverse=True)
    return adv,adv+value


class PPOPolicy:
    head=False
    def __init__(self,seed=0):
        self.params=init_params(seed);self.carry=None;self.key=jax.random.PRNGKey(seed+1234)
        self.optimizer=OPT.init(self.params)
    def policy(self,obs,predict=False):
        if self.carry is None:self.carry=jnp.zeros((len(obs['is_first']),256))
        self.key,key=jax.random.split(self.key)
        old=self.carry
        self.carry,action,raw,lp,value=act(self.params,self.carry,obs,key)
        return action,dict(hidden=old,raw=raw,logprob=lp,value=value)
    def __call__(self,obs):return self.policy(obs)[0]
