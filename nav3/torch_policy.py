"""Environment-independent FP32 encoder/RSSM/actor/traversability inference.

Core forward accepts explicit categorical uniforms for cross-runtime parity.
step(bev, state, carry) accepts batched tensors and returns action, carry, trav_map.
The default actor uses its mean; set sample_actions=True for clipped sampling.
This port targets the official size12m/simple configuration used by nav3.
"""
import numpy as np
import torch
import torch.nn.functional as F


class TorchPolicy(torch.nn.Module):
    def __init__(self,weights,sample_actions=False):
        super().__init__()
        for key,value in weights.items():
            if key.startswith(('enc/','dyn/dyn','dyn/obs','pol/','dec/sp','dec/conv','dec/imgout')):
                self.register_buffer(key.replace('/','__'),torch.from_numpy(np.asarray(value,dtype=np.float32).copy()))
        self.sample_actions=sample_actions
        self.deter_size=self.p('dyn/dynin0/kernel').shape[0]
        self.stoch_size=32;self.classes=16;self.blocks=8
        assert self.deter_size==2048 and self.p('dyn/obslogit/bias').numel()==512

    def p(self,key):return getattr(self,key.replace('/','__'))
    def linear(self,x,key):return x@self.p(key+'/kernel')+self.p(key+'/bias')
    def norm(self,x,key):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-4)*self.p(key+'/scale')
    def block(self,x,key):
        w=self.p(key+'/kernel');b=x.shape[0]
        return torch.einsum('bki,kio->bko',x.reshape(b,8,-1),w).reshape(b,-1)+self.p(key+'/bias')
    def conv(self,x,key):
        return F.conv2d(x.permute(0,3,1,2),self.p(key+'/kernel').permute(3,2,0,1),self.p(key+'/bias'),padding=2).permute(0,2,3,1)

    def forward(self,image,state,deter,stoch,prev,reset,uniforms):
        b=image.shape[0]
        x=torch.sign(state)*torch.log1p(state.abs())
        for i in range(3):x=F.silu(self.norm(self.linear(x,f'enc/mlp{i}'),f'enc/mlp{i}norm'))
        vec=x;x=image.float()/255-.5
        for i in range(4):
            key=f'enc/cnn{i}';x=self.conv(x,key)
            h,w,c=x.shape[1:]
            x=x.reshape(b,h//2,2,w//2,2,c).amax((2,4))
            x=F.silu(self.norm(x,key+'norm'))
        tokens=torch.cat([vec,x.reshape(b,-1)],-1)
        mask=(~reset).float()[:,None]
        deter=deter*mask;stoch=stoch*mask[:,:,None];prev=prev*mask
        prev=prev/torch.maximum(torch.ones_like(prev),prev.abs())
        xs=[]
        for i,x in enumerate((deter,stoch.reshape(b,-1),prev)):
            xs.append(F.silu(self.norm(self.linear(x,f'dyn/dynin{i}'),f'dyn/dynin{i}norm')))
        x=torch.cat(xs,-1)[:,None,:].expand(-1,8,-1)
        x=torch.cat([deter.reshape(b,8,-1),x],-1).reshape(b,-1)
        x=F.silu(self.norm(self.block(x,'dyn/dynhid0'),'dyn/dynhid0norm'))
        gates=self.block(x,'dyn/dyngru').reshape(b,8,-1)
        reset_gate,candidate,update=[v.reshape(b,-1) for v in gates.chunk(3,-1)]
        candidate=torch.tanh(torch.sigmoid(reset_gate)*candidate);update=torch.sigmoid(update-1)
        deter=update*candidate+(1-update)*deter
        x=torch.cat([deter,tokens],-1)
        x=F.silu(self.norm(self.linear(x,'dyn/obs0'),'dyn/obs0norm'))
        logit=self.linear(x,'dyn/obslogit').reshape(b,32,16)
        probs=F.softmax(logit,-1)*.99+.01/16
        index=(uniforms[:,:,None]>torch.cumsum(probs,-1)).sum(-1).clamp(max=15)
        stoch=F.one_hot(index,16).float()
        x=torch.cat([deter,stoch.reshape(b,-1)],-1)
        for i in range(3):x=F.silu(self.norm(self.linear(x,f'pol/mlp/linear{i}'),f'pol/mlp/norm{i}'))
        action=torch.tanh(self.linear(x,'pol/head/action/mean'))
        stddev=.1+.9*torch.sigmoid(self.linear(x,'pol/head/action/stddev')+2)
        # Official block-to-spatial arrangement: (..., g h w c) -> (..., h w g c).
        x0=self.block(deter,'dec/sp0').reshape(b,8,4,4,8).permute(0,2,3,1,4).reshape(b,4,4,64)
        x1=F.silu(self.norm(self.linear(stoch.reshape(b,-1),'dec/sp1'),'dec/sp1norm'))
        x1=self.linear(x1,'dec/sp2').reshape(b,4,4,64)
        x=F.silu(self.norm(x0+x1,'dec/spnorm'))
        for i in (2,1,0):
            x=x.repeat_interleave(2,1).repeat_interleave(2,2)
            x=F.silu(self.norm(self.conv(x,f'dec/conv{i}'),f'dec/conv{i}norm'))
        x=x.repeat_interleave(2,1).repeat_interleave(2,2)
        trav=torch.sigmoid(self.conv(x,'dec/imgout'))[...,6:7]
        return action,deter,stoch,trav,tokens,logit,stddev

    @torch.no_grad()
    def step(self,bev,state,carry=None):
        device=self.p('dyn/dynin0/kernel').device
        bev=torch.as_tensor(bev,dtype=torch.uint8,device=device)
        state=torch.as_tensor(state,dtype=torch.float32,device=device)
        b=bev.shape[0]
        if carry is None:
            carry=dict(deter=torch.zeros(b,2048,device=device),stoch=torch.zeros(b,32,16,device=device),
                prev=torch.zeros(b,3,device=device),reset=torch.ones(b,dtype=torch.bool,device=device))
        reset=carry.get('reset',torch.zeros(b,dtype=torch.bool,device=device))
        uniforms=torch.rand(b,32,device=device)
        action,deter,stoch,trav,_,_,stddev=self(bev,state,carry['deter'],carry['stoch'],carry['prev'],reset,uniforms)
        if self.sample_actions:action=torch.clamp(action+stddev*torch.randn_like(action),-1,1)
        return action,dict(deter=deter,stoch=stoch,prev=action),trav


def load_policy(weights_path,device='cpu',sample_actions=False):
    with np.load(weights_path) as archive:weights={k:archive[k] for k in archive.files}
    return TorchPolicy(weights,sample_actions).to(device).eval()
