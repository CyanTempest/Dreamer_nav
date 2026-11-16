"""Inference-only PyTorch port of trained encoder + RSSM posterior + actor mean."""
import numpy as np
import torch
import torch.nn.functional as F

class TorchPolicy(torch.nn.Module):
    def __init__(self,weights):
        super().__init__()
        for k,v in weights.items():
            if k.startswith(('enc/','dyn/','pol/')) and not k.startswith(('dyn/prior','pol/head/velocity/stddev')):
                self.register_buffer(k.replace('/','__'),torch.from_numpy(np.asarray(v).copy()))
    def p(self,k):return getattr(self,k.replace('/','__'))
    def linear(self,x,k):return x@self.p(k+'/kernel')+self.p(k+'/bias')
    def norm(self,x,k):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-4)*self.p(k+'/scale')
    def block(self,x,k):
        w=self.p(k+'/kernel');b=x.shape[0]
        return torch.einsum('bki,kio->bko',x.reshape(b,8,-1),w).reshape(b,-1)+self.p(k+'/bias')
    def forward(self,image,state,deter,stoch,prev,reset,uniforms):
        b=image.shape[0]
        x=torch.sign(state)*torch.log1p(state.abs())
        for i in range(3):x=F.silu(self.norm(self.linear(x,f'enc/mlp{i}'),f'enc/mlp{i}norm'))
        vec=x
        x=image.float()/255-.5
        for i in range(4):
            k=f'enc/cnn{i}'
            x=F.conv2d(x.permute(0,3,1,2),self.p(k+'/kernel').permute(3,2,0,1),self.p(k+'/bias'),padding=2).permute(0,2,3,1)
            h,w,c=x.shape[1:]
            x=x.reshape(b,h//2,2,w//2,2,c).amax((2,4))
            x=F.silu(self.norm(x,k+'norm'))
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
        r,ca,u=[v.reshape(b,-1) for v in gates.chunk(3,-1)]
        ca=torch.tanh(torch.sigmoid(r)*ca);u=torch.sigmoid(u-1)
        deter=u*ca+(1-u)*deter
        x=torch.cat([deter,tokens],-1)
        x=F.silu(self.norm(self.linear(x,'dyn/obs0'),'dyn/obs0norm'))
        logit=self.linear(x,'dyn/obslogit').reshape(b,16,8)
        probs=F.softmax(logit,-1)*.99+.01/8
        index=(uniforms[:,:,None]>torch.cumsum(probs,-1)).sum(-1).clamp(max=7)
        stoch=F.one_hot(index,8).float()
        x=torch.cat([deter,stoch.reshape(b,-1)],-1)
        for i in range(3):x=F.silu(self.norm(self.linear(x,f'pol/mlp/linear{i}'),f'pol/mlp/norm{i}'))
        action=torch.tanh(self.linear(x,'pol/head/velocity/mean'))
        return action,deter,stoch,tokens,logit
