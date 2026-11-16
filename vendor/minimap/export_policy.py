"""4090 cross-runtime parity, 1-step and 100-step recurrent trajectories + TorchScript."""
import os,sys,argparse,pickle,json,time
from pathlib import Path
import numpy as np
from scene import ROOT
# Deliberately separate vendor directory; do not change installed JAX environment.
sys.path.append(str(ROOT/'torch_vendor'))
import jax,jax.numpy as jnp
import ninjax as nj
import torch
from model import make_model,deployment_policy
from train import Replay
from torch_policy import TorchPolicy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--tag',default='');a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    with open(a.checkpoint,'rb') as f:ck=pickle.load(f)
    weights={k:np.asarray(v) for k,v in ck['params'].items() if k.startswith(('enc/','dyn/','pol/'))}
    params=jax.device_put(ck['params']);del ck
    np.savez_compressed(ROOT/f'exports/{a.tag}policy_weights.npz',**weights)
    m,cfg=make_model();fn=deployment_policy(m);pt=TorchPolicy(weights).cuda().eval()
    # 100 actual sequential observations from independent developer map, no synthetic random BEV.
    from nav_env import NavEnv
    from control import planner
    e=NavEnv(333,obstacles=True);rng=np.random.default_rng(55);jc=m.init_policy(1)
    d=torch.zeros(1,256,device='cuda');s=torch.zeros(1,16,8,device='cuda');prev=torch.zeros(1,3,device='cuda')
    errors=[];inputs=[];single=None;upstream_errors=[]
    def official_features(carry,obs):
        _,_,tokens=m.enc(carry[0],obs,obs['is_first'],False,single=True)
        _,_,feat=m.dyn.observe(carry[1],tokens,carry[3],obs['is_first'],False,single=True)
        return tokens,feat['deter'],feat['logit']
    official_pure=nj.pure(official_features)
    official=jax.jit(lambda p,c,o:official_pure(p,c,o,seed=jax.random.PRNGKey(17))[1])
    with torch.no_grad():
        for t in range(100):
            if e.last:e=NavEnv(334+t,obstacles=True)
            obs=e.observe();o={k:jnp.asarray(v[None]) for k,v in obs.items()};u=rng.random((1,16)).astype(np.float32)
            before=jc
            jc,ja,extra=fn(params,jc,o,jnp.asarray(u));ja=np.asarray(ja)
            if t in (0,50):
                tokens0,deter0,logit0=official(params,before,o)
                upstream_errors.append(max(float(np.max(abs(np.asarray(tokens0)-np.asarray(extra['tokens'])))),float(np.max(abs(np.asarray(deter0)-np.asarray(jc[1]['deter'])))),float(np.max(abs(np.asarray(logit0)-np.asarray(extra['logit']))))))
            args=(torch.from_numpy(obs['bev'][None]).cuda(),torch.from_numpy(obs['state'][None]).cuda(),d,s,prev,torch.tensor([obs['is_first']],device='cuda'),torch.from_numpy(u).cuda())
            ta,d,s,tok,logit=pt(*args)
            err={'step':t,'action_max_abs':float(np.max(abs(ta.cpu().numpy()-ja))),'deter_max_abs':float(np.max(abs(d.cpu().numpy()-np.asarray(jc[1]['deter'])))),'stoch_max_abs':float(np.max(abs(s.cpu().numpy()-np.asarray(jc[1]['stoch'])))),'encoder_max_abs':float(np.max(abs(tok.cpu().numpy()-np.asarray(extra['tokens'])))),'posterior_logits_max_abs':float(np.max(abs(logit.cpu().numpy()-np.asarray(extra['logit']))))}
            errors.append(err)
            if t==0:single=err;trace_args=args
            inputs.append(dict(bev=obs['bev'],state=obs['state'],reset=obs['is_first'],uniforms=u[0]))
            # Both recurrent implementations are closed on their own predicted previous action.
            prev=ta
            e.step(planner(obs['bev'],obs['state']))
        scripted=torch.jit.trace(pt,trace_args,check_trace=True)
        scripted.save(str(ROOT/f'exports/{a.tag}policy_step.pt'))
        for _ in range(30):scripted(*trace_args)
        torch.cuda.synchronize();lat=[]
        for _ in range(200):
            t0=time.perf_counter();scripted(*trace_args);torch.cuda.synchronize();lat.append((time.perf_counter()-t0)*1000)
    np.savez_compressed(ROOT/f'exports/{a.tag}parity_inputs.npz',**{k:np.stack([x[k] for x in inputs]) for k in inputs[0]})
    maxes={k:max(x[k] for x in errors) for k in single if k!='step'}
    result=dict(checkpoint=a.checkpoint,torch=torch.__version__,jax=jax.__version__,device=torch.cuda.get_device_name(),precision='FP32; PyTorch TF32 disabled; JAX default backend matmul precision',categorical_sampling='identical explicit U(0,1), inverse CDF, 1% unimix',actor='mean of bounded_normal; training remains stochastic',single_step=single,sequence_100=maxes,thresholds={'action':1e-4,'deter':1e-4,'stoch':0,'encoder':2e-4,'logits':2e-4},passed=maxes['action_max_abs']<1e-4 and maxes['deter_max_abs']<1e-4 and maxes['stoch_max_abs']==0 and maxes['encoder_max_abs']<2e-4 and maxes['posterior_logits_max_abs']<2e-4,raw_errors=errors,torchscript_latency_ms={'p50':float(np.median(lat)),'p95':float(np.percentile(lat,95)),'raw':lat},export_params=sum(v.size for k,v in weights.items() if not k.startswith(('dyn/prior','pol/head/velocity/stddev'))))
    result['official_observe_max_abs']=max(upstream_errors);result['passed'] &= max(upstream_errors)<1e-5
    (ROOT/f'metrics/{a.tag}export_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='raw_errors'},indent=2));assert result['passed']
if __name__=='__main__':main()
