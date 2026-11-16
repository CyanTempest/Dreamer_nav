"""Isolated-process, same-condition 2048-env observe benchmark; no env.step."""
import argparse,time,json,gc
import numpy as np
import jax
import jax.numpy as jnp
from .r8_p1_common import ROOT,dump,Monitor
from .lwy_bank import LwyBank
from .lwy_env import LwyEnv
from .r8_p1_online import SimpleEnv,Atlas,state_at,fitted_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['training','simple','no_visibility'],required=True);p.add_argument('--n',type=int,default=2048);args=p.parse_args()
    mon=Monitor('throughput_'+args.method);n=args.n
    assert jax.default_backend()=='gpu',jax.devices()
    bank=LwyBank('test',limit=1,root='maps_lwy')
    config={'base_height_jitter':0.}
    if args.method=='training':env=LwyEnv(bank,n,random_obstacles=False,config=config)
    else:
        config.update(fitted_config())
        if args.method=='no_visibility':config['p1_visibility']=False
        env=SimpleEnv(bank,n,config=config)
    stations=np.linspace(3,997,n,dtype=np.float32);mids=jnp.zeros(n,jnp.int32)
    xyz,yaw,_=bank.route_at(mids,jnp.asarray(stations));xyz=np.asarray(xyz);yaw=np.asarray(yaw)
    rng=np.random.default_rng(8814);offset=rng.uniform(-.25,.25,n)
    xy=xyz[:,:2]+np.c_[-np.sin(yaw),np.cos(yaw)]*offset[:,None]
    z=np.asarray(bank.sample(mids,jnp.asarray(xy),0,linear=True))
    poses=np.c_[xy,z,yaw+rng.uniform(-.3,.3,n)].astype(np.float32)
    print('STATE',args.method,n,flush=True)
    state=state_at(env,poses,stations);jax.block_until_ready(state)
    t=time.perf_counter();out=env.observe(state);jax.block_until_ready(out)
    compile_warm=time.perf_counter()-t
    del out;gc.collect()
    # Fixed input prevents state aliasing/donation from giving either method an
    # advantage. Return and synchronize the entire state/obs/extra, including
    # 128x128 memory, privileged decoder target, and 13-vector observation.
    times=[]
    for rep in range(30):
        start=time.perf_counter();out=env.observe(state);jax.block_until_ready(out)
        elapsed=time.perf_counter()-start;times.append(elapsed)
        if rep in [0,9,19,29]:print('OBSERVE',rep,elapsed,flush=True)
        del out
    # Frame-only is supplementary; the acceptance ratio uses complete observe.
    frame=jax.jit(type(env)._frame)
    k=jax.random.PRNGKey(8815);out=frame(env,state,k);jax.block_until_ready(out);del out
    frame_times=[]
    for rep in range(10):
        start=time.perf_counter();out=frame(env,state,k);jax.block_until_ready(out);frame_times.append(time.perf_counter()-start);del out
    stats=jax.devices()[0].memory_stats()
    compiled=jax.jit(type(env)._observe).lower(env,state).compile()
    analysis=compiled.memory_analysis()
    compiled_memory={name:int(getattr(analysis,name)) for name in ['argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes']}
    result=dict(method=args.method,envs=n,jax=jax.__version__,device=str(jax.devices()[0]),compile_warm_seconds=compile_warm,
        observe_seconds=times,median_seconds=float(np.median(times)),min_seconds=min(times),p90_seconds=float(np.percentile(times,90)),
        frame_seconds=frame_times,frame_median_seconds=float(np.median(frame_times)),env_steps_per_second=n/np.median(times),
        gpu_memory_stats=stats,compiled_memory=compiled_memory,config=env.cfg,poses='2048 distinct full-route positions, +/- .25m offset and +/- .3 rad yaw, seed8814',
        synchronization='jax.block_until_ready on entire returned state, obs and extra; fixed identical input state; 30 warm calls',
        scope='Complete LwyEnv.observe including 8s memory, public prior, decoder target, state vector, noise and observation queue. No policy or env.step.',
        host_bank_bytes=bank.host_bytes,atlas_bytes=getattr(getattr(env,'atlas',None),'bytes',0),gpu_shared=True)
    dump(ROOT/f'metrics/r8_p1_throughput_{args.method}.json',result);mon.close()
    print('DONE',args.method,result['median_seconds'],mon.peak,flush=True)

if __name__=='__main__':main()
