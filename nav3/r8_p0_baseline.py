"""Static observation only: no rollout, policy, training or evaluation queue."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from .lwy_bank import LwyBank
from .lwy_env import LwyEnv
from .lwy_env_r7 import LwyEnvR7
from .env import rotate,decode

STATIONS=np.array([8,95,190,320,480,620,715,820,920,985],np.float32)
def main():
 t=time.perf_counter(); bank=LwyBank('test',limit=1,root='maps_lwy');n=len(STATIONS)
 assert bank.records[0]['length_m']==1000 and bank.records[0]['seed']==862428,bank.records[0]
 out={};records=[]
 for label,options in [('clean',dict(dropout_base=0.,dropout_distance=0.,height_std=0.,position_walk_std=0.,yaw_walk_std=0.,max_observation_delay_steps=0)),('noisy',{})]:
  env=LwyEnv(bank,n,config={'base_height_jitter':0.,**options})
  state=env._reset(jax.random.PRNGKey(88001),jnp.zeros(n,jnp.int32),jnp.asarray(STATIONS),observe=False)
  p,yaw,_=bank.route_at(state['mids'],jnp.asarray(STATIONS))
  state['pos']=p[:,:2];state['yaw']=yaw;state['mempos']=jnp.floor(p[:,:2]/.125)*.125-8
  state['height']=jnp.full(n,.54);state['memyaw']=yaw
  ns,obs,ex=env.observe(state);jax.block_until_ready(obs)
  measured,seen,visible,trav,_=jax.jit(LwyEnv._frame)(env,state,jax.random.PRNGKey(88002))
  xy=state['pos'][:,None,None]+rotate(env.local[None],state['yaw']);truth,_=env.truth(state,xy)
  z=np.asarray(bank.sample(state['mids'],state['pos'],0,linear=True))
  poses=np.c_[np.asarray(p[:,:2]),z,np.asarray(yaw)]
  for k,v in dict(bev=np.asarray(obs['bev']),decoded=np.asarray(decode(obs['bev'])),priv_trav=np.asarray(obs['priv_trav']),measured=np.asarray(measured),seen=np.asarray(seen),visible=np.asarray(visible)).items():out[label+'_'+k]=v
  records.append(dict(label=label,config=env.cfg,unknown_fraction=np.asarray(decode(obs['bev']))[...,3].mean((1,2)).tolist()))
  if label=='clean':
   out.update(stations=STATIONS,poses=poses,truth=np.asarray(truth),local=np.asarray(env.local),state=np.asarray(obs['state']),route_positions=np.asarray(p))
   # R7 sensing parity on exactly the same static state; no episodic obstacles.
   r7=LwyEnvR7(bank,n,random_obstacles=False,config={'base_height_jitter':0.,**options})
   _,r7obs,_=r7.observe(state);jax.block_until_ready(r7obs)
   parity=int(np.max(np.abs(np.asarray(r7obs['bev']).astype(int)-np.asarray(obs['bev']).astype(int))))
 np.savez_compressed('fullgeo/r8_baseline.npz',**out)
 spec=json.loads(Path('../../ATEC_Env/config/track_spec.json').read_text())
 rows=[dict(station_m=float(s),zone=next(z['label'] for z in spec['track']['zones'] if z['start_m']<=s<z['end_m']),pose_world_ground_xyz_yaw=poses[i].tolist()) for i,s in enumerate(STATIONS)]
 result=dict(seconds=time.perf_counter()-t,device=str(jax.devices()[0]),map=bank.records[0],map_bytes=bank.host_bytes,stations=rows,observations=records,r7_bev_max_uint8_difference=parity,static_first_frame=True,no_training_or_rollout=True)
 Path('metrics/r8_p0_baseline.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['map','observations']},indent=2),flush=True)
if __name__=='__main__': main()
