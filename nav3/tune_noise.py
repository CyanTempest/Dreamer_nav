"""Fit only distance dropout to 30 calibration poses; reserve 30 for checking."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .bank import MapBank
from .env import FastEnv


def main():
    ref=json.loads(Path('metrics/ray_reference.json').read_text())
    n=len(ref['rows']);bank=MapBank('test');env=FastEnv(bank,n)
    state,_,_=env.reset(jax.random.PRNGKey(4096),jnp.zeros(n,jnp.int32),jnp.array([r['s_m'] for r in ref['rows']]))
    poses=np.array([r['pose'] for r in ref['rows']])
    state['pos']=jnp.array(poses[:,:2],jnp.float32);state['yaw']=jnp.array(poses[:,3],jnp.float32);state['height']=jnp.full(n,.54)
    key=jax.random.PRNGKey(44)
    _,_,visible,_,_=jax.jit(env._frame)(state,key)
    visible=np.asarray(visible)
    key1,_=jax.random.split(key);u=np.asarray(jax.random.uniform(key1,visible.shape))
    distance=np.asarray(jnp.linalg.norm(env.local,axis=-1))
    ray=np.stack([np.load(f'metrics/rays/pose_{i:03d}.npz')['count']>0 for i in range(n)])
    # Alternating poses avoid fitting only to one feature class.
    calibration=np.arange(n)%2==0;validation=~calibration
    trials=[]
    for base in [.015,.03,.05]:
        for coefficient in [.007,.009,.011,.013,.015,.017]:
            known=visible&(u>base+coefficient*distance**2)
            error=float(abs((~known[calibration]).mean()-(~ray[calibration]).mean()))
            trials.append(dict(base=base,distance=coefficient,calibration_unknown_error=error,
                validation_unknown=float((~known[validation]).mean()),validation_ray_unknown=float((~ray[validation]).mean())))
    best=min(trials,key=lambda r:r['calibration_unknown_error'])
    result=dict(best=best,calibration_pose_indices=np.flatnonzero(calibration).tolist(),validation_pose_indices=np.flatnonzero(validation).tolist(),
        objective='Absolute calibration-set unknown fraction error; no performance/label fitting',trials=trials)
    Path('metrics/noise_tuning.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(best,indent=2))


if __name__=='__main__':main()
