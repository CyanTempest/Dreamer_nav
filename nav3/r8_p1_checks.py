"""Functional tests for memory age/reprojection, contracts and vmap parity."""
import numpy as np
import jax
import jax.numpy as jnp
from .r8_p1_common import ROOT,GEO,dump,Monitor
from .r8_p1_online import SimpleEnv,Atlas,state_at
from .r8_p1_validate import first_frame
from .lwy_bank import LwyBank
from .env import decode

def main():
    monitor=Monitor('checks')
    refs=np.load(GEO/'r8_p1_references.npz');bank=LwyBank('test',limit=1,root='maps_lwy');atlas=Atlas()
    cfg={'position_walk_std':0.,'yaw_walk_std':0.,'max_observation_delay_steps':0}
    env=SimpleEnv(bank,3,atlas=atlas,config=cfg);st=state_at(env,refs['poses'][1:4],refs['stations'][1:4])
    key=jax.random.PRNGKey(121)
    m,s,v=jax.jit(SimpleEnv.sensor_frame)(env,st,key)
    def vmapped(env,pos,yaw,mids,key):
        return jax.vmap(lambda p,y,mid:env.sensor_frame({'pos':p[None],'yaw':y[None],'mids':mid[None]},key)[0][0])(pos,yaw,mids)
    vm=jax.jit(vmapped)(env,st['pos'],st['yaw'],st['mids'],key)
    parity=float(np.max(np.abs(np.asarray(m)-np.asarray(vm))));assert parity<1e-5,parity
    _,obs,_=env.observe(st)
    expected=first_frame(np.asarray(m),np.asarray(s),refs['poses'][1:4],refs['local'],refs['clean'][1:4,...,4]/255)
    # Float32 JAX vs NumPy rotations can straddle world-cell boundaries. Count
    # and disclose any such cells instead of changing either scatter contract.
    changed=np.any(np.asarray(obs['bev'])!=expected,axis=-1).sum(axis=(1,2))
    assert np.asarray(obs['bev']).shape==(3,64,64,6)
    assert np.asarray(obs['bev']).dtype==np.uint8
    assert np.all(np.asarray(obs['bev'])[...,3]==np.asarray(obs['bev'])[...,5])
    # Freeze sensing; preload a single observed world cell. Exercise translation,
    # rotation and recentering through the actual inherited complete _observe.
    env=SimpleEnv(bank,1,atlas=atlas,config={**cfg,'p1_sensor_enabled':False})
    st=state_at(env,refs['poses'][2:3],refs['stations'][2:3]);z=float(np.asarray(st['memz'])[0])
    marker=np.array([82,70]);world=np.asarray(st['mempos'])[0]+(marker+.5)*.125
    st['memory']=st['memory'].at[0,82,70].set(jnp.array([z+.25,.4,0.]));st['age']=st['age'].at[0,82,70].set(0.)
    rec=[]
    for step in range(42):
        # Slow subpixel movement with integer recenter crossings; no env.step,
        # policy, collision rollout, or training is invoked.
        st['pos']=jnp.array(refs['poses'][2:3,:2]+[.012*step,.004*step]);st['yaw']=jnp.array(refs['poses'][2:3,3]+.006*step)
        st,obs,_=env.observe(st);bev=np.asarray(decode(obs['bev']))[0]
        mem=np.asarray(st['memory'])[0];age=np.asarray(st['age'])[0];origin=np.asarray(st['mempos'])[0]
        ij=np.floor((world-origin)/.125).astype(int)
        a=float(age[tuple(ij)]);recorded=mem[tuple(ij)]
        assert abs(float(recorded[1])-.4)<1e-5
        assert abs(a-(step+1)*.2)<2e-5,(step,a)
        # Check that the observed age mask agrees with actual world addresses.
        yaw=float(np.asarray(st['yaw'])[0]);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        xy=refs['local']@R.T+np.asarray(st['pos'])[0];idx=np.floor((xy-origin)/.125).astype(int)
        hits=np.all(idx==ij,axis=-1);known=bev[...,3]<.5
        expected_known=hits&(a<8)
        assert np.array_equal(known,expected_known),(step,int(known.sum()),int(expected_known.sum()))
        rec.append(dict(step=step+1,time_s=(step+1)*.2,marker_age_s=a,known_cells=int(known.sum()),marker_world_xy=world.tolist()))
    dump(ROOT/'metrics/r8_p1_checks.json',dict(status='pass',vmap_max_abs=parity,first_frame_numpy_scatter_changed_cells=changed,shape=[3,64,64,6],dtype='uint8',first_frame_age_equals_unknown=True,memory_seconds=8.,memory_moving_marker=rec,uses_env_step=False))
    print('CHECKS PASS',parity,changed,flush=True);monitor.close()

if __name__=='__main__':main()
