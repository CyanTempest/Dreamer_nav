import json
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from nav4 import sensor, train_dreamer
from nav4.env import TrailEnv
from nav4.tests import synthetic
from nav4.tests.helpers import QUIET, bank, load, reset


class R10(unittest.TestCase):
    def test_replay_preset_is_opt_in(self):
        cfg = json.loads((train_dreamer.ROOT/'nav4/config/train.json').read_text())
        self.assertEqual(cfg['replay']['capacity'], 200000)
        train_dreamer.apply_r10(cfg)
        replay = cfg['replay']
        self.assertEqual(replay['capacity'], 1680000)
        self.assertTrue(replay['batched'])
        self.assertFalse(replay['compress'])
        self.assertLess((replay['capacity'] + 1024*(64+4*replay['chunksize']))*53355, 100e9)

    def test_mixture_and_rejection(self):
        data = synthetic.build('flat')
        # Hole within the selected interval. No centre within it can pass the footprint check.
        x = data['origin'][0]+np.arange(len(data['layers']))*data['res']
        data['layers'][(x>10)&(x<11), :, 2] = 2.
        mix = dict(route_start=.2, intervals=[dict(lo=8., hi=14., p=.5)])
        env = TrailEnv(load({'flat': data}), 64, {**QUIET, 'task.start_mixture': mix})
        def sample(key):
            mid=jnp.zeros(env.n, jnp.int32)
            s, _, dyaw, boxes=env.mixture_start(key,mid,jnp.ones(env.n,bool))
            p, yaw, _=env.bank.route_at(mid,s)
            valid=env._feasible(dict(mids=mid,obstacles=boxes),p[:,:2],yaw+dyaw)
            return s,valid
        sample=jax.jit(sample)
        samples=[]
        for key in jax.random.split(jax.random.PRNGKey(77),32):
            s,ok=sample(key)
            self.assertTrue(bool(ok.all()))
            samples.extend(np.asarray(s))
        s=np.asarray(samples)
        self.assertAlmostEqual(np.mean(s==2.), .2, delta=.035)
        self.assertAlmostEqual(np.mean((s>=8)&(s<14)), .5+.3*4/27, delta=.045)
        self.assertFalse(((s>10)&(s<11)).any())
        one=TrailEnv(env.bank,2,{**QUIET,'task.start_mixture':dict(route_start=0.,intervals=[dict(lo=8.,hi=14.,p=1.)])})
        st,_,_=reset(one,2.)
        np.testing.assert_array_equal(st['s'],[2.,2.])
        st['done']=jnp.ones(2,bool)
        fresh,_,_=one.step(st,jnp.zeros((2,3)))
        self.assertTrue(bool(((fresh['s']>=8)&(fresh['s']<14)).all()))
        self.assertTrue(bool(one.feasible(fresh,fresh['pos'],fresh['yaw']).all()))

    def test_effective_finish(self):
        env=TrailEnv(bank('flat'),1,{**QUIET,'task.end_station':28.})
        st,_,_=reset(env,27.99)
        moving=jnp.array([.5,0.,0.])
        st.update(vel=moving[None],queue=jnp.broadcast_to(moving,st['queue'].shape),max_steps=jnp.array([10]))
        st,obs,_=env.step(st,jnp.array([[1.,0.,1.]]))
        self.assertTrue(bool(st['arrived'][0]))
        self.assertEqual(int(st['gates'][0]),8)
        self.assertEqual(float(st['s'][0]),28.)
        self.assertAlmostEqual(float(st['pos'][0,0]),28.,places=4)
        self.assertGreater(float(obs['reward'][0]),9.)
        pose,_=env.restart_pose(st,jnp.array([29.]))
        self.assertAlmostEqual(float(pose[0,0]),27.8,places=4)
        self.assertEqual(float(env.bank.gates[0,-1]),29.)

    def test_failure_beyond_finish_does_not_pollute_final_substep(self):
        data = synthetic.build('flat')
        # Support ends at the far edge of the x=28.5 cell (28.55).
        # Front edge at finish is 28.515. A full decision would drive beyond 28.55.
        x = data['origin'][0] + np.arange(len(data['layers']))*data['res']
        data['layers'][x>28.55, :, 2] = 1.
        env = TrailEnv(load({'edge':data}), 1, {**QUIET,'task.end_station':28.})
        st, _, _ = reset(env,27.99)
        action = jnp.array([[1.,0.,1.]])
        moving = jnp.array([.8,0.,0.])
        st.update(vel=moving[None],queue=jnp.broadcast_to(moving,st['queue'].shape),max_steps=jnp.array([10]))
        st, _, _ = env.step(st,action)
        self.assertTrue(bool(st['arrived'][0]))
        self.assertTrue(bool(env.feasible(st,st['pos'],st['yaw'])[0]))

    def test_opening_ignores_unobserved_truth_and_dropped_points(self):
        h=jnp.arange(81,dtype=jnp.float32).reshape(1,9,9)
        keep=(h%4==0)&(h<69)
        a=sensor.measured_opening(h,keep,5)
        b=sensor.measured_opening(jnp.where(keep,h,jnp.nan),keep,5)
        np.testing.assert_array_equal(a,b)
        self.assertTrue((np.asarray(a)[~np.asarray(keep)]==0).all())
        self.assertTrue(np.isfinite(a).all())

    def test_frame_cannot_read_height_fill_in_unscanned_hole(self):
        m = synthetic.build('flat')
        x = m['origin'][0] + np.arange(len(m['atlas']))*m['res']
        hole = (x > 6.0) & (x < 6.5)
        m['atlas'][hole, :, 19] = 0.
        frames = []
        for fill in (40., 400.):
            m['atlas'][hole, :, 0] = fill
            env = TrailEnv(load({'flat': m}), 1, {**QUIET, 'sensor.version': 'v2'})
            st, _, _ = reset(env, 5.)
            frames.append(jax.device_get(jax.jit(sensor.frame)(env, st, jax.random.PRNGKey(9))))
        for before, after in zip(*frames):
            np.testing.assert_array_equal(before, after)
        measured, keep, _ = frames[0]
        self.assertGreater(keep.sum(), 0)
        self.assertTrue((measured[~keep] == 0).all())

    def test_static_canopy_shadows_ground_without_collision_label(self):
        data=synthetic.build('flat')
        x=data['origin'][0]+np.arange(len(data['layers']))*data['res']
        wall=(x>=7.0)&(x<=7.4)
        data['layers'][wall,:,1]=1.0
        data['layers'][wall,:,6]=2
        b=load({'canopy':data})
        for version in ('v1','v2'):
            env=TrailEnv(b,1,{**QUIET,'sensor.version':version})
            st,_,_=reset(env,5.)
            _,_,seen=jax.jit(lambda st: sensor.frame(env,st,jax.random.PRNGKey(0)))(st)
            mask=(env.local[...,0]>3.)&(env.local[...,0]<4.)&(np.abs(env.local[...,1])<.5)
            rate=np.asarray(seen)[0,mask].mean()
            if version == 'v1':
                self.assertGreater(rate, .95)
            else:
                self.assertEqual(rate, 0.)

    def test_camera_dropout_lasts_across_steps_and_assistance(self):
        env=TrailEnv(bank('flat'),4,{**QUIET,'sensor.version':'v2','sensor.camera_off_probability':.5})
        st,_,_=reset(env,5.,seed=7)
        initial=np.asarray(st['camera_on'])
        for done in (False, True):
            st.update(done=jnp.full((4,),done),restart=jnp.tile(jnp.array([6.,0.]),(4,1)))
            st,_,_=env.step(st,jnp.zeros((4,3)))
            np.testing.assert_array_equal(st['camera_on'],initial)


