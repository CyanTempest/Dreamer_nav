import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from sensors import decode, motion, scan_age, sim_at_wall
from codec import unpack_wire
from control import Monitor, Pursuit, Route, RouteState, upright
from shard import Episode,digest

CFG=json.loads(Path(__file__).with_name('config.json').read_text())


def cloud(h=860,big=False):
    b=bytearray(b'\x00\x01\x00\x00')
    def put(fmt,align,*values):
        b.extend(b'\x00'*(-(len(b)-4)%align))
        b.extend(struct.pack('<'+fmt,*values))
    def string(v):
        s=v.encode()+b'\x00'
        put('I',4,len(s))
        b.extend(s)
    put('iI',4,100,0)
    string('map')
    put('III',4,h,96,6)
    for label,off,t in [('x',0,7),('y',4,7),('z',8,7),('intensity',12,7),('ring',16,4),('timestamp',18,8)]:
        string(label)
        put('I',4,off)
        put('B',1,t)
        put('I',4,1)
    put('B',1,int(big))
    width=96*26+8
    put('III',4,26,width,width*h)
    row=bytearray(width)
    for i in range(96):
        struct.pack_into(('>' if big else '<')+'ffffHd',row,i*26,
                         float('nan') if i==0 else 75. if i==1 else 1.234,0,0,37.,95-i,100.099)
    return bytes(b+row*h+b'\x00')


class Tests(unittest.TestCase):
    def test_wire(self):
        for h,big in [(860,True),(900,False)]:
            raw=cloud(h,big)
            s=decode(raw,8)
            self.assertEqual(s.distance.shape,(96,900))
            self.assertEqual(s.shape,(h,96))
            self.assertEqual(s.finite,95*h)
            self.assertEqual(s.distance[93,0],1234)
            self.assertEqual(s.distance[94,0],65535)
            self.assertEqual(s.distance[95,0],0)
            self.assertFalse(s.distance[:,h:].any())
            self.assertEqual(s.intensity[0,0],37)
            self.assertEqual(s.raw,raw)

    def test_clocks(self):
        s=decode(cloud(1),8)
        self.assertAlmostEqual(scan_age(s,8.2,92),.101)
        self.assertTrue(np.isinf(scan_age(s,7,92)))
        self.assertTrue(np.isinf(scan_age(s,8,92)))
        self.assertTrue(np.isinf(scan_age(None,8,92)))

    def test_sim_interpolation(self):
        self.assertAlmostEqual(sim_at_wall([(100,12),(100.1,12.05)],100.05),12.025)
        for t in (99,101):
            self.assertTrue(np.isnan(sim_at_wall([(100,12),(100.1,12.05)],t)))
        self.assertTrue(np.isnan(sim_at_wall([(100,12),(101,12.1)],100.5)))

    def test_reject_truncation(self):
        with self.assertRaises(ValueError):
            decode(cloud(1)[:-5],0)

    def test_monitor(self):
        good=RouteState(5,0,.5,0,0,0,False)
        m=Monitor(CFG,5,0)
        self.assertTrue(upright(good,CFG,True))
        self.assertEqual(m.reason(good,14),'')
        self.assertEqual(m.reason(good,15),'stuck')
        for prop,value,expected in [('roll',1.2,'overturned'),('clearance',.2,'overturned'),
                                     ('lateral',1.3,'off_route'),('outside',True,'window_exit')]:
            st=RouteState(**{**vars(good),prop:value})
            self.assertEqual(m.reason(st,0),expected)

    def test_window_and_frame(self):
        s=np.arange(201)*.05
        xy=np.c_[s,0*s]
        ground=np.array([[[i,0,4]] for i in s])
        r=Route(s,xy,ground,3,[1,10],CFG)
        st=r.locate([10.2,.1,1.55,0,0,0,1],9.9)
        self.assertTrue(st.outside)
        self.assertAlmostEqual(st.clearance,.55)
        self.assertAlmostEqual(st.lateral,.1)
        self.assertAlmostEqual(st.s,10.2)

    def test_driver_seeds(self):
        for gs,bs,mode in [(1,0,'gaussian'),(0,1,'burst'),(0,0,'clean')]:
            cfg=dict(CFG,gaussian_fraction=gs,burst_fraction=bs)
            drivers=[Pursuit(cfg,np.random.default_rng(9)) for _ in range(2)]
            self.assertEqual(drivers[0].mode,mode)
            acts=[d.command(dict(elapsed_sim_s=10),RouteState(0,0,.5,0,0,.1,False)) for d in drivers]
            np.testing.assert_equal(*acts)

    def test_shard(self):
        with tempfile.TemporaryDirectory() as d:
            ep=Episode(d,'abc',dict(seed=1))
            s=decode(cloud(1),2)
            self.assertEqual(ep.scan('rear',s),ep.scan('rear',s))
            ep.rows.append(dict(qualified=True))
            ep.finish('window_exit')
            with np.load(ep.path,allow_pickle=False) as z:
                self.assertEqual(unpack_wire(z['rear_00000_cdr_zstd']),s.raw)
                self.assertEqual(json.loads(str(z['meta']))['termination'],'window_exit')
            self.assertEqual(ep.path.with_suffix('.sha256').read_text().split()[0],digest(ep.path))

    @unittest.skipUnless('MV4_CDR_DIR' in os.environ,'optional recorded fixture')
    def test_recorded(self):
        root=Path(os.environ['MV4_CDR_DIR'])
        for file in root.glob('rt_*lidar.cdr'):
            s=decode(file.read_bytes(),0)
            self.assertIn(s.shape,[(860,96),(900,96)])
            self.assertGreater(s.finite,100)
        _,q,omega=motion((root/'rt_imu.cdr').read_bytes(),False)
        _,pose,twist=motion((root/'rt_odom.cdr').read_bytes(),True)
        self.assertAlmostEqual(np.linalg.norm(q),1)
        self.assertAlmostEqual(np.linalg.norm(pose[3:]),1)
        self.assertEqual((len(omega),len(twist)),(3,6))


if __name__=='__main__':
    unittest.main()
