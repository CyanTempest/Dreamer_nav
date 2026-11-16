import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_scan_patches import build_patch
from qualify_routes import footprint_ok,swept_footprint_ok


class ScanGeometryTests(unittest.TestCase):
    def test_rotation_sweep_detects_collision_between_clear_endpoints(self):
        layers=np.zeros((41,41,7));layers[...,2]=-1;layers[...,5]=1
        layers[24,24,5]=-1
        data={'layers':layers,'origin':np.array([-2.,-2.])}
        xy=np.zeros((2,2));yaw=np.array([0,np.pi/2])
        self.assertTrue(footprint_ok(data,xy,yaw).all())
        self.assertFalse(swept_footprint_ok(data,xy,yaw).all())

    def test_canopy_does_not_become_body_collision_and_gap_stays_unknown(self):
        x,y=np.meshgrid(np.arange(-8,48,.1),np.arange(-8,8,.1),indexing='ij')
        xyz=np.stack([x.ravel(),y.ravel(),np.zeros(x.size)],-1)
        xyz=xyz[(xyz[:,0]<10)|(xyz[:,0]>12)]
        canopy=xyz[(xyz[:,0]>18)&(xyz[:,0]<22)&(abs(xyz[:,1])<1)].copy();canopy[:,2]=3.5
        xyz=np.concatenate([xyz,canopy])
        cloud=np.zeros(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('normal_z','f4')])
        for k,col in enumerate(['x','y','z']):cloud[col]=xyz[:,k]
        cloud['normal_z']=1
        route=np.stack([np.arange(201)*.2,np.zeros(201),np.full(201,2.)],-1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):build_patch(cloud,route,0,root,'synthetic')
            with np.load(root/'map.npz') as data:
                def ij(x,y):return tuple(np.rint((np.array([x,y])-data['origin'])/.1).astype(int))
                gap=ij(11,0);overhead=ij(20,0)
                self.assertEqual(data['atlas'][gap][19],0)
                self.assertGreater(data['layers'][gap][2],0)
                self.assertEqual(data['atlas'][overhead][19],1)
                self.assertGreater(data['layers'][overhead][5],0)
                self.assertGreater(data['atlas'][overhead][17],3)


if __name__=='__main__':unittest.main()
