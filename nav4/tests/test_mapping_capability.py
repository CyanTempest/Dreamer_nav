"""Map-side confirmed terrain limits and solid obstacle regression tests."""
import unittest
from unittest.mock import patch

import numpy as np

from nav4.mapping.common import RES, load_config
from nav4.mapping.model import max_step, plane_gradient
from nav4.tests.test_mapping_terrain import Scene, flat, DENSITY

CFG, _ = load_config()


class MappingCapability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = np.meshgrid(np.arange(61)*RES, np.arange(61)*RES, indexing='ij')
        cls.known = np.ones_like(cls.x, bool)

    def cap(self, h):
        gx, gy, _ = plane_gradient(h, self.known, CFG['terrain']['grade_radius_m'])
        return ((max_step(h, self.known) <= CFG['terrain']['max_step_m']+1e-6)
                & (np.hypot(gx, gy) <= np.tan(np.deg2rad(CFG['terrain']['max_grade_deg']))+1e-6))

    def test_riser_028_and_ledge_040(self):
        for sign in (-1, 1):
            self.assertTrue(self.cap(np.where(self.x >= 3, sign*.28, 0.))[5:-5,5:-5].all())
            m = self.cap(np.where(self.x >= 3, sign*.4, 0.))
            self.assertFalse(m[(self.x >= 2.9) & (self.x <= 3) & (self.y > 1) & (self.y < 5)].any())

    def test_47_degree_plane_and_50_degree_rejection(self):
        for yaw in (0, np.pi/4, np.pi/2):
            along = self.x*np.cos(yaw)+self.y*np.sin(yaw)
            self.assertTrue(self.cap(np.tan(np.deg2rad(47))*along)[5:-5,5:-5].all())
            self.assertFalse(self.cap(np.tan(np.deg2rad(50))*along)[5:-5,5:-5].any())

    def test_full_model_028_riser_and_040_drop_on_walked_path(self):
        from nav4.mapping.common import footprint_ok
        for height in (.28, -.40):
            zfun = lambda x, y: np.where(np.asarray(x) >= 4., height, 0.)
            sc = Scene(); sc.ground(zfun)
            # Exact synthetic surfaces: capability boundary, not a noise-margin test.
            x,y,z,nz,cls = sc.parts[0]
            sc.parts[0] = (x,y,zfun(x,y),nz,cls)
            # Generic capability is distinct from the user-declared A-trail prior.
            from nav4.tests.test_mapping_terrain import CFG as scene_cfg
            with patch.dict(scene_cfg['terrain'], trail_terrain_exemption=False):
                r = sc.run(zfun)
            ok = footprint_ok(r['sdf']['support'], r['sdf']['obstacle_c'], np.zeros(2),
                              np.array([[4.,3.]]), np.zeros(1))
            self.assertEqual(bool(ok[0]), height == .28)
            if height < 0:
                self.assertFalse(r['exempt'].any(), 'walked path must not override a cliff')

    def test_wall_post_and_large_rock_keep_collision_returns(self):
        sc = Scene(); sc.ground(flat); rng=sc.rng
        # All well outside the existing walked-corridor prior (path y=3).
        n=10000
        theta=rng.uniform(0,2*np.pi,n)
        sc.add(2+.06*np.cos(theta),1+.06*np.sin(theta),rng.uniform(0,1.2,n),np.zeros(n),cls=13)
        sc.add(rng.uniform(3,4,n),np.full(n,1.0),rng.uniform(0,1.3,n),np.zeros(n))
        # Rock: a solid box with 0.6 m sides, below overhead band, above riser capability.
        for axis in (0,1):
            for side in (-.25,.25):
                a=rng.uniform(-.25,.25,n); b=np.full(n,side)
                x,y=(a,b) if axis==0 else (b,a)
                sc.add(5.5+x,1+y,rng.uniform(0,.6,n),np.zeros(n))
        r=sc.run(flat)
        for name,xy,radius in [('post',(2,1),.15),('wall',(3.5,1),.55),('rock',(5.5,1),.4)]:
            zone=np.hypot(r['gx']-xy[0],r['gy']-xy[1])<radius
            hits=zone & r['hard']
            self.assertTrue(hits.any(),name)
            self.assertTrue((r['sdf']['obstacle_c'][hits]<0).all(),name)


if __name__ == '__main__':
    unittest.main()
