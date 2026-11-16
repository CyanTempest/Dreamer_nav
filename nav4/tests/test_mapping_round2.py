"""Round-2 regressions: bank-contaminated stairs and capability-independent hard evidence."""
import copy
import unittest
import numpy as np

from nav4.mapping.common import load_config
from nav4.tests.test_mapping_terrain import Scene, flat, surface_points
from nav4.tests.test_mapping_fixes import run_scene

CFG, _ = load_config()


class Round2(unittest.TestCase):
    def test_high_neighbour_cannot_hide_a_tall_object(self):
        from nav4.mapping.model import collision_returns
        # First two returns are at equal height above a higher neighbour, but
        # only the second is a tall object above its own ground. Last is a
        # terrain face excluded by the pre-existing geometry rule.
        own = np.array([.28, .65, .65])
        above_neighbour = np.array([.25, .25, .10])
        np.testing.assert_array_equal(collision_returns(own, above_neighbour, CFG), [False, True, False])

    def test_observed_treads_are_not_the_adjacent_steep_bank(self):
        def stairs(x, y):
            return .14*np.floor(np.clip(np.asarray(x)-1, 0, 4)/.35)
        def ground(x, y):
            return stairs(x, y)+4*np.maximum(np.asarray(y)-3.35, 0)
        sc = Scene()
        x,y,z,nz = surface_points(sc.rng,0,8,0,6,ground)
        nz = np.where(y<=3.35,1,1/np.sqrt(17)).astype(np.float32)
        sc.add(x,y,z,nz)
        # Supply the measured ground directly to isolate derivative contamination
        # from the independent ground-extraction/vegetation classifier.
        from nav4.mapping.grid import CellIndex
        from nav4.mapping.model import tread_surface, max_step, plane_gradient
        cells=CellIndex(x,y,z,np.zeros(2),(81,61))
        gx,gy=np.meshgrid(np.arange(81)*.1,np.arange(61)*.1,indexing='ij')
        g=ground(gx,gy).astype(np.float32);known=np.ones(g.shape,bool)
        tz,tread,_=tread_surface(cells,np.abs(cells.take(nz)),g,known,CFG)
        dx,dy,_=plane_gradient(g,known,.5);grade=np.rad2deg(np.arctan(np.hypot(dx,dy)))
        tx,ty,_=plane_gradient(tz,tread,.5);tgrade=np.rad2deg(np.arctan(np.hypot(tx,ty)))
        step=max_step(tz,tread)
        edge=(gx>2)&(gx<4)&(gy>=3.1)&(gy<=3.3)
        corrected=edge & (grade>47) & tread & (step<=.28) & (tgrade<=47)
        self.assertGreater(corrected.sum(),10)
        bank=(gy>=3.6)&(gy<=3.7)&(gx>2)&(gx<4)
        self.assertFalse(tread[bank].any())

    def test_tread_selector_does_not_erase_040_drop(self):
        def ground(x,y): return np.where(np.asarray(x)>=4,-.4,0.)
        sc=Scene();sc.ground(ground)
        path=np.c_[np.arange(.2,7.8,.1),np.full(76,3.)]
        r=run_scene(sc,ground,path,path_yaw=np.zeros(76))
        edge=(r['gx']>=3.9)&(r['gx']<=4)&(r['gy']>2.8)&(r['gy']<3.2)
        self.assertTrue(r['tread'][edge].all())
        self.assertFalse(r['tread_exempt'][edge].any())
        self.assertFalse(r['tread_exempt'][~r['known']].any())

    def test_height_cut_preserves_geometry_and_one_remaining_hard_return(self):
        # A thin upright with measured geometry below .30 m and one return above.
        # The full geometry supplies the evidence; raising the collision cutoff
        # must not discard it solely because only one high return remains.
        sc=Scene();sc.ground(flat)
        n=30
        sc.add(np.full(n,2.1),np.full(n,1.2),np.linspace(.23,.29,n),np.zeros(n),cls=13)
        sc.add(np.array([2.1]),np.array([1.2]),np.array([.42]),np.zeros(1),cls=13)
        # A second low object is entirely below the traversable height.
        sc.add(np.full(n,5.1),np.full(n,1.2),np.linspace(.23,.27,n),np.zeros(n),cls=13)
        path=np.c_[np.arange(.2,7.8,.1),np.full(76,3.)]
        r=run_scene(sc,flat,path,path_yaw=np.zeros(76))
        self.assertEqual(r['n_collision'][21,12],1)
        self.assertTrue(r['hard'][21,12]);self.assertLess(r['sdf']['obstacle_c'][21,12],0)
        self.assertEqual(r['n_collision'][51,12],0)
        self.assertFalse(r['hard'][51,12])
        old=copy.deepcopy(CFG);old['obstacles']['body_band_m']=[.22,.8]
        r0=run_scene(sc,flat,path,path_yaw=np.zeros(76),cfg=old)
        for k in ('n_body','peak_subbin','occ_bins','vox_lo_lin','vox_hi_lin'):
            np.testing.assert_array_equal(r['feat'][k],r0['feat'][k])
        self.assertTrue(r0['hard'][51,12])


if __name__ == '__main__':
    unittest.main()
