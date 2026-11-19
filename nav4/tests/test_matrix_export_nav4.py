"""Small scan-coordinate scenes exercise the normal raster and SDF kernels."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from nav4.matrix.export_nav4 import Raster, capsule, hfield, obstacle_sdf, projected_pieces
from nav4.matrix.collision import heightfield


class ExportGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.grid = Raster([0.,0.], (15,15), cls.temp.name)

    def test_step_centres_keep_vertical_edge_and_slope(self):
        z = np.full(self.grid.shape, 0., np.float32)
        v = np.array([[.435,.1,.16],[.835,.1,.18],[.835,1.,.18],[.435,1.,.16]])
        f = np.array([[0,1,2],[0,2,3]])
        self.grid.triangles(v,f,z)
        self.assertEqual(z[4,4], 0.)
        np.testing.assert_allclose(z[5:9,4], .16+(.1*np.arange(5,9)-.435)*.05, atol=1e-7)
        self.assertEqual(z[9,4], 0.)

    def test_heightfield_matches_mujoco_at_off_lattice_points(self):
        root = ET.fromstring("<mujoco><asset/><worldbody/></mujoco>")
        directory = Path(self.temp.name)
        vals = np.array([[.12,.64,.4],[.51,.05,.15],[.02,.75,.06],[.6,.28,.8]])
        g = heightfield(root,directory,"saddle",[.018,.032],vals,.22)
        g.set("rgba","1 1 1 1")
        asset = root.find("asset/hfield")
        z = np.full(self.grid.shape,-np.inf,np.float32)
        hfield(directory,g,asset,self.grid,z)
        path = directory/"test.xml"
        ET.ElementTree(root).write(path)
        m = mujoco.MjModel.from_xml_path(str(path))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        for i,j in np.argwhere(np.isfinite(z)):
            hit = mujoco.mj_ray(m,d,np.array([i*.1,j*.1,2.]),np.array([0.,0.,-1.]),
                                None,1,-1,np.empty(1,np.int32))
            self.assertGreaterEqual(hit,0.)
            self.assertAlmostEqual(2-hit,z[i,j],places=5)

    def test_body_slab_and_distance_inside_union(self):
        ground = np.zeros(self.grid.shape,np.float32)
        band = np.array([.3,.8])
        def box(x0,x1,z0,z1):
            v = np.array([[x,y,z] for x in [x0,x1] for y in [.215,.785] for z in [z0,z1]])
            return projected_pieces(v,ConvexHull(v).simplices,self.grid,ground,band)
        pieces = box(.235,.535,.31,.79) + box(.465,.725,.31,.79)
        self.assertFalse(box(.8,1.,.82,.95))
        self.assertFalse(box(.8,1.,.05,.29))
        sdf = obstacle_sdf(self.grid,pieces)
        self.assertAlmostEqual(sdf[2,5], .035,places=6)
        self.assertAlmostEqual(sdf[5,5], -.225,places=6)
        self.assertAlmostEqual(sdf[8,8], np.hypot(.075,.015),places=6)

    def test_capsule_cross_section_at_band_boundary(self):
        ground = np.zeros(self.grid.shape,np.float32)
        g = ET.Element("geom",fromto=".3 .5 .27 .9 .5 .27",size=".05")
        v,f = capsule(g)
        sdf = obstacle_sdf(self.grid,projected_pieces(v,f,self.grid,ground,np.array([.3,.8])))
        self.assertAlmostEqual(sdf[5,5],-.04,delta=.0002)
        self.assertAlmostEqual(sdf[5,6],.06,delta=.0002)

    def test_soft_voxel_visible_through_cell_edge_without_collision(self):
        shape = self.grid.shape
        ground = np.zeros(shape,np.float32)
        atlas = np.zeros((*shape,20),np.float32)
        atlas[...,2:19:3] = -1
        top = ground.copy()
        soft = np.zeros(shape,np.uint8)
        # Narrow vertical side misses all centres but intersects a real BEV column.
        v = np.array([[.32,.27,.25],[.32,.33,.25],[.32,.33,1.2],[.32,.27,1.2]])
        self.grid.triangles(v,[[0,1,2],[0,2,3]],ground,2,atlas,top,soft)
        self.assertEqual(soft[3,3],1)
        self.assertEqual(soft.sum(),1)
        self.assertAlmostEqual(top[3,3],1.2,places=6)
        np.testing.assert_allclose(atlas[3,3,3:19:3],[0,1,1,1,0,0])
        self.assertTrue((obstacle_sdf(self.grid,[])>0).all())

