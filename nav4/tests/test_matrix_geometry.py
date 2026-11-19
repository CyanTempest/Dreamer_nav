"""CPU checks of synthetic stairs, railing fits and collision semantics."""
import unittest
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nav4.matrix.compose import xml
from nav4.matrix.ground_mesh import in_polygon, underlay_treads
from nav4.matrix.objects import cylinder_section, solid_shapes, tapered_solid
from nav4.matrix.rail_refine import estimate_circle
from nav4.matrix.stairs import prism, riser_edge, tread_plane
from nav4.matrix.survey import Corridor


class PilotGeometryTest(unittest.TestCase):
    def test_corridor_flat_caps_and_width(self):
        x = np.arange(401)*.05
        corridor = Corridor(np.c_[x,np.zeros_like(x)])
        _, _, keep = corridor.project(np.array([[0.,0.],[10.,5.99],[10.,6.02],[-.01,0.],[20.01,0.]]))
        np.testing.assert_array_equal(keep,[True,True,False,False,False])

    def test_tread_planes_and_vertical_riser(self):
        rng=np.random.default_rng(2)
        x=rng.uniform(.03,.50,1200)
        y=rng.uniform(-.45,.45,1200)
        z=.18+.03*x+.01*y+rng.normal(0,.001,1200)
        result,_,_=tread_plane(np.c_[x,y,z],np.zeros(2),.185)
        np.testing.assert_allclose(result,[.03,.01,.18],atol=.001)
        yr=rng.uniform(-.48,.48,1300)
        xr=.031+.15*yr+rng.normal(0,.002,1300)
        zr=rng.uniform(.02,.16,1300)
        pt, normal, stats=riser_edge(np.c_[xr,yr,zr],np.zeros(2),np.array([1.,0.]),
                                    0.,.18,[-.5,.5],{"edge_search_m":.12})
        self.assertTrue(stats["fit"])
        np.testing.assert_allclose(pt,[.031,0.],atol=.003)
        np.testing.assert_allclose(normal[1]/normal[0],-.15,atol=.01)
        poly=np.array([[0,-.5],[.5,-.5],[.5,.5],[0,.5]])
        vertices,faces=prism(poly,.18+.03*poly[:,0],-.2)
        normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        self.assertEqual(float(abs(normals[4:,2]).max()),0.)
        mask,under=underlay_treads(np.full((40,60),.20),np.array([-.2,-.75]),.025,
                                  [{"polygon":poly,"vertices":vertices}])
        self.assertTrue(mask.any())
        self.assertLess(float(under[mask].max()),.12)
        self.assertGreater(float(under[~mask].max()),.19)
        self.assertTrue(in_polygon(np.array([[.1,0.]]),poly).all())

    def test_rail_radius_and_solid(self):
        rng=np.random.default_rng(3)
        axis=np.array([1.,0.,0.])
        angles=np.arange(720)*np.pi/360
        points=np.c_[rng.uniform(-.3,.3,len(angles)),.012+.026*np.cos(angles),
                     -.008+.026*np.sin(angles)]
        center,radius,stats=cylinder_section(points,np.zeros(3),axis,.025,.10)
        self.assertTrue(stats["measured"])
        np.testing.assert_allclose(center,[0.,.012,-.008],atol=.001)
        self.assertAlmostEqual(radius,.026,places=3)
        xyz,vertices,faces=tapered_solid(np.array([0.,0.,0.]),np.array([.02,0.,.50]),.14,.1,10)
        self.assertEqual(len(vertices),20)
        self.assertEqual(len(faces),36)
        self.assertGreater(xyz[2],.2)

    def test_partial_rail_arc_with_background_points(self):
        rng = np.random.default_rng(915)
        angle = rng.uniform(.2,2.85,260)
        origin = np.array([.011,-.009])
        arc = origin + .030*np.c_[np.cos(angle),np.sin(angle)]
        arc += rng.normal(0,.001,arc.shape)
        xy = np.vstack((arc,rng.uniform(-.085,.085,(110,2))))
        result = estimate_circle(xy,rng)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["radius_m"],.030,delta=.004)
        np.testing.assert_allclose(result["center"],origin,atol=.005)

    def test_soft_points_do_not_generate_solid_mesh(self):
        yy, zz = np.meshgrid(np.arange(.12,.40,.012),np.arange(.14,.32,.012))
        points = np.c_[np.full(yy.size,.50),yy.ravel(),zz.ravel()]
        route = np.c_[np.arange(0,1,.05),np.zeros(20)]
        layers = np.zeros((20,20,7))
        ground = np.zeros((20,20))
        cfg = {"max_height_m":1.0,"tile_m":.34,"minimum_thickness_m":.022}
        with patch("nav4.matrix.objects.write_obj"):
            layers[...,6] = 2
            layers[4,:,6] = 1
            soft = solid_shapes(points,route,np.zeros(2),layers,.1,ground,np.zeros(2),.1,[],[],cfg,Path("."))
            self.assertEqual(soft,[])
            layers[...,6] = 1
            hard = solid_shapes(points,route,np.zeros(2),layers,.1,ground,np.zeros(2),.1,[],[],cfg,Path("."))
            self.assertGreater(len(hard),0)

    def test_soft_overlap_has_no_contact(self):
        root=ET.Element("mujoco")
        world=xml(root,"worldbody")
        xml(world,"geom",type="plane",size=[1,1,.1],contype=1,conaffinity=0)
        xml(world,"geom",name="soft",type="sphere",pos=[0,0,.15],size=.20,contype=0,conaffinity=0)
        body=xml(world,"body",pos=[0,0,.055])
        xml(body,"freejoint")
        xml(body,"geom",name="proxy",type="box",size=[.12,.1,.055],contype=2,conaffinity=1)
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
        d=mujoco.MjData(m)
        for _ in range(10):
            mujoco.mj_step(m,d)
        self.assertGreater(d.ncon,0)
        soft=m.geom("soft").id
        self.assertTrue(all(soft not in (d.contact[i].geom1,d.contact[i].geom2) for i in range(d.ncon)))


if __name__ == "__main__":
    unittest.main()
