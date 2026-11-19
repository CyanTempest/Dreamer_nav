"""Serialized CPU collision geometry tests on small synthetic scenes."""
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from nav4.matrix.collision import clear_stairs, hard_copy, heightfield
from nav4.matrix.stairs import prism
from nav4.matrix.compose import save_xml
from nav4.matrix.collision_spawn import set_reference


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.root = ET.fromstring("<mujoco><compiler angle='radian'/><asset/><worldbody/></mujoco>")
        self.world = self.root.find("worldbody")

    def compile(self):
        file = self.folder / "scene.xml"
        save_xml(self.root,file)
        m = mujoco.MjModel.from_xml_path(str(file))
        m.geom_rgba[:,3] = 1
        d = mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        return m,d

    def hit(self, md, xyz, direction=(0.,0.,-1.)):
        g = np.array([-1],np.int32)
        return mujoco.mj_ray(*md,np.asarray(xyz,dtype=float),np.array(direction),None,1,-1,g)

    def test_asymmetric_slope_and_shared_edge(self):
        x,y = np.indices((9,5))*.10
        z = -.7+.21*x-.38*y
        for k,xx in enumerate((slice(0,5),slice(4,9))):
            heightfield(self.root,self.folder,f"tile{k}",np.array([k*.4,0]),z[xx],.1)
        md = self.compile()
        for x in (.023,.39999,.40001,.754):
            for y in (.031,.349):
                actual = 2-self.hit(md,(x,y,2))
                self.assertAlmostEqual(actual,-.7+.21*x-.38*y,places=6)

    def test_flat_tread_vertical_riser_over_ground(self):
        heightfield(self.root,self.folder,"floor",np.zeros(2),np.zeros((11,9)),.1)
        step=ET.Element("geom",name="step",type="box",pos=".6 .4 .065",size=".2 .25 .065")
        self.world.append(hard_copy(step,cKDTree([[.6,.4]])))
        md=self.compile()
        for x,z in ((.399,0),(.401,.13),(.73,.13)):
            self.assertAlmostEqual(1-self.hit(md,(x,.4,1)),z,places=6)
        self.assertAlmostEqual(self.hit(md,(.25,.4,.07),(1.,0.,0.)),.15,places=6)

    def test_recess_exposes_measured_descending_riser(self):
        poly=np.array([[.5,.15],[.9,.15],[.9,.65],[.5,.65]])
        vertices,faces=prism(poly,np.zeros(4),-.18)
        shape=dict(polygon=poly,vertices=vertices,edge=poly[[0,3]],
                   before_plane_coeff=[0.,0.,.2],before_plane_anchor=[.5,.4])
        self.world.append(ET.Element("geom",name="stair_0000",type="box",group="2",
                                    pos=".7 .4 -.09",size=".2 .25 .09"))
        original=np.full((25,20),.16)
        floor=original.copy()
        clear_stairs([self.root],[self.folder],floor,np.zeros(2),original,[shape])
        heightfield(self.root,self.folder,"soil",np.zeros(2),floor,.05)
        md=self.compile()
        before=1-self.hit(md,(.495,.4,1.))
        after=1-self.hit(md,(.505,.4,1.))
        self.assertAlmostEqual(after-before,-.20,places=6)
        self.assertAlmostEqual(1-self.hit(md,(1.1,.4,1.)),.16,places=6)

    def test_post_radius_and_corridor_boundary(self):
        tree=cKDTree([[0.,0.],[.2,0.]])
        post=ET.Element("geom",name="post",type="capsule",fromto="0 4.02 .1 0 4.02 1",size=".04")
        copy=hard_copy(post,tree)
        self.assertIsNotNone(copy)
        self.assertEqual(copy.get("fromto"),post.get("fromto"))
        self.world.append(copy)
        md=self.compile()
        self.assertAlmostEqual(self.hit(md,(0.,3.8,.5),(0.,1.,0.)),.18,places=6)
        post.set("fromto","0 4.2 .1 0 4.2 1")
        self.assertIsNone(hard_copy(post,tree))

    def test_independent_bearing_layer_and_clear_underpass(self):
        for key,z in (("bottom",-.65),("deck",.1)):
            heightfield(self.root,self.folder,key,np.zeros(2),np.full((6,6),z),.1)
        md=self.compile()
        self.assertAlmostEqual(self.hit(md,(.25,.25,1)),.90,places=6)
        self.assertAlmostEqual(self.hit(md,(.25,.25,-.10)),.55,places=6)

    def test_home_and_default_reset_preserve_other_joint_positions(self):
        robot=ET.fromstring("""<mujoco><compiler angle="radian"/><worldbody>
        <body name="base_link"><freejoint/><geom size=".1"/>
        <body pos=".3 0 0"><joint axis="0 1 0"/><geom pos="0 0 -.2" size=".06"/>
        </body></body></worldbody><keyframe><key name="home"/></keyframe></mujoco>""")
        before=mujoco.MjModel.from_xml_string(ET.tostring(robot,encoding="unicode"))
        q=np.array([1.,2.,3.,.8,0,0,.6,.37])
        set_reference(robot,q)
        after=mujoco.MjModel.from_xml_string(ET.tostring(robot,encoding="unicode"))
        np.testing.assert_allclose(after.qpos0,q)
        for angle in (-.8,.37,.72):
            poses=[]
            for m in (before,after):
                d=mujoco.MjData(m)
                d.qpos[:]=q
                d.qpos[-1]=angle
                mujoco.mj_forward(m,d)
                poses.append(d.geom_xpos.copy())
            np.testing.assert_allclose(*poses,atol=1e-8)


if __name__=="__main__":
    unittest.main()
