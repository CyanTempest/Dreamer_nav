"""Synthetic CPU regression checks for full-route reconstruction."""
import io
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nav4.matrix.compose import new_scene, xml
from nav4.matrix.full_ground import (
    LOCAL_SUPPORT_M, MAX_AXIS_GRADE, PITCH, apply_grades, bounded_grades,
    edge_statistics, fill_patch, grid_geometry,
)
from nav4.matrix.full_hard_audit import connected_rock
from nav4.matrix.full_objects import fit_stem, fit_trunk_section, foliage_item
from nav4.matrix.full_scene import CONFIG, place_soft


class FullMatrixGeometryTest(unittest.TestCase):
    def test_grade_limit_removes_spikes_and_preserves_plane(self):
        i, j = np.indices((85, 90))
        plane = (.4*i+.22*j)*PITCH
        bad = plane.copy()
        bad[41, 32] += 1.8
        bad[40:55, 61] -= .95
        smooth = bounded_grades(bad.astype(np.float32))
        for axis in (0,1):
            self.assertLess(np.abs(np.diff(smooth, axis=axis)).max(), MAX_AXIS_GRADE*PITCH+1e-5)
        v, f = grid_geometry(smooth, np.zeros(2))
        self.assertEqual(edge_statistics(v, f.reshape(-1,3))["near_vertical"], 0)
        np.testing.assert_allclose(smooth[:12], plane[:12], atol=1e-5)

    def test_anchors_keep_trail_from_dragging_toward_side_wall(self):
        z = np.zeros((60,80),np.float32)
        z[9:23,20:55] = 4.5
        active = np.ones(z.shape,bool)
        anchors = np.zeros(z.shape,bool)
        anchors[28:34] = True
        z[28:34] = .32
        grade = apply_grades(z,active,anchors)
        np.testing.assert_allclose(grade[anchors], z[anchors], atol=2e-5)
        self.assertLess(np.max(abs(np.diff(grade,axis=0))), .116)

    def test_hole_remains_marked_after_smoothing(self):
        i, j = np.indices((96, 90))
        prior = (.08*i+.05*j)*PITCH
        slab = prior.astype(np.float32)+.03
        slab[29:63,31:62] = np.nan
        slab[8,8] += .9
        filled, distance, rejects = fill_patch(slab, prior.astype(np.float32))
        self.assertEqual(rejects,1)
        self.assertGreater(distance[47,44],LOCAL_SUPPORT_M)
        self.assertLess(distance[29,43],LOCAL_SUPPORT_M)
        self.assertEqual(float(distance[75,52]),0)
        self.assertLess(np.max(abs(np.diff(filled,axis=0))),.07)
        self.assertTrue(np.isfinite(filled).all())

    def test_shared_vertices_match_at_partition(self):
        z = np.arange(91*95).reshape(91,95).astype(float)/1000
        a, _ = grid_geometry(z[:40], np.zeros(2))
        b, _ = grid_geometry(z[39:], np.array([39*PITCH,0.]))
        np.testing.assert_allclose(a[-95:], b[:95], atol=1e-12)

    def test_leaning_trunk_fits_independent_slices(self):
        rng = np.random.default_rng(472)
        z = rng.uniform(.11, 3.70, 7200)
        angle = rng.uniform(-2.2, .9, len(z))
        center = np.c_[.07*z, -.05*z]
        xy = center+.145*np.c_[np.cos(angle),np.sin(angle)]
        points = np.c_[xy, z]+rng.normal(0,.003,(len(z),3))
        xy = points[abs(points[:,2]-.50)<.075,:2]
        circle = fit_trunk_section(xy, np.zeros(2), rng)
        self.assertIsNotNone(circle)
        self.assertAlmostEqual(circle[1],.145,delta=.012)
        trunk = fit_stem(points, lambda xy: np.zeros(len(xy)), np.array([.02,0]), "synthetic", rng)
        self.assertGreater(len(trunk),7)
        self.assertGreater(trunk[-1]["geom"]["pos"][0]-trunk[0]["geom"]["pos"][0],.17)
        self.assertTrue(all(p["kind"]=="trunk" for p in trunk))

    def test_airborne_rock_fragment_is_rejected(self):
        levels = np.array([[x,y,z] for x in (-.021,.021) for y in (-.20,.20) for z in (0.,.42)])
        shapes = [{"world": levels + [0.,0.,h]} for h in (.08,.55,1.33)]
        supported, _ = connected_rock(shapes,lambda xy: np.zeros(len(xy)))
        np.testing.assert_array_equal(supported,[True,True,False])

    def test_batched_vegetation_is_visual_only(self):
        root = new_scene("soft geometry", CONFIG)
        world = root.find("worldbody")
        item = foliage_item("soft_test", np.array([[0.,0.,.16],[.21,0.,.13]]))
        place_soft(root, item)
        xml(world,"geom",type="plane",size=[2.,2.,.1],contype=1,conaffinity=0)
        body = xml(world,"body",pos=[0.,0.,.109])
        xml(body,"freejoint")
        xml(body,"geom",type="box",size=[.33,.22,.11],mass=30,contype=2,conaffinity=1)
        stream = io.StringIO()
        np.savetxt(stream,item["vertices"],fmt="v %.7f %.7f %.7f")
        np.savetxt(stream,item["triangles"]+1,fmt="f %d %d %d")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"),
                                              {item["file"]:stream.getvalue().encode()})
        soft = model.geom(item["name"]).id
        self.assertEqual((model.geom_contype[soft],model.geom_conaffinity[soft]),(0,0))
        d = mujoco.MjData(model)
        for _ in range(100):
            mujoco.mj_step(model,d)
            self.assertTrue(all(soft not in x.geom for x in d.contact))
        self.assertGreater(d.ncon,0)


if __name__ == "__main__":
    unittest.main()
