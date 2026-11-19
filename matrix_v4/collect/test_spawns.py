"""Run under the project's CPU-only mjmap environment with the geometry modules on PYTHONPATH."""
import unittest
import numpy as np
from spawns import tread_center


class Centers(unittest.TestCase):
    def test_skewed_tread(self):
        polygon=np.array([[0,-1],[1,-1],[1.4,1],[.4,1]])
        p,tread=tread_center(np.array([.82,.2]),np.array([1.,0.]),[polygon])
        self.assertTrue(tread)
        np.testing.assert_allclose(p,[.74,.2])

    def test_outside_convex_tread(self):
        p=np.array([.01,.9])
        actual,tread=tread_center(p,np.array([1.,0.]),[np.array([[0,-1],[1,-1],[1.4,1],[.4,1]])])
        self.assertFalse(tread)
        np.testing.assert_equal(actual,p)
