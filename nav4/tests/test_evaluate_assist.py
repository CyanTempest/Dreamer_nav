"""CPU synthetic-map regression for forward assistance at the finish."""
import unittest
from unittest.mock import patch

import numpy as np

from nav4.evaluate import drive
from nav4.tests.helpers import env as make_env, reset


class EvaluateAssist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Force a real timeout after one stationary step on a supported, obstacle-free map.
        cls.env = make_env(('flat',), **{'task.stuck_seconds': .2, 'assist.end_clearance': .25})
        cls.final = float(cls.env.bank.gates[0, -1])

    def run_at(self, station, **kwargs):
        state, obs, _ = reset(self.env, station)
        return drive(self.env, state, obs, lambda _: np.array([[0., 0., -1.]], np.float32),
                     max_steps=3, **kwargs)[0][0]

    def test_finish_assist_counts_and_arrives(self):
        boundary = self.final-self.env.cfg['assist']['end_clearance']-self.env.cfg['assist']['advance']
        for station in (boundary, boundary+.5):
            with self.subTest(station=station), patch.object(self.env, 'restart_pose', wraps=self.env.restart_pose) as restart:
                row = self.run_at(station, assist=True)
                self.assertEqual(row['assists'], 1)
                self.assertTrue(row['arrived'])
                self.assertFalse(row['success'] or row['truncated'])
                self.assertEqual(row['final_station'], self.final)
                self.assertEqual(row['steps'], 1)
                self.assertEqual(row['failures'][0][0], 5)
                self.assertAlmostEqual(row['failures'][0][1], station, places=5)
                restart.assert_not_called()

    def test_far_assist_restarts_forward_as_before(self):
        with patch.object(self.env, 'restart_pose', wraps=self.env.restart_pose) as restart:
            row = self.run_at(5., assist=True, max_assists=1)
        self.assertEqual(row['assists'], 1)
        self.assertFalse(row['arrived'] or row['success'] or row['truncated'])
        self.assertEqual([f[:2] for f in row['failures']], [[5, 5.], [5, 6.]])
        self.assertEqual(row['final_station'], 6.)
        self.assertEqual(row['steps'], 2)
        self.assertEqual(restart.call_count, 1)
        np.testing.assert_array_equal(restart.call_args.args[1], [6.])

    def test_finish_without_available_assist_still_fails(self):
        for kwargs in ({'assist': False}, {'assist': True, 'max_assists': 0}):
            with self.subTest(kwargs=kwargs):
                row = self.run_at(self.final-.5, **kwargs)
                self.assertEqual(row['assists'], 0)
                self.assertFalse(row['arrived'] or row['success'] or row['truncated'])
                self.assertEqual(row['failures'][0][0], 5)


if __name__ == '__main__':
    unittest.main()
