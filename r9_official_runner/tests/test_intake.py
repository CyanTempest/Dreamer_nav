import io
import json
import hashlib
import unittest
import zipfile

import numpy as np
from audit_trajectory import path_statistics, rigid_fit, verify_manifest
from inspect_packages import safe_name


class IntakeTests(unittest.TestCase):
    def test_reject_traversal(self):
        for name in ['../a', '/a', 'x/../../a', 'C:\\a', 'x\\..\\a']:
            self.assertFalse(safe_name(name), name)
        self.assertTrue(safe_name('03_Trajectory/developer_data/poses.csv'))

    def test_reject_corrupt_manifest(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('p/data', b'changed')
            z.writestr('p/_package_info/MANIFEST.json', json.dumps({'files': [
                {'path': 'p/data', 'size_bytes': 7, 'sha256': hashlib.sha256(b'correct').hexdigest()}]}))
        with zipfile.ZipFile(buffer) as z:
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_manifest(z)

    def test_rigid_frame_with_heldout_points(self):
        rng = np.random.default_rng(812)
        source = rng.normal(size=(100, 3))
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = source @ rotation.T + [20, -3, 8]
        r, t = rigid_fit(source[:50], target[:50])
        np.testing.assert_allclose(source[50:] @ r.T + t, target[50:], atol=1e-10)

    def test_reject_timestamp_reversal(self):
        rows = np.zeros((3, 8)); rows[:, 0] = [0, 2, 1]; rows[:, 4] = 1
        with self.assertRaisesRegex(ValueError, 'Timestamps'):
            path_statistics(rows)


if __name__ == '__main__':
    unittest.main()
