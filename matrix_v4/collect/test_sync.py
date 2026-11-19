import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sync import pull


class SyncTests(unittest.TestCase):
    def test_resume_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            raw=b'abcdef'
            expected=hashlib.sha256(raw).hexdigest()
            manifest=f'{expected}  ep.npz\n'
            (dest/'ep.partial').write_bytes(raw[:2])
            def send(args,stdout,check):
                self.assertTrue(check)
                self.assertIn('tail -c +3',args[-1])
                stdout.write(raw[2:])
            with patch.object(subprocess,'check_output',return_value=manifest),patch.object(subprocess,'run',side_effect=send) as transfer:
                pull(dest,['ssh','vm'])
                pull(dest,['ssh','vm'])
                self.assertEqual(transfer.call_count,1)
            self.assertEqual((dest/'ep.npz').read_bytes(),raw)

    def test_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            with patch.object(subprocess,'check_output',return_value='0'*64+'  ep.npz\n'),patch.object(subprocess,'run'):
                with self.assertRaises(IOError):
                    pull(dest,['ssh','vm'])
            self.assertFalse((dest/'ep.sha256').exists())
            self.assertFalse((dest/'ep.npz').exists())
