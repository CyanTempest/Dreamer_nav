"""Check batched insertion against the existing row-at-a-time replay on CPU."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nav4 import policy  # sets the vendored embodied import path
from nav4.train_dreamer import BatchedReplayWriter
import embodied


class BatchedInsertion(unittest.TestCase):
    def test_sequences_eviction_updates_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            def create(name):
                return embodied.Replay(4, capacity=19, chunksize=3, online=True,
                                       seed=17, directory=str(Path(tmp)/name), save_wait=True)
            old, new = create('old'), create('new')
            writer = BatchedReplayWriter(new)
            for t in range(16):
                data = dict(reward=np.arange(4, dtype=np.float32) + 10*t,
                            is_first=np.full(4, t % 5 == 0), is_last=np.full(4, t % 5 == 4))
                for i in range(4):
                    old.add({k: v[i] for k, v in data.items()}, i)
                writer.add(data)
                self.assertEqual(len(old), len(new))
                if t >= 3:
                    a, b = old.sample(3), new.sample(3)
                    for k in data:
                        np.testing.assert_array_equal(a[k], b[k])
                    for rep, batch in ((old, a), (new, b)):
                        rep.update(dict(stepid=batch['stepid'], reward=batch['reward'] + 1))
                if t == 7:
                    for rep in (new, old):
                        state = rep.save(detailed=True)
                        restored = create(rep.directory.name)
                        restored.load(state)
                        if rep is new:
                            new = restored
                            writer = BatchedReplayWriter(new)
                        else:
                            old = restored
