"""Resident-memory telemetry and bounded-host-copy environment checkpoints."""
import gc
import json
import os
from pathlib import Path
import time
import uuid


def rss_bytes():
    # Current RSS, not ru_maxrss (which never decreases after a transient peak).
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def atomic_json(path, data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2))
    os.replace(temporary,path)


def release_host_memory():
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def save_device_tree(tree, directory):
    import jax
    import numpy as np
    directory = Path(directory) / ('env_' + uuid.uuid4().hex)
    directory.mkdir()
    leaves, structure = jax.tree.flatten(tree)
    paths = []
    for index, leaf in enumerate(leaves):
        path = directory / f'{index}.npy'
        np.save(path, np.asarray(leaf), allow_pickle=False)
        paths.append(str(path))
    return dict(structure=structure, paths=paths)


def load_device_tree(snapshot):
    import jax
    import jax.numpy as jnp
    import numpy as np
    leaves = []
    for path in snapshot['paths']:
        leaf = jnp.asarray(np.load(path))
        leaf.block_until_ready()
        leaves.append(leaf)
    return jax.tree.unflatten(snapshot['structure'], leaves)


class MemoryMonitor:
    def __init__(self, name, limit_gb=50):
        self.path = Path('metrics') / f'memory_{name}.jsonl'
        self.limit = limit_gb * 1e9
        self.last = 0
        self.last_trim = 0
        self.exceeded = False

    def check(self, stage, steps, replay=None, force=False):
        now = time.monotonic()
        if not force and now - self.last < 10:
            return self.exceeded
        self.last = now
        rss = rss_bytes()
        before_trim = None
        # Concurrent replay restoration allocates compressed entries in several
        # libc arenas. Later recurrent-state replacements can leave free pages
        # in those arenas that the main training thread does not promptly reuse.
        # Return such pages to the OS without changing any live replay values.
        if rss > 32e9 and now - self.last_trim >= 60:
            before_trim = rss
            release_host_memory()
            self.last_trim = now
            rss = rss_bytes()
        row = dict(time=time.time(), pid=os.getpid(), stage=stage,
                   steps=steps, rss_bytes=rss, rss_gb=rss / 1e9,
                   memory_policy='periodic_allocator_trim_v1')
        if before_trim is not None:
            row['rss_before_trim_bytes'] = before_trim
        if replay is not None:
            chunks = list(replay.chunks.values())
            row.update(replay_items=len(replay), replay_chunks=len(chunks),
                       replay_storage='lz4' if replay.compress else 'mmap',
                       replay_bytes=sum(c.nbytes for c in chunks),
                       replay_rows=sum(c.length for c in chunks),
                       replay_slots=sum(c.size for c in chunks if c.data),
                       workers=len(replay.current))
        with self.path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        if max(rss, before_trim or 0) > self.limit:
            if not self.exceeded:
                print(json.dumps(dict(event='rss_watchdog', **row,
                                      limit_gb=self.limit / 1e9,
                                      action='checkpoint_then_exit', exit_code=75)), flush=True)
            self.exceeded = True
        return self.exceeded
