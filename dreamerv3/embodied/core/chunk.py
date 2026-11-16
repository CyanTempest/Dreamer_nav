import os
import mmap
import tempfile
import sys
import traceback

import elements
import numpy as np
from . import compress as compression


class Chunk:

  __slots__ = ('time', 'uuid', 'succ', 'length', 'size', 'data', 'saved',
               'directory', 'backing', 'mapping', 'compress', 'packed',
               'specs', 'packed_bytes')

  def __init__(self, size=1024, directory=None, compress=False):
    self.time = elements.timestamp(millis=True)
    self.uuid = elements.UUID()
    self.succ = elements.UUID(0)
    # self.uuid = int(np.random.randint(1, 2 * 63))
    # self.succ = 0
    self.length = 0
    self.size = size
    self.data = None
    self.saved = False
    self.directory = directory
    self.backing = self.mapping = None
    assert not (directory and compress)
    self.compress = compress
    self.packed = {}
    self.specs = {}
    self.packed_bytes = 0

  def __del__(self, unlink=os.unlink):
    if self.backing:
      try:
        unlink(self.backing)
      except FileNotFoundError:
        pass

  def _allocate(self, example):
    if not self.directory:
      return {k: np.empty((self.size, *v.shape), v.dtype)
              for k, v in example.items()}
    sizes = {k: self.size * v.nbytes for k, v in example.items()}
    total = sum((n + 63) // 64 * 64 for n in sizes.values())
    fd, self.backing = tempfile.mkstemp(dir=str(self.directory), suffix='.buffer')
    os.ftruncate(fd, total)
    os.close(fd)
    self.mapping = np.memmap(self.backing, np.uint8, 'r+', shape=(total,))
    arrays, offset = {}, 0
    for key, value in example.items():
      arrays[key] = np.ndarray((self.size, *value.shape), value.dtype,
                              buffer=self.mapping, offset=offset)
      offset += (sizes[key] + 63) // 64 * 64
    return arrays

  def __repr__(self):
    return f'Chunk({self.filename})'

  def __lt__(self, other):
    return self.time < other.time

  @property
  def filename(self):
    succ = self.succ.uuid if isinstance(self.succ, type(self)) else self.succ
    return f'{self.time}-{str(self.uuid)}-{str(succ)}-{self.length}.npz'

  @property
  def nbytes(self):
    if not self.data:
      return 0
    return sum(x.nbytes for x in self.data.values()) + self.packed_bytes

  def _prepare(self, example):
    if self.compress:
      self.specs = {k: (v.shape, v.dtype) for k, v in example.items()
                    if v.nbytes >= 1024}
      self.packed = {k: [None] * self.size for k in self.specs}
    return {k: v for k, v in example.items() if k not in self.packed}

  def _pack(self, key, index, value):
    shape, dtype = self.specs[key]
    value = np.asarray(value, dtype)
    if value.shape != shape:
      value = np.broadcast_to(value, shape)
    data = compression.pack(value)
    previous = self.packed[key][index]
    self.packed_bytes += len(data) - (len(previous) if previous else 0)
    self.packed[key][index] = data

  def append(self, step):
    assert self.length < self.size
    if not self.data:
      example = step
      self.data = self._allocate(self._prepare(example))
    for key, value in step.items():
      if key in self.packed:
        self._pack(key, self.length, value)
      else:
        self.data[key][self.length] = value
    self.length += 1
    self.release_pages()
    # if self.length == self.size:
    #   [x.setflags(write=False) for x in self.data.values()]

  def update(self, index, length, mapping):
    assert 0 <= index <= self.length, (index, self.length)
    assert 0 <= index + length <= self.length, (index, length, self.length)
    for key, value in mapping.items():
      if key in self.packed:
        for offset, row in enumerate(value):
          self._pack(key, index + offset, row)
      else:
        self.data[key][index: index + length] = value
    self.release_pages()

  def release_pages(self):
    if self.mapping is not None:
      # MAP_SHARED keeps dirty bytes in the page cache/backing file. DONTNEED
      # releases this process's resident mappings without changing any values.
      self.mapping._mmap.madvise(mmap.MADV_DONTNEED)

  def slice(self, index, length, keys=None):
    assert 0 <= index and index + length <= self.length
    result = {}
    for key in keys or (*self.data, *self.packed):
      if key in self.packed:
        shape, dtype = self.specs[key]
        result[key] = np.empty((length, *shape), dtype)
        for offset in range(length):
          compression.unpack(self.packed[key][index + offset], result[key][offset])
      else:
        result[key] = self.data[key][index: index + length]
    return result

  @elements.timer.section('chunk_save')
  def save(self, directory, log=False):
    assert not self.saved
    filename = elements.Path(directory) / self.filename
    data = self.slice(0, self.length)
    temporary = str(filename) + '.tmp'
    with open(temporary, 'wb') as stream:
      np.savez_compressed(stream, **data)
    os.replace(temporary, str(filename))
    self.saved = True
    self.release_pages()
    log and print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename, error='raise', directory=None, compress=False):
    assert error in ('raise', 'none')
    time, uuid, succ, length = filename.stem.split('-')
    length = int(length)
    try:
      with elements.Path(filename).open('rb') as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    except Exception:
      tb = ''.join(traceback.format_exception(sys.exception()))
      print(f'Error loading chunk {filename}:\n{tb}')
      if error == 'raise':
        raise
      else:
        return None
    chunk = cls(length, directory, compress)
    chunk.time = time
    chunk.uuid = elements.UUID(uuid)
    chunk.succ = elements.UUID(succ)
    chunk.length = length
    if compress:
      chunk._prepare({k: v[0] for k, v in data.items()})
      chunk.data = {k: v for k, v in data.items() if k not in chunk.packed}
      for key in chunk.packed:
        for index, value in enumerate(data[key]):
          chunk._pack(key, index, value)
    elif directory:
      chunk.data = chunk._allocate({k: v[0] for k, v in data.items()})
      for key, value in data.items():
        chunk.data[key][:] = value
      chunk.release_pages()
    else:
      chunk.data = data
    chunk.saved = True
    return chunk


def release_pages(arrays):
  mappings = {}
  for array in arrays:
    while array is not None:
      if isinstance(array, np.memmap):
        mappings[id(array._mmap)] = array._mmap
        break
      array = getattr(array, 'base', None)
  for mapping in mappings.values():
    mapping.madvise(mmap.MADV_DONTNEED)
