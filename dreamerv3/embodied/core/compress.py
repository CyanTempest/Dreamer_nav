"""Lossless LZ4 blocks, using the host library without extra Python packages."""
import ctypes
import ctypes.util
import functools

import numpy as np


@functools.lru_cache(1)
def library():
  lib = ctypes.CDLL(ctypes.util.find_library('lz4') or 'liblz4.so.1')
  lib.LZ4_compressBound.argtypes = [ctypes.c_int]
  lib.LZ4_compressBound.restype = ctypes.c_int
  lib.LZ4_compress_default.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
  lib.LZ4_compress_default.restype = ctypes.c_int
  lib.LZ4_compress_HC.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  lib.LZ4_compress_HC.restype = ctypes.c_int
  lib.LZ4_decompress_safe.argtypes = [
      ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
  lib.LZ4_decompress_safe.restype = ctypes.c_int
  return lib


def pack(value):
  value = np.ascontiguousarray(value)
  lib = library()
  buffer = ctypes.create_string_buffer(lib.LZ4_compressBound(value.nbytes))
  args = (value.ctypes.data, ctypes.addressof(buffer), value.nbytes, len(buffer))
  # Images are written once, while float recurrent entries are updated often.
  # Level 3 saved 35% more image bytes at 140 MB/s in the nav3 workload probe.
  size = (lib.LZ4_compress_HC(*args, 3) if value.dtype == np.uint8 else
          lib.LZ4_compress_default(*args))
  if size <= 0:
    raise RuntimeError('LZ4 compression failed')
  return ctypes.string_at(buffer, size)


def unpack(data, output):
  size = library().LZ4_decompress_safe(
      data, output.ctypes.data, len(data), output.nbytes)
  if size != output.nbytes:
    raise RuntimeError(f'Invalid LZ4 block: {size} != {output.nbytes}')
