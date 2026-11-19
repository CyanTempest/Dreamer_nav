"""Lossless byte shuffle + libzstd for the exact CDR member (including its header and padding)."""
import ctypes as ct
import ctypes.util
import numpy as np

lib=ct.CDLL(ctypes.util.find_library('zstd') or 'libzstd.so.1')
def bind(name,result,*arguments):
    fn=getattr(lib,name)
    fn.restype=result
    fn.argtypes=arguments
    return fn

size=ct.c_size_t
ptr=ct.c_void_p
bound=bind('ZSTD_compressBound',size,size)
compress=bind('ZSTD_compress',size,ptr,size,ptr,size,ct.c_int)
decompress=bind('ZSTD_decompress',size,ptr,size,ptr,size)
content_size=bind('ZSTD_getFrameContentSize',ct.c_ulonglong,ptr,size)
is_error=bind('ZSTD_isError',ct.c_uint,size)


def pack_wire(raw):
    # Fixed shuffle period is reversible for any wire layout; row padding only affects ratio.
    n=len(raw)//26*26
    shuffled=np.frombuffer(raw,np.uint8,count=n).reshape(-1,26).T.tobytes()+raw[n:]
    dst=ct.create_string_buffer(bound(len(raw)))
    written=compress(dst,len(dst),shuffled,len(shuffled),1)
    if is_error(written):
        raise ValueError('Zstd compression failed')
    return dst.raw[:written]


def unpack_wire(packed):
    data=bytes(packed)
    length=content_size(data,len(data))
    if length>32<<20:
        raise ValueError('Unexpected Airy CDR size')
    dst=ct.create_string_buffer(length)
    if decompress(dst,length,data,len(data))!=length:
        raise ValueError('Damaged Zstd frame')
    n=length//26*26
    raw=dst.raw
    return np.frombuffer(raw,np.uint8,count=n).reshape(26,-1).T.tobytes()+raw[n:]
