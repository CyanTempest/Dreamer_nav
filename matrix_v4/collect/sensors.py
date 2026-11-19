"""Wire decoding. All array views respect XCDR1 encapsulation and row padding."""
from dataclasses import dataclass
import struct
import numpy as np


class Reader:
    def __init__(self, payload):
        self.b = payload
        if payload[:2] not in (b'\x00\x01', b'\x00\x00'):
            raise ValueError("Expected plain XCDR1")
        self.order = '<' if payload[1] else '>'
        self.pos = 4

    def take(self, fmt, align):
        self.pos += -(self.pos-4) % align
        layout = struct.Struct(self.order+fmt)
        values = layout.unpack_from(self.b, self.pos)
        self.pos += layout.size
        return values

    def string(self):
        size, = self.take('I', 4)
        end = self.pos + size
        if size < 1 or end > len(self.b) or self.b[end-1]:
            raise ValueError('Invalid CDR string')
        value = self.b[self.pos:end-1].decode()
        self.pos = end
        return value

    def header(self):
        sec, ns = self.take('iI', 4)
        self.string()
        if ns >= 10**9:
            raise ValueError('Invalid timestamp')
        return sec + ns/1e9


def motion(payload, odom):
    r = Reader(payload)
    stamp = r.header()
    if odom:
        r.string()
        pose = r.take('7d', 8)
        r.take('36d', 8)
        return stamp, pose, r.take('6d', 8)
    attitude = r.take('4d', 8)
    r.take('9d', 8)
    return stamp, attitude, r.take('3d', 8)


@dataclass
class Scan:
    stamp: float
    end: float
    received: float
    shape: tuple
    finite: int
    distance: np.ndarray
    intensity: np.ndarray
    raw: bytes


def decode(payload, received):
    r = Reader(payload)
    stamp = r.header()
    h, w, n = r.take('III', 4)
    if w != 96 or not 0 < h <= 900 or n > 32:
        raise ValueError(f'Unexpected Airy layout {h} x {w}, {n} fields')
    fields = {}
    for _ in range(n):
        name = r.string()
        off, = r.take('I', 4)
        kind, = r.take('B', 1)
        count, = r.take('I', 4)
        fields[name] = off, kind, count
    big, = r.take('B', 1)
    point, row, size = r.take('III', 4)
    if row < w*point or size != row*h or r.pos+size+1 > len(payload):
        raise ValueError('Invalid PointCloud2 strides')
    def field(name, kind, dtype):
        off, actual, count = fields[name]
        dtype = np.dtype(('>' if big else '<')+dtype)
        if actual != kind or count != 1 or off+dtype.itemsize > point:
            raise ValueError(f'Invalid {name} field')
        return np.ndarray((h,w), dtype, buffer=payload, offset=r.pos+off, strides=(row,point))
    xyz = [field(k,7,'f4') for k in ('x','y','z')]
    mm = np.rint(1000*np.sqrt(sum(a.astype('f8')**2 for a in xyz)))
    good = np.isfinite(mm)
    rings = field('ring',4,'u2')
    if not np.all(rings == np.arange(w)) and not np.all(np.sort(rings,axis=1) == np.arange(w)):
        raise ValueError('Duplicate/missing rings within firing column')
    columns = np.arange(h)[:,None]
    distance = np.zeros((96,900),np.uint16)
    # 0=no return/padding, 65535=overflow. Exact floats remain in raw CDR.
    distance[rings,columns] = np.where(good, np.clip(mm,1,65535),0).astype('u2')
    intensity = np.zeros((96,900), np.uint8)
    iv = field('intensity',7,'f4')
    intensity[rings,columns] = np.clip(np.nan_to_num(iv),0,255).astype(np.uint8)
    ts = field('timestamp',8,'f8')
    end = float(np.nanmax(ts))
    if not stamp-.001 <= end <= stamp+.101:
        raise ValueError('Scan point clock outside one Airy revolution')
    return Scan(stamp,end,received,(h,w),int(good.sum()),distance,intensity,payload)


def scan_age(scan, mono, epoch_offset):
    if scan is None or scan.received > mono:
        return float('inf')
    # Capture age includes queue/transport delay. Reject future scan ends.
    age = mono + epoch_offset - scan.end
    return age if age >= 0 else float('inf')


def sim_at_wall(history, wall):
    """Piecewise interpolation on received Odom anchors, confined to this episode."""
    if len(history)<2 or not history[0][0] <= wall <= history[-1][0]:
        return float('nan')
    t, s = zip(*history)
    i = np.searchsorted(t,wall,side='right')
    if i == len(t):
        return s[-1]
    if t[i]-t[i-1] > .20 or s[i] < s[i-1]:
        return float('nan')
    return s[i-1] + (s[i]-s[i-1])*(wall-t[i-1])/(t[i]-t[i-1])
