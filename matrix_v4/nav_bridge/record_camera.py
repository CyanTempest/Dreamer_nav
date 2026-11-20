"""Presentation-only subscriber. The actor never receives this topic."""
from pathlib import Path
import signal
import sys
import time
import zenoh
from sensors import Reader

out=Path(sys.argv[1])
cfg=zenoh.Config()
cfg.insert_json5('mode','"client"')
cfg.insert_json5('connect/endpoints','["tcp/127.0.0.1:7447"]')
while True:
    try:
        session=zenoh.open(cfg)
        break
    except Exception:
        time.sleep(.25)

def receive(sample):
    raw=sample.payload.to_bytes()
    reader=Reader(raw)
    stamp=reader.header()
    kind=reader.string()
    size,=reader.take('I',4)
    data=raw[reader.pos:reader.pos+size]
    if len(data)!=size or not data.startswith((b'\xff\xd8',b'\x89PNG')): return
    (out/f'follow-{time.time_ns()}.jpg').write_bytes(data)

sub=session.declare_subscriber('rt/nav_follow/image/compressed',receive)
signal.pause()
