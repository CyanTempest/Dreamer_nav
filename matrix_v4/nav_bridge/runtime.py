import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path('/content/eval')
BRIDGE=Path('/content/nav_bridge')
CATALOG=Path('/content/val/stairs_standing.json')

def set_pose(name, pose):
    data=json.loads(CATALOG.read_text())
    data[name]=pose
    CATALOG.write_text(json.dumps(data))

class Instance:
    def __init__(self, out, placement):
        self.out=out
        out.mkdir(parents=True,exist_ok=True)
        self.file=(out/'console.log').open('w')
        self.proc=subprocess.Popen(['bash',str(BRIDGE/'launch.sh'),str(out),placement],
                                   stdout=self.file,stderr=subprocess.STDOUT,start_new_session=True)
    def ready(self):
        start=time.monotonic()
        with (self.out/'console.log').open(errors='replace') as f:
            while time.monotonic()-start<200:
                if self.proc.poll() is not None:
                    raise RuntimeError('MATRiX launch exited')
                if 'native runtime ready' in f.read(): return
                time.sleep(.15)
        raise TimeoutError('MATRiX native runtime')
    def close(self):
        try:
            os.killpg(self.proc.pid,signal.SIGTERM)
            time.sleep(.8)
            os.killpg(self.proc.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        self.file.close()

if __name__=='__main__':
    from calibrate import measure
    out=ROOT/'calibration'
    set_pose('cal',json.loads(Path('/content/cal_pose.json').read_text()))
    inst=Instance(out,'cal')
    try:
        inst.ready()
        measure(out)
    finally:
        inst.close()
