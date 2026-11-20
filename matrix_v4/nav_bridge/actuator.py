import json
import time
from pathlib import Path
import numpy as np

class Actuator:
    def __init__(self):
        self.index = 0
        self.directory = Path('/content/nav_live/UeSim/Content/JavaScript')

    def send(self, forward, yaw, heading, stair=False):
        path = self.directory / f'nav_cmd_{self.index}.js'
        tmp = path.with_suffix('.tmp')
        packet = dict(f=float(np.clip(forward,-1,1)),w=float(np.clip(yaw,-1,1)),
                      heading=float(heading),stair=stair,until=int(time.time()*1000)+550)
        tmp.write_text('module.exports='+json.dumps(packet)+';')
        tmp.replace(path)
        self.index += 1
