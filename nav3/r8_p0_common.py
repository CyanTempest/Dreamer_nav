"""R8 paths, provenance and bounded GPU monitoring."""
from pathlib import Path
import json,time,os,threading,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT.parents[1]/'ATEC_Env'
GEO=ROOT/'fullgeo'
CLASSES={0:'unclassified',1:'road',2:'stairs',3:'rail',4:'bank_soil',5:'rock',6:'root',7:'trunk_wood',8:'tree_leaves',9:'shrub_leaves',10:'grass',11:'fern',12:'wall',13:'sign',14:'shrub_wood'}
SOFT=[8,9,10,11]
HARD=[3,5,7,12,13,14]
def dump(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
def json_ready(x):
 if isinstance(x,dict):return {k:json_ready(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [json_ready(v) for v in x]
 if isinstance(x,np.ndarray):return json_ready(x.tolist())
 if isinstance(x,np.generic):return json_ready(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
class GPUMonitor:
 """Sample ONLY the current PID allocation; reject before new large allocations."""
 def __init__(self,tag):
  self.tag=tag;self.rows=[];self.stop=False;self.pid=os.getpid();self.peak=0;self.started=time.time()
  self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def loop(self):
  while not self.stop:
   try:
    txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    vals=[int(line.split(',')[1].strip()) for line in txt.splitlines() if line.split(',')[0].strip()==str(self.pid)]
    m=max(vals,default=0);self.peak=max(self.peak,m);self.rows.append([time.time()-self.started,m])
    if m>7800:
     print('MEMORY GUARD exceeds 7800 MiB; exiting own process',flush=True);self.save();os._exit(78)
   except Exception as e:print('monitor:',repr(e),flush=True)
   time.sleep(.25)
 def save(self):dump(ROOT/f'metrics/r8_p0_gpu_{self.tag}.json',dict(pid=self.pid,peak_mib=self.peak,sampling_seconds=.25,limit_mib=8192,rows=self.rows))
 def close(self):self.stop=True;self.thread.join(timeout=2);self.save()
def rot(yaw):
 c,s=np.cos(yaw),np.sin(yaw)
 return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)
