"""P1-only artifacts, provenance, and own-PID GPU guard. No training."""
from pathlib import Path
import os, json, time, threading, subprocess, hashlib
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / 'fullgeo'

def dump(path, obj):
    def convert(x):
        if isinstance(x, dict): return {k: convert(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, np.generic): return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(convert(obj), ensure_ascii=False, indent=2, allow_nan=False))

class Monitor:
    def __init__(self, tag):
        self.tag=tag; self.pid=os.getpid(); self.rows=[]; self.stop=False
        self.start=time.perf_counter(); self.peak=0
        self.thread=threading.Thread(target=self.loop, daemon=True); self.thread.start()
    def loop(self):
        while not self.stop:
            try:
                txt=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
                mem=max([int(s.split(',')[1]) for s in txt.splitlines() if s.split(',')[0].strip()==str(self.pid)],default=0)
                self.peak=max(self.peak,mem); self.rows.append([time.perf_counter()-self.start,mem])
                if mem>7400:
                    self.save(); os._exit(78)  # only this process
            except Exception as exc: print('MONITOR',repr(exc),flush=True)
            time.sleep(.2)
    def save(self):
        dump(ROOT/f'metrics/r8_p1_gpu_{self.tag}.json',dict(pid=self.pid,peak_mib=self.peak,sampling_s=.2,guard_mib=7400,rows=self.rows))
    def close(self):
        self.stop=True; self.thread.join(2); self.save()

def isolate_warp():
    import warp as wp
    wp.config.kernel_cache_dir=str(GEO/'r8_p1_cache/warp')
    from warp._src.build import init_kernel_cache
    init_kernel_cache(wp.config.kernel_cache_dir)

def provenance(after=False):
    path=GEO/'r8_p1_provenance_before.json'
    if not after:
        files=[]
        for directory in ['nav3','configs','scripts','reports','metrics','maps_lwy']:
            files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in str(p) and 'r8_p1_' not in p.name and not p.name.startswith('R8_P1'))
        files.extend(p for p in GEO.glob('r8_*') if p.is_file() and not p.name.startswith('r8_p1_'))
        # Hash existing source/config/P0 artifacts; large geometry arrays also hashed.
        result={}
        for p in files:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
            result[str(p.relative_to(ROOT))]={'sha256':h.hexdigest(),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        dump(path,result)
    else:
        records=json.loads(path.read_text()); changed=[]
        for rel,rec in records.items():
            p=ROOT/rel
            if not p.exists(): changed.append(rel); continue
            # All protected files are rehashed, not merely stat-checked.
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
            if h.hexdigest()!=rec['sha256']: changed.append(rel)
        dump(ROOT/'metrics/r8_p1_integrity.json',{'protected_files':len(records),'changed':changed,'all_rehashed':True})
        assert not changed, changed

if __name__=='__main__':
    import sys
    provenance('--after' in sys.argv)
