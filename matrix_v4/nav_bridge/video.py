"""Encode real UE screenshots at 5 Hz against Odom decision receipt time."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def nearest(a,t):
    k=min(np.searchsorted(a,t),len(a)-1)
    return k-1 if k and abs(a[k-1]-t)<abs(a[k]-t) else k

def preview(bev):
    known=bev[...,3]<128
    h=bev[...,0]/255.
    o=bev[...,1]/255.
    rgb=np.stack([.16+.55*h+.65*o,.17+.47*h+.31*bev[...,2]/255,.19+.5*h],-1)
    rgb[~known]=[.1,.13,.18]
    return Image.fromarray(np.uint8(np.clip(rgb[::-1,::-1],0,1)*255)).resize((144,144),Image.Resampling.NEAREST)

def encode(out):
    files=sorted([*(out/'shots').glob('*.jpg'),*(out/'shots').rglob('*.png')],key=lambda p:p.stat().st_mtime)
    chunks=sorted(out.glob('steps-*.npz'))
    if not files or not chunks: raise ValueError(f'{out}: screenshots={len(files)}, chunks={len(chunks)}')
    keys=['epoch','station','vel','assist','bev']
    data={k:[] for k in keys}
    for p in chunks:
        with np.load(p) as z:
            for k in keys: data[k].append(z[k])
    data={k:np.concatenate(v) for k,v in data.items()}
    times=data['epoch']; ft=np.array([p.stat().st_mtime for p in files])
    end=json.loads((out/'end.json').read_text())
    reason=end['reason']
    assisted=reason not in ('finish','budget')
    targets=np.arange(times[0]-.45,times[-1]+.201,.2)
    from matplotlib import get_data_path
    font=str(Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf')
    large=ImageFont.truetype(font,27)
    small=ImageFont.truetype(font,16)
    tmp=out/'clip.partial.mp4'
    proc=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r','5','-i','-','-an','-c:v','libx264','-threads','2','-preset','veryfast',
        '-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
    last=None
    indices=[]
    for t in targets:
        j=nearest(ft,t); i=nearest(times,t)
        indices.append(j)
        with Image.open(files[j]) as im:
            if im.size!=(1280,720): raise ValueError(f'Unexpected render size {im.size}')
            frame=im.convert('RGB')
        d=ImageDraw.Draw(frame,'RGBA')
        d.rounded_rectangle((20,19,775,73),8,fill=(10,20,27,220))
        speed=np.linalg.norm(data['vel'][i,:2])
        d.text((36,29),f"M1  /  STAIR     {data['station'][i]:7.2f} m      {speed:.2f} m/s      Assists  {end['assists']}",font=large,fill='white')
        frame.paste(preview(data['bev'][i]),(1115,23))
        d.rectangle((1115,168,1258,192),fill=(10,20,27,220))
        d.text((1124,169),'LiDAR BEV',font=small,fill='white')
        if end['assists'] and t<times[0]+2:
            d.rounded_rectangle((23,603,630,660),8,fill=(15,68,78,230))
            d.text((38,615),f"Restart after assist {end['assists']}   |   {end['start']:.2f} m",font=large,fill='white')
        proc.stdin.write(frame.tobytes())
        last=frame
    if assisted:
        d=ImageDraw.Draw(last,'RGBA')
        d.rounded_rectangle((21,522,930,590),8,fill=(156,67,21,238))
        d.text((38,541),f"ASSIST {end['assists']+1}    {reason.replace('_',' ')}    at {end['s']:.2f} m",font=large,fill='white')
        for _ in range(6): proc.stdin.write(last.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('ffmpeg failed')
    tmp.replace(out/'clip.mp4')
    (out/'frame_stats.json').write_text(json.dumps(dict(screenshots=len(files),frames=len(targets),unique=len(set(indices)),
        active_frames_per_wall_second=float(((ft>=times[0])&(ft<=times[-1])).sum()/max(.2,times[-1]-times[0])),
        p95_matching_error_s=float(np.percentile(abs(ft[indices]-targets),95))),indent=2))

if __name__=='__main__': encode(Path(sys.argv[1]))
