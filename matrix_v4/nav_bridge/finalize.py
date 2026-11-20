"""Seal the VM recording, retain controller evidence and check the exported movie."""
import json
from pathlib import Path
import subprocess
import sys
from video import encode

root=Path('/content/eval')
legs=sorted(p for p in root.glob('leg-*') if list(p.glob('steps-*.npz')))
if not legs: raise RuntimeError('no policy decisions recorded')
evidence=[]
for out in legs:
    if not (out/'clip.mp4').exists(): encode(out)
    lines=(out/'console.log').read_text(errors='replace').splitlines()
    ack=[x.strip() for x in lines if 'NAV_STAIR' in x]
    active=[x.strip() for x in lines if 'mode=Stair' in x and 'policy_cmd=' in x]
    evidence.append(dict(leg=out.name,stair_ack=ack[:1],stair_control_samples=len(active),
                         active_example=active[len(active)//2] if active else None,
                         timing=json.loads((out/'frame_stats.json').read_text())))
(root/'controller_checks.json').write_text(json.dumps(evidence,indent=2))
(root/'clips.txt').write_text(''.join("file '"+str(out/'clip.mp4')+"'\n" for out in legs))
subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(root/'clips.txt'),
                '-c','copy','-movflags','+faststart',str(root/'m1_nav4_A.mp4')],check=True)
info=subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames,duration','-of','json',str(root/'m1_nav4_A.mp4')],text=True)
(root/'video_probe.json').write_text(info)
print(info)
