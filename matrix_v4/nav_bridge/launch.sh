#!/bin/bash
# Run once per physical placement on the temporary Colab VM.
set -euo pipefail
run=$1
station=$2
r=/content/matrix-v4
v=/content/val
i=/content/nav_live
model=UeSim/Content/model
js=UeSim/Content/JavaScript
mkdir -p "$run" "$i"
if [[ ! -d "$i/UeSim" ]]; then
    cp -al "$r/MATRiX_v1.1.2/." "$i/"
    rm -rf "$i/UeSim/Saved" "$i/External/zgws"
    install -d -o matrixv4 -g matrixv4 "$i/UeSim/Saved/DLCs" "$i/External/zgws"
    chown matrixv4:matrixv4 "$i/UeSim/Saved"
    ln "$v/CustomWorldBundle.pak" "$i/UeSim/Saved/DLCs/"
    cp -al "$i/$model/zgws/." "$i/External/zgws/"
    rm -f "$i/External/zgws/zgws.xml"
    cp "$i/$model/zgws/zgws_atec_full.xml" "$i/External/zgws/zgws.xml"
fi
cd "$i"
rm -f "$js"/nav_cmd_*.js "$js/QuickStart.js"
cp /content/nav_bridge/QuickStart.js "$js/QuickStart.js"
cp --remove-destination "$model/zgws/zgws_atec_robot_full.xml" External/zgws/zgws_atec_robot_full.xml
python3 - "$r/MATRiX_v1.1.2/$model/config/config.json" "$model/config/config.json" <<'PY'
import json, sys, os
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
d['robot'].update(mujoco_model='/content/nav_live/External/zgws/zgws.xml',zenoh_router='tcp/127.0.0.1:7447')
for sensor in d['robot']['sensors'].values():
    if sensor['sensor_type']=='airy':
        sensor['ignore_mounting_actor']=False
camera=json.loads((Path(sys.argv[1]).parent/'sensors/config_rgb_semantic.json').read_text())['rgb_camera']
camera.update(topic='/nav_follow/image/compressed',width=1280,height=720,frequency=10,ignore_mounting_actor=False,
              position=dict(x=-3.4,y=0,z=2.1),rotation=dict(roll=0,pitch=-27,yaw=0),frame_id='follow_camera')
d['robot']['sensors']['nav_follow']=camera
target=Path(sys.argv[2]); target.unlink(missing_ok=True); target.write_text(json.dumps(d))
PY
python3 "$v/fit_step.py" "$i" full "$station"
python3 - "$js/nav_heading.js" <<'PYHEAD'
import math,sys,xml.etree.ElementTree as E
from pathlib import Path
q=list(map(float,E.parse('External/zgws/zgws_atec_robot_full.xml').find("./keyframe/key[@name='home']").get('qpos').split()))[3:7]
w,x,y,z=q
Path(sys.argv[1]).write_text('module.exports={yaw:'+str(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))+'};')
PYHEAD
mkdir -p "$run/shots"
chown matrixv4:matrixv4 "$run" "$run/shots"
mkdir -p UeSim/Saved/Screenshots
ln -sfnT "$run/shots" UeSim/Saved/Screenshots/Linux
chmod 755 /content/eval
chmod a+r External/zgws/zgws.xml External/zgws/zgws_atec_robot_full.xml "$model/config/config.json" "$js/QuickStart.js"
cp "$model/config/config.json" "$run/config.json"
ulimit -c 0
PYTHONPATH=/content/collect /content/matrix-v4/venvs/bridge/bin/python /content/nav_bridge/record_camera.py "$run/shots" >"$run/camera.log" 2>&1 &
exec runuser -u matrixv4 -- env -u DISPLAY bash -c '
    source /content/matrix-v4/graphics.env
    export SDL_AUDIODRIVER=dummy
    exec ./UeSim.sh /Game/Maps/CustomWorld -RenderOffScreen -unattended -nosound -NoVSync -windowed -ResX=1280 -ResY=720 -stdout -FullStdOutLogOutput "-abslog=$1/ue.log"
' _ "$run"
