#!/bin/bash
# VM-only cold reset. Private net, mount and IPC for each slot, including its collector.
set -euo pipefail
[[ $(id -u) == 0 && -d /content/matrix-v4 && -f /content/val/CustomWorldBundle.pak ]]
r=/content/matrix-v4
c=/content/col
run=$1
slot=$2
seed=$3
end=$4
n=$5
gpu=$6
scene=$(printf 'window_%02d' "$slot")
i=$c/instances/$slot
py=$r/venvs/bridge/bin/python
mkdir -p "$run" "$i"
if [[ ! -e "$i/UeSim" ]]; then
 cp -al "$r/MATRiX_v1.1.2/." "$i/"
 rm -rf "$i/UeSim/Saved"
 install -d -o matrixv4 -g matrixv4 "$i/UeSim/Saved/DLCs"
 chown matrixv4:matrixv4 "$i/UeSim/Saved"
 ln /content/val/CustomWorldBundle.pak "$i/UeSim/Saved/DLCs/"
fi
js=$i/UeSim/Content/JavaScript
rm -f "$js"/col_cmd_*.js "$js/QuickStart.js"
cp "$c/collect/QuickStart.js" "$js/QuickStart.js"
"$py" "$c/collect/prepare.py" "$i" "$c/inputs/$scene.json" "$seed" "$run/spawn.json"
chown matrixv4:matrixv4 "$run" "$i/External/zgws"
shm=/dev/shm/$(basename "$run")
install -d -m 1777 "$shm"
unshare -Urnmi --propagation private bash -c 'ip link set lo up && mount --bind "$1" /dev/shm && exec sleep infinity' _ "$shm" &
holder=$!
ue=0
cleanup(){
 if (( ue )); then
   kill -TERM -- "-$ue" 2>/dev/null || true
   sleep 1
   kill -KILL -- "-$ue" 2>/dev/null || true
 fi
 kill "$holder" 2>/dev/null || true
 wait "$holder" 2>/dev/null || true
 rm -rf "$shm"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
enter=(nsenter -t "$holder" -n -m -i)
for k in {1..80}; do
 "${enter[@]}" ip -br addr show lo 2>/dev/null | grep -Fq '127.0.0.1' && [[ $(readlink /proc/"$holder"/ns/net) != $(readlink /proc/self/ns/net) ]] && break
 sleep .1
done
[[ $(readlink /proc/"$holder"/ns/net) != $(readlink /proc/self/ns/net) ]]
# Concurrent cold PSO initialization can trip UE's 60 s render fence on 2 vCPUs.
# Serialize only the initial map/render warm-up; all physics and driving run concurrently.
exec 8>"$c/renderer-start.lock"
while ! flock -n 8; do
 (( $(date +%s) < end )) || exit 4
 sleep .5
done
ulimit -c 0
setsid "${enter[@]}" runuser -u matrixv4 -- env -u DISPLAY bash -c '
 source /content/matrix-v4/graphics.env
 export SDL_AUDIODRIVER=dummy
 cd "$1"
 exec ./UeSim.sh /Game/Maps/CustomWorld -RenderOffScreen -unattended -nosound -NoVSync -ResX=64 -ResY=64 -windowed -stdout -FullStdOutLogOutput "-abslog=$2/ue.log" "-port=$((7777+$3))" "-JsEnvDebugPort=$((9201+$3))" "-ini:Engine:[SystemSettings]:g.TimeoutForBlockOnRenderFence=240000" "-ExecCmds=g.TimeoutForBlockOnRenderFence 240000,r.ShadowQuality 0,r.PostProcessAAQuality 0,r.DefaultFeature.AntiAliasing 0,r.BloomQuality 0,r.MotionBlurQuality 0,r.SSR.Quality 0,r.ViewDistanceScale 0.3,sg.PostProcessQuality 0"
 ' _ "$i" "$run" "$slot" 8>&- >"$run/console.log" 2>&1 &
ue=$!
for k in {1..360}; do
 grep -aq 'spawn settings source=config' "$run/console.log" && break
 kill -0 "$ue" || exit 3
 (( $(date +%s) < end )) || exit 4
 sleep .5
done
flock -u 8
exec 8>&-
for k in {1..480}; do
 grep -aq 'native runtime ready' "$run/console.log" && break
 kill -0 "$ue" || exit 3
 (( $(date +%s) < end )) || exit 4
 sleep .5
done
grep -aq 'native runtime ready' "$run/console.log"
export PYTHONPATH="$r/MATRiX_v1.1.2/Tools"
"${enter[@]}" "$py" -u "$c/collect/worker.py" --run "$run" --js "$js" --output "$c/data" --config "$c/collect/config.json" --end "$end" --n "$n" --gpu "$gpu" &
worker=$!
while kill -0 "$worker" 2>/dev/null; do
 if ! kill -0 "$ue" 2>/dev/null; then
   # The worker seals the open episode on odometry timeout.
   echo 'UE exited during active episode'
 fi
 sleep 1
done
wait "$worker"
