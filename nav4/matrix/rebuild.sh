#!/usr/bin/env bash
set -euo pipefail
cd /home/hpf/wang-sm/ATEC_NAV_WSM/research/exp_nav3
export PYTHONPATH=../envs/mjmap/mujoco330
out=data/trail_a_mujoco/pilot_0_30
test -f "$out/geometry.json"
# Preserve the first pass and its geometry for diagnosis.
stamp=$(date +%Y%m%d_%H%M%S)
mkdir "$out/pass_$stamp"
for f in hard terrain stairs scene.xml; do
    mv -- "$out/$f" "$out/pass_$stamp/"
done
/usr/bin/time -v bash nav4/matrix/cpu.sh -m nav4.matrix.build --source removed > "$out/build.log" 2>&1
bash nav4/matrix/cpu.sh -m nav4.matrix.rail_refine --apply > "$out/rail_validate.log" 2>&1
/usr/bin/time -v bash nav4/matrix/cpu.sh -m nav4.matrix.verify > "$out/verify.log" 2>&1
