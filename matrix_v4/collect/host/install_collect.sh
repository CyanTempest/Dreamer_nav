#!/bin/bash
set -euo pipefail
cd /content
v=/content/val
curl -fsSL --retry 3 --max-time 240 http://39.96.31.125:18080/matrix/v1.1.2/Linux/CustomWorldBundle.pak -o "$v/CustomWorldBundle.pak" &
pid=$!
while [[ ! -s "$v/system.exit" ]]; do sleep 2; done
[[ $(cat "$v/system.exit") == 0 ]]
while [[ ! -s /content/matrix-v4-private.tar.gz.sha256 ]]; do sleep 2; done
MATRIX_SCENE=window_00 bash "$v/setup/colab_setup.sh" collect
wait "$pid"
echo 'e9de0f83b8eb9d9b454ce3149bbb1641eaa79081a0f2a935efadbe001574d7b6  /content/val/CustomWorldBundle.pak' | sha256sum -c -
