#!/bin/bash
set -euo pipefail
host=$(cd "$(dirname "$0")" && pwd)
cd "${MATRIX_COLLECTION_DIR:?}"
card=$1
mkdir -p logs results data
[[ $card =~ ^(T4|L4|G4)$ ]]
export MV4_SESSION=mv4col-${card,,}
cli=$HOME/.local/bin/colab
test ! -e logs/budget_stop
"$cli" sessions | tee "logs/$MV4_SESSION.before_sessions"
grep -q 'No active sessions' "logs/$MV4_SESSION.before_sessions"
"$cli" usage | tee "logs/$MV4_SESSION.before_new"
grep -q 'Active assignments: 0' "logs/$MV4_SESSION.before_new"
[[ -s logs/usage_initial.log ]] || cp "logs/$MV4_SESSION.before_new" logs/usage_initial.log
"$cli" new -s "$MV4_SESSION" --gpu "$card" |& tee "logs/new_${card,,}.log"
"$cli" usage | tee "logs/$MV4_SESSION.usage"
bash "$host/bootstrap.sh"
tar -C ../exp_nav3_matrix_v4/matrix_v4 -cf - collect -C "$PWD" inputs | bash "$host/mux.sh" 'mkdir -p /content/col && tar -xf - -C /content/col'
for i in {1..150}; do
 if result=$(bash "$host/mux.sh" 'cat /content/val/install.exit' 2>/dev/null); then
   [[ $result == 0 ]]
   break
 fi
 test ! -e logs/budget_stop
 sleep 5
done
bash "$host/mux.sh" 'test "$(cat /content/val/install.exit)" = 0 && test -f /content/matrix-v4/graphics.env'
nohup ~/.local/share/uv/tools/google-colab-cli/bin/python "$host/http_pull.py" --watch </dev/null >"logs/sync_${card,,}.log" 2>&1 &
echo "$!" >"logs/sync_${card,,}.pid"
python3 "$host/run_tiers.py"
