#!/bin/bash
# Run from the research server. All calls to UeSim and GPU telemetry occur in the VM.
set -euo pipefail
host=$(cd "$(dirname "$0")" && pwd)
cd "${MATRIX_COLLECTION_DIR:?}"
mkdir -p logs results data
cli="$HOME/.local/bin/colab"
[[ ! -e logs/usage_initial.log ]] || { echo 'Use a fresh results directory for a new budget.' >&2; exit 1; }
"$cli" sessions | tee logs/sessions_initial.log
grep -q 'No active sessions' logs/sessions_initial.log
"$cli" usage | tee logs/usage_initial.log
grep -q 'Active assignments: 0' logs/usage_initial.log
nohup python3 "$host/guard.py" </dev/null >logs/guard22.out 2>&1 &
on_error(){
 rc=$?
 if (( rc )) && [[ -n ${MV4_SESSION:-} ]]; then
   "$cli" sessions >logs/sessions_on_error.log || true
   if grep -Fq "[$MV4_SESSION]" logs/sessions_on_error.log; then
     "$cli" stop -s "$MV4_SESSION" >logs/stop_on_error.log 2>&1 || true
   fi
   "$cli" sessions >logs/sessions_final.log || true
   "$cli" usage >logs/usage_final.log || true
   if grep -q 'No active sessions' logs/sessions_final.log && grep -q 'Active assignments: 0' logs/usage_final.log; then
     date -Is >logs/closed
   fi
 fi
 return "$rc"
}
trap on_error EXIT
for card in T4 L4 G4; do
 [[ ! -e logs/budget_stop ]] || break
 export MV4_SESSION=mv4col-${card,,}
 bash "$host/open_card.sh" "$card"
 [[ ! -e logs/budget_stop ]] || break
 ~/.local/share/uv/tools/google-colab-cli/bin/python "$host/http_pull.py"
 bash "$host/mux.sh" 'cd /content/col/data && cat -- *.sha256' >"logs/${card,,}.manifest"
 (cd data && sha256sum -c "../logs/${card,,}.manifest") >"logs/${card,,}.verified"
 bash "$host/close.sh"
done
"$cli" sessions | tee logs/sessions_final.log
"$cli" usage | tee logs/usage_final.log
grep -q 'No active sessions' logs/sessions_final.log
grep -q 'Active assignments: 0' logs/usage_final.log
date -Is >logs/closed
