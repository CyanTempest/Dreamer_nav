#!/bin/bash
set -euo pipefail
host=$(cd "$(dirname "$0")" && pwd)
cd "${MATRIX_COLLECTION_DIR:?}"
s=${MV4_SESSION:?}
cli="$HOME/.local/bin/colab"
"$cli" usage | tee "logs/$s.before_stop"
if [[ -f logs/"sync_${s#mv4col-}".pid ]]; then
 kill "$(cat logs/"sync_${s#mv4col-}".pid)" 2>/dev/null || true
fi
if ssh -S "$PWD/logs/$s.sock" -O check root@colab-runtime 2>/dev/null; then
 ssh -S "$PWD/logs/$s.sock" -O exit root@colab-runtime || true
fi
"$cli" stop -s "$s" |& tee "logs/$s.stop"
for i in {1..12}; do
 "$cli" sessions | tee "logs/$s.sessions_closed"
 if grep -Fq 'No active sessions' "logs/$s.sessions_closed"; then
   "$cli" usage | tee "logs/$s.after_stop"
   grep -Fq 'Active assignments: 0' "logs/$s.after_stop" && exit 0
 fi
 sleep 5
done
exit 1
