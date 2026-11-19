#!/bin/bash
set -eu
base=${MATRIX_COLLECTION_DIR:?}
name=${MV4_SESSION:?}
sock="$base/logs/$name.sock"
key="$HOME/.ssh/colab_mv4_ed25519"
cli="$HOME/.local/bin/colab"
if ! ssh -S "$sock" -O check root@colab-runtime 2>/dev/null; then
 "$cli" sessions | grep -F "[$name]" >/dev/null
 ssh -MNf -S "$sock" -i "$key" -o ControlPersist=15m -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$base/logs/$name.known" -o "ProxyCommand=$cli ssh -s $name -i $key --proxy-mode" root@colab-runtime
fi
exec ssh -T -S "$sock" -o ProxyCommand=false root@colab-runtime "$@"
