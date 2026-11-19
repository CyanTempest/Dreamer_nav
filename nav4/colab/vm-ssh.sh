#!/usr/bin/env bash
# rsync invokes this as: vm-ssh.sh SESSION rsync --server ...
set -euo pipefail
home_dir=$(cd "$(dirname "$0")/../../.." && pwd)/nav4_colab
session=$1
shift
[[ $session =~ ^[a-zA-Z0-9_-]+$ && $# -gt 0 ]]
umask 077
mkdir -p "$home_dir/transport"
exec 9>"$home_dir/transport/$session.lock"
flock -w 240 9
identity=${COLAB_IDENTITY:-$home_dir/transport/id_ed25519}
if [[ ! -f "$identity" && -z "${COLAB_IDENTITY:-}" ]]; then
  ssh-keygen -q -t ed25519 -N '' -f "$identity"
fi
exec "${COLAB:-$HOME/.local/bin/colab}" ssh -s "$session" -i "$identity" -- "$@"

