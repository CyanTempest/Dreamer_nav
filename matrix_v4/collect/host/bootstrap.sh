#!/bin/bash
set -euo pipefail
host=$(cd "$(dirname "$0")" && pwd)
cd "${MATRIX_COLLECTION_DIR:?}"
cli="$HOME/.local/bin/colab"
mux=(bash "$host/mux.sh")
"$cli" usage | tee "logs/$MV4_SESSION.usage"
"${mux[@]}" 'mkdir -p /content/val/setup; nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /content'
tar -C ../exp_nav3_matrix_v4/envs/matrix_v4 -cf - . | "${mux[@]}" 'tar -C /content/val/setup -xf -'
"${mux[@]}" 'nohup bash -c "bash /content/val/setup/colab_setup.sh system; echo \$? >/content/val/system.exit" </dev/null >/content/val/system.log 2>&1 &'
"${mux[@]}" 'cat >/content/val/install_vm.sh' <"$host/install_collect.sh"
"${mux[@]}" 'nohup bash -c "bash /content/val/install_vm.sh; echo \$? >/content/val/install.exit" </dev/null >/content/val/install.log 2>&1 &'
for f in ../matrix_v4_val/transfer/matrix-v4-private.tar.gz{,.sha256}; do
 n=${f##*/}
 "${mux[@]}" "cat >/content/$n.tmp && mv /content/$n.tmp /content/$n" <"$f"
done
"${mux[@]}" 'cd /content && sha256sum -c matrix-v4-private.tar.gz.sha256'
