#!/usr/bin/env bash
# Run inside the temporary Colab VM. Inputs are staged under /content by launch.sh.
set -euo pipefail
umask 077
cd /content
revision=6405fc4d1cea875b0dfae6ea3bca6083b480d6fc
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | grep A100
if [[ ! -d nav/.git ]]; then
  git init -q nav
  git -C nav remote add origin https://github.com/CyanTempest/Dreamer_nav.git
fi
if [[ $(git -C nav rev-parse --verify HEAD 2>/dev/null || true) != "$revision" ]]; then
  git -C nav fetch -q --depth=1 origin "$revision"
  git -C nav checkout -q --detach FETCH_HEAD
fi
[[ $(git -C nav rev-parse HEAD) == "$revision" ]]
tar -xf nav4-code.tar -C nav
mkdir -p /content/nav4-map
tar -xf /content/nav4-map.tar -C /content/nav4-map
test -s /content/nav4-map/manifest.json
if [[ ! -x /content/nav4-tools/bin/uv ]]; then
  . /etc/os-release
  src=/tmp/nav4-apt.list
  for suite in "$VERSION_CODENAME" "$VERSION_CODENAME-updates" "$VERSION_CODENAME-security"; do
    echo "deb https://azure.archive.ubuntu.com/ubuntu/ $suite main universe"
  done > "$src"
  apt_opts=(-o Acquire::Retries=2 -o Acquire::https::Timeout=30 -o "Dir::Etc::sourcelist=$src" -o Dir::Etc::sourceparts=-)
  distro_python=$(awk '$1=="default-version"{print $3}' /usr/share/python3/debian_defaults)
  if ! "/usr/bin/$distro_python" -m venv /content/nav4-tools; then
    chmod 644 "$src"
    apt-get "${apt_opts[@]}" update -qq
    DEBIAN_FRONTEND=noninteractive apt-get "${apt_opts[@]}" install -y python3-venv liblz4-1
    "/usr/bin/$distro_python" -m venv /content/nav4-tools
  fi
  /content/nav4-tools/bin/python -m pip install --disable-pip-version-check uv==0.13.7
fi
uv=/content/nav4-tools/bin/uv
export UV_NO_PROGRESS=1
if [[ ! -x /content/nav4-env/bin/python ]]; then
  "$uv" venv --python 3.11.18 --managed-python /content/nav4-env
fi
"$uv" pip sync --python /content/nav4-env/bin/python /content/nav/nav4/colab/requirements.txt
"$uv" pip check --python /content/nav4-env/bin/python
JAX_PLATFORMS=cuda /content/nav4-env/bin/python -c 'import jax; print(jax.__version__,jax.devices()); assert jax.default_backend()=="gpu"'

