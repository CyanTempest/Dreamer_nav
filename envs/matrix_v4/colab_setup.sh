#!/usr/bin/env bash
# Prepare an existing Linux GPU VM. This script never starts UeSim or a cloud session.
set -euo pipefail
umask 077

here=$(cd -- "$(dirname -- "$0")" && pwd)
root=${MATRIX_V4_ROOT:-/content/matrix-v4}
inbox=${MATRIX_INBOX:-/content}
scene=${MATRIX_SCENE:-full}
mode=${1:-all}
release=MATRiX_v1.1.2
upstream_rev=3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9
nav_url=https://github.com/CyanTempest/Dreamer_nav.git
wm_url=https://github.com/nicklashansen/mmbench2.git
assets_url=https://github.com/zsibot/matrix/releases/download/v1.1.2
uv=/opt/matrix-v4-tools/bin/uv

die() { echo "$*" >&2; exit 1; }

system_packages() {
  [[ $EUID == 0 ]] || die "system installs Debian packages and requires the VM's root user"
  export DEBIAN_FRONTEND=noninteractive
  local apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
  # Colab's default HTTP mirror can stall. Limit this override to the setup invocation.
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ $ID == ubuntu ]]; then
    apt_source=$(mktemp /tmp/matrix-v4-apt.XXXXXXXX.list)
    trap 'rm -f -- "$apt_source"' EXIT
    for suite in "$VERSION_CODENAME" "$VERSION_CODENAME-updates" "$VERSION_CODENAME-security"; do
      echo "deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] https://azure.archive.ubuntu.com/ubuntu/ $suite main universe"
    done > "$apt_source"
    chmod 0644 "$apt_source"
    apt_opts+=(-o "Dir::Etc::sourcelist=$apt_source" -o Dir::Etc::sourceparts=-)
  fi
  apt-get "${apt_opts[@]}" update -qq
  apt-get "${apt_opts[@]}" install -y --no-install-recommends \
    ca-certificates curl git build-essential python3-venv passwd util-linux xz-utils \
    libvulkan1 vulkan-tools libegl1 libgl1 libglu1-mesa libsdl2-2.0-0 \
    libx11-6 libxkbcommon-x11-0 libxi6 libxrandr2 libxcursor1 libxinerama1 libxss1 \
    libfontconfig1 libsm6 libpulse0 libgomp1
  if [[ $ID == ubuntu ]]; then rm -f -- "$apt_source"; trap - EXIT; fi
  # Shared bootstrap must be executable by the unprivileged simulation user.
  # Colab may repoint /usr/bin/python3 while APT installs the distro's venv modules.
  local distro_python
  distro_python=$(awk '$1 == "default-version" {print $3}' /usr/share/python3/debian_defaults)
  [[ $distro_python =~ ^python3\.[0-9]+$ ]] || die "Cannot determine APT Python version"
  (umask 022
   "/usr/bin/$distro_python" -m venv /opt/matrix-v4-tools
   "/opt/matrix-v4-tools/bin/$distro_python" -m pip install --disable-pip-version-check uv==0.13.7)
  if ! id matrixv4 >/dev/null 2>&1; then useradd --create-home --user-group --shell /bin/bash matrixv4; fi
}

clone_sources() {
  if [[ ! -d "$root/nav" ]]; then
    git clone --depth=1 --branch matrix-v4 --single-branch "$nav_url" "$scratch/nav"
    mv "$scratch/nav" "$root/nav"
  fi
  [[ $(git -C "$root/nav" branch --show-current) == matrix-v4 ]] || die "Expected matrix-v4 in $root/nav"
  [[ $(git -C "$root/nav" remote get-url origin) == "$nav_url" ]] || die "Unexpected navigation remote"
  if [[ ! -d "$root/mmbench2" ]]; then
    git init -q "$scratch/mmbench2"
    git -C "$scratch/mmbench2" remote add origin "$wm_url"
    git -C "$scratch/mmbench2" fetch --depth=1 origin "$upstream_rev"
    git -C "$scratch/mmbench2" checkout --detach -q FETCH_HEAD
    mv "$scratch/mmbench2" "$root/mmbench2"
  fi
  [[ $(git -C "$root/mmbench2" rev-parse HEAD) == "$upstream_rev" ]] || die "Unexpected MMbench2 revision"
  echo "Navigation $(git -C "$root/nav" rev-parse HEAD), mmbench2 $upstream_rev"
}

install_envs() {
  local locks=$1 name py
  export UV_CACHE_DIR="$root/.cache/uv" UV_PYTHON_INSTALL_DIR="$root/python"
  export UV_PYTHON_BIN_DIR="$root/bin" UV_NO_PROGRESS=1
  mkdir -p "$root/venvs" "$UV_PYTHON_BIN_DIR"
  "$uv" python install 3.10.21 3.11.18
  for name in v3 v4 bridge; do
    py=3.11.18
    if [[ $name == v4 ]]; then py=3.10.21; fi
    if [[ ! -x "$root/venvs/$name/bin/python" ]]; then
      "$uv" venv --managed-python --python "$py" "$root/venvs/$name"
    fi
    [[ $("$root/venvs/$name/bin/python" -c 'import platform; print(platform.python_version())') == "$py" ]] ||
      die "Unexpected Python in $root/venvs/$name"
    "$uv" pip sync --strict --python "$root/venvs/$name/bin/python" "$locks/requirements-$name.txt"
    "$uv" pip check --python "$root/venvs/$name/bin/python"
  done
}

fetch_release() {
  local downloads="$root/downloads/v1.1.2" name expected part
  local manifest_sha=fde01b0f0240d4fc46ed7e85ac03a9b63799feb389ff1453afbd7b266c89b5b7
  if [[ -e "$root/$release" ]]; then
    [[ -f "$root/$release/.archive-verified" && $(cat "$root/$release/.archive-verified") == "$manifest_sha" &&
       -f "$root/$release/UeSim/Binaries/Linux/UeSim" ]] || die "Incomplete or unmanaged MATRiX directory"
    return
  fi
  mkdir -p "$downloads"
  local curl_args=(--proto '=https' --fail --location --retry 3 --retry-all-errors --connect-timeout 20)
  curl "${curl_args[@]}" "$assets_url/SHA256SUMS" -o "$scratch/SHA256SUMS"
  (cd "$scratch" && echo "$manifest_sha  SHA256SUMS" | sha256sum --check --status)
  cp "$scratch/SHA256SUMS" "$downloads/SHA256SUMS"
  local parts=()
  for part in 001 002 003; do
    name="$release.tar.gz.part-$part"
    parts+=("$downloads/$name")
    expected=$(awk -v name="$name" '$2 == name {print $1}' "$downloads/SHA256SUMS")
    [[ ${#expected} == 64 ]] || die "Missing release entry $name"
    if ! (cd "$downloads" && echo "$expected  $name" | sha256sum --check --status 2>/dev/null); then
      if ! curl "${curl_args[@]}" -C - -o "$downloads/$name.partial" "$assets_url/$name" ||
          ! (cd "$downloads" && echo "$expected  $name.partial" | sha256sum --check --status); then
        curl "${curl_args[@]}" -o "$downloads/$name.partial" "$assets_url/$name"
      fi
      (cd "$downloads" && echo "$expected  $name.partial" | sha256sum --check --status)
      mv -f "$downloads/$name.partial" "$downloads/$name"
    fi
  done
  (cd "$downloads" && sha256sum -c SHA256SUMS)
  cat "${parts[@]}" | tar --no-same-owner -xz -C "$scratch"
  [[ -f "$scratch/$release/UeSim.sh" && -f "$scratch/$release/RELEASE_BUILD.json" ]] || die "Unexpected archive layout"
  echo "$manifest_sha" > "$scratch/$release/.archive-verified"
  mv "$scratch/$release" "$root/$release"
}

install_assets() {
  [[ -r "$inbox/matrix-v4-private.tar.gz" && -r "$inbox/matrix-v4-private.tar.gz.sha256" ]] ||
    die "Place the private bundle and its .sha256 in $inbox; rerun the assets phase."
  "$root/venvs/bridge/bin/python" "$root/nav/envs/matrix_v4/private_assets.py" install "$inbox" "$root" --scene "$scene"
}

[[ $(uname -sm) == 'Linux x86_64' ]] || die "Linux x86_64 required"
case "$mode" in
  system) system_packages; exit ;;
  all|assets|envs) ;;
  *) die "Usage: $0 [all|assets|envs|system]" ;;
esac
[[ $root == /* && $root != / ]] || die "MATRIX_V4_ROOT must be an absolute workspace path"
if [[ $EUID == 0 ]]; then
  if [[ $mode == all ]]; then system_packages; fi
  id matrixv4 >/dev/null 2>&1 || die "Run the system phase first"
  [[ $mode != envs ]] || die "Use envs as matrixv4 with the locks beside this script"
  install -d -m 0700 -o matrixv4 -g matrixv4 "$root"
  handoff=$(mktemp -d /tmp/matrix-v4-setup.XXXXXXXX)
  trap 'rm -rf -- "$handoff"' EXIT
  chgrp matrixv4 "$handoff"
  chmod 0750 "$handoff"
  bundle=matrix-v4-private.tar.gz
  if [[ -f "$inbox/$bundle.sha256" ]]; then
    # The Contents API sends a whole file as base64. Accept numbered small uploads too.
    if [[ -f "$inbox/$bundle" ]]; then
      cp --reflink=auto "$inbox/$bundle" "$handoff/$bundle"
    else
      shopt -s nullglob
      parts=("$inbox/$bundle".part-[0-9][0-9][0-9])
      (( ${#parts[@]} > 0 )) || die "No archive or numbered parts in $inbox"
      cat "${parts[@]}" > "$handoff/$bundle"
    fi
    cp "$inbox/$bundle.sha256" "$handoff/$bundle.sha256"
    chgrp matrixv4 "$handoff/$bundle" "$handoff/$bundle.sha256"
    chmod 0640 "$handoff/$bundle" "$handoff/$bundle.sha256"
    inbox=$handoff
  fi
  # A standalone script downloaded by root may also have mode 0600.
  install -m 0755 "$0" "$handoff/setup.sh"
  runuser -u matrixv4 -- env MATRIX_V4_ROOT="$root" MATRIX_INBOX="$inbox" MATRIX_SCENE="$scene" bash "$handoff/setup.sh" "$mode"
  exit
fi
mkdir -p "$root"
exec 9>"$root/.setup.lock"
flock 9
scratch=$(mktemp -d "$root/.setup.XXXXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
if [[ $mode == assets ]]; then install_assets; exit; fi
[[ -x "$uv" ]] || die "Run the system phase in the VM/image first"
if [[ $mode == envs ]]; then install_envs "$here"; exit; fi
clone_sources
bash "$root/nav/envs/matrix_v4/graphics_setup.sh" "$root"
install_envs "$root/nav/envs/matrix_v4"
fetch_release
install_assets
echo "Prepared files in $root. Host graphics and the closed loop require a separate runtime test."
