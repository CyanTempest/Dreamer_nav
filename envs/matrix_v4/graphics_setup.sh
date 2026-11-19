#!/usr/bin/env bash
# Runtime-only. Invoke as the setup user with the runtime root; never during image build.
# Write literal parameter expansions for the shell which later sources graphics.env.
# shellcheck disable=SC2016
set -euo pipefail
umask 077
root=${1:?runtime root required}
[[ $(id -u) != 0 ]] || { echo "Run graphics preparation as the simulation user" >&2; exit 1; }
tmp=$(mktemp -d "$root/.graphics.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$root/xdg"
export XDG_RUNTIME_DIR="$root/xdg"
if [[ -d /usr/lib64-nvidia ]]; then export LD_LIBRARY_PATH="/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
printf 'export XDG_RUNTIME_DIR=%q\n' "$XDG_RUNTIME_DIR" > "$tmp/graphics.env"
if [[ -d /usr/lib64-nvidia ]]; then
  echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> "$tmp/graphics.env"
fi
probe() {
  vulkaninfo --summary > "$root/vulkan-summary.log" 2>&1 &&
    grep -Eq 'vendorID[[:space:]]*=[[:space:]]*0x10de' "$root/vulkan-summary.log" &&
    python3 -c 'import ctypes; ctypes.CDLL("libnvoptix.so.1")'
}
if ! probe; then
  # Compute-only VM images can omit user-space graphics libraries. Keep the
  # matching EGL ICD private to UeSim. The host owns the loaded kernel module.
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p')
  [[ $driver =~ ^[0-9]+(\.[0-9]+)+$ ]] || { echo "Cannot identify the NVIDIA driver" >&2; exit 1; }
  libs="$root/nvidia-$driver-egl"
  if [[ ! -f "$libs/icd.json" ]]; then
    name="NVIDIA-Linux-x86_64-$driver.run"
    url="https://us.download.nvidia.com/XFree86/Linux-x86_64/$driver/$name"
    curl_args=(--proto '=https' --fail --location --retry 3 --connect-timeout 20)
    curl "${curl_args[@]}" "$url.sha256sum" -o "$tmp/nvidia.sha256"
    curl "${curl_args[@]}" "$url" -o "$tmp/$name"
    digest=$(awk -v name="$name" '$2 == name {print $1}' "$tmp/nvidia.sha256")
    [[ $digest =~ ^[a-fA-F0-9]{64}$ ]]
    (cd "$tmp" && echo "$digest  $name" | sha256sum -c -)
    sh "$tmp/$name" --extract-only --target "$tmp/vendor"
    mkdir "$tmp/libs"
    shopt -s nullglob
    for file in "$tmp/vendor"/lib{EGL_nvidia,nvoptix,nvidia-{allocator,eglcore,glcore,glsi,glvkspirv,gpucomp,rtcore,tls}}.so."$driver"; do
      [[ -f $file ]] || continue
      cp -L "$file" "$tmp/libs/"
    done
    /sbin/ldconfig -n "$tmp/libs"
    test -f "$tmp/libs/libEGL_nvidia.so.0" && test -f "$tmp/libs/libnvoptix.so.1" || { echo "Missing EGL/OptiX in vendor package" >&2; exit 1; }
    python3 - "$tmp/vendor/nvidia_icd.json" "$tmp/libs/icd.json" <<'PY'
import json, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
manifest = json.loads(src.read_text())
manifest["ICD"]["library_path"] = "libEGL_nvidia.so.0"
dst.write_text(json.dumps(manifest) + "\n")
PY
    mv "$tmp/libs" "$libs"
  fi
  export VK_ICD_FILENAMES="$libs/icd.json" VK_DRIVER_FILES="$libs/icd.json"
  export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  probe || { echo "NVIDIA Vulkan/OptiX probe failed. See $root/vulkan-summary.log" >&2; exit 1; }
  printf 'export VK_DRIVER_FILES=%q VK_ICD_FILENAMES=%q\n' "$VK_DRIVER_FILES" "$VK_ICD_FILENAMES" >> "$tmp/graphics.env"
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$libs" >> "$tmp/graphics.env"
fi
mv "$tmp/graphics.env" "$root/graphics.env"
echo "NVIDIA Vulkan device enumerated; OptiX library loads. UeSim runtime settings: $root/graphics.env"
