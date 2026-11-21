#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MATRIX_DEPS="$PROJECT_ROOT/third_party/runtimes/MATRiX_Robot_MC-v0.6.6/local_deps"
test -f "$MATRIX_DEPS/usr/include/boost/version.hpp"
rg -q 'BOOST_LIB_VERSION "1_74"' "$MATRIX_DEPS/usr/include/boost/version.hpp"
mkdir -p build/matrix
g++ -std=c++17 -shared -fPIC -O2 -pthread -I "$MATRIX_DEPS/usr/include" \
  src/atec_nav/robots/matrix_shm.cpp -o build/matrix/libatec_matrix_shm.new.so
mv build/matrix/libatec_matrix_shm.new.so build/matrix/libatec_matrix_shm.so
"$MATRIX_DEPS/usr/bin/lcm-gen" -p --ppath build/matrix configs/matrix_gamepad.lcm
