#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script builds the HLS part of the tool  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE=${SCRIPT_DIR}/..

# --------------------------------------------------------------------
# Build MLIR and LLVM
# --------------------------------------------------------------------
echo ""
echo ">>> Build LLVM and MLIR "
echo ""

# If ninja is available, use it.
CMAKE_GENERATOR="Unix Makefiles"
if which ninja &>/dev/null; then
  CMAKE_GENERATOR="Ninja"
fi

# Go to the llvm directory and carry out installation.
LLVM_DIR="${MASE}/llvm"
MASE_HLS="${MASE}/hls"
cd "${MASE_HLS}"
mkdir -p build
cd build
CC=gcc CXX=g++ cmake .. \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR="${LLVM_DIR}/build/lib/cmake/mlir/" \
  -DLLVM_DIR="${LLVM_DIR}/build/lib/cmake/llvm/" \
  -DLLVM_EXTERNAL_LIT="${LLVM_DIR}/build/bin/llvm-lit" 

# ------------------------- Build and test ---------------------

cmake --build . --target check-mase
