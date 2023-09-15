#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script builds the AIE and AIR part of the tool  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE=$SCRIPT_DIR/..
LLVM_DIR="${MASE}/llvm"

# Build MLIR-AIE
# export PATH=$PATH:${VHLS}/Vitis/2023.1/aietools/bin:${VHLS}/Vitis/2023.1/bin
bash ${MASE}/mlir-air/utils/build-mlir-aie-local.sh \
  ${LLVM_DIR} \
  ${MASE}/mlir-aie/cmake/modulesXilinx \
  ${MASE}/aienginev2/install \
  ${MASE}/mlir-aie
