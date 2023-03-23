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

# Temporarily declare sysroot
SYS_ROOT=${MASE}/vck190_air_sysroot
mkdir -p $SYS_ROOT

# Build MLIR-AIE
bash ${MASE}/mlir-air/utils/build-mlir-air.sh \
  ${SYS_ROOT} \
  ${LLVM_DIR} \
  ${MASE}/mlir-air/cmakeModules \
  ${MASE}/mlir-aie \
  ${MASE}/mlir-air/build


