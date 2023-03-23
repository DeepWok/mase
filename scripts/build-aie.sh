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
bash ${MASE}/mlir-air/utils/build-mlir-aie-local.sh \
  ${LLVM_DIR} \
  ${MASE}/cmakeModules/cmakeModulesXilinx \
  ${MASE}/mlir-aie

