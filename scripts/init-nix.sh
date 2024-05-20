#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script installs environment vars for the mase nix shell 
# --------------------------------------------------------------------

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE="$(dirname "$SCRIPT_DIR")"

# Basic PATH setup 
export PATH=$MASE/scripts:$MASE/hls/build/bin:$MASE/llvm/build/bin:$MASE/mlir-aie/install/bin:$MASE/mlir-air/install/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$MASE:$MASE/src:$MASE/mlir-aie/install/python:$MASE/mlir-air/install/python
export LIBRARY_PATH=${LIBRARY_PATH:+LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MASE/mlir-aie/lib:$MASE/mlir-air/lib:/opt/xaiengine
