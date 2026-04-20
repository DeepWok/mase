#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script installs environment vars for the mase nix shell 
# --------------------------------------------------------------------

# The absolute path to the directory of this script.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE="$(dirname "$SCRIPT_DIR")"

# Basic PATH setup 
export PATH=$MASE/scripts:$MASE/hls/build/bin:$MASE/llvm/build/bin:$PATH
export PYTHONPATH=$PYTHONPATH:$MASE:$MASE/src
export LIBRARY_PATH=${LIBRARY_PATH:+LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu
