#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script installs environment vars for the mase nix shell 
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

# The absolute path to the directory of this script.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MASE="$(dirname "$SCRIPT_DIR")"

# Basic PATH setup 
export PATH=$MASE/scripts:$MASE/hls/build/bin:$MASE/llvm/build/bin:$MASE/mlir-aie/install/bin:$MASE/mlir-air/install/bin:$PATH
export PYTHONPATH=$MASE:$MASE/src:$MASE/mlir-aie/install/python:$MASE/mlir-air/install/python:$PYTHONPATH
export LIBRARY_PATH=${LIBRARY_PATH:+LIBRARY_PATH:}/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+LD_LIBRARY_PATH:}$MASE/mlir-aie/lib:$MASE/mlir-air/lib:/opt/xaiengine

# Terminal color
export PS1="[\\[$(tput setaf 3)\\]\t\\[$(tput setaf 2)\\] \u\\[$(tput sgr0)\\]@\\[$(tput setaf 2)\\]\h \\[$(tput setaf 7)\\]\w \\[$(tput sgr0)\\]] \\[$(tput setaf 6)\\]$ \\[$(tput sgr0)\\]"
export LS_COLORS='rs=0:di=01;96:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01'
