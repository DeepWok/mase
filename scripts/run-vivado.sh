#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script searches for Vivado and calls Vivado  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

source $VHLS/Vivado/$XLNX_VERSION/settings64.sh
source $VHLS/Vitis_HLS/$XLNX_VERSION/settings64.sh
LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 vivado -mode batch -source $1

