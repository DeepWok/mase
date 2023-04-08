#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script searches for Vivado and calls Vivado  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

source $VHLS/Vivado/2020.2/settings64.sh
source $VHLS/Vitis_HLS/2020.2/settings64.sh
LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 vivado -mode batch -source $1

