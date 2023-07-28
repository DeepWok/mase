#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script searches for Vitis HLS and calls Vitis HLS  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

source $VHLS/Vitis_HLS/$XLNX_VERSION/settings64.sh
vitis_hls $1
