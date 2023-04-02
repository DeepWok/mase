#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script searches for Vitis HLS and calls Vitis HLS  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

source $VHLS/Vitis_HLS/2020.2/settings64.sh
vitis_hls $1
