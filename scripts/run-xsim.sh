#!/usr/bin/env bash
# --------------------------------------------------------------------
#    This script searches for XSIM and runs co-simulation  
# --------------------------------------------------------------------

set -o errexit
set -o pipefail
set -o nounset

source $VHLS/Vivado/$XLNX_VERSION/settings64.sh

xelab $1 -prj proj.prj -L smartconnect_v1_0 -L axi_protocol_checker_v1_1_12 -L axi_protocol_checker_v1_1_13 -L axis_protocol_checker_v1_1_11 -L axis_protocol_checker_v1_1_12 -L xil_defaultlib -L unisims -L unisims_ver -L xpm  -L floating_point_v7_0_18 -L floating_point_v7_1_11 --lib "ieee_proposed=./ieee_proposed" -s proj
xsim --noieeewarnings proj -tclbatch proj.tcl

