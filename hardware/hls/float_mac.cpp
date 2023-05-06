// This file implements an floating-point MAC operator.
// open_project -reset float_mac
// set_top float_mac
// add_files { ./float_mac.cpp } -cflags "-DTY=float"
// open_solution -reset "solution1"
// set_part {xcu250-figd2104-2L-e}
// create_clock -period 4 -name default
// config_bind -effort high
// config_compile -pipeline_loops 1
// csynth_design
// export_design -flow syn -format ip_catalog

#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

using namespace std;

TY float_mac(TY a, TY b, TY c) {
#pragma HLS PIPELINE II = 1
  TY ab = a * b;
#pragma HLS BIND_OP variable = ab op = add impl = fabric
  TY abc = ab + c;
#pragma HLS BIND_OP variable = abc op = add impl = fabric

  return abc;
}
