// This file implements an 8-bit fixed-point MAC operator.
// open_project -reset fixed_mac
// set_top fixed_mac
// add_files { ./fixed_mac.cpp }
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

ap_int<8> fixed_mac(ap_int<8> a, ap_int<8> b, ap_int<8> c) {
#pragma HLS PIPELINE II = 1
  ap_uint<16> res = a * b + c;
  ap_int<8> r;
  if (res > 127)
    r = 127;
  else if (res < -128)
    r = -128;
  else
    r = res;
  return r;
}
