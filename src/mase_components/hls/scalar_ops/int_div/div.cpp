#include "ap_int.h"
#include "hls_stream.h"

#define total_width_0 32

#define total_width_1 32

#define total_width_2 16

void div(hls::stream<ap_int<total_width_0>> &data_in_0,
         hls::stream<ap_int<total_width_1>> &data_in_1,
         hls::stream<ap_int<total_width_2>> &data_out_0) {
#pragma HLS PIPELINE II = 1
  if (data_in_0.empty() || data_in_1.empty())
    return;
  ap_int<total_width_0> in0;
  ap_int<total_width_1> in1;
  data_in_0.read_nb(in0);
  data_in_1.read_nb(in1);
  ap_int<total_width_2> res = in0 / in1;
  // TODO: #pragma HLS bind_op variable=res op= impl=fabric
  data_out_0.write_nb(res);
}
