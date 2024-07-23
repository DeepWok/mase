#include "ap_fixed.h"
#include "hls_stream.h"

#define frac_width_0 8
#define total_width_0 16 
#define int_width_0 total_width_0 - frac_width_0

#define frac_width_1 8
#define total_width_1 16 
#define int_width_1 total_width_1 - frac_width_1

#define frac_width_2 8
#define total_width_2 16 
#define int_width_2 total_width_2 - frac_width_2

void div(
		hls::stream<ap_fixed<total_width_0, int_width_0>> &data_in_0, 
		hls::stream<ap_fixed<total_width_1, int_width_1>> &data_in_1, 
		hls::stream<ap_fixed<total_width_2, int_width_2>> &data_out_0) {
#pragma HLS PIPELINE II=1
 if (data_in_0.empty() || data_in_1.empty())
    return;
	ap_fixed<total_width_0, int_width_0> in0;
       ap_fixed<total_width_1, int_width_1> in1;
       data_in_0.read_nb(in0);
data_in_1.read_nb(in1);
	ap_fixed<total_width_2, int_width_2> res = in0/in1;
// TODO: #pragma HLS bind_op variable=res op= impl=fabric
	data_out_0.write_nb(res);
}	
		
