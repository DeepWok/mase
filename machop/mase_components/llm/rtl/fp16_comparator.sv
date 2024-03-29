`timescale 1ns / 1ps


// for floating-point 16
// module fp16_comparator # (
//     parameter THRES = 30
// )(
//     input [16-1: 0] data_in,  // in fp-16 format
//     // S | E  E  E  E  E  | M M M ... M
//     // 15| 14 13 12 11 10 | 9 8 7 ... 0

//     output result
// );
//     logic [5-1: 0] exponent_biased;
//     logic sign;

//     assign sign = data_in[15];
//     assign exponent_biased = data_in[14:10];  // biased by 15. i.e., exponent_biased = exponenet + 15;
//     assign result = exponent_biased[4] && (exponent_biased[3] || exponent_biased[2] || exponent_biased[1]);  // exponent_biase >= 18, i.e., exponent >= 3
// endmodule

// for fixed-point 16
module fp16_comparator #(
    parameter IN_WIDTH = 16,
    parameter IN_FRAC_WIDTH = 0,
    parameter THRES = 127  // currently deprecated
) (
    input [IN_WIDTH-1:0] data_in,  // by default ap_int<16>
    output result
);
  initial begin
    assert (IN_WIDTH - IN_FRAC_WIDTH > 1)
    else $fatal("IN_WIDTH must be larger than IN_FRAC_WIDTH!");
    assert (THRES > 0)
    else $fatal("THRES must be positive!");
  end

  // retrieve integer part of data_in
  localparam IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;
  logic [IN_INT_WIDTH-1 : 0] data_in_int;
  assign data_in_int = data_in[IN_WIDTH-1:IN_FRAC_WIDTH];

  logic sign;
  assign sign = data_in[IN_WIDTH-1];

  // find absolute value
  logic [IN_INT_WIDTH-1 : 0] data_in_int_abs;
  assign data_in_int_abs = (sign == 1'b1) ? -$signed(data_in_int) : data_in_int;

  assign result = (data_in_int_abs > THRES) ? 1'b1 : 1'b0;

  // logic result_reg;
  // always_comb begin
  //     if (sign) begin // negative
  //         result_reg = (!(& data_in[14:7])) || (data_in[7:0] == 8'b1000_0000); // -128 is treated as large number //(!data_in[14]) || (!data_in[13]);
  //     end else begin  // positive
  //         result_reg = | data_in[14:7];
  //     end
  // end
  // assign result = result_reg;
endmodule
