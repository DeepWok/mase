`timescale 1ns / 1ps
/*
Module      : mxint_register_slice
Description : This module does the same function as register slice
              But for datatype mxint.
*/

module mxint_register_slice #(
    // precision represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_PRECISION_0 = 8,
    parameter DATA_PRECISION_1 = 8,
    parameter IN_NUM = 6

) (
    input clk,
    input rst,
    // m -> mantissa, e -> exponent
    input logic [DATA_PRECISION_0-1:0] mdata_in[IN_NUM - 1:0],
    input logic [DATA_PRECISION_1-1:0] edata_in,
    input data_in_valid,
    output data_in_ready,

    output logic [DATA_PRECISION_0-1:0] mdata_out[IN_NUM - 1:0],
    output logic [DATA_PRECISION_1-1:0] edata_out,
    output data_out_valid,
    input data_out_ready
);
  initial begin
    assert (DATA_PRECISION_0 >= DATA_PRECISION_1)
    else $fatal("DATA_PRECISION_0 must larger than PRECISION_1");
  end
  logic [DATA_PRECISION_0 - 1:0] packed_data_in [IN_NUM:0];
  logic [DATA_PRECISION_0 - 1:0] packed_data_out[IN_NUM:0];
  always_comb begin : data_pack
    packed_data_in[IN_NUM-1:0] = mdata_in;
    packed_data_in[IN_NUM] = $signed(edata_in);
    mdata_out = packed_data_out[IN_NUM-1:0];
    edata_out = packed_data_out[IN_NUM];
  end

  unpacked_register_slice #(
      .DATA_WIDTH(DATA_PRECISION_0),
      .IN_SIZE(IN_NUM + 1)
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in       (packed_data_in),
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out      (packed_data_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );

endmodule
