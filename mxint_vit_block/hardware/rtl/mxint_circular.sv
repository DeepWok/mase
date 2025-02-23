`timescale 1ns / 1ps
/*
Module      : Mxint circular
Description : Cicular buffer for mxint datatype.
*/

module mxint_circular #(
    // precision represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_PRECISION_0 = 8,
    parameter DATA_PRECISION_1 = 8,
    parameter IN_NUM = -1,
    parameter REPEAT = -1,
    parameter BUFFER_SIZE = -1
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
  logic [DATA_PRECISION_0 * IN_NUM + DATA_PRECISION_1 - 1:0] data_in_flatten[0:0];
  logic [DATA_PRECISION_0 * IN_NUM + DATA_PRECISION_1 - 1:0] data_out_flatten[0:0];
  logic [DATA_PRECISION_0 * IN_NUM + DATA_PRECISION_1 - 1:0] packed_data_out_flatten;
  logic [DATA_PRECISION_0 * IN_NUM + DATA_PRECISION_1 - 1:0] packed_data_in_flatten;
  assign data_in_flatten[0] = packed_data_in_flatten;
  assign packed_data_out_flatten = data_out_flatten[0];
  for (genvar i = 0; i < IN_NUM; i++) begin : reshape
    assign packed_data_in_flatten[(i+1)*DATA_PRECISION_0-1:i*DATA_PRECISION_0] = mdata_in[i];
  end
  assign packed_data_in_flatten[DATA_PRECISION_0*IN_NUM+DATA_PRECISION_1-1:DATA_PRECISION_0*IN_NUM] = edata_in;
  input_buffer #(
      .DATA_WIDTH (DATA_PRECISION_0 * IN_NUM + DATA_PRECISION_1),
      .IN_NUM     (1),
      .REPEAT     (REPEAT),
      .BUFFER_SIZE(BUFFER_SIZE)
  ) mdata_in_0_buffer (
      .clk,
      .rst,
      // Input streaming port
      .data_in(data_in_flatten),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      // Output streaming port
      .data_out(data_out_flatten),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
  for (genvar i = 0; i < IN_NUM; i++) begin : unreshape
    assign mdata_out[i] = packed_data_out_flatten[(i+1)*DATA_PRECISION_0-1:i*DATA_PRECISION_0];
  end
  assign edata_out = packed_data_out_flatten[DATA_PRECISION_0*IN_NUM+DATA_PRECISION_1-1:DATA_PRECISION_0*IN_NUM];

endmodule
