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
  localparam EDATA_BREAK_NUM = DATA_PRECISION_1/DATA_PRECISION_0 + 1;
  logic [DATA_PRECISION_0 - 1:0] breaked_edata_in [EDATA_BREAK_NUM - 1:0];
  logic [DATA_PRECISION_0 - 1:0] breaked_edata_out [EDATA_BREAK_NUM - 1:0];
  logic [DATA_PRECISION_0 - 1:0] packed_data_in [IN_NUM - 1 + EDATA_BREAK_NUM:0];
  logic [DATA_PRECISION_0 - 1:0] packed_data_out[IN_NUM:0];
  break_data #(
    .IN_WIDTH(DATA_PRECISION_1),
    .OUT_WIDTH(DATA_PRECISION_0)
  ) bd_inst (
    .data_in(edata_in),
    .data_out(breaked_edata_in)
  );
  pack_data #(
    .IN_WIDTH(DATA_PRECISION_0),
    .BREAK_NUM(EDATA_BREAK_NUM),
    .OUT_WIDTH(DATA_PRECISION_0)
  ) pk_data (
    .data_in(breaked_edata_out),
    .data_out(edata_out)
  );

  always_comb begin : data_pack
    packed_data_in[IN_NUM + EDATA_BREAK_NUM-1:IN_NUM] = breaked_edata_in;
    packed_data_in[IN_NUM-1:0] = mdata_in;
    breaked_edata_out = packed_data_out[IN_NUM + EDATA_BREAK_NUM-1:IN_NUM];
    mdata_out = packed_data_out[IN_NUM-1:0];
  end

  unpacked_register_slice #(
      .DATA_WIDTH(DATA_PRECISION_0),
      .IN_SIZE(IN_NUM + EDATA_BREAK_NUM)
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

module break_data #(
  /*
  If input data with [IN_WIDTH - 1:0]
  This module will actually break it into smaller pieces
  Redundant bit width will be filled with 0
  */
  parameter IN_WIDTH = -1,
  parameter OUT_WIDTH = -1,
  parameter BREAK_NUM = IN_WIDTH/OUT_WIDTH + 1
) (
  input logic [IN_WIDTH - 1:0] data_in,
  output logic [OUT_WIDTH - 1:0] data_out [BREAK_NUM - 1:0]
);
  logic [BREAK_NUM * OUT_WIDTH - 1:0] extended_data_in;
  assign extended_data_in = {{(BREAK_NUM*OUT_WIDTH - IN_WIDTH){0}}, data_in};
  for(genvar i=0; i<BREAK_NUM; i++)
    assign data_out[i] = extended_data_in[(i+1)*OUT_WIDTH - 1:i*OUT_WIDTH];
endmodule

module pack_data #(
  /*
  If input data with [IN_WIDTH - 1:0]
  This module will actually break it into smaller pieces
  Redundant bit width will be filled with 0
  */
  parameter IN_WIDTH = -1,
  parameter BREAK_NUM = -1,
  parameter OUT_WIDTH = -1
) (
  input logic [IN_WIDTH - 1:0] data_in [BREAK_NUM - 1:0],
  output logic [OUT_WIDTH - 1:0] data_out
);
  logic [BREAK_NUM * IN_WIDTH - 1:0] extended_data_out;
  for(genvar i=0; i<BREAK_NUM; i++)
    assign extended_data_out[(i+1)*OUT_WIDTH - 1:i*OUT_WIDTH] = data_in[i];
  assign data_out = extended_data_out;
endmodule