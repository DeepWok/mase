`timescale 1ns / 1ps
module mxint_vector_mult #(
    // precision_0 represent mantissa width
    // precision_1 represent exponent width
    // 
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 8,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 - 1,
    parameter DATA_OUT_0_PRECISION_1 = (DATA_IN_0_PRECISION_1 > WEIGHT_PRECISION_1)? DATA_IN_0_PRECISION_1 + 1 : WEIGHT_PRECISION_1 + 1,
    parameter BLOCK_SIZE = 6

) (
    input clk,
    input rst,
    // m -> mantissa, e -> exponent
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input data_in_0_valid,
    output data_in_0_ready,

    input logic [WEIGHT_PRECISION_0-1:0] mweight[BLOCK_SIZE - 1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight,
    input weight_valid,
    output weight_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0[BLOCK_SIZE - 1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output data_out_0_valid,
    input data_out_0_ready
);
  logic data_out_0_reg_in_valid, data_out_0_reg_in_ready;
  logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0_reg_in [BLOCK_SIZE - 1:0];
  logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0_reg_in;

  join2 #() join_inst (
      .data_in_ready ({weight_ready, data_in_0_ready}),
      .data_in_valid ({weight_valid, data_in_0_valid}),
      .data_out_valid(data_out_0_reg_in_valid),
      .data_out_ready(data_out_0_reg_in_ready)
  );
  for (genvar i = 0; i < BLOCK_SIZE; i = i + 1) begin : mantissa_mult
    assign mdata_out_0_reg_in[i] = $signed(mdata_in_0[i]) * $signed(mweight[i]);
  end
  assign edata_out_0_reg_in = $signed(edata_in_0) + $signed(eweight);

  mxint_register_slice #(
      .DATA_PRECISION_0($bits(mdata_out_0_reg_in[0])),
      .DATA_PRECISION_1($bits(edata_out_0_reg_in)),
      .IN_NUM(BLOCK_SIZE)
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .mdata_in      (mdata_out_0_reg_in),
      .edata_in      (edata_out_0_reg_in),
      .data_in_valid (data_out_0_reg_in_valid),
      .data_in_ready (data_out_0_reg_in_ready),
      .mdata_out     (mdata_out_0),
      .edata_out     (edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );
endmodule
