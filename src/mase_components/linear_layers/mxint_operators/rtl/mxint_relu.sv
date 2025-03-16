/*
Module      : mxint_relu
Description : This module performs relu(x) on the input function

              Python equivalent:
              out = torch.nn.functional.relu(x)

              x should be the dimension of (DATA_IN_0_TENSOR_SIZE_DIM_1, DATA_IN_0_TENSOR_SIZE_DIM_0)
*/
`timescale 1ns / 1ps

module mxint_relu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 1,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 20,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,

    localparam IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    localparam IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1
) (
    input clk,
    input rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_IN_0_PRECISION_0-1:0] mdata_out_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_IN_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready

);
  logic [DATA_IN_0_PRECISION_0-1:0] mdata_out_0_i [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_1-1:0] edata_out_0_i;
  logic data_out_0_valid_i;
  logic data_out_0_ready_i;

  always_comb begin
    edata_out_0_i = edata_in_0;

    for (int i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin
      mdata_out_0_i[i] = mdata_in_0[i][DATA_IN_0_PRECISION_0-1] ? '0 : mdata_in_0[i];
    end
    data_out_0_valid_i = data_in_0_valid;
  end

  mxint_cast #(
      .IN_MAN_WIDTH (DATA_IN_0_PRECISION_0),
      .IN_EXP_WIDTH (DATA_IN_0_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_IN_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .BLOCK_SIZE   (DATA_IN_0_PARALLELISM_DIM_1 * DATA_IN_0_PARALLELISM_DIM_0)
  ) cast_i (
      .clk           (clk),
      .rst           (rst),
      .mdata_in      (mdata_out_0_i),
      .edata_in      (edata_out_0_i),
      .data_in_valid (data_out_0_valid_i),
      .data_in_ready (data_in_0_ready),
      .mdata_out     (mdata_out_0),
      .edata_out     (edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule

