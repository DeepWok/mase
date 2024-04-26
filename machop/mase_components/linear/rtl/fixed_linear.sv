`timescale 1ns / 1ps

module fixed_linear #(
    /* verilator lint_off UNUSEDPARAM */
    parameter HAS_BIAS = 0,

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,

    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 32,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 1,
    parameter WEIGHT_PARALLELISM_DIM_0 = 1,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,  // must equal DATA_IN_0_PARALLELISM_DIM_0

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DATA_IN_0_TENSOR_SIZE_DIM_0
    ) + $clog2(
        IN_0_DEPTH
    ) + HAS_BIAS,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,
    parameter BIAS_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = 1,
    parameter BIAS_PARALLELISM_DIM_1 = 1
) (
    input clk,
    input rst,

    // input port for data_inivations
    input  [DATA_IN_0_PRECISION_0-1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input data_in_0_valid,
    output data_in_0_ready,

    // input port for weight
    input [WEIGHT_PRECISION_0-1:0] weight[WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input weight_valid,
    output weight_ready,

    /* verilator lint_off UNUSEDSIGNAL */
    input [BIAS_PRECISION_0-1:0] bias[BIAS_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    input bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output bias_ready,

    output [DATA_OUT_0_PRECISION_0-1:0] data_out_0      [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output data_out_0_valid,
    input data_out_0_ready
);

  matmul #(
      // Total dimensions
      .A_TOTAL_DIM0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .A_TOTAL_DIM1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .B_TOTAL_DIM0(WEIGHT_TENSOR_SIZE_DIM_0),
      .B_TOTAL_DIM1(WEIGHT_TENSOR_SIZE_DIM_1),

      .A_COMPUTE_DIM0(DATA_IN_0_PARALLELISM_DIM_0),
      .A_COMPUTE_DIM1(DATA_IN_0_PARALLELISM_DIM_1),
      .B_COMPUTE_DIM0(WEIGHT_PARALLELISM_DIM_0),
      .B_COMPUTE_DIM1(WEIGHT_PARALLELISM_DIM_1),

      .A_WIDTH     (DATA_IN_0_PRECISION_0),
      .A_FRAC_WIDTH(DATA_IN_0_PRECISION_1),
      .B_WIDTH     (WEIGHT_PRECISION_0),
      .B_FRAC_WIDTH(WEIGHT_PRECISION_1),

      .OUT_WIDTH     (DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
      .OUT_SYMMETRIC (0)
  ) matmul_i (
      .clk,
      .rst,

      .a_data (data_in_0),
      .a_valid(data_in_0_valid),
      .a_ready(data_in_0_ready),

      .b_data (weight),
      .b_valid(weight_valid),
      .b_ready(weight_ready),

      .out_data (data_out_0),
      .out_valid(data_out_0_valid),
      .out_ready(data_out_0_ready)
  );

  // ! TO DO: add bias
  assign bias_ready = '0;

endmodule
