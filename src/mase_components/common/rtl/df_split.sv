`timescale 1ns / 1ps

module df_split #(
    /* verilator lint_off UNUSEDPARAM */

    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,

    // Output 1
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,

    // Output 2
    parameter DATA_OUT_1_PRECISION_0 = 16,
    parameter DATA_OUT_1_PRECISION_1 = 3,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_1_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_1_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_1_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2
) (
    input logic clk,
    input logic rst,

    // input port for data_inivations
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready,

    output logic [DATA_OUT_1_PRECISION_0-1:0] data_out_1 [DATA_OUT_1_PARALLELISM_DIM_0*DATA_OUT_1_PARALLELISM_DIM_1-1:0],
    output logic data_out_1_valid,
    input logic data_out_1_ready
);

  split2 split_i (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid({data_out_0_valid, data_out_1_valid}),
      .data_out_ready({data_out_0_ready, data_out_1_ready})
  );

  for (genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++) begin
    always_comb begin
      data_out_0[i] = data_in_0[i];
      data_out_1[i] = data_in_0[i];
    end
  end

endmodule
