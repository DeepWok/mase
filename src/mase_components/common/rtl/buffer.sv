`timescale 1ns / 1ps

module buffer #(
    /* verilator lint_off UNUSEDPARAM */
    parameter SELECT = 0,

    // Input 1
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter IN_0_DEPTH_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,
    parameter IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1,

    // Input 2
    parameter DATA_IN_1_PRECISION_0 = 16,
    parameter DATA_IN_1_PRECISION_1 = 3,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_1_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_0 = 4,  // must equal WEIGHT_PARALLELISM_DIM_1
    parameter DATA_IN_1_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_1_PARALLELISM_DIM_2 = 1,
    parameter IN_1_DEPTH_DIM_0 = DATA_IN_1_TENSOR_SIZE_DIM_0 / DATA_IN_1_PARALLELISM_DIM_0,
    parameter IN_1_DEPTH_DIM_1 = DATA_IN_1_TENSOR_SIZE_DIM_1 / DATA_IN_1_PARALLELISM_DIM_1,

    // Output 1
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = DATA_IN_0_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,

    // Chosen input
    localparam CHOSEN_DATA_IN_PRECISION_0 = (SELECT == 0) ? DATA_IN_0_PRECISION_0 : DATA_IN_1_PRECISION_0,
    localparam CHOSEN_DATA_IN_PRECISION_1 = (SELECT == 0) ? DATA_IN_0_PRECISION_1 : DATA_IN_1_PRECISION_1,
    localparam CHOSEN_DATA_IN_TENSOR_SIZE_DIM_0 = (SELECT == 0) ? DATA_IN_0_TENSOR_SIZE_DIM_0 : DATA_IN_1_TENSOR_SIZE_DIM_0,
    localparam CHOSEN_DATA_IN_TENSOR_SIZE_DIM_1 = (SELECT == 0) ? DATA_IN_0_TENSOR_SIZE_DIM_1 : DATA_IN_1_TENSOR_SIZE_DIM_1,
    localparam CHOSEN_DATA_IN_TENSOR_SIZE_DIM_2 = (SELECT == 0) ? DATA_IN_0_TENSOR_SIZE_DIM_2 : DATA_IN_1_TENSOR_SIZE_DIM_2,
    localparam CHOSEN_DATA_IN_PARALLELISM_DIM_0 = (SELECT == 0) ? DATA_IN_0_PARALLELISM_DIM_0 : DATA_IN_1_PARALLELISM_DIM_0,
    localparam CHOSEN_DATA_IN_PARALLELISM_DIM_1 = (SELECT == 0) ? DATA_IN_0_PARALLELISM_DIM_1 : DATA_IN_1_PARALLELISM_DIM_1,
    localparam CHOSEN_DATA_IN_PARALLELISM_DIM_2 = (SELECT == 0) ? DATA_IN_0_PARALLELISM_DIM_2 : DATA_IN_1_PARALLELISM_DIM_2,
    parameter CHOSEN_DEPTH_DIM_0 = (SELECT == 0) ? IN_0_DEPTH_DIM_0 : IN_1_DEPTH_DIM_0,
    parameter CHOSEN_DEPTH_DIM_1 = (SELECT == 0) ? IN_0_DEPTH_DIM_1 : IN_1_DEPTH_DIM_1
) (
    input logic clk,
    input logic rst,

    // Input 0
    input logic [CHOSEN_DATA_IN_PRECISION_0-1:0] data_in_0 [CHOSEN_DATA_IN_PARALLELISM_DIM_0*CHOSEN_DATA_IN_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  unpacked_fifo #(
      .DEPTH     (CHOSEN_DEPTH_DIM_0 * CHOSEN_DEPTH_DIM_1),
      .DATA_WIDTH(CHOSEN_DATA_IN_PRECISION_0),
      .IN_NUM    (CHOSEN_DATA_IN_PARALLELISM_DIM_0 * CHOSEN_DATA_IN_PARALLELISM_DIM_1)
  ) buffer_i (
      .clk,
      .rst,

      .data_in      (data_in_0),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),

      .data_out      (data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
