`timescale 1ns / 1ps

module fork2 #(
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 3,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_1_PRECISION_0 = 8,
    parameter DATA_OUT_1_PRECISION_1 = 3,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = -1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = -1,

    parameter DATA_OUT_1_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_OUT_1_PARALLELISM_DIM_0 = -1,
    parameter DATA_OUT_1_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_OUT_1_PARALLELISM_DIM_1 = -1,

    parameter DATA_OUT_1_FIFO_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1 / (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
    parameter DATA_OUT_0_FIFO_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1 / (DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_IN_0_PRECISION_0-1:0] data_out_0    [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready,

    output logic [DATA_IN_0_PRECISION_0-1:0] data_out_1     [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_1_valid,
    input logic data_out_1_ready
);
  logic buffered_data_out_1_valid, buffered_data_out_0_valid;
  logic buffered_data_out_1_ready, buffered_data_out_0_ready;

  split2 #() split2_inst (
      .data_out_valid({buffered_data_out_1_valid, buffered_data_out_0_valid}),
      .data_out_ready({buffered_data_out_1_ready, buffered_data_out_0_ready}),
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready)
  );
  unpacked_fifo #(
      .DEPTH(DATA_OUT_0_FIFO_DEPTH),
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
  ) data_out_0_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(buffered_data_out_0_valid),
      .data_in_ready(buffered_data_out_0_ready),  // write enable
      .data_out(data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)  // read enable
  );
  unpacked_fifo #(
      .DEPTH(DATA_OUT_1_FIFO_DEPTH),
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .IN_NUM(DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1)
  ) data_out_1_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(buffered_data_out_1_valid),
      .data_in_ready(buffered_data_out_1_ready),  // write enable
      .data_out(data_out_1),
      .data_out_valid(data_out_1_valid),
      .data_out_ready(data_out_1_ready)  // read enable
  );
endmodule
