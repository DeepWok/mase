`timescale 1ns / 1ps
module self_attention_head_scatter #(
    parameter NUM_HEADS = 12,

    // * Queries, keys and values are assumed to have the same
    // * precision, dimensions and parallelism
    parameter IN_DATA_TENSOR_SIZE_DIM_0 = 64,
    parameter IN_DATA_TENSOR_SIZE_DIM_1 = 32,
    parameter IN_DATA_PARALLELISM_DIM_0 = 4,
    parameter IN_DATA_PARALLELISM_DIM_1 = 4
) (
    input logic clk,
    input logic rst,

    input  logic query_valid,
    output logic query_ready,

    input  logic key_valid,
    output logic key_ready,

    input  logic value_valid,
    output logic value_ready,

    output logic [NUM_HEADS-1:0] split_query_valid,
    input  logic [NUM_HEADS-1:0] split_query_ready,

    output logic [NUM_HEADS-1:0] split_key_valid,
    input  logic [NUM_HEADS-1:0] split_key_ready,

    output logic [NUM_HEADS-1:0] split_value_valid,
    input  logic [NUM_HEADS-1:0] split_value_ready
);

  // -----
  // Modules
  // -----

  // Instantiate QKV scatters

  self_attention_head_single_scatter #(
      .NUM_HEADS                (NUM_HEADS),
      .IN_DATA_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) q_scatter (
      .clk      (clk),
      .rst      (rst),
      .in_valid (query_valid),
      .in_ready (query_ready),
      .out_valid(split_query_valid),
      .out_ready(split_query_ready)
  );

  self_attention_head_single_scatter #(
      .NUM_HEADS                (NUM_HEADS),
      .IN_DATA_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) k_scatter (
      .clk      (clk),
      .rst      (rst),
      .in_valid (key_valid),
      .in_ready (key_ready),
      .out_valid(split_key_valid),
      .out_ready(split_key_ready)
  );

  self_attention_head_single_scatter #(
      .NUM_HEADS                (NUM_HEADS),
      .IN_DATA_TENSOR_SIZE_DIM_0(IN_DATA_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(IN_DATA_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(IN_DATA_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(IN_DATA_PARALLELISM_DIM_1)
  ) v_scatter (
      .clk      (clk),
      .rst      (rst),
      .in_valid (value_valid),
      .in_ready (value_ready),
      .out_valid(split_value_valid),
      .out_ready(split_value_ready)
  );

endmodule
