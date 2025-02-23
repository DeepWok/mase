/*
Module      : fixed_gqa_wrapper
Description : This module wrapps the fixed_grouped_query_attention module and
              implements the interface required for the MASE software stack.
*/

`timescale 1ns / 1ps

module fixed_gqa_wrapper #(
    parameter NUM_HEADS = -1,
    parameter NUM_GROUPS = -1,
    parameter WEIGHTS_PRE_TRANSPOSED = -1,
    parameter HAS_BIAS = -1,

    parameter DATA_IN_0_PRECISION_0 = -1,
    parameter DATA_IN_0_PRECISION_1 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = -1,

    parameter Q_PROJECTION_WEIGHT_PRECISION_0 = -1,
    parameter Q_PROJECTION_WEIGHT_PRECISION_1 = -1,
    parameter Q_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter Q_PROJECTION_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter Q_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter Q_PROJECTION_WEIGHT_PARALLELISM_DIM_1 = -1,

    parameter K_PROJECTION_WEIGHT_PRECISION_0 = -1,
    parameter K_PROJECTION_WEIGHT_PRECISION_1 = -1,
    parameter K_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter K_PROJECTION_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter K_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter K_PROJECTION_WEIGHT_PARALLELISM_DIM_1 = -1,

    parameter V_PROJECTION_WEIGHT_PRECISION_0 = -1,
    parameter V_PROJECTION_WEIGHT_PRECISION_1 = -1,
    parameter V_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter V_PROJECTION_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter V_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter V_PROJECTION_WEIGHT_PARALLELISM_DIM_1 = -1,

    parameter O_PROJECTION_WEIGHT_PRECISION_0 = -1,
    parameter O_PROJECTION_WEIGHT_PRECISION_1 = -1,
    parameter O_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter O_PROJECTION_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter O_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter O_PROJECTION_WEIGHT_PARALLELISM_DIM_1 = -1,

    parameter DATA_OUT_0_PRECISION_0 = -1,
    parameter DATA_OUT_0_PRECISION_1 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = -1,

    localparam DATA_IN_0_PARALLELISM = DATA_IN_0_PARALLELISM_DIM_0 *
                                       DATA_IN_0_PARALLELISM_DIM_1 *
                                       DATA_IN_0_PARALLELISM_DIM_2,

    localparam Q_PROJECTION_WEIGHT_PARALLELISM = Q_PROJECTION_WEIGHT_PARALLELISM_DIM_0 *
                                                 Q_PROJECTION_WEIGHT_PARALLELISM_DIM_1,

    localparam K_PROJECTION_WEIGHT_PARALLELISM = K_PROJECTION_WEIGHT_PARALLELISM_DIM_0 *
                                                 K_PROJECTION_WEIGHT_PARALLELISM_DIM_1,

    localparam V_PROJECTION_WEIGHT_PARALLELISM = V_PROJECTION_WEIGHT_PARALLELISM_DIM_0 *
                                                 V_PROJECTION_WEIGHT_PARALLELISM_DIM_1,

    localparam O_PROJECTION_WEIGHT_PARALLELISM = O_PROJECTION_WEIGHT_PARALLELISM_DIM_0 *
                                                 O_PROJECTION_WEIGHT_PARALLELISM_DIM_1,

    localparam DATA_OUT_0_PARALLELISM = DATA_OUT_0_PARALLELISM_DIM_0 *
                                        DATA_OUT_0_PARALLELISM_DIM_1 *
                                        DATA_OUT_0_PARALLELISM_DIM_2
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0      [DATA_IN_0_PARALLELISM-1:0],
    input  logic                             data_in_0_valid,
    output logic                             data_in_0_ready,

    input  logic [Q_PROJECTION_WEIGHT_PRECISION_0-1:0] q_projection_weight [Q_PROJECTION_WEIGHT_PARALLELISM-1:0],
    input logic q_projection_weight_valid,
    output logic q_projection_weight_ready,

    input  logic [K_PROJECTION_WEIGHT_PRECISION_0-1:0] k_projection_weight [K_PROJECTION_WEIGHT_PARALLELISM-1:0],
    input logic k_projection_weight_valid,
    output logic k_projection_weight_ready,

    input  logic [V_PROJECTION_WEIGHT_PRECISION_0-1:0] v_projection_weight [V_PROJECTION_WEIGHT_PARALLELISM-1:0],
    input logic v_projection_weight_valid,
    output logic v_projection_weight_ready,

    input  logic [O_PROJECTION_WEIGHT_PRECISION_0-1:0] o_projection_weight [O_PROJECTION_WEIGHT_PARALLELISM-1:0],
    input logic o_projection_weight_valid,
    output logic o_projection_weight_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0      [DATA_OUT_0_PARALLELISM-1:0],
    output logic                              data_out_0_valid,
    input  logic                              data_out_0_ready
);

  fixed_grouped_query_attention #(
      .NUM_HEADS(NUM_HEADS),
      .NUM_GROUPS(NUM_GROUPS),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
      // Only specify Q weight dims, KV can be derived using num_groups
      .WEIGHT_TENSOR_SIZE_DIM_0(Q_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(Q_PROJECTION_WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(Q_PROJECTION_WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(Q_PROJECTION_WEIGHT_PARALLELISM_DIM_1),
      .WEIGHT_PRECISION_0(Q_PROJECTION_WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1(Q_PROJECTION_WEIGHT_PRECISION_1),
      .WEIGHTS_PRE_TRANSPOSED(WEIGHTS_PRE_TRANSPOSED),
      .HAS_BIAS(0)  // TODO: enable when we support bias
  ) gqa_inst (
      .clk(clk),
      .rst(rst),

      .data_in_0      (data_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),

      .weight_query      (q_projection_weight),
      .weight_query_valid(q_projection_weight_valid),
      .weight_query_ready(q_projection_weight_ready),
      .bias_query        (  /* unconnected */),
      .bias_query_valid  ('0),
      .bias_query_ready  (  /* unconnected */),

      .weight_key      (k_projection_weight),
      .weight_key_valid(k_projection_weight_valid),
      .weight_key_ready(k_projection_weight_ready),
      .bias_key        (  /* unconnected */),
      .bias_key_valid  ('0),
      .bias_key_ready  (  /* unconnected */),

      .weight_value      (v_projection_weight),
      .weight_value_valid(v_projection_weight_valid),
      .weight_value_ready(v_projection_weight_ready),
      .bias_value        (  /* unconnected */),
      .bias_value_valid  ('0),
      .bias_value_ready  (  /* unconnected */),

      .weight_output      (o_projection_weight),
      .weight_output_valid(o_projection_weight_valid),
      .weight_output_ready(o_projection_weight_ready),
      .bias_output        (  /* unconnected */),
      .bias_output_valid  ('0),
      .bias_output_ready  (  /* unconnected */),

      .data_out_0      (data_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );

endmodule
