`timescale 1ns / 1ps

/*
 * This is a workaround to use attention in single precision
 * in emitted verilog, where separate precision parameters are
 * emitted for each model submodule.
 */

module fixed_self_attention_single_precision_wrapper #(
    parameter NUM_HEADS = 12,
    parameter ACTIVATION = 0,
    parameter CHOSEN_PRECISION = "QUERY",
    parameter OUTPUT_ATTENTIONS = 0,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 128,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter QUERY_WEIGHTS_PRE_TRANSPOSED = 0,
    parameter QUERY_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter QUERY_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter QUERY_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter QUERY_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter QUERY_WEIGHT_PRECISION_0 = 16,
    parameter QUERY_WEIGHT_PRECISION_1 = 3,

    parameter KEY_WEIGHTS_PRE_TRANSPOSED = 0,
    parameter KEY_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter KEY_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter KEY_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter KEY_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter KEY_WEIGHT_PRECISION_0 = 16,
    parameter KEY_WEIGHT_PRECISION_1 = 3,

    parameter VALUE_WEIGHTS_PRE_TRANSPOSED = 0,
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter VALUE_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter VALUE_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter VALUE_WEIGHT_PRECISION_0 = 16,
    parameter VALUE_WEIGHT_PRECISION_1 = 3,

    parameter QUERY_HAS_BIAS = 0,
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter QUERY_BIAS_PARALLELISM_DIM_0 = 4,
    parameter QUERY_BIAS_PARALLELISM_DIM_1 = 4,
    parameter QUERY_BIAS_PRECISION_0 = 16,
    parameter QUERY_BIAS_PRECISION_1 = 3,

    parameter KEY_HAS_BIAS = 0,
    parameter KEY_BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter KEY_BIAS_TENSOR_SIZE_DIM_1 = 20,
    parameter KEY_BIAS_PARALLELISM_DIM_0 = 4,
    parameter KEY_BIAS_PARALLELISM_DIM_1 = 4,
    parameter KEY_BIAS_PRECISION_0 = 16,
    parameter KEY_BIAS_PRECISION_1 = 3,

    parameter VALUE_HAS_BIAS = 0,
    parameter VALUE_BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter VALUE_BIAS_TENSOR_SIZE_DIM_1 = 20,
    parameter VALUE_BIAS_PARALLELISM_DIM_0 = 4,
    parameter VALUE_BIAS_PARALLELISM_DIM_1 = 4,
    parameter VALUE_BIAS_PRECISION_0 = 16,
    parameter VALUE_BIAS_PRECISION_1 = 3,

    parameter CHOSEN_WEIGHTS_PRE_TRANSPOSED = QUERY_WEIGHTS_PRE_TRANSPOSED,
    parameter CHOSEN_WEIGHT_TENSOR_SIZE_DIM_0 = QUERY_WEIGHT_TENSOR_SIZE_DIM_0,
    parameter CHOSEN_WEIGHT_TENSOR_SIZE_DIM_1 = QUERY_WEIGHT_TENSOR_SIZE_DIM_1,
    parameter CHOSEN_WEIGHT_PARALLELISM_DIM_0 = QUERY_WEIGHT_PARALLELISM_DIM_0,
    parameter CHOSEN_WEIGHT_PARALLELISM_DIM_1 = QUERY_WEIGHT_PARALLELISM_DIM_1,
    parameter CHOSEN_WEIGHT_PRECISION_0 = QUERY_WEIGHT_PRECISION_0,
    parameter CHOSEN_WEIGHT_PRECISION_1 = QUERY_WEIGHT_PRECISION_1,
    parameter CHOSEN_HAS_BIAS = QUERY_HAS_BIAS,
    parameter CHOSEN_BIAS_TENSOR_SIZE_DIM_0 = QUERY_BIAS_TENSOR_SIZE_DIM_0,
    parameter CHOSEN_BIAS_TENSOR_SIZE_DIM_1 = QUERY_BIAS_TENSOR_SIZE_DIM_1,
    parameter CHOSEN_BIAS_PARALLELISM_DIM_0 = QUERY_BIAS_PARALLELISM_DIM_0,
    parameter CHOSEN_BIAS_PARALLELISM_DIM_1 = QUERY_BIAS_PARALLELISM_DIM_1,
    parameter CHOSEN_BIAS_PRECISION_0 = QUERY_BIAS_PRECISION_0,
    parameter CHOSEN_BIAS_PRECISION_1 = QUERY_BIAS_PRECISION_1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = CHOSEN_WEIGHT_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = DATA_IN_0_TENSOR_SIZE_DIM_2,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = CHOSEN_WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = DATA_IN_0_PARALLELISM_DIM_2,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1

) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Query weights
    input logic [QUERY_WEIGHT_PRECISION_0-1:0] query_weight [QUERY_WEIGHT_PARALLELISM_DIM_0 * QUERY_WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic query_weight_valid,
    output logic query_weight_ready,

    // Query bias
    input logic [QUERY_BIAS_PRECISION_0-1:0] query_bias [QUERY_BIAS_PARALLELISM_DIM_0 * QUERY_BIAS_PARALLELISM_DIM_1 -1:0],
    input logic query_bias_valid,
    output logic query_bias_ready,

    // Key weights
    input logic [KEY_WEIGHT_PRECISION_0-1:0] key_weight [KEY_WEIGHT_PARALLELISM_DIM_0 * KEY_WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic key_weight_valid,
    output logic key_weight_ready,

    // Key bias
    input logic [KEY_BIAS_PRECISION_0-1:0] key_bias [KEY_BIAS_PARALLELISM_DIM_0 * KEY_BIAS_PARALLELISM_DIM_1 -1:0],
    input logic key_bias_valid,
    output logic key_bias_ready,

    // Value weights
    input logic [VALUE_WEIGHT_PRECISION_0-1:0] value_weight [VALUE_WEIGHT_PARALLELISM_DIM_0 * VALUE_WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic value_weight_valid,
    output logic value_weight_ready,

    // Value bias
    input logic [VALUE_BIAS_PRECISION_0-1:0] value_bias [VALUE_BIAS_PARALLELISM_DIM_0 * VALUE_BIAS_PARALLELISM_DIM_1 -1:0],
    input logic value_bias_valid,
    output logic value_bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  fixed_self_attention #(
      .NUM_HEADS (NUM_HEADS),
      .ACTIVATION(ACTIVATION),

      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),

      .WEIGHTS_PRE_TRANSPOSED(CHOSEN_WEIGHTS_PRE_TRANSPOSED),
      .WEIGHT_TENSOR_SIZE_DIM_0(CHOSEN_WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(CHOSEN_WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(CHOSEN_WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(CHOSEN_WEIGHT_PARALLELISM_DIM_1),
      .WEIGHT_PRECISION_0(CHOSEN_WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1(CHOSEN_WEIGHT_PRECISION_1),

      .HAS_BIAS   (CHOSEN_HAS_BIAS),
      .BIAS_TENSOR_SIZE_DIM_0 (CHOSEN_BIAS_TENSOR_SIZE_DIM_0),
      .BIAS_TENSOR_SIZE_DIM_1 (CHOSEN_BIAS_TENSOR_SIZE_DIM_1),
      .BIAS_PARALLELISM_DIM_0 (CHOSEN_BIAS_PARALLELISM_DIM_0),
      .BIAS_PARALLELISM_DIM_1 (CHOSEN_BIAS_PARALLELISM_DIM_1),
      .BIAS_PRECISION_0   (CHOSEN_BIAS_PRECISION_0),
      .BIAS_PRECISION_1   (CHOSEN_BIAS_PRECISION_1),

      .DATA_OUT_0_TENSOR_SIZE_DIM_0   (DATA_OUT_0_TENSOR_SIZE_DIM_0),
      .DATA_OUT_0_TENSOR_SIZE_DIM_1   (DATA_OUT_0_TENSOR_SIZE_DIM_1),
      .DATA_OUT_0_PARALLELISM_DIM_0   (DATA_OUT_0_PARALLELISM_DIM_0),
      .DATA_OUT_0_PARALLELISM_DIM_1   (DATA_OUT_0_PARALLELISM_DIM_1),
      .DATA_OUT_0_PRECISION_0 (DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1 (DATA_OUT_0_PRECISION_1)
  ) encoder_layer_0_attention_self_inst (
      .clk(clk),
      .rst(rst),

      .data_in_0      (data_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),

      .weight_query      (query_weight),
      .weight_query_valid(query_weight_valid),
      .weight_query_ready(query_weight_ready),

      .bias_query      (query_bias),
      .bias_query_valid(query_bias_valid),
      .bias_query_ready(query_bias_ready),

      .weight_key      (key_weight),
      .weight_key_valid(key_weight_valid),
      .weight_key_ready(key_weight_ready),

      .bias_key      (key_bias),
      .bias_key_valid(key_bias_valid),
      .bias_key_ready(key_bias_ready),

      .weight_value      (value_weight),
      .weight_value_valid(value_weight_valid),
      .weight_value_ready(value_weight_ready),

      .bias_value      (value_bias),
      .bias_value_valid(value_bias_valid),
      .bias_value_ready(value_bias_ready),

      .data_out_0      (data_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );

endmodule
