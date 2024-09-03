`timescale 1ns / 1ps

/*
 * This is a workaround to use attention in single precision
 * in emitted verilog, where separate precision parameters are
 * emitted for each model submodule.
 */

module fixed_vit_attention_single_precision_wrapper #(
    parameter NUM_HEADS = 12,
    parameter CHOSEN_PRECISION = "QUERY",

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 128,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter QUERY_WEIGHTS_PRE_TRANSPOSED = 1,
    parameter QUERY_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter QUERY_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter QUERY_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter QUERY_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter QUERY_WEIGHT_PRECISION_0 = 16,
    parameter QUERY_WEIGHT_PRECISION_1 = 3,

    parameter KEY_WEIGHTS_PRE_TRANSPOSED = 1,
    parameter KEY_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter KEY_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter KEY_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter KEY_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter KEY_WEIGHT_PRECISION_0 = 16,
    parameter KEY_WEIGHT_PRECISION_1 = 3,

    parameter VALUE_WEIGHTS_PRE_TRANSPOSED = 1,
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter VALUE_WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter VALUE_WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter VALUE_WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter VALUE_WEIGHT_PRECISION_0 = 16,
    parameter VALUE_WEIGHT_PRECISION_1 = 3,

    parameter QUERY_HAS_BIAS = 1,
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter QUERY_BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter QUERY_BIAS_PARALLELISM_DIM_0 = 4,
    parameter QUERY_BIAS_PARALLELISM_DIM_1 = 4,
    parameter QUERY_BIAS_PRECISION_0 = 16,
    parameter QUERY_BIAS_PRECISION_1 = 3,

    parameter KEY_HAS_BIAS = 1,
    parameter KEY_BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter KEY_BIAS_TENSOR_SIZE_DIM_1 = 20,
    parameter KEY_BIAS_PARALLELISM_DIM_0 = 4,
    parameter KEY_BIAS_PARALLELISM_DIM_1 = 4,
    parameter KEY_BIAS_PRECISION_0 = 16,
    parameter KEY_BIAS_PRECISION_1 = 3,

    parameter VALUE_HAS_BIAS = 1,
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

    parameter QKV_PRECISION_0 = -1,
    parameter QKV_PRECISION_1 = -1,
    parameter QKMM_OUT_PRECISION_0 = -1,
    parameter QKMM_OUT_PRECISION_1 = -1,
    parameter SOFTMAX_EXP_PRECISION_0 = -1,
    parameter SOFTMAX_EXP_PRECISION_1 = -1,
    parameter SOFTMAX_OUT_PRECISION_1 = -1,
    parameter SVMM_OUT_PRECISION_0 = -1,
    parameter SVMM_OUT_PRECISION_1 = -1,

    parameter PROJ_WEIGHT_TENSOR_SIZE_DIM_0 = -1,
    parameter PROJ_WEIGHT_TENSOR_SIZE_DIM_1 = -1,
    parameter PROJ_WEIGHT_PARALLELISM_DIM_0 = -1,
    parameter PROJ_WEIGHT_PARALLELISM_DIM_1 = -1,
    parameter PROJ_WEIGHT_PRECISION_0 = -1,
    parameter PROJ_WEIGHT_PRECISION_1 = -1,
    parameter PROJ_BIAS_TENSOR_SIZE_DIM_0 = -1,
    parameter PROJ_BIAS_TENSOR_SIZE_DIM_1 = -1,
    parameter PROJ_BIAS_PARALLELISM_DIM_0 = -1,
    parameter PROJ_BIAS_PARALLELISM_DIM_1 = -1,
    parameter PROJ_BIAS_PRECISION_0 = -1,
    parameter PROJ_BIAS_PRECISION_1 = -1,
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

    // Proj weights
    input logic [PROJ_WEIGHT_PRECISION_0-1:0] proj_weight [PROJ_WEIGHT_PARALLELISM_DIM_0 * PROJ_WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic proj_weight_valid,
    output logic proj_weight_ready,

    // Proj bias
    input logic [PROJ_BIAS_PRECISION_0-1:0] proj_bias [PROJ_BIAS_PARALLELISM_DIM_0 * PROJ_BIAS_PARALLELISM_DIM_1 -1:0],
    input logic proj_bias_valid,
    output logic proj_bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  fixed_vit_attention #(
      .NUM_HEADS(NUM_HEADS),

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
      .BIAS_PRECISION_0   (CHOSEN_BIAS_PRECISION_0),
      .BIAS_PRECISION_1   (CHOSEN_BIAS_PRECISION_1),

      .QKV_PRECISION_0(QKV_PRECISION_0),
      .QKV_PRECISION_1(QKV_PRECISION_1),
      .QKMM_OUT_PRECISION_0(QKMM_OUT_PRECISION_0),
      .QKMM_OUT_PRECISION_1(QKMM_OUT_PRECISION_1),
      .SOFTMAX_EXP_PRECISION_0(SOFTMAX_EXP_PRECISION_0),
      .SOFTMAX_EXP_PRECISION_1(SOFTMAX_EXP_PRECISION_1),
      .SOFTMAX_OUT_DATA_PRECISION_1(SOFTMAX_OUT_PRECISION_1),
      .SVMM_OUT_PRECISION_0(SVMM_OUT_PRECISION_0),
      .SVMM_OUT_PRECISION_1(SVMM_OUT_PRECISION_1),

      .WEIGHT_PROJ_TENSOR_SIZE_DIM_0(PROJ_WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_PROJ_TENSOR_SIZE_DIM_1(PROJ_WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PROJ_PARALLELISM_DIM_0(PROJ_WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PROJ_PARALLELISM_DIM_1(PROJ_WEIGHT_PARALLELISM_DIM_1),
      .WEIGHT_PROJ_PRECISION_0(PROJ_WEIGHT_PRECISION_0),
      .WEIGHT_PROJ_PRECISION_1(PROJ_WEIGHT_PRECISION_1),
      .BIAS_PROJ_PRECISION_0(PROJ_BIAS_PRECISION_0),
      .BIAS_PROJ_PRECISION_1(PROJ_BIAS_PRECISION_1),


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

      .query_weight      (query_weight),
      .query_weight_valid(query_weight_valid),
      .query_weight_ready(query_weight_ready),

      .query_bias      (query_bias),
      .query_bias_valid(query_bias_valid),
      .query_bias_ready(query_bias_ready),

      .key_weight      (key_weight),
      .key_weight_valid(key_weight_valid),
      .key_weight_ready(key_weight_ready),

      .key_bias      (key_bias),
      .key_bias_valid(key_bias_valid),
      .key_bias_ready(key_bias_ready),

      .value_weight      (value_weight),
      .value_weight_valid(value_weight_valid),
      .value_weight_ready(value_weight_ready),

      .value_bias      (value_bias),
      .value_bias_valid(value_bias_valid),
      .value_bias_ready(value_bias_ready),

      .proj_weight(proj_weight),
      .proj_weight_valid(proj_weight_valid),
      .proj_weight_ready(proj_weight_ready),

      .proj_bias(proj_bias),
      .proj_bias_valid(proj_bias_valid),
      .proj_bias_ready(proj_bias_ready),

      .data_out_0      (data_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );

endmodule
