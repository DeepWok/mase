`timescale 1ns / 1ps
module mxint_vit_attention_input_block_batched #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,

    parameter HAS_BIAS = 1,
    parameter BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = 4,
    parameter BIAS_PARALLELISM_DIM_1 = 1,
    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0 = 16,
    parameter DATA_OUT_0_PRECISION_1 = 3

) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Query weights
    input logic [WEIGHT_PRECISION_0-1:0] mweight_query [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight_query,
    input logic weight_query_valid,
    output logic weight_query_ready,

    // Query bias
    input logic [BIAS_PRECISION_0-1:0] mbias_query [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] ebias_query,
    input logic bias_query_valid,
    output logic bias_query_ready,

    // Key weights
    input logic [WEIGHT_PRECISION_0-1:0] mweight_key [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight_key,
    input logic weight_key_valid,
    output logic weight_key_ready,

    // Key bias
    input logic [BIAS_PRECISION_0-1:0] mbias_key [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] ebias_key,
    input logic bias_key_valid,
    output logic bias_key_ready,

    // Value weights
    input logic [WEIGHT_PRECISION_0-1:0] mweight_value [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] eweight_value,
    input logic weight_value_valid,
    output logic weight_value_ready,

    // Value bias
    input logic [BIAS_PRECISION_0-1:0] mbias_value [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] ebias_value,
    input logic bias_value_valid,
    output logic bias_value_ready,

    // Query
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_query [DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_query,
    output logic data_out_query_valid,
    input logic data_out_query_ready,

    // Key
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_key [DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_key,
    output logic data_out_key_valid,
    input logic data_out_key_ready,

    // Value
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_value [DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_0-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_value,
    output logic data_out_value_valid,
    input logic data_out_value_ready
);

  // ! TO DO: add assertions about bias parallelism matching weight parallelism

  // * Inferred parameters
  parameter DATA_IN_0_DEPTH_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1 / DATA_IN_0_PARALLELISM_DIM_1;
  parameter WEIGHT_DEPTH_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_0 / WEIGHT_PARALLELISM_DIM_0;

  // * Declarations
  // * =================================================================

  logic query_data_in_valid, query_data_in_ready;
  logic key_data_in_valid, key_data_in_ready;
  logic value_data_in_valid, value_data_in_ready;

  logic [DATA_OUT_0_PRECISION_0-1:0] query_buffer_mdata [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_1-1:0];
  logic [DATA_OUT_0_PRECISION_1-1:0] query_buffer_edata;
  logic query_buffer_valid;
  logic query_buffer_ready;

  // * Instances
  // * =================================================================

  // * Split the incoming data over the QKV projections
  split_n #(
      .N(3)
  ) split_i (
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .data_out_valid({query_data_in_valid, key_data_in_valid, value_data_in_valid}),
      .data_out_ready({query_data_in_ready, key_data_in_ready, value_data_in_ready})
  );

  // * Query linear

  mxint_linear #(
      .HAS_BIAS              (HAS_BIAS),

      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),

      .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1),

      .BIAS_PRECISION_0      (BIAS_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PRECISION_1),
      .BIAS_TENSOR_SIZE_DIM_0(BIAS_TENSOR_SIZE_DIM_0),
      .BIAS_TENSOR_SIZE_DIM_1(BIAS_TENSOR_SIZE_DIM_1),
      .BIAS_PARALLELISM_DIM_0(BIAS_PARALLELISM_DIM_0),
      .BIAS_PARALLELISM_DIM_1(BIAS_PARALLELISM_DIM_1),

      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)

  ) mxint_linear_query (
      .clk,
      .rst,

      // input port for data_inivations
      .mdata_in_0      (mdata_in_0),
      .edata_in_0      (edata_in_0),
      .data_in_0_valid(query_data_in_valid),
      .data_in_0_ready(query_data_in_ready),

      // input port for weight
      .mweight      (mweight_query),
      .eweight      (eweight_query),
      .weight_valid(weight_query_valid),
      .weight_ready(weight_query_ready),

      .mbias      (mbias_query),
      .ebias      (ebias_query),
      .bias_valid(bias_query_valid),
      .bias_ready(bias_query_ready),

      .mdata_out_0      (query_buffer_mdata),
      .edata_out_0      (query_buffer_edata),
      .data_out_0_valid(query_buffer_valid),
      .data_out_0_ready(query_buffer_ready)
  );

  // * We must buffer the queries to latency match the key transpose path
  // * since the matmul for QK^T buffers K^T but streams Q
  unpacked_mx_fifo #(
      .MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .IN_SIZE(DATA_OUT_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1),
      .DEPTH(DATA_IN_0_DEPTH_DIM_1 * DATA_OUT_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_PARALLELISM_DIM_0)
  ) query_buffer_i (
      .clk(clk),
      .rst(rst),
      .mdata_in(query_buffer_mdata),
      .edata_in(query_buffer_edata),
      .data_in_valid(query_buffer_valid),
      .data_in_ready(query_buffer_ready),
      .mdata_out(mdata_out_query),
      .edata_out(edata_out_query),
      .data_out_valid(data_out_query_valid),
      .data_out_ready(data_out_query_ready)
  );

  // * Key linear

  mxint_linear #(
      .HAS_BIAS              (HAS_BIAS),

      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),

      .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1),

      .BIAS_PRECISION_0      (BIAS_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PRECISION_1),
      .BIAS_TENSOR_SIZE_DIM_0(BIAS_TENSOR_SIZE_DIM_0),
      .BIAS_TENSOR_SIZE_DIM_1(BIAS_TENSOR_SIZE_DIM_1),
      .BIAS_PARALLELISM_DIM_0(BIAS_PARALLELISM_DIM_0),
      .BIAS_PARALLELISM_DIM_1(BIAS_PARALLELISM_DIM_1),

      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)

  ) mxint_linear_key (
      .clk,
      .rst,

      // input port for data_inivations
      .mdata_in_0      (mdata_in_0),
      .edata_in_0      (edata_in_0),
      .data_in_0_valid(key_data_in_valid),
      .data_in_0_ready(key_data_in_ready),

      // input port for weight
      .mweight      (mweight_key),
      .eweight      (eweight_key),
      .weight_valid(weight_key_valid),
      .weight_ready(weight_key_ready),

      .mbias      (mbias_key),
      .ebias      (ebias_key),
      .bias_valid(bias_key_valid),
      .bias_ready(bias_key_ready),

      .mdata_out_0      (mdata_out_key),
      .edata_out_0      (edata_out_key),
      .data_out_0_valid(data_out_key_valid),
      .data_out_0_ready(data_out_key_ready)
  );

  // * Value linear

  mxint_linear #(
      .HAS_BIAS              (HAS_BIAS),

      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),

      .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1),

      .BIAS_PRECISION_0      (BIAS_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PRECISION_1),
      .BIAS_TENSOR_SIZE_DIM_0(BIAS_TENSOR_SIZE_DIM_0),
      .BIAS_TENSOR_SIZE_DIM_1(BIAS_TENSOR_SIZE_DIM_1),
      .BIAS_PARALLELISM_DIM_0(BIAS_PARALLELISM_DIM_0),
      .BIAS_PARALLELISM_DIM_1(BIAS_PARALLELISM_DIM_1),

      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)

  ) mxint_linear_value (
      .clk,
      .rst,

      // input port for data_inivations
      .mdata_in_0      (mdata_in_0),
      .edata_in_0      (edata_in_0),
      .data_in_0_valid(value_data_in_valid),
      .data_in_0_ready(value_data_in_ready),

      // input port for weight
      .mweight      (mweight_value),
      .eweight      (eweight_value),
      .weight_valid(weight_value_valid),
      .weight_ready(weight_value_ready),

      .mbias      (mbias_value),
      .ebias      (ebias_value),
      .bias_valid(bias_value_valid),
      .bias_ready(bias_value_ready),

      .mdata_out_0      (mdata_out_value),
      .edata_out_0      (edata_out_value),
      .data_out_0_valid(data_out_value_valid),
      .data_out_0_ready(data_out_value_ready)
  );

endmodule
