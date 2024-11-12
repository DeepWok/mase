`timescale 1ns / 1ps
module mxint_vit_attention #(
    // currently assume weights are all transposed
    // currently force weight dim keep same

    parameter NUM_HEADS = 4,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 8,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 8,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 3,

    parameter HAS_BIAS = 1,
    parameter BIAS_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1,
    parameter BIAS_PARALLELISM_DIM_1 = 1,
    parameter BIAS_PRECISION_0 = 8,
    parameter BIAS_PRECISION_1 = 3,

    parameter QKV_PRECISION_0 = 16,
    parameter QKV_PRECISION_1 = 3,
    parameter QKMM_OUT_PRECISION_0 = 16,
    parameter QKMM_OUT_PRECISION_1 = 3,
    parameter SOFTMAX_EXP_PRECISION_0 = 16,
    parameter SOFTMAX_EXP_PRECISION_1 = 3,
    parameter SOFTMAX_OUT_DATA_PRECISION_1 = 3,
    parameter SVMM_OUT_PRECISION_0 = 8,
    parameter SVMM_OUT_PRECISION_1 = 3,

    parameter WEIGHT_PROJ_PRECISION_0 = 12,
    parameter WEIGHT_PROJ_PRECISION_1 = 3,

    parameter WEIGHT_PROJ_TENSOR_SIZE_DIM_0 = 8,
    parameter WEIGHT_PROJ_TENSOR_SIZE_DIM_1 = 8,
    parameter WEIGHT_PROJ_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PROJ_PARALLELISM_DIM_1 = 4,

    parameter BIAS_PROJ_PRECISION_0 = 8,
    parameter BIAS_PROJ_PRECISION_1 = 3,
    parameter BIAS_PROJ_TENSOR_SIZE_DIM_0 = WEIGHT_PROJ_TENSOR_SIZE_DIM_1,
    parameter BIAS_PROJ_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_PROJ_PARALLELISM_DIM_0 = WEIGHT_PROJ_PARALLELISM_DIM_1,
    parameter BIAS_PROJ_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_PROJ_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PROJ_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1

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
    input logic query_weight_valid,
    output logic query_weight_ready,

    // Query bias
    input logic [BIAS_PRECISION_0-1:0] mquery_bias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] equery_bias,
    input logic query_bias_valid,
    output logic query_bias_ready,

    // Key weights
    input logic [WEIGHT_PRECISION_0-1:0] mkey_weight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] ekey_weight,
    input logic key_weight_valid,
    output logic key_weight_ready,

    // Key bias
    input logic [BIAS_PRECISION_0-1:0] mkey_bias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] ekey_bias,
    input logic key_bias_valid,
    output logic key_bias_ready,

    // Value weights
    input logic [WEIGHT_PRECISION_0-1:0] mvalue_weight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PRECISION_1-1:0] evalue_weight,
    input logic value_weight_valid,
    output logic value_weight_ready,

    // Value bias
    input logic [BIAS_PRECISION_0-1:0] mvalue_bias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic [BIAS_PRECISION_1-1:0] evalue_bias,
    input logic value_bias_valid,
    output logic value_bias_ready,

    // Proj weights
    input logic [WEIGHT_PROJ_PRECISION_0-1:0] mproj_weight [WEIGHT_PROJ_PARALLELISM_DIM_0 * WEIGHT_PROJ_PARALLELISM_DIM_1-1:0],
    input logic [WEIGHT_PROJ_PRECISION_1-1:0] eproj_weight,
    input logic proj_weight_valid,
    output logic proj_weight_ready,

    // Proj bias
    input logic [BIAS_PROJ_PRECISION_0-1:0] proj_bias [BIAS_PROJ_PARALLELISM_DIM_0 * BIAS_PROJ_PARALLELISM_DIM_1 -1:0],
    input logic proj_bias_valid,
    output logic proj_bias_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  // * Declarations
  // * =================================================================

  localparam HEAD_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_1;
  localparam HEAD_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1;
  localparam HEAD_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_1;
  localparam HEAD_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1;
  // Query
  logic [QKV_PRECISION_0-1:0] query[DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [QKV_PRECISION_1-1:0] equery;
  logic joint_query_valid, joint_query_ready;
  logic [NUM_HEADS-1:0] split_query_valid, split_query_ready;

  // Key
  logic [QKV_PRECISION_0-1:0] key[DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [QKV_PRECISION_1-1:0] ekey;
  logic joint_key_valid, joint_key_ready;
  logic [NUM_HEADS-1:0] split_key_valid, split_key_ready;

  // Value
  logic [QKV_PRECISION_0-1:0] value[DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [QKV_PRECISION_1-1:0] evalue;
  logic joint_value_valid, joint_value_ready;
  logic [NUM_HEADS-1:0] split_value_valid, split_value_ready;

  logic [QKV_PRECISION_0-1:0] fifo_key[DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0-1:0];
  logic fifo_key_valid, fifo_key_ready;
  logic [QKV_PRECISION_0-1:0] mfifo_value[DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0-1:0];
  logic [QKV_PRECISION_1-1:0] efifo_value;
  logic fifo_value_valid, fifo_value_ready;

  // Head output
  logic [SVMM_OUT_PRECISION_0-1:0] mhead_out [NUM_HEADS-1:0] [HEAD_OUT_0_PARALLELISM_DIM_0 * HEAD_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [QKV_PRECISION_1-1:0] ehead_out [NUM_HEADS-1:0];
  logic [NUM_HEADS-1:0] head_out_valid;
  logic [NUM_HEADS-1:0] head_out_ready;

  logic [SVMM_OUT_PRECISION_0-1:0] proj_in [HEAD_OUT_0_PARALLELISM_DIM_0 * HEAD_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [SVMM_OUT_PRECISION_1-1:0] eproj_in; // Add this signal declaration
  logic proj_in_valid, proj_in_ready;

  // * Instances
  // * =================================================================

  mxint_vit_attention_input_block_batched #(
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),

      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1),
      .WEIGHT_PRECISION_0      (WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PRECISION_1),

      .HAS_BIAS              (HAS_BIAS),
      .BIAS_TENSOR_SIZE_DIM_0(BIAS_TENSOR_SIZE_DIM_0),
      .BIAS_TENSOR_SIZE_DIM_1(BIAS_TENSOR_SIZE_DIM_1),
      .BIAS_PARALLELISM_DIM_0(BIAS_PARALLELISM_DIM_0),
      .BIAS_PARALLELISM_DIM_1(BIAS_PARALLELISM_DIM_1),
      .BIAS_PRECISION_0      (BIAS_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PRECISION_1),

      .DATA_OUT_0_PRECISION_0(QKV_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(QKV_PRECISION_1)
  ) batched_input_block_i (
      .clk(clk),
      .rst(rst),

      .mdata_in_0(mdata_in_0),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),

      // Query parameters
      .mweight_query(mweight_query),
      .eweight_query(eweight_query),
      .weight_query_valid(query_weight_valid),
      .weight_query_ready(query_weight_ready),

      .mquery_bias(mquery_bias),
      .equery_bias(equery_bias),
      .query_bias_valid(query_bias_valid),
      .query_bias_ready(query_bias_ready),

      // Key parameters
      .mweight_key(mkey_weight),
      .eweight_key(ekey_weight),
      .weight_key_valid(key_weight_valid),
      .weight_key_ready(key_weight_ready),

      .mkey_bias(mkey_bias),
      .ekey_bias(ekey_bias),
      .key_bias_valid(key_bias_valid),
      .key_bias_ready(key_bias_ready),

      // Value parameters
      .mweight_value(mvalue_weight),
      .eweight_value(evalue_weight),
      .weight_value_valid(value_weight_valid),
      .weight_value_ready(value_weight_ready),

      .mvalue_bias(mvalue_bias),
      .evalue_bias(evalue_bias),
      .value_bias_valid(value_bias_valid),
      .value_bias_ready(value_bias_ready),

      // Query output
      .mdata_out_query(query),
      .edata_out_query(equery),
      .data_out_query_valid(joint_query_valid),
      .data_out_query_ready(joint_query_ready),

      // Key output
      .mdata_out_key(key),
      .edata_out_key(ekey),
      .data_out_key_valid(joint_key_valid),
      .data_out_key_ready(joint_key_ready),

      // Value output
      .mdata_out_value(mfifo_value),
      .edata_out_value(efifo_value),
      .data_out_value_valid(fifo_value_valid),
      .data_out_value_ready(fifo_value_ready)
  );

  unpacked_mx_fifo #(
      .DEPTH(DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1 / (DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0)),
      .MAN_WIDTH(QKV_PRECISION_0),
      .EXP_WIDTH(QKV_PRECISION_1),
      .IN_SIZE(DATA_IN_0_PARALLELISM_DIM_1 * HEAD_OUT_0_PARALLELISM_DIM_0)
  ) value_in_buffer (
      .clk(clk),
      .rst(rst),
      .mdata_in(mfifo_value),
      .edata_in(efifo_value),
      .data_in_valid(fifo_value_valid),
      .data_in_ready(fifo_value_ready),
      .mdata_out(value),
      .edata_out(evalue),
      .data_out_valid(joint_value_valid),
      .data_out_ready(joint_value_ready)
  );

  // * Scatter query, key, value

  self_attention_head_scatter #(
      .NUM_HEADS(NUM_HEADS),

      .IN_DATA_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1)

  ) scatter_qkv_i (
      .clk,
      .rst,

      .query_valid(joint_query_valid),
      .query_ready(joint_query_ready),

      .key_valid(joint_key_valid),
      .key_ready(joint_key_ready),

      .value_valid(joint_value_valid),
      .value_ready(joint_value_ready),

      .split_query_valid(split_query_valid),
      .split_query_ready(split_query_ready),

      .split_key_valid(split_key_valid),
      .split_key_ready(split_key_ready),

      .split_value_valid(split_value_valid),
      .split_value_ready(split_value_ready)
  );

  // * Heads

  for (genvar head = 0; head < NUM_HEADS; head++) begin : g_attention_head
    mxint_vit_attention_head #(
        .IN_DATA_TENSOR_SIZE_DIM_0   (HEAD_OUT_0_TENSOR_SIZE_DIM_0 / NUM_HEADS),
        .IN_DATA_TENSOR_SIZE_DIM_1   (HEAD_OUT_0_TENSOR_SIZE_DIM_1),
        .IN_DATA_PARALLELISM_DIM_0   (HEAD_OUT_0_PARALLELISM_DIM_0),
        .IN_DATA_PARALLELISM_DIM_1   (HEAD_OUT_0_PARALLELISM_DIM_1),
        .IN_DATA_PRECISION_0         (QKV_PRECISION_0),
        .IN_DATA_PRECISION_1         (QKV_PRECISION_1),
        .QKMM_OUT_PRECISION_0        (QKMM_OUT_PRECISION_0),
        .QKMM_OUT_PRECISION_1        (QKMM_OUT_PRECISION_1),
        .SOFTMAX_EXP_PRECISION_0     (SOFTMAX_EXP_PRECISION_0),
        .SOFTMAX_EXP_PRECISION_1     (SOFTMAX_EXP_PRECISION_1),
        .SOFTMAX_OUT_DATA_PRECISION_1(SOFTMAX_OUT_DATA_PRECISION_1),
        .OUT_DATA_PRECISION_0        (SVMM_OUT_PRECISION_0),
        .OUT_DATA_PRECISION_1        (SVMM_OUT_PRECISION_1)
    ) head_i (
        .clk,
        .rst,

        .mquery      (query),
        .equery      (equery),
        .query_valid(split_query_valid[head]),
        .query_ready(split_query_ready[head]),

        .mkey      (key),
        .ekey      (ekey),
        .key_valid(split_key_valid[head]),
        .key_ready(split_key_ready[head]),

        .mvalue      (value),
        .evalue      (evalue),
        .value_valid(split_value_valid[head]),
        .value_ready(split_value_ready[head]),

        .mout      (mhead_out[head]),
        .eout      (ehead_out[head]),
        .out_valid(head_out_valid[head]),
        .out_ready(head_out_ready[head])
    );
  end

  // * Gather heads

  mxint_vit_attention_head_gather #(
      .NUM_HEADS(NUM_HEADS),
      .IN_DATA_TENSOR_SIZE_DIM_0(HEAD_OUT_0_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(HEAD_OUT_0_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(HEAD_OUT_0_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(HEAD_OUT_0_PARALLELISM_DIM_1),
      .MAN_WIDTH(SVMM_OUT_PRECISION_0),
      .EXP_WIDTH(SVMM_OUT_PRECISION_1)
  ) gather_qkv_i (
      .clk,
      .rst,
      .msplit_head_out(mhead_out),
      .esplit_head_out(ehead_out),
      .split_head_out_valid(head_out_valid),
      .split_head_out_ready(head_out_ready),
      .mupdated_tokens(proj_in),
      .eupdated_tokens(eproj_in),
      .updated_tokens_valid(proj_in_valid),
      .updated_tokens_ready(proj_in_ready)
  );

  mxint_linear #(
      .HAS_BIAS              (HAS_BIAS),

      .DATA_IN_0_PRECISION_0      (SVMM_OUT_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (SVMM_OUT_PRECISION_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(HEAD_OUT_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(HEAD_OUT_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(HEAD_OUT_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(HEAD_OUT_0_PARALLELISM_DIM_1),

      .WEIGHT_PRECISION_0      (WEIGHT_PROJ_PRECISION_0),
      .WEIGHT_PRECISION_1      (WEIGHT_PROJ_PRECISION_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_PROJ_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_PROJ_TENSOR_SIZE_DIM_1),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PROJ_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PROJ_PARALLELISM_DIM_1),

      .BIAS_PRECISION_0      (BIAS_PROJ_PRECISION_0),
      .BIAS_PRECISION_1      (BIAS_PROJ_PRECISION_1),
      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
  ) proj (
      .clk(clk),
      .rst(rst),

      // input port for data_inivations
      .mdata_in_0      (proj_in),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(proj_in_valid),
      .data_in_0_ready(proj_in_ready),

      // input port for weight
      .mweight      (mproj_weight),
      .eweight(eproj_weight),
      .weight_valid(proj_weight_valid),
      .weight_ready(proj_weight_ready),

      .mbias      (proj_bias),
      .ebias(ebias_proj),
      .bias_valid(proj_bias_valid),
      .bias_ready(proj_bias_ready),

      .mdata_out_0(mdata_out_0),
      .edata_out_0(edata_out_0),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready)
  );
endmodule
