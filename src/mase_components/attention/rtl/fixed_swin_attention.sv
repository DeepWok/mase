`timescale 1ns / 1ps
module fixed_swin_attention #(
    parameter NUM_HEADS  = 12,
    parameter ACTIVATION = 0,

    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_0 = 768,
    parameter DATA_IN_0_MAX_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter WEIGHTS_PRE_TRANSPOSED = 0,
    parameter WEIGHT_MAX_TENSOR_SIZE_DIM_0 = 768,
    parameter WEIGHT_MAX_TENSOR_SIZE_DIM_1 = 768,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,

    parameter HAS_BIAS = 1,
    parameter BIAS_MAX_TENSOR_SIZE_DIM_0 = 64,
    parameter BIAS_MAX_TENSOR_SIZE_DIM_1 = 20,
    parameter BIAS_PARALLELISM_DIM_0 = 4,
    parameter BIAS_PARALLELISM_DIM_1 = 4,
    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3,

    parameter POS_EMBED_SIZE_DIM_0 = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter POS_EMBED_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_0*2 - 1,
    parameter POS_EMBED_PARALLELISM_DIM_0 = 1,
    parameter POS_EMBED_PARALLELISM_DIM_1 = 1,
    parameter POS_EMBED_PRECISION_0 = 16,
    parameter POS_EMBED_PRECISION_1 = 3,


    parameter DATA_OUT_0_MAX_TENSOR_SIZE_DIM_0 = WEIGHT_MAX_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_MAX_TENSOR_SIZE_DIM_1 = DATA_IN_0_MAX_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1

) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Query weights
    input logic [WEIGHT_PRECISION_0-1:0] weight_query [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_query_valid,
    output logic weight_query_ready,

    // Query bias
    input logic [BIAS_PRECISION_0-1:0] bias_query [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_query_valid,
    output logic bias_query_ready,

    // Content bias
    input logic [BIAS_PRECISION_0-1:0] bias_con [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_con_valid,
    output logic bias_con_ready,

    // Positional bias
    input logic [BIAS_PRECISION_0-1:0] bias_pos [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_pos_valid,
    output logic bias_pos_ready,

    // Key weights
    input logic [WEIGHT_PRECISION_0-1:0] weight_key [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_key_valid,
    output logic weight_key_ready,

    // Key bias
    input logic [BIAS_PRECISION_0-1:0] bias_key [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_key_valid,
    output logic bias_key_ready,

    // Value weights
    input logic [WEIGHT_PRECISION_0-1:0] weight_value [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_value_valid,
    output logic weight_value_ready,

    // Value bias
    input logic [BIAS_PRECISION_0-1:0] bias_value [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_value_valid,
    output logic bias_value_ready,

    input logic [WEIGHT_PRECISION_0-1:0] pos_embed [POS_EMBED_PARALLELISM_DIM_0 * POS_EMBED_PARALLELISM_DIM_1-1:0],
    input logic pos_embed_valid,
    output logic pos_embed_ready,
    

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  // * Declarations
  // * =================================================================

  // Query
  logic [DATA_OUT_0_PRECISION_0-1:0] query[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
  logic joint_query_valid, joint_query_ready;
  logic [NUM_HEADS-1:0] split_query_valid, split_query_ready;

  // Key
  logic [DATA_OUT_0_PRECISION_0-1:0] key[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
  logic joint_key_valid, joint_key_ready;
  logic [NUM_HEADS-1:0] split_key_valid, split_key_ready;

  // Value
  logic [DATA_OUT_0_PRECISION_0-1:0] value[DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
  logic joint_value_valid, joint_value_ready;
  logic [NUM_HEADS-1:0] split_value_valid, split_value_ready;

  // Head output
  logic [DATA_OUT_0_PRECISION_0-1:0] head_out [NUM_HEADS-1:0] [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic [NUM_HEADS-1:0] head_out_valid;
  logic [NUM_HEADS-1:0] head_out_ready;

  // Qpos output
  logic [DATA_OUT_0_PRECISION_0-1:0] query_pos [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic query_pos_valid, query_pos_ready;
  logic [NUM_HEADS-1:0] split_query_pos_valid, split_query_pos_ready;

  // Qcon output
  logic [DATA_OUT_0_PRECISION_0-1:0] query_con [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  logic query_con_valid, query_con_ready;
  logic [NUM_HEADS-1:0] split_query_con_valid, split_query_con_ready;

  logic [NUM_HEADS-1:0] split_pos_embed_valid, split_pos_embed_ready;

  // Qcon output
  //logic [DATA_OUT_0_PRECISION_0-1:0] query_con_out [NUM_HEADS-1:0] [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0];
  //logic [NUM_HEADS-1:0] query_con_valid;
  //logic [NUM_HEADS-1:0] query_con_ready;



  // * Instances
  // * =================================================================

  fixed_self_attention_input_block_batched #(
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
      .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
      .DATA_IN_0_PRECISION_0      (DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1      (DATA_IN_0_PRECISION_1),

      .WEIGHTS_PRE_TRANSPOSED  (WEIGHTS_PRE_TRANSPOSED),
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

      .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
      .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1)
  ) batched_input_block_i (
      .clk(clk),
      .rst(rst),

      .data_in_0(data_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),

      // Query parameters
      .weight_query(weight_query),
      .weight_query_valid(weight_query_valid),
      .weight_query_ready(weight_query_ready),

      .bias_query(bias_query),
      .bias_query_valid(bias_query_valid),
      .bias_query_ready(bias_query_ready),

      // Key parameters
      .weight_key(weight_key),
      .weight_key_valid(weight_key_valid),
      .weight_key_ready(weight_key_ready),

      .bias_key(bias_key),
      .bias_key_valid(bias_key_valid),
      .bias_key_ready(bias_key_ready),

      // Value parameters
      .weight_value(weight_value),
      .weight_value_valid(weight_value_valid),
      .weight_value_ready(weight_value_ready),

      .bias_value(bias_value),
      .bias_value_valid(bias_value_valid),
      .bias_value_ready(bias_value_ready),

      // Query output
      .data_out_query(query),
      .data_out_query_valid(joint_query_valid),
      .data_out_query_ready(joint_query_ready),

      // Key output
      .data_out_key(key),
      .data_out_key_valid(joint_key_valid),
      .data_out_key_ready(joint_key_ready),

      // Value output
      .data_out_value(value),
      .data_out_value_valid(joint_value_valid),
      .data_out_value_ready(joint_value_ready)
  );

  fixed_adder #(
    .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2 (0),
    .DATA_IN_0_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2 (0),

    .DATA_IN_1_PRECISION_0 (DATA_IN_0_PRECISION_0),
    .DATA_IN_1_PRECISION_1 (DATA_IN_0_PRECISION_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_1_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_2 (0),
    .DATA_IN_1_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_1_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_1_PARALLELISM_DIM_2 (0),

    .DATA_OUT_0_PRECISION_0 (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1 (DATA_OUT_0_PRECISION_1)
  ) adder_i_con(
      .clk(clk),
      .rst(rst),

      .data_in_0(query),
      .data_in_0_valid(joint_query_valid),
      .data_in_0_ready(joint_query_ready),

      .data_in_1(bias_con),
      .data_in_1_valid(bias_con_valid),
      .data_in_1_ready(bias_con_ready),

      .data_out_0(query_con),
      .data_out_0_valid(query_con_valid),
      .data_out_0_ready(query_con_ready)
  );

    fixed_adder #(
    .DATA_IN_0_PRECISION_0 (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1 (DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2 (0),
    .DATA_IN_0_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_2 (0),

    .DATA_IN_1_PRECISION_0 (DATA_IN_0_PRECISION_0),
    .DATA_IN_1_PRECISION_1 (DATA_IN_0_PRECISION_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_1_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_1_TENSOR_SIZE_DIM_2 (0),
    .DATA_IN_1_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_1_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_1_PARALLELISM_DIM_2 (0),

    .DATA_OUT_0_PRECISION_0 (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1 (DATA_OUT_0_PRECISION_1)
    ) adder_i_pos(
      .clk(clk),
      .rst(rst),

      .data_in_0(query),
      .data_in_0_valid(joint_query_valid),
      .data_in_0_ready(joint_query_ready),

      .data_in_1(bias_pos),
      .data_in_1_valid(bias_pos_valid),
      .data_in_1_ready(bias_pos_ready),

      .data_out_0(query_pos),
      .data_out_0_valid(query_pos_valid),
      .data_out_0_ready(query_pos_ready)
  );

  // * Scatter query, key, value

  swin_attention_head_scatter #(
      .NUM_HEADS(NUM_HEADS),

      .IN_DATA_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1)

  ) scatter_qkv_i (
      .clk,
      .rst,

      .query_con_valid(query_con_valid),
      .query_con_ready(query_con_ready),

      .query_pos_valid(query_pos_valid),
      .query_pos_ready(query_pos_ready),

      .key_valid(joint_key_valid),
      .key_ready(joint_key_ready),

      .value_valid(joint_value_valid),
      .value_ready(joint_value_ready),

      .pos_embed_valid(pos_embed_valid),
      .pos_embed_ready(pos_embed_ready),

      .split_query_con_valid(split_query_con_valid),
      .split_query_con_ready(split_query_con_ready),

      .split_query_pos_valid(split_query_pos_valid),
      .split_query_pos_ready(split_query_pos_ready),

      .split_key_valid(split_key_valid),
      .split_key_ready(split_key_ready),

      .split_value_valid(split_value_valid),
      .split_value_ready(split_value_ready),

      .split_pos_embed_valid(split_pos_embed_valid),
      .split_pos_embed_ready(split_pos_embed_ready)
  );

  // * Heads

  for (genvar head = 0; head < NUM_HEADS; head++) begin

    fixed_swin_attention_head #(
        .IN_DATA_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0 / NUM_HEADS),
        .IN_DATA_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
        .IN_DATA_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
        .IN_DATA_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
        .IN_DATA_PRECISION_0      (DATA_OUT_0_PRECISION_0),
        .IN_DATA_PRECISION_1      (DATA_OUT_0_PRECISION_1),

        .POS_EMBED_PARALLELISM_DIM_0(POS_EMBED_PARALLELISM_DIM_0),
        .POS_EMBED_PARALLELISM_DIM_1(POS_EMBED_PARALLELISM_DIM_1),

        .OUT_DATA_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0 / NUM_HEADS),
        .OUT_DATA_TENSOR_SIZE_DIM_1(DATA_OUT_0_TENSOR_SIZE_DIM_1),
        .OUT_DATA_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
        .OUT_DATA_PARALLELISM_DIM_1(DATA_OUT_0_PARALLELISM_DIM_1),
        .OUT_DATA_PRECISION_0      (DATA_OUT_0_PRECISION_0),
        .OUT_DATA_PRECISION_1      (DATA_OUT_0_PRECISION_1)

    ) head_i (
        .clk,
        .rst,

        .query_con      (query_con),
        .query_con_valid(split_query_con_valid[head]),
        .query_con_ready(split_query_con_ready[head]),

        .query_pos      (query_pos),
        .query_pos_valid(split_query_pos_valid[head]),
        .query_pos_ready(split_query_pos_ready[head]),

        .key      (key),
        .key_valid(split_key_valid[head]),
        .key_ready(split_key_ready[head]),

        .value      (value),
        .value_valid(split_value_valid[head]),
        .value_ready(split_value_ready[head]),

        .pos_embed     (pos_embed),
        .pos_embed_valid (split_pos_embed_valid[head]),
        .pos_embed_ready (split_pos_embed_ready[head]),

        .out      (head_out[head]),
        .out_valid(head_out_valid[head]),
        .out_ready(head_out_ready[head])
    );

  end

  // * Gather heads

  self_attention_head_gather #(
      .NUM_HEADS(NUM_HEADS),

      .IN_DATA_TENSOR_SIZE_DIM_0(DATA_OUT_0_TENSOR_SIZE_DIM_0),
      .IN_DATA_TENSOR_SIZE_DIM_1(DATA_OUT_0_TENSOR_SIZE_DIM_1),
      .IN_DATA_PARALLELISM_DIM_0(DATA_OUT_0_PARALLELISM_DIM_0),
      .IN_DATA_PARALLELISM_DIM_1(DATA_OUT_0_PARALLELISM_DIM_1),
      .IN_DATA_PRECISION_0      (DATA_OUT_0_PRECISION_0),
      .IN_DATA_PRECISION_1      (DATA_OUT_0_PRECISION_1)

  ) gather_qkv_i (
      .clk,
      .rst,

      .split_head_out      (head_out),
      .split_head_out_valid(head_out_valid),
      .split_head_out_ready(head_out_ready),

      .updated_tokens      (data_out_0),
      .updated_tokens_valid(data_out_0_valid),
      .updated_tokens_ready(data_out_0_ready)
  );

endmodule