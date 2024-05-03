`timescale 1ns / 1ps
module fixed_self_attention #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 768,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 20,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,

    parameter WEIGHTS_PRE_TRANSPOSED = 0,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 768,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 768,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_1 = 3,

    parameter HAS_BIAS = 1,
    parameter BIAS_TENSOR_SIZE_DIM_0 = 64,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 20,
    parameter BIAS_PARALLELISM_DIM_0 = 4,
    parameter BIAS_PARALLELISM_DIM_1 = 4,
    parameter BIAS_PRECISION_0 = 16,
    parameter BIAS_PRECISION_1 = 3

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

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);


// * Inferred parameters
parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_0;
parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1;
parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0;
parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1;

// * Precision parameters for intermediate signals

parameter QKV_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0
                                + $clog2(DATA_IN_0_PARALLELISM_DIM_0)
                                + $clog2(WEIGHT_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1)
                                + HAS_BIAS_QUERY;
parameter QKV_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;


// * Declarations
// * =================================================================

// Query
logic [QKV_PRECISION_0-1:0] joint_query [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
logic joint_query_valid;
logic joint_query_ready;

// Key
logic [QKV_PRECISION_0-1:0] joint_key [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
logic joint_key_valid;
logic joint_key_ready;

// Value
logic [QKV_PRECISION_0-1:0] joint_value [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0];
logic joint_value_valid;
logic joint_value_ready;

// * Instances
// * =================================================================

fixed_self_attention_batched_input_block #(
    .DATA_IN_0_TENSOR_SIZE_DIM_0    (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1    (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0    (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1    (DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_PRECISION_0          (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1          (DATA_IN_0_PRECISION_1),

    .WEIGHTS_PRE_TRANSPOSED         (WEIGHTS_PRE_TRANSPOSED),
    .WEIGHT_TENSOR_SIZE_DIM_0       (WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1       (WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0       (WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1       (WEIGHT_PARALLELISM_DIM_1),
    .WEIGHT_PRECISION_0             (WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1             (WEIGHT_PRECISION_1),

    .HAS_BIAS                       (HAS_BIAS),
    .BIAS_TENSOR_SIZE_DIM_0         (BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1         (BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0         (BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1         (BIAS_PARALLELISM_DIM_1),
    .BIAS_PRECISION_0               (BIAS_PRECISION_0),
    .BIAS_PRECISION_1               (BIAS_PRECISION_1)
) batched_input_block_i (
    .clk (clk),
    .rst (rst),

    .data_in_0 (data_in_0),
    .data_in_0_valid (data_in_0_valid),
    .data_in_0_ready (data_in_0_ready),

    // Query parameters
    .weight_query (weight_query),
    .weight_query_valid (weight_query_valid),
    .weight_query_ready (weight_query_ready),

    .bias_query (bias_query),
    .bias_query_valid (bias_query_valid),
    .bias_query_ready (bias_query_ready),

    // Key parameters
    .weight_key (weight_key),
    .weight_key_valid (weight_key_valid),
    .weight_key_ready (weight_key_ready),

    .bias_key (bias_key),
    .bias_key_valid (bias_key_valid),
    .bias_key_ready (bias_key_ready),

    // Value parameters
    .weight_value (weight_value),
    .weight_value_valid (weight_value_valid),
    .weight_value_ready (weight_value_ready),

    .bias_value (bias_value),
    .bias_value_valid (bias_value_valid),
    .bias_value_ready (bias_value_ready),

    // Query output
    .joint_query (joint_query),
    .joint_query_valid (joint_query_valid),
    .joint_query_ready (joint_query_ready),

    // Key output
    .joint_key (joint_key),
    .joint_key_valid (joint_key_valid),
    .joint_key_ready (joint_key_ready),

    // Value output
    .joint_value (joint_value),
    .joint_value_valid (joint_value_valid),
    .joint_value_ready (joint_value_ready)
);

// * Scatter query, key, value

bert_self_attention_head_scatter #(
    .NUM_HEADS                 (NUM_HEADS),

    .IN_DATA_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .IN_DATA_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .IN_DATA_PARALLELISM_DIM_0 (WEIGHT_PARALLELISM_DIM_0),
    .IN_DATA_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),
    .IN_DATA_PRECISION_0       (QKV_PRECISION_0)
    .IN_DATA_PRECISION_1       (QKV_PRECISION_1),

) scatter_qkv_i (
    .clk,
    .rst,

    .query          (joint_query),
    .query_valid    (joint_query_valid),
    .query_ready    (joint_query_ready),

    .key            (joint_key),
    .key_valid      (joint_key_valid),
    .key_ready      (joint_key_ready),

    .value          (joint_value),
    .value_valid    (joint_value_valid),
    .value_ready    (joint_value_ready),

    .split_query       (query_head),
    .split_query_valid (query_head_valid),
    .split_query_ready (query_head_ready),

    .split_key         (key_head),
    .split_key_valid   (key_head_valid),
    .split_key_ready   (key_head_ready),

    .split_value       (value_head),
    .split_value_valid (value_head_valid),
    .split_value_ready (value_head_ready)
);

// * Heads

for (genvar head = 0; head < NUM_HEADS; head++) begin

    bert_self_attention_head_fixed #(
        .IN_DATA_TENSOR_SIZE_DIM_0    (),
        .IN_DATA_TENSOR_SIZE_DIM_1    (),
        .IN_DATA_PARALLELISM_DIM_0    (),
        .IN_DATA_PARALLELISM_DIM_1    (),
        .IN_DATA_PRECISION_0          (),
        .IN_DATA_PRECISION_1          (),

        .OUT_DATA_TENSOR_SIZE_DIM_0   (),
        .OUT_DATA_TENSOR_SIZE_DIM_1   (),
        .OUT_DATA_PARALLELISM_DIM_0   (),
        .OUT_DATA_PARALLELISM_DIM_1   (),
        .OUT_DATA_PRECISION_0         (),
        .OUT_DATA_PRECISION_1         ()

    ) head_i (
        .clk,
        .rst,

        .query (query_head[head]),
        .query_valid (query_head_valid[head]),
        .query_ready (query_head_valid[head]),

        .key (key_head[head]),
        .key_valid (key_head_valid[head]),
        .key_ready (key_head_ready[head]),

        .value (value_head[head]),
        .value_valid (value_head_valid[head]),
        .value_ready (value_head_ready[head]),

        .out (out_head[head]),
        .out_valid (out_head_valid[head]),
        .out_ready (out_head_ready[head])
    );

end

// * Gather heads

bert_self_attention_head_gather #(
    .NUM_HEADS                 (NUM_HEADS),

    .IN_DATA_TENSOR_SIZE_DIM_0 (DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .IN_DATA_TENSOR_SIZE_DIM_1 (DATA_OUT_0_TENSOR_SIZE_DIM_1),
    .IN_DATA_PARALLELISM_DIM_0 (DATA_OUT_0_PARALLELISM_DIM_0),
    .IN_DATA_PARALLELISM_DIM_1 (DATA_OUT_0_PARALLELISM_DIM_1),
    .IN_DATA_PRECISION_0       ()
    .IN_DATA_PRECISION_1       (),

) scatter_qkv_i (
    .clk,
    .rst,

    .split_head_out          (out_head),
    .split_head_out_valid    (out_head_valid),
    .split_head_out_ready    (out_head_ready),

    .updated_tokens       (updated_tokens),
    .updated_tokens_valid (updated_tokens_valid),
    .updated_tokens_ready (updated_tokens_ready)
);

// * Output block

endmodule