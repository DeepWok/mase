`timescale 1ns / 1ps
module fixed_self_attention_batched_input_block #(
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

    // Query
    output logic [QKV_PRECISION_0-1:0] data_out_query [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0],
    output logic data_out_query_valid,
    input  logic data_out_query_ready,

    // Key
    output logic [QKV_PRECISION_0-1:0] data_out_key [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0],
    output logic data_out_key_valid,
    input  logic data_out_key_ready,

    // Value
    output logic [QKV_PRECISION_0-1:0] data_out_value [DATA_IN_0_PARALLELISM_DIM_1 * WEIGHT_PARALLELISM_DIM_0-1:0],
    output logic data_out_value_valid,
    input  logic data_out_value_ready
);

// ! TO DO: add assertions about bias parallelism matching weight parallelism

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

// * Instances
// * =================================================================

// * Query linear

fixed_linear # (
    .HAS_BIAS                            (HAS_BIAS_QUERY),
    .WEIGHTS_PRE_TRANSPOSED              (WEIGHTS_PRE_TRANSPOSED_QUERY),

    .DATA_IN_0_PRECISION_0               (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1               (DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0         (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1         (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0         (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1         (DATA_IN_0_PARALLELISM_DIM_1),

    .WEIGHT_PRECISION_0                  (WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_PARALLELISM_DIM_1)

) fixed_linear_query (
    .clk,
    .rst,

    // input port for data_inivations
    .data_in_0                          (data_in_0),
    .data_in_0_valid                    (data_in_0_valid),
    .data_in_0_ready                    (data_in_0_ready),

    // input port for weight
    .weight                             (weight_query),
    .weight_valid                       (weight_query_valid),
    .weight_ready                       (weight_query_ready),

    .bias                               (bias_query),
    .bias_valid                         (bias_query_valid),
    .bias_ready                         (bias_query_ready),

    .data_out_0                         (data_out_query),
    .data_out_0_valid                   (data_out_query_valid),
    .data_out_0_ready                   (data_out_query_ready)
);

// * Key linear

fixed_linear # (
    .HAS_BIAS                            (HAS_BIAS_KEY),
    .WEIGHTS_PRE_TRANSPOSED              (WEIGHTS_PRE_TRANSPOSED_KEY),

    .DATA_IN_0_PRECISION_0               (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1               (DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0         (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1         (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0         (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1         (DATA_IN_0_PARALLELISM_DIM_1),

    .WEIGHT_PRECISION_0                  (WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_PARALLELISM_DIM_1)

) fixed_linear_key (
    .clk,
    .rst,

    // input port for data_inivations
    .data_in_0                          (data_in_0),
    .data_in_0_valid                    (data_in_0_valid),
    .data_in_0_ready                    (data_in_0_ready),

    // input port for weight
    .weight                             (weight_key),
    .weight_valid                       (weight_key_valid),
    .weight_ready                       (weight_key_ready),

    .bias                               (bias_key),
    .bias_valid                         (bias_key_valid),
    .bias_ready                         (bias_key_ready),

    .data_out_0                         (data_out_key),
    .data_out_0_valid                   (data_out_key_valid),
    .data_out_0_ready                   (data_out_key_ready)
);

// * Value linear

fixed_linear # (
    .HAS_BIAS                            (HAS_BIAS_VALUE),
    .WEIGHTS_PRE_TRANSPOSED              (WEIGHTS_PRE_TRANSPOSED_VALUE),

    .DATA_IN_0_PRECISION_0               (DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1               (DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0         (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1         (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0         (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1         (DATA_IN_0_PARALLELISM_DIM_1),

    .WEIGHT_PRECISION_0                  (WEIGHT_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_PARALLELISM_DIM_1)

) fixed_linear_value (
    .clk,
    .rst,

    // input port for data_inivations
    .data_in_0                          (data_in_0),
    .data_in_0_valid                    (data_in_0_valid),
    .data_in_0_ready                    (data_in_0_ready),

    // input port for weight
    .weight                             (weight_value),
    .weight_valid                       (weight_value_valid),
    .weight_ready                       (weight_value_ready),

    .bias                               (bias_value),
    .bias_valid                         (bias_value_valid),
    .bias_ready                         (bias_value_ready),

    .data_out_0                         (data_out_value),
    .data_out_0_valid                   (data_out_value_valid),
    .data_out_0_ready                   (data_out_value_ready)
);

endmodule