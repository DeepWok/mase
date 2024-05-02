`timescale 1ns / 1ps
module fixed_self_attention #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter IN_0_DEPTH = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_IN_0_PARALLELISM_DIM_0,

    parameter WEIGHTS_PRE_TRANSPOSED_QUERY = 0,
    parameter WEIGHT_QUERY_PRECISION_0 = 16,
    parameter WEIGHT_QUERY_PRECISION_1 = 3,
    parameter WEIGHT_QUERY_TENSOR_SIZE_DIM_0 = 32,
    parameter WEIGHT_QUERY_TENSOR_SIZE_DIM_1 = 1,
    parameter WEIGHT_QUERY_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_QUERY_PARALLELISM_DIM_1 = 4,

    parameter HAS_BIAS_QUERY = 0,
    parameter BIAS_QUERY_PRECISION_0 = 16,
    parameter BIAS_QUERY_PRECISION_1 = 3,
    parameter BIAS_QUERY_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_QUERY_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_QUERY_PARALLELISM_DIM_0 = 1,
    parameter BIAS_QUERY_PARALLELISM_DIM_1 = 1,

    parameter WEIGHTS_PRE_TRANSPOSED_KEY = 0,
    parameter WEIGHT_KEY_PRECISION_0 = 16,
    parameter WEIGHT_KEY_PRECISION_1 = 3,
    parameter WEIGHT_KEY_TENSOR_SIZE_DIM_0 = 32,
    parameter WEIGHT_KEY_TENSOR_SIZE_DIM_1 = 1,
    parameter WEIGHT_KEY_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_KEY_PARALLELISM_DIM_1 = 4,

    parameter HAS_BIAS_KEY = 0,
    parameter BIAS_KEY_PRECISION_0 = 16,
    parameter BIAS_KEY_PRECISION_1 = 3,
    parameter BIAS_KEY_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_KEY_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_KEY_PARALLELISM_DIM_0 = 1,
    parameter BIAS_KEY_PARALLELISM_DIM_1 = 1,

    parameter WEIGHTS_PRE_TRANSPOSED_VALUE = 0,
    parameter WEIGHT_VALUE_PRECISION_0 = 16,
    parameter WEIGHT_VALUE_PRECISION_1 = 3,
    parameter WEIGHT_VALUE_TENSOR_SIZE_DIM_0 = 32,
    parameter WEIGHT_VALUE_TENSOR_SIZE_DIM_1 = 1,
    parameter WEIGHT_VALUE_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_VALUE_PARALLELISM_DIM_1 = 4,

    parameter HAS_BIAS_VALUE = 0,
    parameter BIAS_VALUE_PRECISION_0 = 16,
    parameter BIAS_VALUE_PRECISION_1 = 3,
    parameter BIAS_VALUE_TENSOR_SIZE_DIM_0 = DATA_OUT_0_TENSOR_SIZE_DIM_0,
    parameter BIAS_VALUE_TENSOR_SIZE_DIM_1 = 1,
    parameter BIAS_VALUE_PARALLELISM_DIM_0 = 1,
    parameter BIAS_VALUE_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1

) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Query weights
    input logic [WEIGHT_QUERY_PRECISION_0-1:0] weight_query [WEIGHT_QUERY_PARALLELISM_DIM_0 * WEIGHT_QUERY_PARALLELISM_DIM_1-1:0],
    input logic weight_query_valid,
    output logic weight_query_ready,

    // Query bias
    input logic [BIAS_PRECISION_0-1:0] bias_query [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_query_valid,
    output logic bias_query_ready,

    // Key weights
    input logic [WEIGHT_KEY_PRECISION_0-1:0] weight_key [WEIGHT_KEY_PARALLELISM_DIM_0 * WEIGHT_KEY_PARALLELISM_DIM_1-1:0],
    input logic weight_key_valid,
    output logic weight_key_ready,

    // Key bias
    input logic [BIAS_PRECISION_0-1:0] bias_key [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_key_valid,
    output logic bias_key_ready,

    // Value weights
    input logic [WEIGHT_VALUE_PRECISION_0-1:0] weight_value [WEIGHT_VALUE_PARALLELISM_DIM_0 * WEIGHT_VALUE_PARALLELISM_DIM_1-1:0],
    input logic weight_value_valid,
    output logic weight_value_ready,

    // Value bias
    input logic [BIAS_PRECISION_0-1:0] bias_value [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_value_valid,
    output logic bias_value_ready,

    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready,

    // * Intermediate signals (for debug)

    // Query
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_query [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_query_valid,
    input  logic data_out_query_ready,

    // Key
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_key [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_key_valid,
    input  logic data_out_key_ready,

    // Value
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_value [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_value_valid,
    input logic data_out_value_ready
);

// data in = (1 x 784)
// weight = (784 x 784)
// dout = (1 x 784)

// * Declarations
// * =================================================================

logic [DATA_OUT_0_PRECISION_0-1:0] key_transpose [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0];
logic key_transpose_valid;
logic key_transpose_ready;

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

    .WEIGHT_PRECISION_0                  (WEIGHT_QUERY_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_QUERY_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_QUERY_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_QUERY_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_QUERY_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_QUERY_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_QUERY_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_QUERY_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_QUERY_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_QUERY_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_QUERY_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_QUERY_PARALLELISM_DIM_1),

    .DATA_OUT_0_PRECISION_0              (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1              (DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0        (DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1        (DATA_OUT_0_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_0        (DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1        (DATA_OUT_0_PARALLELISM_DIM_1)
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

    .WEIGHT_PRECISION_0                  (WEIGHT_KEY_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_KEY_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_KEY_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_KEY_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_KEY_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_KEY_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_KEY_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_KEY_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_KEY_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_KEY_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_KEY_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_KEY_PARALLELISM_DIM_1),

    .DATA_OUT_0_PRECISION_0              (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1              (DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0        (DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1        (DATA_OUT_0_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_0        (DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1        (DATA_OUT_0_PARALLELISM_DIM_1)
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

    .WEIGHT_PRECISION_0                  (WEIGHT_VALUE_PRECISION_0),
    .WEIGHT_PRECISION_1                  (WEIGHT_VALUE_PRECISION_1),
    .WEIGHT_TENSOR_SIZE_DIM_0            (WEIGHT_VALUE_TENSOR_SIZE_DIM_0),
    .WEIGHT_TENSOR_SIZE_DIM_1            (WEIGHT_VALUE_TENSOR_SIZE_DIM_1),
    .WEIGHT_PARALLELISM_DIM_0            (WEIGHT_VALUE_PARALLELISM_DIM_0),
    .WEIGHT_PARALLELISM_DIM_1            (WEIGHT_VALUE_PARALLELISM_DIM_1),

    .BIAS_PRECISION_0                    (BIAS_VALUE_PRECISION_0),
    .BIAS_PRECISION_1                    (BIAS_VALUE_PRECISION_1),
    .BIAS_TENSOR_SIZE_DIM_0              (BIAS_VALUE_TENSOR_SIZE_DIM_0),
    .BIAS_TENSOR_SIZE_DIM_1              (BIAS_VALUE_TENSOR_SIZE_DIM_1),
    .BIAS_PARALLELISM_DIM_0              (BIAS_VALUE_PARALLELISM_DIM_0),
    .BIAS_PARALLELISM_DIM_1              (BIAS_VALUE_PARALLELISM_DIM_1),

    .DATA_OUT_0_PRECISION_0              (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1              (DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0        (DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1        (DATA_OUT_0_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_0        (DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1        (DATA_OUT_0_PARALLELISM_DIM_1)
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

// * Transpose projected keys

// matrix_stream_transpose #(
//     .TOTAL_DIM0 (),
//     .TOTAL_DIM1 (),

//     .COMPUTE_DIM0 (),
//     .COMPUTE_DIM1 (),

//     .DATA_WIDTH (8)
// ) key_transpose_i (
//     .clk,
//     .rst,

//     // In Matrix
//     .in_data    (),
//     .in_valid   (),
//     .in_ready   (),

//     // Out Matrix
//     .out_data    (key_transpose),
//     .out_valid   (key_transpose_valid),
//     .out_ready   (key_transpose_ready),
// );

// * Logic
// * =================================================================

endmodule