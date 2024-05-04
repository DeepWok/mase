`timescale 1ns / 1ps
module fixed_bert_attention_output_block #(
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
    parameter BIAS_PRECISION_1 = 3,

    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0,
    parameter DATA_OUT_0_PRECISION_1 = DATA_IN_0_PRECISION_1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = WEIGHT_TENSOR_SIZE_DIM_0,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1

) (
    input logic clk,
    input logic rst,

    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Query weights
    input logic [WEIGHT_PRECISION_0-1:0] weight [WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1-1:0],
    input logic weight_valid,
    output logic weight_ready,

    // Query bias
    input logic [BIAS_PRECISION_0-1:0] bias [BIAS_PARALLELISM_DIM_0 * BIAS_PARALLELISM_DIM_1 -1:0],
    input logic bias_valid,
    output logic bias_ready,

    // Query
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

// * Inferred parameters for intermediate signals

parameter LINEAR_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0
                                + $clog2(DATA_IN_0_PARALLELISM_DIM_0)
                                + $clog2(WEIGHT_TENSOR_SIZE_DIM_1 / WEIGHT_PARALLELISM_DIM_1)
                                + HAS_BIAS_QUERY;
parameter LINEAR_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
parameter LINEAR_PARALLELISM_DIM_0 = WEIGHT_PARALLELISM_DIM_0;
parameter LINEAR_PARALLELISM_DIM_1 = DATA_IN_0_PARALLELISM_DIM_1;

logic [LINEAR_PRECISION_0-1:0] out_linear [LINEAR_PARALLELISM_DIM_0*LINEAR_PARALLELISM_DIM_1-1:0];
logic out_linear_valid;
logic out_linear_ready;

// * Instances
// * =================================================================

// * Query linear

fixed_linear # (
    .HAS_BIAS                            (HAS_BIAS),
    .WEIGHTS_PRE_TRANSPOSED              (WEIGHTS_PRE_TRANSPOSED),

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

    .data_in_0                          (data_in_0),
    .data_in_0_valid                    (data_in_0_valid),
    .data_in_0_ready                    (data_in_0_ready),

    .weight                             (weight),
    .weight_valid                       (weight_valid),
    .weight_ready                       (weight_ready),

    .bias                               (bias),
    .bias_valid                         (bias_valid),
    .bias_ready                         (bias_ready),

    .data_out_0                         (out_linear),
    .data_out_0_valid                   (out_linear_valid),
    .data_out_0_ready                   (out_linear_ready)
);

norm #(
    .DATA_IN_0_PRECISION_0       (LINEAR_PRECISION_0),
    .DATA_IN_0_PRECISION_1       (LINEAR_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0 (DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1 (DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_0 (DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_1 (DATA_IN_0_PARALLELISM_DIM_1),

    .DATA_OUT_0_PRECISION_0       (DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1       (DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_PARALLELISM_DIM_0 (DATA_OUT_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_1 (DATA_OUT_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0 (DATA_OUT_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1 (DATA_OUT_0_TENSOR_SIZE_DIM_1),

    .NORM_TYPE                    ("LAYER_NORM")
) norm_inst (
    .clk                 (clk),
    .rst                 (rst),

    .data_in_0           (out_linear),
    .data_in_0_valid     (out_linear_valid),
    .data_in_0_ready     (out_linear_ready),

    // Weights unused (only for RMS norm)
    .weight              ('0),
    .weight_valid        ('0),
    .weight_ready        (),

    .data_out_0          (data_out_0),
    .data_out_0_valid    (data_out_0_valid),
    .data_out_0_ready    (data_out_0_ready)
);


endmodule