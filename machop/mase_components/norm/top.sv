/*
Module      : top
Description : Module for Vivado Synthesis & Implementation
*/

`timescale 1ns/1ps

module top #(

    // -----
    // SOFTWARE PARAMETERS
    // -----

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0   = 128, // TOTAL_DIM0
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1   = 128, // TOTAL_DIM1
    parameter DATA_IN_0_PARALLELISM_DIM_0   = 4, // COMPUTE_DIM0
    parameter DATA_IN_0_PARALLELISM_DIM_1   = 4, // COMPUTE_DIM1
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2   = 64, // CHANNELS

    parameter DATA_IN_0_PRECISION_0         = 8, // IN_WIDTH
    parameter DATA_IN_0_PRECISION_1         = 4, // IN_FRAC_WIDTH

    parameter DATA_OUT_0_PRECISION_0        = 8, // OUT_WIDTH
    parameter DATA_OUT_0_PRECISION_1        = 4, // OUT_FRAC_WIDTH

    // Inverse sqrt unit LUT file
    parameter ISQRT_LUT_MEMFILE    = "",

    // Batch norm LUT files
    parameter SCALE_LUT_MEMFILE    = "",
    parameter SHIFT_LUT_MEMFILE    = "",

    // Norm select: BATCH_NORM, LAYER_NORM, INSTANCE_NORM, GROUP_NORM, RMS_NORM
    parameter NORM_TYPE            = "LAYER_NORM",

    localparam COMPUTE_DIM0        = DATA_IN_0_PARALLELISM_DIM_0,
    localparam COMPUTE_DIM1        = DATA_IN_0_PARALLELISM_DIM_1,
    localparam IN_WIDTH            = DATA_IN_0_PRECISION_0,
    localparam OUT_WIDTH           = DATA_OUT_0_PRECISION_0
) (
    input  logic                 clk,
    input  logic                 rst,

    input  logic [IN_WIDTH-1:0]  data_in_0  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                 data_in_0_valid,
    output logic                 data_in_0_ready,

    output logic [OUT_WIDTH-1:0] data_out_0 [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 data_out_0_valid,
    input  logic                 data_out_0_ready
);

norm #(
    .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
    .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_IN_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_IN_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_IN_0_TENSOR_SIZE_DIM_2(DATA_IN_0_TENSOR_SIZE_DIM_2), // Channels
    .DATA_IN_0_PARALLELISM_DIM_2('0),
    .DATA_IN_0_TENSOR_SIZE_DIM_3('0),
    .DATA_IN_0_PARALLELISM_DIM_3('0),
    .DATA_OUT_0_PRECISION_0(DATA_OUT_0_PRECISION_0),
    .DATA_OUT_0_PRECISION_1(DATA_OUT_0_PRECISION_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
    .DATA_OUT_0_PARALLELISM_DIM_0(DATA_IN_0_PARALLELISM_DIM_0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
    .DATA_OUT_0_PARALLELISM_DIM_1(DATA_IN_0_PARALLELISM_DIM_1),
    .DATA_OUT_0_TENSOR_SIZE_DIM_2('0),
    .DATA_OUT_0_PARALLELISM_DIM_2('0),
    .DATA_OUT_0_TENSOR_SIZE_DIM_3('0),
    .DATA_OUT_0_PARALLELISM_DIM_3('0),
    .ISQRT_LUT_MEMFILE(ISQRT_LUT_MEMFILE),
    .SCALE_LUT_MEMFILE(SCALE_LUT_MEMFILE),
    .SHIFT_LUT_MEMFILE(SHIFT_LUT_MEMFILE),
    .NORM_TYPE(NORM_TYPE)
) norm_inst (
    .clk(clk),
    .rst(rst),
    .data_in_0(data_in_0),
    .data_in_0_valid(data_in_0_valid),
    .data_in_0_ready(data_in_0_ready),
    .data_out_0(data_out_0),
    .data_out_0_valid(data_out_0_valid),
    .data_out_0_ready(data_out_0_ready)
);

endmodule