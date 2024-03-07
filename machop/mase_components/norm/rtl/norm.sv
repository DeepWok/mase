/*
Module      : norm
Description : Module which unifies all types of normalization.

              Currently supports:
              - Layer Norm
              - Instance Norm
              - Group Norm
              - RMS Norm
*/

`timescale 1ns/1ps
`default_nettype none

module norm #(

    // -----
    // SOFTWARE PARAMETERS
    // -----

    // DIMENSIONS:
    // batch (3), channel (2), dim1 (1), dim0 (0)

    // PRECISION:
    // width (1), frac_width (0)

    parameter DATA_IN_0_PRECISION_0         = -1,
    parameter DATA_IN_0_PRECISION_1         = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0   = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_0   = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1   = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_1   = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2   = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_2   = -1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3   = -1,
    parameter DATA_IN_0_PARALLELISM_DIM_3   = -1,

    parameter DATA_OUT_0_PRECISION_0        = -1,
    parameter DATA_OUT_0_PRECISION_1        = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0  = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0  = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1  = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1  = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2  = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2  = -1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3  = -1,
    parameter DATA_OUT_0_PARALLELISM_DIM_3  = -1,

    // -----
    // HARDWARE ALIASES
    // -----

    // Dimensions
    parameter TOTAL_DIM0          = DATA_IN_0_TENSOR_SIZE_DIM_0,
    parameter TOTAL_DIM1          = DATA_IN_0_TENSOR_SIZE_DIM_1,
    parameter COMPUTE_DIM0        = DATA_IN_0_PARALLELISM_DIM_0,
    parameter COMPUTE_DIM1        = DATA_IN_0_PARALLELISM_DIM_1,

    // Layer: CHANNELS should be set to total number of channels
    // RMS: CHANNELS should be set to total number of channels
    // Group: CHANNELS can be set to any factor of total channels
    parameter CHANNELS            = DATA_IN_0_PARALLELISM_DIM_2,

    // Data widths
    parameter IN_WIDTH            = DATA_IN_0_PRECISION_0,
    parameter IN_FRAC_WIDTH       = DATA_IN_0_PRECISION_1,
    parameter OUT_WIDTH           = DATA_OUT_0_PRECISION_0,
    parameter OUT_FRAC_WIDTH      = DATA_OUT_0_PRECISION_1,

    // Precision of inverse sqrt unit
    parameter INV_SQRT_WIDTH      = 16,
    parameter INV_SQRT_FRAC_WIDTH = 10,

    // Norm select
    parameter LAYER_NORM          = 0,
    parameter INSTANCE_NORM       = 0,
    parameter GROUP_NORM          = 0,
    parameter RMS_NORM            = 0
) (
    input  logic                clk,
    input  logic                rst,

    input  logic [IN_WIDTH-1:0] data_in_0  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                data_in_0_valid,
    output logic                data_in_0_ready,

    output logic [IN_WIDTH-1:0] data_out_0 [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                data_out_0_valid,
    input  logic                data_out_0_ready
);

initial begin
    // Only one normalization should be selected
    assert (LAYER_NORM + INSTANCE_NORM + GROUP_NORM + RMS_NORM == 1);
end


generate

if (LAYER_NORM || INSTANCE_NORM || GROUP_NORM) begin : group_norm

    localparam NORM_CHANNELS = (INSTANCE_NORM) ? 1: CHANNELS;

    group_norm_2d #(
        .TOTAL_DIM0(TOTAL_DIM0),
        .TOTAL_DIM1(TOTAL_DIM1),
        .COMPUTE_DIM0(COMPUTE_DIM0),
        .COMPUTE_DIM1(COMPUTE_DIM1),
        .GROUP_CHANNELS(NORM_CHANNELS),
        .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .INV_SQRT_WIDTH(INV_SQRT_WIDTH),
        .INV_SQRT_FRAC_WIDTH(INV_SQRT_FRAC_WIDTH)
    ) group_norm_inst (
        .clk(clk),
        .rst(rst),
        .in_data(data_in_0),
        .in_valid(data_in_0_valid),
        .in_ready(data_in_0_ready),
        .out_data(data_out_0),
        .out_valid(data_out_0_valid),
        .out_ready(data_out_0_ready)
    );

end else if (RMS_NORM) begin : rms_norm

    rms_norm_2d #(
        .TOTAL_DIM0(TOTAL_DIM0),
        .TOTAL_DIM1(TOTAL_DIM1),
        .COMPUTE_DIM0(COMPUTE_DIM0),
        .COMPUTE_DIM1(COMPUTE_DIM1),
        .CHANNELS(CHANNELS),
        .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
        .IN_WIDTH(IN_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .INV_SQRT_WIDTH(INV_SQRT_WIDTH),
        .INV_SQRT_FRAC_WIDTH(INV_SQRT_FRAC_WIDTH)
    ) rms_norm_inst (
        .clk(clk),
        .rst(rst),
        .in_data(data_in_0),
        .in_valid(data_in_0_valid),
        .in_ready(data_in_0_ready),
        .out_data(data_out_0),
        .out_valid(data_out_0_valid),
        .out_ready(data_out_0_ready)
    );

end

endgenerate

endmodule
