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
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,

    // Layer: CHANNELS should be set to total number of channels
    // RMS: CHANNELS should be set to total number of channels
    // Group: CHANNELS can be set to any factor of total channels
    parameter CHANNELS            = 2,

    // Data widths
    parameter IN_FRAC_WIDTH       = 2,
    parameter IN_WIDTH            = 8,
    parameter OUT_WIDTH           = 8,
    parameter OUT_FRAC_WIDTH      = 4,

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

    input  logic [IN_WIDTH-1:0] in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    output logic [IN_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                out_valid,
    input  logic                out_ready
);

initial begin
    // Only one normalization should be selected
    assert (LAYER_NORM + INSTANCE_NORM + GROUP_NORM + RMS_NORM == 1);

    //
end

generate

if (LAYER_NORM || INSTANCE_NORM || GROUP_NORM) begin : group_norm

group_norm_2d #(
    .TOTAL_DIM0(TOTAL_DIM0),
    .TOTAL_DIM1(TOTAL_DIM1),
    .COMPUTE_DIM0(COMPUTE_DIM0),
    .COMPUTE_DIM1(COMPUTE_DIM1),
    .GROUP_CHANNELS(CHANNELS),
    .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
    .IN_WIDTH(IN_WIDTH),
    .OUT_WIDTH(OUT_WIDTH),
    .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
    .INV_SQRT_WIDTH(INV_SQRT_WIDTH),
    .INV_SQRT_FRAC_WIDTH(INV_SQRT_FRAC_WIDTH)
) group_norm_inst (
    .clk(clk),
    .rst(rst),
    .in_data(in_data),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .out_data(out_data),
    .out_valid(out_valid),
    .out_ready(out_ready)
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
    .in_data(in_data),
    .in_valid(in_valid),
    .in_ready(in_ready),
    .out_data(out_data),
    .out_valid(out_valid),
    .out_ready(out_ready)
);

end

endgenerate

endmodule
