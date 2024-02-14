/*
Module      : group_norm_2d
Description : This module calculates the generalised group norm.
              https://arxiv.org/abs/1803.08494v3

              This module can be easily trivially specialised into layer norm or
              instance norm by setting the GROUP_CHANNELS param to equal C or 1
              respectively.

              Group norm is independent of batch size, so the input shape is:
              (GROUP, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
*/

`timescale 1ns/1ps

module group_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter GROUP_CHANNELS      = 2,

    // Data widths
    parameter WIDTH               = 8,
    parameter FRAC_WIDTH          = 8
) (
    input  logic             clk,
    input  logic             rst,

    input  logic [WIDTH-1:0] in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic             in_valid,
    output logic             in_ready,

    output logic [WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic             out_valid,
    input  logic             out_ready
);

// Constant derived params
localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;

always_comb begin

end

localparam DATA_FLAT_WIDTH = WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1;
localparam FIFO_DEPTH = GROUP_CHANNELS * DEPTH_DIM0 * DEPTH_DIM1;

logic [DATA_FLAT_WIDTH-1:0] in_data_flat;
matrix_flatten #(
    .DATA_WIDTH(WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) input_flatten (
    .data_in(in_data),
    .data_out(in_data_flat)
);

fifo #(
    .DEPTH(FIFO_DEPTH),
    .WIDTH(DATA_FLAT_WIDTH)
) fifo_inst (

);

endmodule
