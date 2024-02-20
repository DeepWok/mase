/*
Module      : online_layer_norm
Description : This module calculates the generalised group norm.
*/

`timescale 1ns/1ps

module online_layer_norm #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter CHANNELS            = 2,

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

endmodule
