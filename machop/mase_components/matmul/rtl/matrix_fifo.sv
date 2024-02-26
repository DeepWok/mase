/*
Module      : matrix_fifo
Description : FIFO to buffer matrices or 2D data.
*/

`timescale 1ns/1ps
`default_nettype none

module matrix_fifo #(
    // Dimensions
    parameter DATA_WIDTH = 8,
    parameter DIM0       = 4,
    parameter DIM1       = 4,
    parameter FIFO_SIZE  = 32
) (
    input  logic                  clk,
    input  logic                  rst,

    input  logic [DATA_WIDTH-1:0] in_data  [DIM0*DIM1-1:0],
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data [DIM0*DIM1-1:0],
    output logic                  out_valid,
    input  logic                  out_ready
);

// Wires
localparam FLAT_DATA_WIDTH = DATA_WIDTH * DIM0 * DIM1;

logic [DATA_FLAT_WIDTH-1:0] in_data_flat, out_data_flat;
logic [DATA_WIDTH-1:0] fifo_data [DIM0*DIM1-1:0];
logic fifo_out_valid, fifo_out_ready;
logic fifo_in_valid, fifo_in_ready;

// Modules
matrix_flatten #(
    .DATA_WIDTH(DATA_WIDTH),
    .DIM0(DIM0),
    .DIM1(DIM1)
) input_flatten (
    .data_in(in_data),
    .data_out(in_data_flat)
);

fifo_v2 #(
    .SIZE(FIFO_SIZE),
    .DATA_WIDTH(FLAT_DATA_WIDTH)
) input_fifo_inst (
    .clk(clk),
    .rst(rst),
    .in_data(in_data_flat),
    .in_valid(fifo_in_valid),
    .in_ready(fifo_in_ready),
    .out_data(out_data_flat),
    .out_valid(fifo_out_valid),
    .out_ready(fifo_out_ready)
);

matrix_unflatten #(
    .DATA_WIDTH(DATA_WIDTH),
    .DIM0(DIM0),
    .DIM1(DIM1)
) fifo_unflatten (
    .data_in(out_data_flat),
    .data_out(fifo_data)
);

endmodule
