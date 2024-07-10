/*
Module      : matrix_fifo
Description : FIFO to buffer matrices or 2D data.
*/

`timescale 1ns / 1ps

module matrix_fifo #(
    // Dimensions
    parameter DATA_WIDTH = 8,
    parameter DIM0       = 4,
    parameter DIM1       = 4,
    parameter FIFO_SIZE  = 32
) (
    input logic clk,
    input logic rst,

    input  logic [DATA_WIDTH-1:0] in_data [DIM0*DIM1-1:0],
    input  logic                  in_valid,
    output logic                  in_ready,

    output logic [DATA_WIDTH-1:0] out_data [DIM0*DIM1-1:0],
    output logic                  out_valid,
    input  logic                  out_ready
);

  // Wires
  localparam FLAT_DATA_WIDTH = DATA_WIDTH * DIM0 * DIM1;

  logic [FLAT_DATA_WIDTH-1:0] in_data_flat, out_data_flat;

  // Modules
  matrix_flatten #(
      .DATA_WIDTH(DATA_WIDTH),
      .DIM0(DIM0),
      .DIM1(DIM1)
  ) input_flatten (
      .data_in (in_data),
      .data_out(in_data_flat)
  );

  fifo #(
      .DATA_WIDTH(FLAT_DATA_WIDTH),
      .DEPTH(FIFO_SIZE)
  ) input_fifo_inst (
      .clk(clk),
      .rst(rst),
      .in_data(in_data_flat),
      .in_valid(in_valid),
      .in_ready(in_ready),
      .out_data(out_data_flat),
      .out_valid(out_valid),
      .out_ready(out_ready),
      .empty(),
      .full()
  );

  matrix_unflatten #(
      .DATA_WIDTH(DATA_WIDTH),
      .DIM0(DIM0),
      .DIM1(DIM1)
  ) fifo_unflatten (
      .data_in (out_data_flat),
      .data_out(out_data)
  );

endmodule
