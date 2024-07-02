/*
Module      : softermax
Description : This module implements softermax.
              https://arxiv.org/abs/2103.09301

              It depends on the "softermax_local_window" and
              "softermax_global_norm" modules.
*/
`timescale 1ns / 1ps
module fixed_softermax_1d #(
    // Shape Parameters
    parameter TOTAL_DIM   = 16,
    parameter PARALLELISM = 4,

    // Width Parameters
    parameter  IN_WIDTH        = 8,
    parameter  IN_FRAC_WIDTH   = 4,
    parameter  POW2_WIDTH      = 16,
    // POW2_FRAC_WIDTH should always be POW2_WIDTH - 1, since local values are
    // two to the power of a number in the range of (-inf, 0].
    localparam POW2_FRAC_WIDTH = 15,
    parameter  OUT_WIDTH       = 8,
    parameter  OUT_FRAC_WIDTH  = 7
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] in_data [PARALLELISM-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    output logic [OUT_WIDTH-1:0] out_data [PARALLELISM-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Params
  // -----

  localparam MAX_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;


  // -----
  // Wires
  // -----

  logic [ MAX_WIDTH-1:0] local_max;
  logic [POW2_WIDTH-1:0] local_values[PARALLELISM-1:0];
  logic local_window_valid, local_window_ready;

  // -----
  // Modules
  // -----

  softermax_local_window #(
      .PARALLELISM(PARALLELISM),
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .OUT_WIDTH(POW2_WIDTH),
      .OUT_FRAC_WIDTH(POW2_FRAC_WIDTH)
  ) local_window_inst (
      .clk(clk),
      .rst(rst),
      .in_data(in_data),
      .in_valid(in_valid),
      .in_ready(in_ready),
      .out_values(local_values),
      .out_max(local_max),
      .out_valid(local_window_valid),
      .out_ready(local_window_ready)
  );

  softermax_global_norm #(
      .TOTAL_DIM(TOTAL_DIM),
      .PARALLELISM(PARALLELISM),
      .IN_VALUE_WIDTH(POW2_WIDTH),
      .IN_MAX_WIDTH(MAX_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
  ) global_norm_inst (
      .clk(clk),
      .rst(rst),
      .in_values(local_values),
      .in_max(local_max),
      .in_valid(local_window_valid),
      .in_ready(local_window_ready),
      .out_data(out_data),
      .out_valid(out_valid),
      .out_ready(out_ready)
  );

endmodule
