/*
Module      : fixed_softermax_2d
Description : This module extends the softermax operation into an additional
              parallelism dimension by instantiating N fixed_softermax modules.

              !! This module only does softmax across DIM0 !!
*/

`timescale 1ns / 1ps

module fixed_softermax_2d #(
    // Shape Parameters
    parameter TOTAL_DIM0   = 16,
    parameter TOTAL_DIM1   = 16,
    parameter COMPUTE_DIM0 = 4,
    parameter COMPUTE_DIM1 = 4,

    // Width Parameters
    parameter IN_WIDTH       = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter POW2_WIDTH     = 16,
    parameter OUT_WIDTH      = 8,
    parameter OUT_FRAC_WIDTH = 7
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] in_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    output logic [OUT_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // -----
  // Wires
  // -----

  // Shape is (COMPUTE_DIM1, COMPUTE_DIM0, IN_WIDTH)
  logic [IN_WIDTH-1:0] softermax_in_rows[COMPUTE_DIM1-1:0][COMPUTE_DIM0-1:0];
  logic softermax_in_ready[COMPUTE_DIM1-1:0];

  logic [OUT_WIDTH-1:0] softermax_out_rows[COMPUTE_DIM1-1:0][COMPUTE_DIM0-1:0];
  logic softermax_out_valid[COMPUTE_DIM1-1:0];


  // -----
  // Modules
  // -----

  for (genvar i = 0; i < COMPUTE_DIM1; i++) begin : softermax_row

    assign softermax_in_rows[i] = in_data[(i+1)*COMPUTE_DIM1-1:i*COMPUTE_DIM1];
    assign out_data[(i+1)*COMPUTE_DIM1-1:i*COMPUTE_DIM1] = softermax_out_rows[i];

    fixed_softermax #(
        .TOTAL_DIM     (TOTAL_DIM0),
        .PARALLELISM   (COMPUTE_DIM0),
        .IN_WIDTH      (IN_WIDTH),
        .IN_FRAC_WIDTH (IN_FRAC_WIDTH),
        .POW2_WIDTH    (POW2_WIDTH),
        .OUT_WIDTH     (OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
    ) softermax_inst (
        .clk      (clk),
        .rst      (rst),
        .in_data  (softermax_in_rows[i]),
        .in_valid (in_valid),
        .in_ready (softermax_in_ready[i]),
        .out_data (softermax_out_rows[i]),
        .out_valid(softermax_out_valid[i]),
        .out_ready(out_ready)
    );

  end

  assign in_ready  = softermax_in_ready[0];
  assign out_valid = softermax_out_valid[0];

endmodule
