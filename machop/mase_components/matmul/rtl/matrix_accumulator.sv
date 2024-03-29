/*
Module      : matrix_accumulator
Description : This module instantiated a 2D array of accumulators.
*/

`timescale 1ns / 1ps

module matrix_accumulator #(
    parameter IN_DEPTH = 4,
    parameter IN_WIDTH = 32,
    parameter DIM0 = 2,
    parameter DIM1 = 2,

    // Derived parameter
    localparam OUT_WIDTH = $clog2(IN_DEPTH) + IN_WIDTH
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [ IN_WIDTH-1:0] in_data  [DIM0*DIM1-1:0],
    input  logic                 in_valid,
    output logic                 in_ready,
    output logic [OUT_WIDTH-1:0] out_data [DIM0*DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  for (genvar i = 0; i < DIM1; i++) begin : rows
    for (genvar j = 0; j < DIM0; j++) begin : columns
      /* verilator lint_off UNUSEDSIGNAL */
      logic in_ready_signal, out_valid_signal;
      /* verilator lint_on UNUSEDSIGNAL */
      fixed_accumulator #(
          .IN_DEPTH(IN_DEPTH),
          .IN_WIDTH(IN_WIDTH)
      ) acc_inst (
          .clk           (clk),
          .rst           (rst),
          .data_in       (in_data[i*DIM0+j]),
          .data_in_valid (in_valid),
          .data_in_ready (in_ready_signal),
          .data_out      (out_data[i*DIM0+j]),
          .data_out_valid(out_valid_signal),
          .data_out_ready(out_ready)
      );
    end
  end

  // All accumulators should be synchronised, so we can take a single signal for
  // out_valid and in_ready
  assign in_ready  = rows[0].columns[0].in_ready_signal;
  assign out_valid = rows[0].columns[0].out_valid_signal;

endmodule
