/*
Module      : fixed_range_reduction
Description : This module finds the MSB of the number. If there is no MSB, then
              the "not_found" wire will be driven HIGH.
*/

`timescale 1ns / 1ps

module fixed_range_reduction #(
    parameter WIDTH = 16,
    localparam MSB_WIDTH = $clog2(WIDTH)
) (
    // Original input x
    input logic [WIDTH-1:0] data_a,  // FORMAT: Q(INT_WIDTH).(FRAC_WIDTH).
    // Reduced x
    output logic [WIDTH-1:0] data_out,  // FORMAT: Q1.(WIDTH-1).
    // msb_index
    output logic [MSB_WIDTH-1:0] msb_index,
    output logic not_found
);

  // Find MSB index. Rightmost position = 0
  /* verilator lint_off LATCH */
  integer i;
  always @* begin
    for (i = WIDTH - 1; i >= 0; i = i - 1) begin
      if (data_a[i] == 1) begin
        msb_index = i;
        break;
      end
    end
    // NOTE: when the input is 0 then this whole block will be ignored
    // by top level module.
  end
  /* verilator lint_on LATCH */

  // Shift by the correct amount to set format to Q1.(WIDTH-1)
  assign data_out  = data_a << (WIDTH - 1 - msb_index);
  assign not_found = data_a == '0;

endmodule
