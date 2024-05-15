`timescale 1ns / 1ps
module fixed_series_approx #(
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_OUT_0_PRECISION_0 = 3 * DATA_IN_0_PRECISION_0 + 5
) (
    /* verilator lint_off UNUSEDSIGNAL */
    /* verilator lint_off SELRANGE */
    input  logic [ DATA_IN_0_PRECISION_0-1:0] data_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0
);
  logic [DATA_IN_0_PRECISION_0 - 1+1:0] x_shift_1;
  logic [DATA_IN_0_PRECISION_0 - 1+2:0] x_shift_2;
  logic [DATA_IN_0_PRECISION_0 - 1+4:0] x_shift_4;
  logic [DATA_IN_0_PRECISION_0 - 1+4:0] term1;
  logic [DATA_IN_0_PRECISION_0 - 1+4:0] term2;
  logic [2*DATA_IN_0_PRECISION_0 + 5 - 1:0] term3;
  logic [2*DATA_IN_0_PRECISION_0 + 5 - 1:0] term4;
  logic [3*DATA_IN_0_PRECISION_0 + 5 - 1:0] product;
  logic [3*DATA_IN_0_PRECISION_0 + 5 - 1:0] result;

  // Shift right operations
  assign x_shift_1 = data_in_0 >> 1;
  assign x_shift_2 = data_in_0 >> 2;
  assign x_shift_4 = data_in_0 >> 4;

  // Calculation of terms
  assign term1 = x_shift_4 + x_shift_2;
  assign term2 = ~term1;
  assign term3 = x_shift_1 * term2;
  assign term4 = ~term3;

  // Multiplication of terms
  assign product = data_in_0 * term4;

  // Inversion of the product
  assign result = ~product;

  // Output
  assign data_out_0 = result[3*DATA_IN_0_PRECISION_0 + 5 - 1: 3*DATA_IN_0_PRECISION_0 + 5 - DATA_OUT_0_PRECISION_0];

endmodule
