/*
Module      : signed_clamp
Description : Clamps a signed number
*/

`timescale 1ns / 1ps

module signed_clamp #(
    parameter IN_WIDTH  = 8,
    parameter OUT_WIDTH = 8,
    parameter SYMMETRIC = 0
) (
    input  logic signed [ IN_WIDTH-1:0] in_data,
    output logic signed [OUT_WIDTH-1:0] out_data
);

  localparam logic signed [OUT_WIDTH-1:0] MIN_VAL = SYMMETRIC ?
                                                  -(2 ** (OUT_WIDTH-1)) + 1 :
                                                  -(2 ** (OUT_WIDTH-1));
  localparam logic signed [OUT_WIDTH-1:0] MAX_VAL = (2 ** (OUT_WIDTH - 1)) - 1;

  always_comb begin
    if (in_data > MAX_VAL) begin
      out_data = MAX_VAL;
    end else if (in_data < MIN_VAL) begin
      out_data = MIN_VAL;
    end else begin
      out_data = in_data;
    end
  end

endmodule
