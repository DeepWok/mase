/*
Module      : fixed_signed_cast
Description : Cast a fixed point signed number into another.

              Types of rounding when OUT_FRAC_WIDTH < IN_FRAC_WIDTH:
              - Floor
              - Truncation
              - Round to nearest int, round half to even
*/

`timescale 1ns / 1ps

module fixed_signed_cast #(
    parameter IN_WIDTH       = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter OUT_WIDTH      = 8,
    parameter OUT_FRAC_WIDTH = 4,
    // SYMMETRIC=1 means that for 8-bit number the range is +127 to -127
    // SYMMETRIC=0 means that it is +127 to -128
    parameter SYMMETRIC      = 0,

    // Rounding types for when OUT_FRAC_WIDTH < IN_FRAC_WIDTH
    // One of these needs to be set to 1
    parameter ROUND_FLOOR                 = 0,
    parameter ROUND_TRUNCATE              = 0,
    parameter ROUND_NEAREST_INT_HALF_EVEN = 0
) (
    input  logic signed [ IN_WIDTH-1:0] in_data,
    output logic signed [OUT_WIDTH-1:0] out_data
);


  initial begin
    assert (IN_WIDTH > 0);
    assert (OUT_WIDTH > 0);
    assert (IN_FRAC_WIDTH <= IN_WIDTH);
    assert (IN_FRAC_WIDTH >= 0);
    assert (OUT_FRAC_WIDTH <= OUT_WIDTH);
    assert (OUT_FRAC_WIDTH >= 0);
    assert (ROUND_FLOOR + ROUND_TRUNCATE + ROUND_NEAREST_INT_HALF_EVEN == 1);

    // TODO: Remove this
    assert (ROUND_FLOOR == 1);  // Currently only supports floor rounding
  end


  localparam ROUND_OUT_WIDTH = (OUT_FRAC_WIDTH > IN_FRAC_WIDTH) ?
                             IN_WIDTH + (OUT_FRAC_WIDTH - IN_FRAC_WIDTH) :
                             IN_WIDTH;

  logic [ROUND_OUT_WIDTH-1:0] round_out;

  floor_round #(
      .IN_WIDTH(IN_WIDTH),
      .OUT_WIDTH(ROUND_OUT_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
  ) floor_round_inst (
      .in_data (in_data),
      .out_data(round_out)
  );

  signed_clamp #(
      .IN_WIDTH (ROUND_OUT_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .SYMMETRIC(SYMMETRIC)
  ) clamp_inst (
      .in_data (round_out),
      .out_data(out_data)
  );

endmodule
