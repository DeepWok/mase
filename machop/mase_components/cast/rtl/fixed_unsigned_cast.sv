/*
Module      : fixed_unsigned_cast
Description : Cast a fixed point unsigned number into another.

              Types of rounding when OUT_FRAC_WIDTH < IN_FRAC_WIDTH:
              - Floor
*/

`timescale 1ns / 1ps

module fixed_unsigned_cast #(
    parameter IN_WIDTH       = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter OUT_WIDTH      = 8,
    parameter OUT_FRAC_WIDTH = 4,

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

    // Currently only supports floor rounding
    assert (ROUND_FLOOR == 1);
  end

  localparam MAX_WIDTH = IN_WIDTH > OUT_WIDTH ? IN_WIDTH : OUT_WIDTH;

  localparam ROUND_OUT_WIDTH = (OUT_FRAC_WIDTH > IN_FRAC_WIDTH) ?
                               MAX_WIDTH + (OUT_FRAC_WIDTH - IN_FRAC_WIDTH) :
                               MAX_WIDTH;

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

  // Unsigned clamp
  always_comb begin
    if (&round_out[ROUND_OUT_WIDTH-1:ROUND_OUT_WIDTH-OUT_WIDTH]) begin
      out_data = '1;
    end else begin
      out_data = round_out[OUT_WIDTH-1:0];
    end
  end

endmodule
