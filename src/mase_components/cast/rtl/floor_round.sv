/*
Module      : floor_round
Description : Rounds a fixed point number towards negative infinity.
*/

`timescale 1ns / 1ps

module floor_round #(
    parameter IN_WIDTH       = 8,
    parameter OUT_WIDTH      = 8,
    parameter IN_FRAC_WIDTH  = 4,
    parameter OUT_FRAC_WIDTH = 4
) (
    input  logic signed [IN_WIDTH-1:0] in_data,
    output logic signed [ OUT_WIDTH:0] out_data
);

  initial begin
    if (OUT_FRAC_WIDTH > IN_FRAC_WIDTH) begin
      assert (OUT_WIDTH >= IN_WIDTH + (OUT_FRAC_WIDTH - IN_FRAC_WIDTH));
    end
  end

  generate
    if (OUT_FRAC_WIDTH > IN_FRAC_WIDTH) begin : gen_out_frac_larger
      assign out_data = in_data <<< (OUT_FRAC_WIDTH - IN_FRAC_WIDTH);
    end else if (OUT_FRAC_WIDTH == IN_FRAC_WIDTH) begin : gen_out_frac_same
      assign out_data = in_data;
    end else begin : gen_out_frac_smaller  // OUT_FRAC_WIDTH < IN_FRAC_WIDTH
      assign out_data = in_data >>> (IN_FRAC_WIDTH - OUT_FRAC_WIDTH);
    end
  endgenerate

endmodule
