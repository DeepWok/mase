`timescale 1ns / 1ps
/* verilator lint_off UNUSEDSIGNAL */
module fixed_lut_index #(
    parameter WIDTH = 16,
    parameter LUT_POW = 5,
    localparam MSB_WIDTH = $clog2(WIDTH)
) (
    // X reduced
    input  logic [    WIDTH-1:0] data_a,   // FORMAT: Q(INT_WIDTH).(FRAC_WIDTH).
    // MSB index
    input  logic [MSB_WIDTH-1:0] data_b,   // FORMAT: Q(WIDTH).0.
    output logic [  LUT_POW-1:0] data_out  // FORMAT: Q(WIDTH).0.

);

  logic [WIDTH-1:0] temp;

  generate
    // Get rid of the MSB 1.
    if (WIDTH == 1) begin
      assign temp = 0;
    end else begin
      assign temp = {1'b0, data_a[WIDTH-2:0]};
    end

    // Multiply by LUT size.
    // Changing format: Q1.(WIDTH-1) to Q(WIDTH).0
    // Getting rid of fractional bits.
    if (LUT_POW - WIDTH + 1 <= 0) begin
      assign data_out = temp >> (WIDTH - LUT_POW - 1);
    end else begin
      assign data_out = temp << (LUT_POW - WIDTH + 1);
    end
  endgenerate

endmodule

/* verilator lint_on UNUSEDSIGNAL */
