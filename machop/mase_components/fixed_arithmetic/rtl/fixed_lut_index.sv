`timescale 1ns / 1ps
module fixed_lut_index #(
    parameter WIDTH = 16,
    parameter LUT_POW = 5,
    localparam MSB_WIDTH = $clog2(WIDTH)
) (
    // Input number x.
    input logic[2*WIDTH-1:0] data_a,            // FORMAT: Q(INT_WIDTH).(FRAC_WIDTH).
    // MSB index
    input logic[MSB_WIDTH-1:0] data_b,      // FORMAT: Q(WIDTH).0.
    output logic[LUT_POW-1:0] data_out          // FORMAT: Q(WIDTH).0.

);

    logic[2*WIDTH-1:0] temp;
    logic[2*WIDTH-1:0] temp2;
    logic[2*WIDTH-1:0] temp3;
    logic[2*WIDTH-1:0] temp4;

    assign temp = data_a << (WIDTH - 1 - data_b); // FORMAT: Q1.(WIDTH-1)
    // Subtract 1.
    assign temp2 = {1'b0, temp[WIDTH-2:0]};
    // Multiply by LUT size.
    assign temp3 = temp2 << (LUT_POW);
    // Changing format: Q1.(WIDTH-1) to Q(WIDTH).0
    // Getting rid of fractional bits.
    assign data_out = temp3 >> (WIDTH - 1);

endmodule
