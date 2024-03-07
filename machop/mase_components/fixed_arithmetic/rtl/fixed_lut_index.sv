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

    assign data_out = (data_a >> data_b) - (1'b1 << LUT_POW);

endmodule
