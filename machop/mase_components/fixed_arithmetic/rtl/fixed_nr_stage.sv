`timescale 1ns / 1ps
module fixed_nr_stage #(
    parameter WIDTH = 16,
    localparam THREEHALFS = 3 << (WIDTH - 2)
) (
    // Input x.
    input logic[WIDTH-1:0] data_a,    // FORMAT: Q1.(WIDTH-1).
    // Initial LUT guess.
    input logic[WIDTH-1:0] data_b,    // FORMAT: Q1.(WIDTH-1).
    output logic[2*WIDTH-1:0] data_out  // FORMAT: Q1.(WIDTH-1)
);
    logic[2*WIDTH-1:0] yy;
    logic[2*WIDTH-1:0] mult;

    assign yy = (data_b * data_b) >> (WIDTH - 1);
    assign mult = ((data_a >> 1) * yy) >> (WIDTH - 1);
    assign data_out = (data_b * (THREEHALFS - mult)) >> (WIDTH - 1);

endmodule
