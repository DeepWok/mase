`timescale 1ns / 1ps
module fixed_nr_stage #(
    parameter INT_WIDTH = 1,
    parameter FRAC_WIDTH = 15,
    parameter WIDTH = INT_WIDTH + FRAC_WIDTH,
    parameter THREEHALFS = 3 << (WIDTH - 2)
) (
    input logic[2*WIDTH-1:0] data_a, // FORMAT: Q1.(WIDTH-1).
    input logic[2*WIDTH-1:0] data_b, // FORMAT: Q1.(WIDTH-1).
    output logic[2*WIDTH-1:0] data_out // FORMAT: Q1.(WIDTH-1)
);
    logic[2*WIDTH-1:0] yy;
    logic[2*WIDTH-1:0] mult;

    assign yy = (data_b * data_b) >> (WIDTH - 1);
    assign mult = ((data_a >> 1) * yy) >> (WIDTH - 1);
    assign data_out = (data_b * (THREEHALFS - mult)) >> (WIDTH - 1);

endmodule
