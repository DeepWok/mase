`timescale 1ns / 1ps
module fixed_isqrt #(
    parameter INT_WIDTH = 8,
    parameter FRAC_WIDTH = 0,
    parameter WIDTH = INT_WIDTH + FRAC_WIDTH,
    parameter MAX_NUM = (1 << WIDTH) - 1,
    parameter THREEHALFS = 2'b11 << (WIDTH - 2)
) (
    input logic[WIDTH-1:0] data_a,
    output logic[WIDTH-1:0] isqrt
);

    logic[2*WIDTH-1:0] y;
    logic[2*WIDTH-1:0] y_aug;
    logic[2*WIDTH-1:0] isqrt_hat;
    logic[2*WIDTH-1:0] x_red;
    logic[2*WIDTH-1:0] nr_stage;
    logic overflow;

    assign x_red = x;
    // TODO: get value from LUT
    assign y = (x_red == 1) ? x_red : nr_stage;
    
    range_augmentation #(
        .INT_WIDTH(INT_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .WIDTH(WIDTH)
    ) range_augmentation_0 (
        .data_a(y),
        .data_a_aug(y_aug)
    );

    assign overflow = (y_aug > MAX_NUM) ? 1'b1 : 1'b0;
    assign isqrt_hat = (overflow) ? MAX_NUM : y_aug;
    assign isqrt = (data_a == 0) ? MAX_NUM : isqrt_hat[WIDTH-1:0];

endmodule

/*
 * INPUT FORMAT: Q1.(WIDTH-1)
 * OUTPUT FORMAT: Q(INT_WIDTH).(FRAC_WIDTH)
 * */
`timescale 1ns / 1ps
module range_augmentation #(
    parameter INT_WIDTH = 8,
    parameter FRAC_WIDTH = 0,
    parameter WIDTH = INT_WIDTH + FRAC_WIDTH
) (
    input logic[2*WIDTH-1:0] data_a,
    output logic[2*WIDTH-1:0] data_a_aug
);

    // TODO: finish this design.
    assign data_a_aug = data_a >> 1;

endmodule


