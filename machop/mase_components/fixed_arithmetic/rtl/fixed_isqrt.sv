`timescale 1ns / 1ps
module fixed_isqrt #(
    parameter INT_WIDTH = 8,
    parameter FRAC_WIDTH = 8,
    parameter WIDTH = INT_WIDTH + FRAC_WIDTH,
    parameter MAX_NUM = (1 << WIDTH) - 1,
    parameter LUT_POW = 5,
    parameter THREEHALFS = 3 << (WIDTH - 2),
    parameter ONE = 1 << (WIDTH-1) // FORMAT: Q1.(WIDTH-1)
) (
    input logic[2*WIDTH-1:0] data_a,
    output logic[2*WIDTH-1:0] isqrt
);

    logic[2*WIDTH-1:0] x_reduced;
    logic[2*WIDTH-1:0] msb_index;
    logic[2*WIDTH-1:0] lut_index;
    logic[2*WIDTH-1:0] lut_value;
    logic[2*WIDTH-1:0] y;
    logic[2*WIDTH-1:0] y_aug;
    
    fixed_range_reduction #(
        .WIDTH(WIDTH)
    ) fixed_range_reduction_inst (
        .data_a(data_a),
        .data_out(x_reduced),
        .msb_index(msb_index)
    );

    fixed_lut_index #(
        .WIDTH(WIDTH),
        .LUT_POW(LUT_POW)
    ) fixed_lut_index_inst (
        .data_a(data_a),
        .data_b(msb_index),
        .data_out(lut_index)
    );

    fixed_lut #(
        .WIDTH(WIDTH),
        .LUT_POW(LUT_POW)
    ) fixed_lut_inst (
        .data_a(lut_index),
        .data_out(lut_value)
    );

    fixed_nr_stage #(
        .INT_WIDTH(INT_WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .WIDTH(WIDTH),
        .THREEHALFS(THREEHALFS) // TODO: make this a local parameter.
    ) fixed_nr_stage_inst_1 (
        .data_a(x_reduced),
        .data_b(lut_value),
        .data_out(y)
    );

    assign y = (x_reduced == ONE) ? x_reduced : y;
    
    fixed_range_augmentation #(
        .WIDTH(WIDTH),
        .FRAC_WIDTH(FRAC_WIDTH),
        .SQRT2(16'b1011010100000100), // TODO: make this a local parameter.
        .ISQRT2(16'b0101101010000010) // TODO: make this a local parameter.
    ) fixed_range_augmentation_inst (
        .data_a(y),
        .data_b(msb_index),
        .data_out(y_aug)
    );

    assign isqrt = 
        // Fishing for 0s.
        (data_a == 0) ? 
            MAX_NUM 
            : 
            (
                // Fishing for overflows.
                (y_aug > MAX_NUM) ?
                    MAX_NUM
                    :
                    y_aug
            );

endmodule

