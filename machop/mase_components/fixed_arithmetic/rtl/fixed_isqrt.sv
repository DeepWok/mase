`timescale 1ns / 1ps
module fixed_isqrt #(
    parameter IN_WIDTH = 16,
    parameter IN_FRAC_WIDTH = 7,
    parameter LUT_POW = 5,
    // TODO: how to use these? Will the output width not always be the same as
    // the input width?
    parameter OUT_WIDTH = 16,
    parameter OUT_FRAC_WIDTH = 7,
    // TODO: the design is stateless therefore no cycles needed.
    // if the critical path is too large for this module then it can be
    // pipelined.
    parameter PIPELINE_CYCLES = 0,

    // LUT parameters
    parameter bit[IN_WIDTH-1:0] LUT00,
    parameter bit[IN_WIDTH-1:0] LUT01,
    parameter bit[IN_WIDTH-1:0] LUT02,
    parameter bit[IN_WIDTH-1:0] LUT03,
    parameter bit[IN_WIDTH-1:0] LUT04,
    parameter bit[IN_WIDTH-1:0] LUT05,
    parameter bit[IN_WIDTH-1:0] LUT06,
    parameter bit[IN_WIDTH-1:0] LUT07,
    parameter bit[IN_WIDTH-1:0] LUT08,
    parameter bit[IN_WIDTH-1:0] LUT09,
    parameter bit[IN_WIDTH-1:0] LUT10,
    parameter bit[IN_WIDTH-1:0] LUT11,
    parameter bit[IN_WIDTH-1:0] LUT12,
    parameter bit[IN_WIDTH-1:0] LUT13,
    parameter bit[IN_WIDTH-1:0] LUT14,
    parameter bit[IN_WIDTH-1:0] LUT15,
    parameter bit[IN_WIDTH-1:0] LUT16,
    parameter bit[IN_WIDTH-1:0] LUT17,
    parameter bit[IN_WIDTH-1:0] LUT18,
    parameter bit[IN_WIDTH-1:0] LUT19,
    parameter bit[IN_WIDTH-1:0] LUT20,
    parameter bit[IN_WIDTH-1:0] LUT21,
    parameter bit[IN_WIDTH-1:0] LUT22,
    parameter bit[IN_WIDTH-1:0] LUT23,
    parameter bit[IN_WIDTH-1:0] LUT24,
    parameter bit[IN_WIDTH-1:0] LUT25,
    parameter bit[IN_WIDTH-1:0] LUT26,
    parameter bit[IN_WIDTH-1:0] LUT27,
    parameter bit[IN_WIDTH-1:0] LUT28,
    parameter bit[IN_WIDTH-1:0] LUT29,
    parameter bit[IN_WIDTH-1:0] LUT30,
    parameter bit[IN_WIDTH-1:0] LUT31,

    localparam INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH,
    localparam MAX_NUM = (1 << IN_WIDTH) - 1,
    localparam MSB_WIDTH = $clog2(IN_WIDTH),
    localparam ONE = 1 << (IN_WIDTH-1) // FORMAT: Q1.(WIDTH-1)
) (
    // TODO: stateless design would not need these pins.
    // input logic clk,
    // input logic rst,

    input   logic[2*IN_WIDTH-1:0] in_data,
    // TODO: usage of these pins depends on whether or not the design is
    // pipelined whether.
    input   logic               in_valid,
    output  logic               in_ready,

    output  logic[2*IN_WIDTH-1:0] out_data,
    // TODO: usage of these pins depends on whether or not the design is
    // pipelined whether.
    output  logic               out_valid,
    input   logic               out_ready
);

    logic[2*IN_WIDTH-1:0] x_reduced;
    logic[MSB_WIDTH-1:0] msb_index;
    logic[2*IN_WIDTH-1:0] lut_index;
    logic[2*IN_WIDTH-1:0] lut_value;
    logic[2*IN_WIDTH-1:0] y;
    logic[2*IN_WIDTH-1:0] y_aug;

    fixed_range_reduction #(
        .WIDTH(IN_WIDTH)
    ) fixed_range_reduction_inst (
        .data_a(in_data),
        .data_out(x_reduced),
        .msb_index(msb_index)
    );

    fixed_lut_index #(
        .WIDTH(IN_WIDTH),
        .LUT_POW(LUT_POW)
    ) fixed_lut_index_inst (
        .data_a(x_reduced),
        .data_b(msb_index),
        .data_out(lut_index)
    );

    fixed_lut #(
        .WIDTH(IN_WIDTH),
        .LUT_POW(LUT_POW),
        .LUT00(LUT00), .LUT01(LUT01), .LUT02(LUT02), .LUT03(LUT03), .LUT04(LUT04),
        .LUT05(LUT05), .LUT06(LUT06), .LUT07(LUT07), .LUT08(LUT08), .LUT09(LUT09),
        .LUT10(LUT10), .LUT11(LUT11), .LUT12(LUT12), .LUT13(LUT13), .LUT14(LUT14),
        .LUT15(LUT15), .LUT16(LUT16), .LUT17(LUT17), .LUT18(LUT18), .LUT19(LUT19),
        .LUT20(LUT20), .LUT21(LUT21), .LUT22(LUT22), .LUT23(LUT23), .LUT24(LUT24),
        .LUT25(LUT25), .LUT26(LUT26), .LUT27(LUT27), .LUT28(LUT28), .LUT29(LUT29),
        .LUT30(LUT30), .LUT31(LUT31)
    ) fixed_lut_inst (
        .data_a(lut_index),
        .data_out(lut_value)
    );

    fixed_nr_stage #(
        .WIDTH(IN_WIDTH)
    ) fixed_nr_stage_inst_1 (
        .data_a(x_reduced),
        .data_b(lut_value),
        .data_out(y)
    );

    assign y = (x_reduced == ONE) ? ONE : y;

    fixed_range_augmentation #(
        .WIDTH(IN_WIDTH),
        .FRAC_WIDTH(IN_FRAC_WIDTH)
    ) fixed_range_augmentation_inst (
        .data_a(y),
        .data_b(msb_index),
        .data_out(y_aug)
    );

    assign out_data =
        // Fishing for 0s.
        (in_data == 0) ?
            MAX_NUM
            :
            (
                // Fishing for overflows.
                (y_aug > MAX_NUM) ?
                    MAX_NUM
                    :
                    y_aug
            );

    assign out_valid = in_valid;
    assign in_ready = out_ready;

endmodule
