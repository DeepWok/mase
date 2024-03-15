`timescale 1ns / 1ps
module fixed_isqrt #(
    parameter IN_WIDTH = 16,
    parameter IN_FRAC_WIDTH = 7,
    parameter LUT_POW = 5,
    // TODO: how to use these? Will the output width not always be the same as
    // the input width?
    // parameter OUT_WIDTH = 16,
    // parameter OUT_FRAC_WIDTH = 7,
    // TODO: the design is stateless therefore no cycles needed.
    // if the critical path is too large for this module then it can be
    // pipelined.
    // parameter PIPELINE_CYCLES = 0,

    // LUT parameters
    parameter string LUT_MEMFILE = "",

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
    logic[2*IN_WIDTH-1:0] y_or_one;
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

    lut #(
        .DATA_WIDTH(IN_WIDTH),
        .SIZE(2 ** LUT_POW),
        .OUTPUT_REG(0),
        .MEM_FILE(LUT_MEMFILE)
    ) fixed_lut_inst (
        .clk('0), // Tie off clock
        .addr(lut_index),
        .data(lut_value)
    );

    fixed_nr_stage #(
        .WIDTH(IN_WIDTH)
    ) fixed_nr_stage_inst_1 (
        .data_a(x_reduced),
        .data_b(lut_value),
        .data_out(y)
    );

    assign y_or_one = (x_reduced == ONE) ? ONE : y;

    fixed_range_augmentation #(
        .WIDTH(IN_WIDTH),
        .FRAC_WIDTH(IN_FRAC_WIDTH)
    ) fixed_range_augmentation_inst (
        .data_a(y_or_one),
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
