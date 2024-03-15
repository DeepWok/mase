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
    localparam MSB_WIDTH = IN_WIDTH == 1 ? 1 : $clog2(IN_WIDTH),
    localparam ONE = 1 << (IN_WIDTH-1) // FORMAT: Q1.(WIDTH-1)
) (
    // TODO: stateless design would not need these pins.
    input logic clk,
    input logic rst,

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

    logic[2*IN_WIDTH-1:0] x_reduced [3:0];
    logic[MSB_WIDTH-1:0] msb_index [3:0];
    logic[2*IN_WIDTH-1:0] lut_index;
    logic[2*IN_WIDTH-1:0] lut_value [2:1];
    logic[2*IN_WIDTH-1:0] y [3:2];
    logic[2*IN_WIDTH-1:0] y_or_one;
    logic[2*IN_WIDTH-1:0] y_aug;

    fixed_range_reduction #(
        .WIDTH(IN_WIDTH)
    ) fixed_range_reduction_inst (
        .data_a(in_data),
        .data_out(x_reduced[0]),
        .msb_index(msb_index[0])
    );

    logic pipe_1_valid, pipe_1_ready;

    skid_buffer #(
        .DATA_WIDTH(2*IN_WIDTH + MSB_WIDTH)
    ) pipe_reg_0 (
        .clk(clk),
        .rst(rst),
        .data_in({x_reduced[0], msb_index[0]}),
        .data_in_valid(in_valid),
        .data_in_ready(in_ready),
        .data_out({x_reduced[1], msb_index[1]}),
        .data_out_valid(pipe_1_valid),
        .data_out_ready(pipe_1_ready)
    );

    fixed_lut_index #(
        .WIDTH(IN_WIDTH),
        .LUT_POW(LUT_POW)
    ) fixed_lut_index_inst (
        .data_a(x_reduced[1]),
        .data_b(msb_index[1]),
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
        .data(lut_value[1])
    );

    logic pipe_2_valid, pipe_2_ready;

    skid_buffer #(
        .DATA_WIDTH(2*IN_WIDTH + MSB_WIDTH + IN_WIDTH)
    ) pipe_reg_1 (
        .clk(clk),
        .rst(rst),
        .data_in({x_reduced[1], msb_index[1], lut_value[1]}),
        .data_in_valid(pipe_1_valid),
        .data_in_ready(pipe_1_ready),
        .data_out({x_reduced[2], msb_index[2], lut_value[2]}),
        .data_out_valid(pipe_2_valid),
        .data_out_ready(pipe_2_ready)
    );

    fixed_nr_stage #(
        .WIDTH(IN_WIDTH)
    ) fixed_nr_stage_inst_1 (
        .data_a(x_reduced[2]),
        .data_b(lut_value[2]),
        .data_out(y[2])
    );

    logic pipe_3_valid, pipe_3_ready;

    skid_buffer #(
        .DATA_WIDTH(2*IN_WIDTH + MSB_WIDTH + 2*IN_WIDTH)
    ) pipe_reg_2 (
        .clk(clk),
        .rst(rst),
        .data_in({y[2], msb_index[2], x_reduced[2]}),
        .data_in_valid(pipe_2_valid),
        .data_in_ready(pipe_2_ready),
        .data_out({y[3], msb_index[3], x_reduced[3]}),
        .data_out_valid(pipe_3_valid),
        .data_out_ready(pipe_3_ready)
    );

    assign y_or_one = (x_reduced[3] == ONE) ? ONE : y[3];

    fixed_range_augmentation #(
        .WIDTH(IN_WIDTH),
        .FRAC_WIDTH(IN_FRAC_WIDTH)
    ) fixed_range_augmentation_inst (
        .data_a(y_or_one),
        .data_b(msb_index[3]),
        .data_out(y_aug)
    );

    logic [2*IN_WIDTH-1:0] isqrt_data_out;

    assign isqrt_data_out =
        // Fishing for 0s.
        (x_reduced[3] == 0) ?
            MAX_NUM
            :
            (
                // Fishing for overflows.
                (y_aug > MAX_NUM) ?
                    MAX_NUM
                    :
                    y_aug
            );

    skid_buffer #(
        .DATA_WIDTH(2*IN_WIDTH)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(isqrt_data_out),
        .data_in_valid(pipe_3_valid),
        .data_in_ready(pipe_3_ready),
        .data_out(out_data),
        .data_out_valid(out_valid),
        .data_out_ready(out_ready)
    );

endmodule
