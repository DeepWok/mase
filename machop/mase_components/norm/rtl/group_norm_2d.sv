/*
Module      : group_norm_2d
Description : This module calculates the generalised group norm.
              https://arxiv.org/abs/1803.08494v3

              This module can be easily trivially specialised into layer norm or
              instance norm by setting the GROUP_CHANNELS param to equal C or 1
              respectively.

              Group norm is independent of batch size, so the input shape is:
              (GROUP, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
*/

`timescale 1ns/1ps

module group_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter GROUP_CHANNELS      = 2,

    // Data widths
    parameter WIDTH               = 8,
    parameter FRAC_WIDTH          = 8
) (
    input  logic             clk,
    input  logic             rst,

    input  logic [WIDTH-1:0] in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic             in_valid,
    output logic             in_ready,

    output logic [WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic             out_valid,
    input  logic             out_ready
);

// Constant derived params
localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;

localparam NUM_ITERS = TOTAL_DIM0 * TOTAL_DIM1 * GROUP_CHANNELS;
localparam ITER_WIDTH = $clog2(NUM_ITERS);

// State
struct {
    logic [ITER_WIDTH+WIDTH-1:0] acc_sum;
    logic [ITER_WIDTH+WIDTH-1:0] sum;
    logic [ITER_WIDTH-1:0] sum_counter;
    logic [ITER_WIDTH+WIDTH-1:0] variance;
    logic [ITER_WIDTH-1:0] variance_counter;
} self, next_self;


localparam DATA_FLAT_WIDTH = WIDTH * COMPUTE_DIM0 * COMPUTE_DIM1;
localparam FIFO_DEPTH = GROUP_CHANNELS * DEPTH_DIM0 * DEPTH_DIM1;

logic [DATA_FLAT_WIDTH-1:0] in_data_flat, out_data_flat;
logic fifo_valid, fifo_ready;

matrix_flatten #(
    .DATA_WIDTH(WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) input_flatten (
    .data_in(in_data),
    .data_out(in_data_flat)
);

fifo #(
    .DEPTH(FIFO_DEPTH),
    .WIDTH(DATA_FLAT_WIDTH)
) fifo_inst (
    .in_data(in_data_flat),
    .in_valid(), // TODO
    .in_ready(),
    .out_data(out_data_flat),
    .out_valid(fifo_valid),
    .out_ready(fifo_ready),
    .full(),
    .empty(),
    .count()
);

matrix_unflatten #(
    .DATA_WIDTH(WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1)
) fifo_unflatten (
    .data_in(out_data_flat),
    .data_out(in_data_flat)
);

localparam ADDER_TREE_IN_SIZE = COMPUTE_DIM0 * COMPUTE_DIM1;
localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + WIDTH;

logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
logic adder_tree_valid, adder_tree_ready;

fixed_adder_tree #(
    .IN_SIZE(COMPUTE_DIM0 * COMPUTE_DIM1),
    .IN_WIDTH(WIDTH),
) sum_adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(in_data),
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out(adder_tree_data),
    .data_out_valid(adder_tree_valid),
    .data_out_ready(adder_tree_ready)
);

always_comb begin
    next_self = self;

    // Accumulation logic: mu
    if (adder_tree_valid && adder_tree_ready) begin
        if (self.sum_counter == NUM_ITERS-1) begin
            // Output sum
            next_self.sum = self.acc_sum + adder_tree_data;
            next_self.acc_sum = 0;
        end else begin
            // Accumulate temporary sum
            next_self.acc_sum = self.acc_sum + adder_tree_data;
        end
    end
end

always_ff @(posedge clk) begin
    if (rst) begin
        self <= '{default: '0};
    end else begin
        self <= next_self;
    end
end

endmodule
