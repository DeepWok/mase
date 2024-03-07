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
`default_nettype none

module group_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter GROUP_CHANNELS      = 2,

    // Data widths
    parameter IN_WIDTH            = 8,
    parameter IN_FRAC_WIDTH       = 4,
    parameter OUT_WIDTH           = 8,
    parameter OUT_FRAC_WIDTH      = 4
) (
    input  logic                 clk,
    input  logic                 rst,

    input  logic [IN_WIDTH-1:0]  in_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                 in_valid,
    output logic                 in_ready,

    output logic [OUT_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

// Derived params
localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;

localparam NUM_VALUES = TOTAL_DIM0 * TOTAL_DIM1 * GROUP_CHANNELS;
localparam logic signed [16:0] INV_NUM_VALUES = (1 << 16) / NUM_VALUES;

localparam NUM_ITERS = DEPTH_DIM0 * DEPTH_DIM1 * GROUP_CHANNELS;
localparam ITER_WIDTH = $clog2(NUM_ITERS);

localparam DIFF_WIDTH = IN_WIDTH + 1;
localparam DIFF_FRAC_WIDTH = IN_FRAC_WIDTH;

localparam VARIANCE_WIDTH = IN_WIDTH * 2;
localparam VARIANCE_FRAC_WIDTH = IN_FRAC_WIDTH * 2;

parameter INV_SQRT_WIDTH      = VARIANCE_WIDTH;
parameter INV_SQRT_FRAC_WIDTH = VARIANCE_FRAC_WIDTH;

localparam NORM_WIDTH = INV_SQRT_WIDTH + DIFF_WIDTH;
localparam NORM_FRAC_WIDTH = INV_SQRT_FRAC_WIDTH + DIFF_FRAC_WIDTH;

// Input FIFO
logic [IN_WIDTH-1:0] fifo_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic fifo_out_valid, fifo_out_ready;
logic fifo_in_valid, fifo_in_ready;

matrix_fifo #(
    .DATA_WIDTH  (IN_WIDTH),
    .DIM0        (COMPUTE_DIM0),
    .DIM1        (COMPUTE_DIM1),
    .FIFO_SIZE   (2*NUM_ITERS)
) input_fifo_inst (
    .clk(clk),
    .rst(rst),
    .in_data(in_data),
    .in_valid(fifo_in_valid),
    .in_ready(fifo_in_ready),
    .out_data(fifo_data),
    .out_valid(fifo_out_valid),
    .out_ready(fifo_out_ready)
);

// Input Adder Tree
localparam ADDER_TREE_IN_SIZE = COMPUTE_DIM0 * COMPUTE_DIM1;
localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + IN_WIDTH;

logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
logic adder_tree_out_valid, adder_tree_out_ready;
logic adder_tree_in_valid, adder_tree_in_ready;

fixed_adder_tree #(
    .IN_SIZE(COMPUTE_DIM0 * COMPUTE_DIM1),
    .IN_WIDTH(IN_WIDTH),
) sum_adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(in_data),
    .data_in_valid(adder_tree_in_valid),
    .data_in_ready(adder_tree_in_ready),
    .data_out(adder_tree_data),
    .data_out_valid(adder_tree_out_valid),
    .data_out_ready(adder_tree_out_ready)
);

// Split2 for input to FIFO & Adder Tree
split2 input_fifo_adder_split (
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out_valid({adder_tree_in_valid, fifo_in_valid}),
    .data_out_ready({adder_tree_in_ready, fifo_in_ready})
);

// Accumulator for mu
localparam ACC_OUT_WIDTH = $clog2(NUM_ITERS) + ADDER_TREE_OUT_WIDTH;

logic [ACC_OUT_WIDTH-1:0] mu_acc;
logic mu_acc_valid, mu_acc_ready;

fixed_accumulator #(
    .IN_DEPTH(NUM_ITERS),
    .IN_WIDTH(ADDER_TREE_OUT_WIDTH)
) mu_accumulator (
    .clk(clk),
    .rst(rst),
    .data_in(adder_tree_data),
    .data_in_valid(adder_tree_out_valid),
    .data_in_ready(adder_tree_out_ready),
    .data_out(mu_acc),
    .data_out_valid(mu_acc_valid),
    .data_out_ready(mu_acc_ready)
);

// Division logic for mu
logic [IN_WIDTH-1:0] mu_in, mu_out;
logic mu_out_valid, mu_out_ready;

logic [ACC_OUT_WIDTH+16-1:0] mu_acc_div;

assign mu_acc_div = ($signed(mu_acc) * INV_NUM_VALUES) >>> 16;
assign mu_in = mu_acc_div[IN_WIDTH-1:0];

repeat_circular_buffer #(
    .DATA_WIDTH(IN_WIDTH),
    .REPEAT(NUM_ITERS),
    .SIZE(1)
) mu_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(mu_in),
    .in_valid(mu_acc_valid),
    .in_ready(mu_acc_ready),
    .out_data(mu_out),
    .out_valid(mu_out_valid),
    .out_ready(mu_out_ready)
);

// Join 2 for combining fifo and mu buffer signals
logic mu_fifo_valid, mu_fifo_ready;

join2 mu_fifo_join2 (
    .data_in_valid({mu_out_valid, fifo_out_valid}),
    .data_in_ready({mu_out_ready, fifo_out_ready}),
    .data_out_valid(mu_fifo_valid),
    .data_out_ready(compute_pipe[0].diff_in_ready)
);

// Compute pipeline
for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : compute_pipe

    // Take the difference between input and mean: (X - mu)
    logic signed [DIFF_WIDTH-1:0] diff_in, diff_out;
    logic diff_in_ready;
    logic diff_out_valid;
    assign diff_in = $signed(fifo_data[i]) - $signed(mu_out);

    skid_buffer #(
        .DATA_WIDTH(DIFF_WIDTH)
    ) subtract_reg (
        .clk(clk),
        .rst(rst),
        .data_in(diff_in),
        .data_in_valid(mu_fifo_valid),
        .data_in_ready(diff_in_ready),
        .data_out(diff_out),
        .data_out_valid(diff_out_valid),
        .data_out_ready(fifo_diff_in_ready)
    );

    // Assign the output of diff int batch to be buffered
    assign diff_batch_in[i] = diff_out;

    // There will be a split in the pipline here, split2 is down below.

    // Take the difference and square it: (X - mu) ^ 2
    logic [VARIANCE_WIDTH-1:0] square_in, square_out;
    logic square_in_ready;
    logic square_out_valid, square_out_ready;
    assign square_in = diff_out * diff_out;

    skid_buffer #(
        .DATA_WIDTH(VARIANCE_WIDTH)
    ) square_reg (
        .clk(clk),
        .rst(rst),
        .data_in(square_in),
        .data_in_valid(fifo_diff_out_valid),
        .data_in_ready(square_in_ready),
        .data_out(square_out),
        .data_out_valid(square_out_valid),
        .data_out_ready(square_out_ready)
    );

    // Take the square and divide it to get variance: (X - mu) ^ 2 / N
    logic [VARIANCE_WIDTH-1:0] variance_in, variance_out;
    logic variance_out_valid, variance_out_ready;

    // TODO: This probably needs to change into multiplication + shift
    assign variance_in = square_out / NUM_VALUES;

    skid_buffer #(
        .DATA_WIDTH(VARIANCE_WIDTH)
    ) variance_reg (
        .clk(clk),
        .rst(rst),
        .data_in(variance_in),
        .data_in_valid(square_out_valid),
        .data_in_ready(square_out_ready),
        .data_out(variance_out),
        .data_out_valid(variance_out_valid),
        .data_out_ready(variance_out_ready)
    );

    // Take inverse square root of variance: 1/root(Var(X))
    logic [INV_SQRT_WIDTH-1:0] inv_sqrt_data;
    logic inv_sqrt_valid;

    fixed_isqrt #(
        .IN_WIDTH(VARIANCE_WIDTH),
        .IN_FRAC_WIDTH(VARIANCE_FRAC_WIDTH),
        .OUT_WIDTH(INV_SQRT_WIDTH),
        .OUT_FRAC_WIDTH(INV_SQRT_FRAC_WIDTH)
    ) inv_sqrt_inst (
        .in_data(variance_out),
        .in_valid(variance_out_valid),
        .in_ready(variance_out_ready),
        .out_data(inv_sqrt_data),
        .out_valid(inv_sqrt_valid),
        .out_ready(inv_sqrt_ready)
    );

    // Multiply difference with 1/sqrt(var) to get normalized result
    // Will need a join2 inserted to join the diff_buffer with the sqrt pipeline
    logic [NORM_WIDTH-1:0] norm_in_data;
    logic norm_in_ready;
    logic [NORM_WIDTH-1:0] norm_out_data;
    logic norm_out_valid, norm_batch_ready;

    assign norm_in_data = $signed({1'b0, inv_sqrt_data}) * $signed(diff_batch_out[i]);

    skid_buffer #(
        .DATA_WIDTH(NORM_WIDTH)
    ) norm_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_in_data),
        .data_in_valid(norm_in_valid),
        .data_in_ready(norm_in_ready),
        .data_out(norm_out_data),
        .data_out_valid(norm_out_valid),
        .data_out_ready(norm_batch_ready)
    );

    // Output Rounding Stage
    logic [OUT_WIDTH-1:0] norm_round_out;
    logic [OUT_WIDTH-1:0] output_reg_data;
    logic output_reg_valid;

    fixed_signed_cast #(
        .IN_WIDTH(NORM_WIDTH),
        .IN_FRAC_WIDTH(NORM_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data(norm_out_data),
        .out_data(norm_round_out)
    );

    skid_buffer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_round_out),
        .data_in_valid(norm_out_valid),
        .data_in_ready(norm_batch_ready),
        .data_out(output_reg_data),
        .data_out_valid(output_reg_valid),
        .data_out_ready(output_reg_ready)
    );

    assign norm_batch_data[i] = output_reg_data;
end

// Split2 for split in pipeline from diff
logic fifo_diff_in_ready;
logic fifo_diff_out_valid;
split2 fifo_diff_split (
    .data_in_valid(compute_pipe[0].diff_out_valid),
    .data_in_ready(fifo_diff_in_ready),
    .data_out_valid({diff_batch_in_valid, fifo_diff_out_valid}),
    .data_out_ready({diff_batch_in_ready, compute_pipe[0].square_in_ready})
);

// Join2 for pipeline join at sqrt and diff fifo
logic inv_sqrt_ready;
logic norm_in_valid;
join2 diff_fifo_sqrt_join (
    .data_in_valid({diff_batch_out_valid, compute_pipe[0].inv_sqrt_valid}),
    .data_in_ready({diff_batch_out_ready, inv_sqrt_ready}),
    .data_out_valid(norm_in_valid),
    .data_out_ready(compute_pipe[0].norm_in_ready)
);

// FIFO for storing X-mu differences
logic [DIFF_WIDTH-1:0] diff_batch_in [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic diff_batch_in_valid, diff_batch_in_ready;
logic [DIFF_WIDTH-1:0] diff_batch_out [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic diff_batch_out_valid, diff_batch_out_ready;

matrix_fifo #(
    .DATA_WIDTH(DIFF_WIDTH),
    .DIM0(COMPUTE_DIM0),
    .DIM1(COMPUTE_DIM1),
    .FIFO_SIZE(NUM_ITERS) // TODO: Change
) diff_fifo_inst (
    .clk(clk),
    .rst(rst),
    .in_data(diff_batch_in),
    .in_valid(diff_batch_in_valid),
    .in_ready(diff_batch_in_ready),
    .out_data(diff_batch_out),
    .out_valid(diff_batch_out_valid),
    .out_ready(diff_batch_out_ready)
);

// Final connection to output
logic [OUT_WIDTH-1:0] norm_batch_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic output_reg_ready;

assign out_data = norm_batch_data;
assign out_valid = compute_pipe[0].output_reg_valid;
assign output_reg_ready = out_ready;

endmodule
