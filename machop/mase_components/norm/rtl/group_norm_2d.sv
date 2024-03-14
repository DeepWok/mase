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
    parameter IN_WIDTH            = 8,
    parameter IN_FRAC_WIDTH       = 4,
    parameter OUT_WIDTH           = 8,
    parameter OUT_FRAC_WIDTH      = 4,

    // Inverse Sqrt LUT (TODO: Change to .mem file)
    parameter LUT00               = 0,
    parameter LUT01               = 0,
    parameter LUT02               = 0,
    parameter LUT03               = 0,
    parameter LUT04               = 0,
    parameter LUT05               = 0,
    parameter LUT06               = 0,
    parameter LUT07               = 0,
    parameter LUT08               = 0,
    parameter LUT09               = 0,
    parameter LUT10               = 0,
    parameter LUT11               = 0,
    parameter LUT12               = 0,
    parameter LUT13               = 0,
    parameter LUT14               = 0,
    parameter LUT15               = 0,
    parameter LUT16               = 0,
    parameter LUT17               = 0,
    parameter LUT18               = 0,
    parameter LUT19               = 0,
    parameter LUT20               = 0,
    parameter LUT21               = 0,
    parameter LUT22               = 0,
    parameter LUT23               = 0,
    parameter LUT24               = 0,
    parameter LUT25               = 0,
    parameter LUT26               = 0,
    parameter LUT27               = 0,
    parameter LUT28               = 0,
    parameter LUT29               = 0,
    parameter LUT30               = 0,
    parameter LUT31               = 0
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

localparam NUM_ITERS = DEPTH_DIM0 * DEPTH_DIM1 * GROUP_CHANNELS;
localparam ITER_WIDTH = $clog2(NUM_ITERS);

// Compute Pipeline Widths

localparam ADDER_TREE_IN_SIZE = COMPUTE_DIM0 * COMPUTE_DIM1;
localparam ADDER_TREE_OUT_WIDTH = $clog2(ADDER_TREE_IN_SIZE) + IN_WIDTH;

localparam ACC_OUT_WIDTH = ITER_WIDTH + ADDER_TREE_OUT_WIDTH;

localparam DIFF_WIDTH = IN_WIDTH + 1;
localparam DIFF_FRAC_WIDTH = IN_FRAC_WIDTH;

localparam SQUARE_WIDTH = DIFF_WIDTH * 2;
localparam SQUARE_FRAC_WIDTH = DIFF_FRAC_WIDTH * 2;

localparam SQUARES_ADDER_TREE_IN_SIZE = COMPUTE_DIM0 * COMPUTE_DIM1;
localparam SQUARES_ADDER_TREE_OUT_WIDTH = $clog2(SQUARES_ADDER_TREE_IN_SIZE) + SQUARE_WIDTH;
localparam SQUARES_ADDER_TREE_OUT_FRAC_WIDTH = SQUARE_FRAC_WIDTH;

localparam VARIANCE_WIDTH = ITER_WIDTH + SQUARES_ADDER_TREE_OUT_WIDTH;
localparam VARIANCE_FRAC_WIDTH = SQUARES_ADDER_TREE_OUT_FRAC_WIDTH;

// Must be same as variance
localparam ISQRT_WIDTH = 16;
localparam ISQRT_FRAC_WIDTH = VARIANCE_FRAC_WIDTH;

localparam NORM_WIDTH = ISQRT_WIDTH + DIFF_WIDTH;
localparam NORM_FRAC_WIDTH = ISQRT_FRAC_WIDTH + DIFF_FRAC_WIDTH;

// Input FIFO
logic [IN_WIDTH-1:0] fifo_data  [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic fifo_out_valid, fifo_out_ready;
logic fifo_in_valid, fifo_in_ready;

matrix_fifo #(
    .DATA_WIDTH  (IN_WIDTH),
    .DIM0        (COMPUTE_DIM0),
    .DIM1        (COMPUTE_DIM1),
    .FIFO_SIZE   (10*NUM_ITERS) // TODO: Change
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
logic [ADDER_TREE_OUT_WIDTH-1:0] adder_tree_data;
logic adder_tree_out_valid, adder_tree_out_ready;
logic adder_tree_in_valid, adder_tree_in_ready;

fixed_adder_tree #(
    .IN_SIZE(COMPUTE_DIM0 * COMPUTE_DIM1),
    .IN_WIDTH(IN_WIDTH)
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

logic [ACC_OUT_WIDTH-1:0] mu_acc_div;

// Division by NUM_VALUES
localparam INV_NUMVALUES_0 = ((1 << ACC_OUT_WIDTH) / NUM_VALUES);
assign mu_acc_div = ($signed(mu_acc) * $signed({1'b0, INV_NUMVALUES_0})) >>> ACC_OUT_WIDTH;
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
assign mu_fifo_ready = compute_pipe[0].diff_in_ready;

join2 mu_fifo_join2 (
    .data_in_valid({mu_out_valid, fifo_out_valid}),
    .data_in_ready({mu_out_ready, fifo_out_ready}),
    .data_out_valid(mu_fifo_valid),
    .data_out_ready(mu_fifo_ready)
);

// Compute pipeline
logic [SQUARE_WIDTH-1:0] square_out [COMPUTE_DIM0*COMPUTE_DIM1-1:0];

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
    logic [SQUARE_WIDTH-1:0] square_in;
    logic square_in_ready;
    logic square_out_valid;
    assign square_in = $signed(diff_batch_in[i]) * $signed(diff_batch_in[i]);

    skid_buffer #(
        .DATA_WIDTH(SQUARE_WIDTH)
    ) square_reg (
        .clk(clk),
        .rst(rst),
        .data_in(square_in),
        .data_in_valid(fifo_diff_out_valid),
        .data_in_ready(square_in_ready),
        .data_out(square_out[i]),
        .data_out_valid(square_out_valid),
        .data_out_ready(squares_adder_tree_in_ready)
    );
end

// Split2 for split in pipeline from diff
logic fifo_diff_in_valid, fifo_diff_in_ready;
logic fifo_diff_out_valid;

assign fifo_diff_in_valid = compute_pipe[0].diff_out_valid;
split2 fifo_diff_split (
    .data_in_valid(fifo_diff_in_valid),
    .data_in_ready(fifo_diff_in_ready),
    .data_out_valid({diff_batch_in_valid, fifo_diff_out_valid}),
    .data_out_ready({diff_batch_in_ready, compute_pipe[0].square_in_ready})
);

// Squares adder tree
logic [SQUARES_ADDER_TREE_OUT_WIDTH-1:0] squares_adder_tree_data;
logic squares_adder_tree_out_valid, squares_adder_tree_out_ready;
logic squares_adder_tree_in_valid, squares_adder_tree_in_ready;

assign squares_adder_tree_in_valid = compute_pipe[0].square_out_valid;

fixed_adder_tree #(
    .IN_SIZE(SQUARES_ADDER_TREE_IN_SIZE),
    .IN_WIDTH(SQUARE_WIDTH)
) squares_adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(square_out),
    .data_in_valid(squares_adder_tree_in_valid),
    .data_in_ready(squares_adder_tree_in_ready),
    .data_out(squares_adder_tree_data),
    .data_out_valid(squares_adder_tree_out_valid),
    .data_out_ready(squares_adder_tree_out_ready)
);

// Squares Accumulator
logic [VARIANCE_WIDTH-1:0] squares_acc;
logic squares_acc_valid, squares_acc_ready;

fixed_accumulator #(
    .IN_DEPTH(NUM_ITERS),
    .IN_WIDTH(SQUARES_ADDER_TREE_OUT_WIDTH)
) squares_accumulator (
    .clk(clk),
    .rst(rst),
    .data_in(squares_adder_tree_data),
    .data_in_valid(squares_adder_tree_out_valid),
    .data_in_ready(squares_adder_tree_out_ready),
    .data_out(squares_acc),
    .data_out_valid(squares_acc_valid),
    .data_out_ready(squares_acc_ready)
);

// Take the accumulated squares and divide it to get variance
logic [SQUARES_ADDER_TREE_OUT_WIDTH+VARIANCE_WIDTH:0] variance_buffer;
logic [VARIANCE_WIDTH-1:0] variance_in, variance_out;
logic variance_out_valid, variance_out_ready;

// Division by NUM_VALUES
localparam INV_NUMVALUES_1 = ((1 << SQUARES_ADDER_TREE_OUT_WIDTH) / NUM_VALUES);
assign variance_buffer = (squares_acc * INV_NUMVALUES_1) >> SQUARES_ADDER_TREE_OUT_WIDTH;
assign variance_in = variance_buffer[VARIANCE_WIDTH-1:0];

skid_buffer #(
    .DATA_WIDTH(VARIANCE_WIDTH)
) variance_reg (
    .clk(clk),
    .rst(rst),
    .data_in(variance_in),
    .data_in_valid(squares_acc_valid),
    .data_in_ready(squares_acc_ready),
    .data_out(variance_out),
    .data_out_valid(variance_out_valid),
    .data_out_ready(variance_out_ready)
);

// Clamp the variance
logic [ISQRT_WIDTH-1:0] variance_clamp_in, variance_clamp_out;
logic variance_clamp_valid, variance_clamp_ready;

always_comb begin
    if(variance_out > 2**ISQRT_WIDTH-1) begin
        variance_clamp_in = 2**ISQRT_WIDTH-1;
    end else begin
        variance_clamp_in = variance_out;
    end
end

skid_buffer #(
    .DATA_WIDTH(ISQRT_WIDTH)
) variance_cast_reg (
    .clk(clk),
    .rst(rst),
    .data_in(variance_clamp_in),
    .data_in_valid(variance_out_valid),
    .data_in_ready(variance_out_ready),
    .data_out(variance_clamp_out),
    .data_out_valid(variance_clamp_valid),
    .data_out_ready(variance_clamp_ready)
);

// Take inverse square root of variance
logic [ISQRT_WIDTH-1:0] inv_sqrt_data;
logic inv_sqrt_valid, inv_sqrt_ready;

fixed_isqrt #(
    .IN_WIDTH(ISQRT_WIDTH),
    .IN_FRAC_WIDTH(ISQRT_FRAC_WIDTH),
    // Module ignores these anyway
    // .OUT_WIDTH(ISQRT_WIDTH),
    // .OUT_FRAC_WIDTH(ISQRT_FRAC_WIDTH),
    .LUT00(LUT00), .LUT01(LUT01), .LUT02(LUT02), .LUT03(LUT03),
    .LUT04(LUT04), .LUT05(LUT05), .LUT06(LUT06), .LUT07(LUT07),
    .LUT08(LUT08), .LUT09(LUT09), .LUT10(LUT10), .LUT11(LUT11),
    .LUT12(LUT12), .LUT13(LUT13), .LUT14(LUT14), .LUT15(LUT15),
    .LUT16(LUT16), .LUT17(LUT17), .LUT18(LUT18), .LUT19(LUT19),
    .LUT20(LUT20), .LUT21(LUT21), .LUT22(LUT22), .LUT23(LUT23),
    .LUT24(LUT24), .LUT25(LUT25), .LUT26(LUT26), .LUT27(LUT27),
    .LUT28(LUT28), .LUT29(LUT29), .LUT30(LUT30), .LUT31(LUT31)
) inv_sqrt_inst (
    .in_data(variance_clamp_out),
    .in_valid(variance_clamp_valid),
    .in_ready(variance_clamp_ready),
    .out_data(inv_sqrt_data),
    .out_valid(inv_sqrt_valid),
    .out_ready(inv_sqrt_ready)
);

// Repeat circular buffer to hold inverse square root of variance during mult
logic [ISQRT_WIDTH-1:0] isqrt_circ_data;
logic isqrt_circ_valid, isqrt_circ_ready;

repeat_circular_buffer #(
    .DATA_WIDTH(ISQRT_WIDTH),
    .REPEAT(NUM_ITERS),
    .SIZE(1)
) isqrt_var_circ_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(inv_sqrt_data),
    .in_valid(inv_sqrt_valid),
    .in_ready(inv_sqrt_ready),
    .out_data(isqrt_circ_data),
    .out_valid(isqrt_circ_valid),
    .out_ready(isqrt_circ_ready)
);

// Join2 for pipeline join at sqrt and diff fifo
// logic inv_sqrt_ready;
logic norm_in_valid;
join2 diff_fifo_isqrt_join (
    .data_in_valid({diff_batch_out_valid, isqrt_circ_valid}),
    .data_in_ready({diff_batch_out_ready, isqrt_circ_ready}),
    .data_out_valid(norm_in_valid),
    .data_out_ready(out_mult_pipe[0].norm_in_ready)
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
    .FIFO_SIZE(10*NUM_ITERS) // TODO: Change
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

logic [NORM_WIDTH-1:0] norm_in_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic [NORM_WIDTH-1:0] norm_out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic [OUT_WIDTH-1:0] norm_round_out [COMPUTE_DIM0*COMPUTE_DIM1-1:0];

// Signals for cocotb monitor
logic norm_monitor_valid, norm_monitor_ready;
assign norm_monitor_valid = out_mult_pipe[0].norm_out_valid;
assign norm_monitor_ready = out_mult_pipe[0].norm_batch_ready;

// Output chunks compute pipeline: final multiply and output cast
for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : out_mult_pipe

    // Multiply difference with 1/sqrt(var) to get normalized result
    logic norm_in_ready;
    logic norm_out_valid, norm_batch_ready;
    assign norm_in_data[i] = $signed({1'b0, isqrt_circ_data}) * $signed(diff_batch_out[i]);

    skid_buffer #(
        .DATA_WIDTH(NORM_WIDTH)
    ) norm_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_in_data[i]),
        .data_in_valid(norm_in_valid),
        .data_in_ready(norm_in_ready),
        .data_out(norm_out_data[i]),
        .data_out_valid(norm_out_valid),
        .data_out_ready(norm_batch_ready)
    );

    // Output Rounding Stage
    fixed_signed_cast #(
        .IN_WIDTH(NORM_WIDTH),
        .IN_FRAC_WIDTH(NORM_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data(norm_out_data[i]),
        .out_data(norm_round_out[i])
    );

    logic output_reg_valid;
    skid_buffer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_round_out[i]),
        .data_in_valid(norm_out_valid),
        .data_in_ready(norm_batch_ready),
        .data_out(norm_batch_data[i]),
        .data_out_valid(output_reg_valid),
        .data_out_ready(output_reg_ready)
    );
end

// Final connection to output
logic [OUT_WIDTH-1:0] norm_batch_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic output_reg_ready;

assign out_data = norm_batch_data;
assign out_valid = out_mult_pipe[0].output_reg_valid;
assign output_reg_ready = out_ready;

endmodule
