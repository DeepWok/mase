/*
Module      : rms_norm_2d
Description : This module calculates root-mean-square norm.
              https://arxiv.org/abs/1910.07467
*/

`timescale 1ns/1ps

module rms_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0          = 4,
    parameter TOTAL_DIM1          = 4,
    parameter COMPUTE_DIM0        = 2,
    parameter COMPUTE_DIM1        = 2,
    parameter CHANNELS            = 2,

    // Data widths
    parameter IN_WIDTH            = 8,
    parameter IN_FRAC_WIDTH       = 2,
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

localparam NUM_VALUES = TOTAL_DIM0 * TOTAL_DIM1 * CHANNELS;
localparam logic signed [16:0] INV_NUM_VALUES = (1 << 16) / NUM_VALUES;

localparam NUM_ITERS = DEPTH_DIM0 * DEPTH_DIM1 * CHANNELS;
localparam ITER_WIDTH = $clog2(NUM_ITERS);

localparam SQUARE_WIDTH = IN_WIDTH * 2;
localparam SQUARE_FRAC_WIDTH = IN_FRAC_WIDTH * 2;

localparam ADDER_WIDTH = $clog2(COMPUTE_DIM0 * COMPUTE_DIM1) + SQUARE_WIDTH;
localparam ADDER_FRAC_WIDTH = SQUARE_FRAC_WIDTH;

localparam ACC_WIDTH = ADDER_WIDTH + ITER_WIDTH;
localparam ACC_FRAC_WIDTH = ADDER_FRAC_WIDTH;

localparam INV_SQRT_WIDTH = ACC_WIDTH;
localparam INV_SQRT_FRAC_WIDTH = ACC_FRAC_WIDTH;

localparam NORM_WIDTH = INV_SQRT_WIDTH + IN_WIDTH;
localparam NORM_FRAC_WIDTH = INV_SQRT_FRAC_WIDTH + IN_FRAC_WIDTH;

// Split2 for input to FIFO & Compute Pipeline
logic compute_in_valid, compute_in_ready;

split2 input_fifo_compute_split (
    .data_in_valid(in_valid),
    .data_in_ready(in_ready),
    .data_out_valid({fifo_in_valid, square_in_valid}),
    .data_out_ready({fifo_in_ready, squares[0].square_in_ready})
);

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
    .clk         (clk),
    .rst         (rst),
    .in_data     (in_data),
    .in_valid    (fifo_in_valid),
    .in_ready    (fifo_in_ready),
    .out_data    (fifo_data),
    .out_valid   (fifo_out_valid),
    .out_ready   (fifo_out_ready)
);

// Compute Squares
logic [SQUARE_WIDTH-1:0] square_batch [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic square_in_valid;
logic square_out_ready;

for (genvar i = 0; i < COMPUTE_DIM0*COMPUTE_DIM1; i++) begin : squares
    logic [SQUARE_WIDTH-1:0] square_in, square_out;
    logic square_in_ready;
    logic square_out_valid;
    assign square_in = $signed(fifo_data[i]) * $signed(fifo_data[i]);
    skid_buffer #(
        .DATA_WIDTH(SQUARE_WIDTH)
    ) square_reg (
        .clk(clk),
        .rst(rst),
        .data_in(square_in),
        .data_in_valid(square_in_valid),
        .data_in_ready(square_in_ready),
        .data_out(square_out),
        .data_out_valid(square_out_valid),
        .data_out_ready(square_out_ready)
    );
    assign square_batch[i] = square_out;
end

// Add reduce squares and accumulate
logic [ADDER_WIDTH-1:0] adder_tree_data;
logic adder_tree_valid, adder_tree_ready;

fixed_adder_tree #(
    .IN_SIZE(COMPUTE_DIM0 * COMPUTE_DIM1),
    .IN_WIDTH(SQUARE_WIDTH),
) adder_tree (
    .clk(clk),
    .rst(rst),
    .data_in(square_batch),
    .data_in_valid(squares[0].square_out_valid),
    .data_in_ready(square_out_ready),
    .data_out(adder_tree_data),
    .data_out_valid(adder_tree_valid),
    .data_out_ready(adder_tree_ready)
);

logic [ACC_WIDTH-1:0] squares_acc_data;
logic squares_acc_valid, squares_acc_ready;

fixed_accumulator #(
    .IN_DEPTH(NUM_ITERS),
    .IN_WIDTH(ADDER_WIDTH)
) squares_accumulator (
    .clk(clk),
    .rst(rst),
    .data_in(adder_tree_data),
    .data_in_valid(adder_tree_valid),
    .data_in_ready(adder_tree_ready),
    .data_out(squares_acc_data),
    .data_out_valid(squares_acc_valid),
    .data_out_ready(squares_acc_ready)
);

// Divide by N to get mean of squares
// TODO: change bitwidth here
logic [ACC_WIDTH-1:0] mean_in_data, mean_out_data;
logic mean_out_valid, mean_out_ready;

assign mean_in_data = ($signed(squares_acc_data) * INV_NUM_VALUES) >>> 16;

skid_buffer #(
    .DATA_WIDTH(ACC_WIDTH)
) mean_reg (
    .clk(clk),
    .rst(rst),
    .data_in(mean_in_data),
    .data_in_valid(squares_acc_valid),
    .data_in_ready(squares_acc_ready),
    .data_out(mean_out_data),
    .data_out_valid(mean_out_valid),
    .data_out_ready(mean_out_ready)
);

// Inverse Square Root of mean square
logic [INV_SQRT_WIDTH-1:0] inv_sqrt_data;
logic inv_sqrt_valid, inv_sqrt_ready;

fixed_isqrt #(
    .IN_WIDTH(ACC_WIDTH),
    .IN_FRAC_WIDTH(ACC_FRAC_WIDTH)
) inv_sqrt_inst (
    .in_data(mean_out_data),
    .in_valid(mean_out_valid),
    .in_ready(mean_out_ready),
    .out_data(inv_sqrt_data),
    .out_valid(inv_sqrt_valid),
    .out_ready(inv_sqrt_ready)
);

// Buffer the Inverse SQRT for NUM_ITERS
logic [INV_SQRT_WIDTH-1:0] inv_sqrt_buff_data;
logic inv_sqrt_buff_valid, inv_sqrt_buff_ready;

repeat_circular_buffer #(
    .DATA_WIDTH(INV_SQRT_WIDTH),
    .REPEAT(NUM_ITERS),
    .SIZE(1)
) inv_sqrt_circ_buffer (
    .clk(clk),
    .rst(rst),
    .in_data(inv_sqrt_data),
    .in_valid(inv_sqrt_valid),
    .in_ready(inv_sqrt_ready),
    .out_data(inv_sqrt_buff_data),
    .out_valid(inv_sqrt_buff_valid),
    .out_ready(inv_sqrt_buff_ready)
);

// Pipeline join: FIFO & Inverse SQRT Repeat Buffer
logic norm_in_valid;
join2 fifo_inv_sqrt_join2 (
    .data_in_valid({inv_sqrt_buff_valid, fifo_out_valid}),
    .data_in_ready({inv_sqrt_buff_ready, fifo_out_ready}),
    .data_out_valid(norm_in_valid),
    .data_out_ready(mult_cast[0].norm_in_ready)
);

// Batched Multiply & Output Casting
for (genvar i = 0; i < COMPUTE_DIM0*COMPUTE_DIM1; i++) begin : mult_cast

    // Multiplication with inverse sqrt
    logic [NORM_WIDTH-1:0] norm_in_data, norm_out_data;
    logic norm_in_ready;
    logic norm_out_valid, norm_out_ready;

    assign norm_in_data = $signed({1'b0, inv_sqrt_buff_data}) * $signed(fifo_data[i]);

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
        .data_out_ready(norm_out_ready)
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
        .data_in_ready(norm_out_ready),
        .data_out(output_reg_data),
        .data_out_valid(output_reg_valid),
        .data_out_ready(output_reg_ready)
    );

    assign batched_norm_out_data[i] = output_reg_data;
end

// Output assignments
logic [OUT_WIDTH-1:0] batched_norm_out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0];
logic output_reg_ready;

assign out_data = batched_norm_out_data;
assign out_valid = mult_cast[0].output_reg_valid;
assign output_reg_ready = out_ready;

endmodule
