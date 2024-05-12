/*
Module      : rms_norm_2d
Description : This module calculates root-mean-square norm.
              https://arxiv.org/abs/1910.07467

              RMS norm is independent of batch size, like layer norm so the
              input shape is:
              (CHANNELS, DEPTH_DIM1 * DEPTH_DIM0, COMPUTE_DIM1 * COMPUTE_DIM0)
*/

`timescale 1ns / 1ps

module rms_norm_2d #(
    // Dimensions
    parameter TOTAL_DIM0   = 4,
    parameter TOTAL_DIM1   = 4,
    parameter COMPUTE_DIM0 = 2,
    parameter COMPUTE_DIM1 = 2,
    parameter CHANNELS     = 2,

    // Data widths
    parameter IN_WIDTH         = 8,
    parameter IN_FRAC_WIDTH    = 2,
    parameter SCALE_WIDTH      = 8,
    parameter SCALE_FRAC_WIDTH = 2,
    parameter OUT_WIDTH        = 8,
    parameter OUT_FRAC_WIDTH   = 4,

    // Inverse Sqrt LUT
    parameter ISQRT_LUT_MEMFILE = "",
    parameter ISQRT_LUT_POW     = 5
) (
    input logic clk,
    input logic rst,

    input  logic [IN_WIDTH-1:0] in_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                in_valid,
    output logic                in_ready,

    input  logic [SCALE_WIDTH-1:0] weight_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    input  logic                   weight_valid,
    output logic                   weight_ready,

    output logic [OUT_WIDTH-1:0] out_data [COMPUTE_DIM0*COMPUTE_DIM1-1:0],
    output logic                 out_valid,
    input  logic                 out_ready
);

  // Derived params
  localparam DEPTH_DIM0 = TOTAL_DIM0 / COMPUTE_DIM0;
  localparam DEPTH_DIM1 = TOTAL_DIM1 / COMPUTE_DIM1;

  localparam NUM_VALUES = TOTAL_DIM0 * TOTAL_DIM1 * CHANNELS;
  localparam NUM_ITERS = DEPTH_DIM0 * DEPTH_DIM1 * CHANNELS;
  localparam ITER_WIDTH = $clog2(NUM_ITERS);

  localparam SQUARE_WIDTH = IN_WIDTH * 2;
  localparam SQUARE_FRAC_WIDTH = IN_FRAC_WIDTH * 2;

  localparam ADDER_WIDTH = $clog2(COMPUTE_DIM0 * COMPUTE_DIM1) + SQUARE_WIDTH;
  localparam ADDER_FRAC_WIDTH = SQUARE_FRAC_WIDTH;

  localparam ACC_WIDTH = ADDER_WIDTH + ITER_WIDTH;
  localparam ACC_FRAC_WIDTH = ADDER_FRAC_WIDTH;

  localparam bit [ACC_WIDTH+1:0] INV_NUMVALUES_0 = ((1 << ACC_WIDTH) / NUM_VALUES);

  localparam ISQRT_WIDTH = ACC_WIDTH;
  localparam ISQRT_FRAC_WIDTH = ACC_FRAC_WIDTH;

  localparam NORM_WIDTH = ISQRT_WIDTH + IN_WIDTH;
  localparam NORM_FRAC_WIDTH = ISQRT_FRAC_WIDTH + IN_FRAC_WIDTH;

  localparam AFFINE_WIDTH = NORM_WIDTH + SCALE_WIDTH;
  localparam AFFINE_FRAC_WIDTH = NORM_FRAC_WIDTH + SCALE_FRAC_WIDTH;

  // Input FIFO
  logic [IN_WIDTH-1:0] fifo_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic fifo_out_valid, fifo_out_ready;
  logic fifo_in_valid, fifo_in_ready;

  // Compute Squares
  logic square_in_valid;
  logic square_out_ready;

  logic [SQUARE_WIDTH-1:0] square_in[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [SQUARE_WIDTH-1:0] square_out[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic square_in_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic square_out_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  // Add reduce squares and accumulate
  logic [ADDER_WIDTH-1:0] adder_tree_data;
  logic adder_tree_valid, adder_tree_ready;

  logic [ACC_WIDTH-1:0] squares_acc_data;
  logic squares_acc_valid, squares_acc_ready;

  logic [ACC_WIDTH-1:0] mean_in_data, mean_out_data;
  logic mean_out_valid, mean_out_ready;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [ACC_WIDTH*2+1:0] mean_in_buffer;
  /* verilator lint_on UNUSEDSIGNAL */
  // Inverse Square Root of mean square
  logic [ISQRT_WIDTH-1:0] inv_sqrt_data;
  logic inv_sqrt_valid, inv_sqrt_ready;


  // Buffer the Inverse SQRT for NUM_ITERS
  logic [ISQRT_WIDTH-1:0] inv_sqrt_buff_data;
  logic inv_sqrt_buff_valid, inv_sqrt_buff_ready;

  logic norm_in_valid;

  // Batched Multiply & Output Casting
  logic [NORM_WIDTH-1:0] norm_in_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [NORM_WIDTH-1:0] norm_out_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [AFFINE_WIDTH-1:0] affine_in_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [AFFINE_WIDTH-1:0] affine_out_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  logic [OUT_WIDTH-1:0] affine_round_out[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic [OUT_WIDTH-1:0] output_reg_data[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  logic affine_in_valid;
  logic norm_out_ready;

  logic norm_in_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic norm_out_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  logic affine_in_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic affine_out_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic affine_out_ready[COMPUTE_DIM0*COMPUTE_DIM1-1:0];

  logic output_reg_valid[COMPUTE_DIM0*COMPUTE_DIM1-1:0];
  logic output_reg_ready;

  split2 input_fifo_compute_split (
      .data_in_valid (in_valid),
      .data_in_ready (in_ready),
      .data_out_valid({fifo_in_valid, square_in_valid}),
      .data_out_ready({fifo_in_ready, square_in_ready[0]})
  );

  matrix_fifo #(
      .DATA_WIDTH(IN_WIDTH),
      .DIM0      (COMPUTE_DIM0),
      .DIM1      (COMPUTE_DIM1),
      .FIFO_SIZE (4 * NUM_ITERS)
  ) input_fifo_inst (
      .clk      (clk),
      .rst      (rst),
      .in_data  (in_data),
      .in_valid (fifo_in_valid),
      .in_ready (fifo_in_ready),
      .out_data (fifo_data),
      .out_valid(fifo_out_valid),
      .out_ready(fifo_out_ready)
  );


  for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : squares
    assign square_in[i] = $signed(in_data[i]) * $signed(in_data[i]);
    skid_buffer #(
        .DATA_WIDTH(SQUARE_WIDTH)
    ) square_reg (
        .clk(clk),
        .rst(rst),
        .data_in(square_in[i]),
        .data_in_valid(square_in_valid),
        .data_in_ready(square_in_ready[i]),
        .data_out(square_out[i]),
        .data_out_valid(square_out_valid[i]),
        .data_out_ready(square_out_ready)
    );
  end

  fixed_adder_tree #(
      .IN_SIZE (COMPUTE_DIM0 * COMPUTE_DIM1),
      .IN_WIDTH(SQUARE_WIDTH)
  ) adder_tree (
      .clk(clk),
      .rst(rst),
      .data_in(square_out),
      .data_in_valid(square_out_valid[0]),
      .data_in_ready(square_out_ready),
      .data_out(adder_tree_data),
      .data_out_valid(adder_tree_valid),
      .data_out_ready(adder_tree_ready)
  );

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

  // Division by NUM_VALUES
  assign mean_in_buffer = ($signed(
      squares_acc_data
  ) * $signed(
      {1'b0, INV_NUMVALUES_0}
  )) >> ACC_WIDTH;
  assign mean_in_data = mean_in_buffer[ACC_WIDTH-1:0];
  // assign mean_in_data = ($signed(squares_acc_data) * INV_NUM_VALUES) >>> 16;

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

  fixed_isqrt #(
      .IN_WIDTH(ISQRT_WIDTH),
      .IN_FRAC_WIDTH(ISQRT_FRAC_WIDTH),
      .LUT_MEMFILE(ISQRT_LUT_MEMFILE),
      .LUT_POW(ISQRT_LUT_POW)
  ) inv_sqrt_inst (
      .clk(clk),
      .rst(rst),
      .in_data(mean_out_data),
      .in_valid(mean_out_valid),
      .in_ready(mean_out_ready),
      .out_data(inv_sqrt_data),
      .out_valid(inv_sqrt_valid),
      .out_ready(inv_sqrt_ready)
  );

  single_element_repeat #(
      .DATA_WIDTH(ISQRT_WIDTH),
      .REPEAT(NUM_ITERS)
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
  join2 fifo_inv_sqrt_join2 (
      .data_in_valid ({inv_sqrt_buff_valid, fifo_out_valid}),
      .data_in_ready ({inv_sqrt_buff_ready, fifo_out_ready}),
      .data_out_valid(norm_in_valid),
      .data_out_ready(norm_in_ready[0])
  );

  join2 weight_norm_affine_join (
      .data_in_valid ({weight_valid, norm_out_valid[0]}),
      .data_in_ready ({weight_ready, norm_out_ready}),
      .data_out_valid(affine_in_valid),
      .data_out_ready(affine_in_ready[0])
  );

  for (genvar i = 0; i < COMPUTE_DIM0 * COMPUTE_DIM1; i++) begin : mult_cast

    // Multiplication with inverse sqrt

    assign norm_in_data[i] = $signed({1'b0, inv_sqrt_buff_data}) * $signed(fifo_data[i]);

    skid_buffer #(
        .DATA_WIDTH(NORM_WIDTH)
    ) norm_reg (
        .clk(clk),
        .rst(rst),
        .data_in(norm_in_data[i]),
        .data_in_valid(norm_in_valid),
        .data_in_ready(norm_in_ready[i]),
        .data_out(norm_out_data[i]),
        .data_out_valid(norm_out_valid[i]),
        .data_out_ready(norm_out_ready)
    );

    // Here is where the join2 between norm_out & weight input is placed

    // Affine Scale Transform

    assign affine_in_data[i] = $signed(norm_out_data[i]) * $signed(weight_data[i]);

    skid_buffer #(
        .DATA_WIDTH(AFFINE_WIDTH)
    ) affine_reg (
        .clk(clk),
        .rst(rst),
        .data_in(affine_in_data[i]),
        .data_in_valid(affine_in_valid),
        .data_in_ready(affine_in_ready[i]),
        .data_out(affine_out_data[i]),
        .data_out_valid(affine_out_valid[i]),
        .data_out_ready(affine_out_ready[i])
    );

    // Output Rounding Stage

    fixed_signed_cast #(
        .IN_WIDTH(AFFINE_WIDTH),
        .IN_FRAC_WIDTH(AFFINE_FRAC_WIDTH),
        .OUT_WIDTH(OUT_WIDTH),
        .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
        .SYMMETRIC(0),
        .ROUND_FLOOR(1)
    ) output_cast (
        .in_data (affine_out_data[i]),
        .out_data(affine_round_out[i])
    );

    skid_buffer #(
        .DATA_WIDTH(OUT_WIDTH)
    ) output_reg (
        .clk(clk),
        .rst(rst),
        .data_in(affine_round_out[i]),
        .data_in_valid(affine_out_valid[i]),
        .data_in_ready(affine_out_ready[i]),
        .data_out(output_reg_data[i]),
        .data_out_valid(output_reg_valid[i]),
        .data_out_ready(output_reg_ready)
    );
  end

  // Output assignments
  assign out_data = output_reg_data;
  assign out_valid = output_reg_valid[0];
  assign output_reg_ready = out_ready;

endmodule
