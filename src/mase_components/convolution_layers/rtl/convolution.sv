`timescale 1ns / 1ps
module convolution #(
    // TODO: parameter name
    parameter DATA_WIDTH      = 16,
    parameter DATA_FRAC_WIDTH = 3,
    parameter W_WIDTH         = 8,
    parameter W_FRAC_WIDTH    = 4,
    parameter OUT_WIDTH       = 16,
    parameter OUT_FRAC_WIDTH  = 3,
    parameter BIAS_WIDTH      = 8,
    parameter BIAS_FRAC_WIDTH = 4,

    parameter IN_X    = 3,
    parameter IN_Y   = 2,
    parameter IN_C = 4,
    parameter UNROLL_IN_C = 2,

    parameter KERNEL_X = 2,
    parameter KERNEL_Y = 2,
    parameter OUT_C = 4,

    parameter UNROLL_KERNEL_OUT = 4,
    parameter UNROLL_OUT_C = 2,

    parameter SLIDING_NUM = 8,

    parameter BIAS_SIZE = UNROLL_OUT_C,
    parameter STRIDE    = 1,

    parameter PADDING_Y = 1,
    parameter PADDING_X = 2
) (
    input clk,
    input rst,

    input  [DATA_WIDTH - 1:0] data_in_0      [UNROLL_IN_C - 1 : 0],
    input                     data_in_0_valid,
    output                    data_in_0_ready,

    input  [W_WIDTH-1:0] weight      [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0],
    input                weight_valid,
    output               weight_ready,

    input  [BIAS_WIDTH-1:0] bias      [BIAS_SIZE-1:0],
    input                   bias_valid,
    output                  bias_ready,

    output [OUT_WIDTH - 1:0] data_out_0      [UNROLL_OUT_C - 1:0],
    output                   data_out_0_valid,
    input                    data_out_0_ready
);
  logic [DATA_WIDTH * UNROLL_IN_C - 1:0] packed_kernel[KERNEL_Y * KERNEL_X - 1:0];
  logic [DATA_WIDTH - 1:0] kernel[KERNEL_Y * KERNEL_X * UNROLL_IN_C - 1:0];
  logic kernel_valid;
  logic kernel_ready;

  localparam UNCAST_OUT_WIDTH = DATA_WIDTH + W_WIDTH + $clog2(KERNEL_Y * KERNEL_X * IN_C) + 1;
  localparam UNCAST_OUT_FRAC_WIDTH = DATA_FRAC_WIDTH + W_FRAC_WIDTH;
  logic [DATA_WIDTH * UNROLL_IN_C - 1:0] packed_data_in;
  logic [UNCAST_OUT_WIDTH - 1:0] uncast_data_out[UNROLL_OUT_C - 1:0];

  localparam KERNEL_SIZE = KERNEL_Y * KERNEL_X * UNROLL_IN_C;


  logic [DATA_WIDTH - 1:0] rolled_k[UNROLL_KERNEL_OUT - 1:0];
  logic rolled_k_valid;
  logic rolled_k_ready;
  for (genvar i = 0; i < UNROLL_IN_C; i++)
  for (genvar j = 0; j < DATA_WIDTH; j++) assign packed_data_in[i*DATA_WIDTH+j] = data_in_0[i][j];

  sliding_window #(
      .IMG_WIDTH     (IN_X),
      .IMG_HEIGHT    (IN_Y),
      .KERNEL_WIDTH  (KERNEL_X),
      .KERNEL_HEIGHT (KERNEL_Y),
      .PADDING_WIDTH (PADDING_X),
      .PADDING_HEIGHT(PADDING_Y),
      .CHANNELS      (IN_C / UNROLL_IN_C),
      .DATA_WIDTH    (UNROLL_IN_C * DATA_WIDTH),
      .STRIDE        (STRIDE)
      /* verilator lint_off PINMISSING */
  ) sw_inst (
      .data_in(packed_data_in),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),

      .data_out(packed_kernel),
      .data_out_valid(kernel_valid),
      .data_out_ready(kernel_ready),
      .*
  );
  /* verilator lint_on PINMISSING */
  for (genvar i = 0; i < KERNEL_Y * KERNEL_X; i++)
  for (genvar j = 0; j < UNROLL_IN_C; j++)
  for (genvar k = 0; k < DATA_WIDTH; k++)
    assign kernel[i*UNROLL_IN_C+j][k] = packed_kernel[i][j*DATA_WIDTH+k];

  roller #(
      .DATA_WIDTH(DATA_WIDTH),
      .NUM(KERNEL_SIZE),
      .ROLL_NUM(UNROLL_KERNEL_OUT)
  ) roller_inst (
      .data_in(kernel),
      .data_in_valid(kernel_valid),
      .data_in_ready(kernel_ready),
      .data_out(rolled_k),
      .data_out_valid(rolled_k_valid),
      .data_out_ready(rolled_k_ready),
      .*
  );
  logic [DATA_WIDTH-1:0] ib_rolled_k[UNROLL_KERNEL_OUT -1:0];
  logic ib_rolled_k_valid, ib_rolled_k_ready;
  input_buffer #(
      .IN_WIDTH(DATA_WIDTH),
      .IN_PARALLELISM(1),
      .IN_SIZE(UNROLL_KERNEL_OUT),
      .BUFFER_SIZE(KERNEL_Y * KERNEL_X * IN_C / UNROLL_KERNEL_OUT),
      .REPEAT(OUT_C / UNROLL_OUT_C)
  ) roller_buffer (
      .data_in(rolled_k),
      .data_in_valid(rolled_k_valid),
      .data_in_ready(rolled_k_ready),
      .data_out(ib_rolled_k),
      .data_out_valid(ib_rolled_k_valid),
      .data_out_ready(ib_rolled_k_ready),
      .*
  );

  logic [W_WIDTH-1:0] ib_weight[UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0];
  logic ib_weight_valid, ib_weight_ready;
  input_buffer #(
      .IN_WIDTH(W_WIDTH),
      .IN_PARALLELISM(UNROLL_OUT_C),
      .IN_SIZE(UNROLL_KERNEL_OUT),
      .BUFFER_SIZE(OUT_C * KERNEL_Y * KERNEL_X * IN_C / (UNROLL_KERNEL_OUT * UNROLL_OUT_C)),
      .REPEAT(SLIDING_NUM)
  ) weight_buffer (
      .data_in(weight),
      .data_in_valid(weight_valid),
      .data_in_ready(weight_ready),
      .data_out(ib_weight),
      .data_out_valid(ib_weight_valid),
      .data_out_ready(ib_weight_ready),
      .*
  );

  logic [BIAS_WIDTH-1:0] ib_bias[BIAS_SIZE -1:0];
  logic ib_bias_valid, ib_bias_ready;
  input_buffer #(
      .IN_WIDTH(BIAS_WIDTH),
      .IN_PARALLELISM(BIAS_SIZE),
      .IN_SIZE(1),
      .BUFFER_SIZE(OUT_C / UNROLL_OUT_C),
      .REPEAT(SLIDING_NUM)
  ) bias_buffer (
      .data_in(bias),
      .data_in_valid(bias_valid),
      .data_in_ready(bias_ready),
      .data_out(ib_bias),
      .data_out_valid(ib_bias_valid),
      .data_out_ready(ib_bias_ready),
      .*
  );
  fixed_linear #(
      .DATA_IN_0_PRECISION_0(DATA_WIDTH),
      .DATA_IN_0_PRECISION_1(DATA_FRAC_WIDTH),
      .DATA_IN_0_PARALLELISM_DIM_0(UNROLL_KERNEL_OUT),
      .IN_0_DEPTH(KERNEL_Y * KERNEL_X * IN_C / UNROLL_KERNEL_OUT),
      .WEIGHT_PRECISION_0(W_WIDTH),
      .WEIGHT_PRECISION_1(W_FRAC_WIDTH),
      .BIAS_PRECISION_0(BIAS_WIDTH),
      .BIAS_PRECISION_1(BIAS_FRAC_WIDTH),
      .DATA_OUT_0_PARALLELISM_DIM_0(UNROLL_OUT_C),
      .HAS_BIAS(1),
      /* verilator lint_off PINMISSING */
  ) fl_instance (
      .data_in_0(ib_rolled_k),
      .data_in_0_valid(ib_rolled_k_valid),
      .data_in_0_ready(ib_rolled_k_ready),
      .weight(ib_weight),
      .weight_valid(ib_weight_valid),
      .weight_ready(ib_weight_ready),
      .bias(ib_bias),
      .bias_valid(ib_bias_valid),
      .bias_ready(ib_bias_ready),
      .data_out_0(uncast_data_out),
      .data_out_0_valid(data_out_0_valid),
      .data_out_0_ready(data_out_0_ready),
      .*
  );
  fixed_rounding #(
      .IN_SIZE(UNROLL_OUT_C),
      .IN_WIDTH(UNCAST_OUT_WIDTH),
      .IN_FRAC_WIDTH(UNCAST_OUT_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH)
  ) inst_cast (
      .data_in (uncast_data_out),
      .data_out(data_out_0)
  );


endmodule
