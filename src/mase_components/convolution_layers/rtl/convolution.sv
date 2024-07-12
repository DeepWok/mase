`timescale 1ns / 1ps
module convolution #(
    // 
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter WEIGHT_PRECISION_0    = 8,
    parameter WEIGHT_PRECISION_1    = 4,
    parameter BIAS_PRECISION_0      = 8,
    parameter BIAS_PRECISION_1      = 4,

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
    parameter PADDING_X = 2,
    parameter HAS_BIAS  = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4
) (
    input clk,
    input rst,

    input  [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [UNROLL_IN_C - 1 : 0],
    input                                data_in_0_valid,
    output                               data_in_0_ready,

    input  [WEIGHT_PRECISION_0-1:0] weight      [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0],
    input                           weight_valid,
    output                          weight_ready,

    input  [BIAS_PRECISION_0-1:0] bias      [BIAS_SIZE-1:0],
    input                         bias_valid,
    output                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0      [UNROLL_OUT_C - 1:0],
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  initial begin
    assert (((UNROLL_IN_C * KERNEL_X * KERNEL_Y) % UNROLL_KERNEL_OUT == 0) & ((UNROLL_IN_C * KERNEL_X * KERNEL_Y) >= UNROLL_KERNEL_OUT) & (UNROLL_IN_C <= IN_C) & (UNROLL_OUT_C <= OUT_C) & (IN_C % UNROLL_IN_C==0)&(OUT_C % UNROLL_OUT_C==0))
    else $fatal("UNROLL parameter not set correctly");
  end

  localparam UNCAST_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_Y * KERNEL_X * IN_C
  ) + 1;
  localparam UNCAST_OUT_FRAC_WIDTH = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic [DATA_IN_0_PRECISION_0 * UNROLL_IN_C - 1:0] packed_data_in;
  logic [UNCAST_OUT_WIDTH - 1:0] uncast_data_out[UNROLL_OUT_C - 1:0];

  localparam ROLL_IN_NUM = KERNEL_Y * KERNEL_X * UNROLL_IN_C;


  logic [DATA_IN_0_PRECISION_0 - 1:0] rolled_k[UNROLL_KERNEL_OUT - 1:0];
  logic rolled_k_valid;
  logic rolled_k_ready;
  for (genvar i = 0; i < UNROLL_IN_C; i++)
  for (genvar j = 0; j < DATA_IN_0_PRECISION_0; j++)
    assign packed_data_in[i*DATA_IN_0_PRECISION_0+j] = data_in_0[i][j];

  logic [DATA_IN_0_PRECISION_0 * UNROLL_IN_C - 1:0] packed_kernel[KERNEL_Y * KERNEL_X - 1:0];
  logic [DATA_IN_0_PRECISION_0 - 1:0] kernel[KERNEL_Y * KERNEL_X * UNROLL_IN_C - 1:0];
  logic kernel_valid;
  logic kernel_ready;
  sliding_window #(
      .IMG_WIDTH     (IN_X),
      .IMG_HEIGHT    (IN_Y),
      .KERNEL_WIDTH  (KERNEL_X),
      .KERNEL_HEIGHT (KERNEL_Y),
      .PADDING_WIDTH (PADDING_X),
      .PADDING_HEIGHT(PADDING_Y),
      .CHANNELS      (IN_C / UNROLL_IN_C),
      .DATA_WIDTH    (UNROLL_IN_C * DATA_IN_0_PRECISION_0),
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
  for (genvar k = 0; k < DATA_IN_0_PRECISION_0; k++)
    assign kernel[i*UNROLL_IN_C+j][k] = packed_kernel[i][j*DATA_IN_0_PRECISION_0+k];

  roller #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .NUM(ROLL_IN_NUM),
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

  localparam ROUND_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_X * KERNEL_Y * IN_C
  );
  localparam ROUND_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic [ROUND_PRECISION_0 -1:0] round_in[UNROLL_OUT_C-1:0];
  convolution_arith #(
      // assume output will only unroll_out_channels
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
      .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
      .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
      .BIAS_PRECISION_0(BIAS_PRECISION_0),
      .BIAS_PRECISION_1(BIAS_PRECISION_1),
      .ROLL_IN_NUM(ROLL_IN_NUM),
      .ROLL_OUT_NUM(UNROLL_KERNEL_OUT),
      .IN_CHANNELS_DEPTH(IN_C / UNROLL_IN_C),
      .OUT_CHANNELS_PARALLELISM(UNROLL_OUT_C),
      .OUT_CHANNELS_DEPTH(OUT_C / UNROLL_OUT_C),
      .WEIGHT_REPEATS(SLIDING_NUM),
      .HAS_BIAS(HAS_BIAS)
  ) convolution_arith_inst (
      .data_in_0(rolled_k),
      .data_in_0_valid(rolled_k_valid),
      .data_in_0_ready(rolled_k_ready),
      .data_out_0(round_in),
      .*
  );

  fixed_rounding #(
      .IN_SIZE(UNROLL_OUT_C),
      .IN_WIDTH(ROUND_PRECISION_0),
      .IN_FRAC_WIDTH(ROUND_PRECISION_1),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  ) round_inst (
      .data_in (round_in),
      .data_out(data_out_0)
  );

endmodule
