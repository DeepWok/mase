`timescale 1ns / 1ps
module fixed_patch_embed #(
    parameter IN_WIDTH        = 6,
    parameter IN_FRAC_WIDTH   = 1,
    parameter W_WIDTH         = 4,
    parameter W_FRAC_WIDTH    = 1,
    parameter BIAS_WIDTH      = 4,
    parameter BIAS_FRAC_WIDTH = 1,
    parameter OUT_WIDTH       = 6,
    parameter OUT_FRAC_WIDTH  = 1,

    parameter IN_C = 3,
    parameter IN_Y = 16,
    parameter IN_X = 16,

    parameter OUT_C = 4,
    parameter KERNEL_SIZE = 2,
    parameter KERNEL_Y = KERNEL_SIZE,
    parameter KERNEL_X = KERNEL_SIZE,

    parameter PADDING_Y = 0,
    parameter PADDING_X = 0,

    parameter UNROLL_KERNEL_OUT = 2,
    parameter UNROLL_OUT_C = 2,
    parameter UNROLL_IN_C = 2,

    parameter SLIDING_NUM = 8,

    parameter STRIDE = KERNEL_SIZE
) (
    input clk,
    input rst,

    input  [IN_WIDTH - 1:0] data_in      [UNROLL_IN_C - 1 : 0],
    input                   data_in_valid,
    output                  data_in_ready,

    input  [W_WIDTH-1:0] weight      [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0],
    input                weight_valid,
    output               weight_ready,

    input  [BIAS_WIDTH-1:0] bias      [UNROLL_OUT_C-1:0],
    input                   bias_valid,
    output                  bias_ready,

    output [OUT_WIDTH - 1:0] data_out      [UNROLL_OUT_C - 1:0],
    output                   data_out_valid,
    input                    data_out_ready
);

  convolution #(
      .DATA_WIDTH(IN_WIDTH),
      .DATA_FRAC_WIDTH(IN_FRAC_WIDTH),
      .W_WIDTH(W_WIDTH),
      .W_FRAC_WIDTH(W_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN_X(IN_X),
      .IN_Y(IN_Y),
      .IN_C(IN_C),
      .KERNEL_X(KERNEL_X),
      .KERNEL_Y(KERNEL_Y),
      .OUT_C(OUT_C),
      .UNROLL_IN_C(UNROLL_IN_C),
      .UNROLL_KERNEL_OUT(UNROLL_KERNEL_OUT),
      .UNROLL_OUT_C(UNROLL_OUT_C),
      .SLIDING_NUM(SLIDING_NUM),
      .STRIDE(STRIDE),
      .PADDING_Y(PADDING_Y),
      .PADDING_X(PADDING_X)
  ) conv_inst (
      .*
  );
endmodule
