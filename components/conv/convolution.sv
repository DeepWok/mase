`timescale 1ns / 1ps
module convolution #(
    parameter DATA_WIDTH      = 16,
    parameter DATA_FRAC_WIDTH = 3,
    parameter W_WIDTH         = 8,
    parameter W_FRAC_WIDTH    = 4,
    parameter BIAS_WIDTH      = 8,
    parameter BIAS_FRAC_WIDTH = 4,

    parameter IN_WIDTH    = 3,
    parameter IN_HEIGHT   = 2,
    parameter IN_CHANNELS = 4,

    parameter KERNEL_WIDTH  = 2,
    parameter KERNEL_HEIGHT = 2,
    parameter OUT_CHANNELS  = 2,

    parameter IN_SIZE  = 2,
    parameter W_SIZE   = 4,
    parameter OUT_SIZE = OUT_CHANNELS,

    parameter SLIDING_SIZE = 8,

    parameter BIAS_SIZE = OUT_SIZE,
    parameter STRIDE    = 1,

    parameter PADDING_HEIGHT = 1,
    parameter PADDING_WIDTH  = 2
) (
    input clk,
    input rst,

    input  [DATA_WIDTH - 1:0] data_in      [IN_SIZE - 1 : 0],
    input                     data_in_valid,
    output                    data_in_ready,

    input  [W_WIDTH-1:0] weight      [W_SIZE * OUT_CHANNELS -1:0],
    input                weight_valid,
    output               weight_ready,

    input  [BIAS_WIDTH-1:0] bias      [BIAS_SIZE-1:0],
    input                   bias_valid,
    output                  bias_ready,

    output [DATA_WIDTH - 1:0] data_out      [OUT_SIZE - 1:0],
    output                    data_out_valid,
    input                     data_out_ready

);
  logic [DATA_WIDTH * IN_SIZE - 1:0] packed_kernel[KERNEL_HEIGHT * KERNEL_WIDTH - 1:0];
  logic [DATA_WIDTH - 1:0] kernel[KERNEL_HEIGHT * KERNEL_WIDTH * IN_SIZE - 1:0];
  logic kernel_valid;
  logic kernel_ready;

  localparam UNCAST_OUT_WIDTH = DATA_WIDTH + W_WIDTH + $clog2(
      KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS
  ) + 1;
  localparam UNCAST_OUT_FRAC_WIDTH = DATA_FRAC_WIDTH + W_FRAC_WIDTH;
  logic [DATA_WIDTH * IN_SIZE - 1:0] packed_data_in;
  logic [UNCAST_OUT_WIDTH - 1:0] uncast_data_out[OUT_SIZE - 1:0];

  localparam KERNEL_SIZE = KERNEL_HEIGHT * KERNEL_WIDTH * IN_SIZE;


  logic [DATA_WIDTH - 1:0] rolled_k[W_SIZE - 1:0];
  logic rolled_k_valid;
  logic rolled_k_ready;
  for (genvar i = 0; i < IN_SIZE; i++)
  for (genvar j = 0; j < DATA_WIDTH; j++) assign packed_data_in[i*DATA_WIDTH+j] = data_in[i][j];

  sliding_window #(
      .IMG_WIDTH     (IN_WIDTH),
      .IMG_HEIGHT    (IN_HEIGHT),
      .KERNEL_WIDTH  (KERNEL_WIDTH),
      .KERNEL_HEIGHT (KERNEL_HEIGHT),
      .PADDING_WIDTH (PADDING_WIDTH),
      .PADDING_HEIGHT(PADDING_HEIGHT),
      .CHANNELS      (IN_CHANNELS / IN_SIZE),
      .DATA_WIDTH    (IN_SIZE * DATA_WIDTH),
      .STRIDE        (STRIDE)
      /* verilator lint_off PINMISSING */
  ) sw_inst (
      .data_in(packed_data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),

      .data_out(packed_kernel),
      .data_out_valid(kernel_valid),
      .data_out_ready(kernel_ready),
      .*
  );
  /* verilator lint_on PINMISSING */
  for (genvar i = 0; i < KERNEL_HEIGHT * KERNEL_WIDTH; i++)
  for (genvar j = 0; j < IN_SIZE; j++)
  for (genvar k = 0; k < DATA_WIDTH; k++)
    assign kernel[i*IN_SIZE+j][k] = packed_kernel[i][j*DATA_WIDTH+k];

  roller #(
      .DATA_WIDTH(DATA_WIDTH),
      .NUM(KERNEL_SIZE),
      .IN_SIZE(IN_SIZE),
      .ROLL_NUM(W_SIZE)
  ) roller_inst (
      .data_in(kernel),
      .data_in_valid(kernel_valid),
      .data_in_ready(kernel_ready),
      .data_out(rolled_k),
      .data_out_valid(rolled_k_valid),
      .data_out_ready(rolled_k_ready),
      .*
  );

  logic [W_WIDTH-1:0] ib_weight[W_SIZE * OUT_CHANNELS -1:0];
  logic ib_weight_valid, ib_weight_ready;
  input_buffer #(
      .IN_WIDTH(W_WIDTH),
      .IN_PARALLELISM(OUT_CHANNELS),
      .IN_SIZE(W_SIZE),
      .BUFFER_SIZE(KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS / W_SIZE),
      .REPEAT(SLIDING_SIZE)
  ) weight_buffer (
      .data_in(weight),
      .data_in_valid(weight_valid),
      .data_in_ready(weight_ready),
      .data_out(ib_weight),
      .data_out_valid(ib_weight_valid),
      .data_out_ready(ib_weight_ready),
      .*
  );

  logic [W_WIDTH-1:0] ib_bias[BIAS_SIZE -1:0];
  logic ib_bias_valid, ib_bias_ready;
  input_buffer #(
      .IN_WIDTH(W_WIDTH),
      .IN_PARALLELISM(BIAS_SIZE),
      .IN_SIZE(1),
      .BUFFER_SIZE(1),
      .REPEAT(SLIDING_SIZE)
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
      .IN_WIDTH(DATA_WIDTH),
      .IN_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .IN_SIZE(W_SIZE),
      .IN_DEPTH(KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS / W_SIZE),

      .WEIGHT_WIDTH(W_WIDTH),
      .WEIGHT_FRAC_WIDTH(W_FRAC_WIDTH),

      .PARALLELISM(OUT_CHANNELS),

      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .HAS_BIAS(1)
      /* verilator lint_off PINMISSING */
  ) fl_instance (
      .data_in(rolled_k),
      .data_in_valid(rolled_k_valid),
      .data_in_ready(rolled_k_ready),
      .weight(ib_weight),
      .weight_valid(ib_weight_valid),
      .weight_ready(ib_weight_ready),
      .bias(ib_bias),
      .bias_valid(ib_bias_valid),
      .bias_ready(ib_bias_ready),
      .data_out(uncast_data_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready),
      .*
  );


  fixed_cast #(
      .IN_SIZE(OUT_SIZE),
      .IN_WIDTH(UNCAST_OUT_WIDTH),
      .IN_FRAC_WIDTH(UNCAST_OUT_FRAC_WIDTH),
      .OUT_WIDTH(DATA_WIDTH),
      .OUT_FRAC_WIDTH(DATA_FRAC_WIDTH)
  ) inst_cast (
      .data_in (uncast_data_out),
      .data_out(data_out)
  );


endmodule
