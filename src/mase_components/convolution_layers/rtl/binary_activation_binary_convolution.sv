`timescale 1ns / 1ps
module binary_activation_binary_convolution #(
    // input feature map pixel precision
    parameter DATA_WIDTH      = 1,
    parameter DATA_FRAC_WIDTH = 0,
    // weight precision
    parameter W_WIDTH         = 1,
    parameter W_FRAC_WIDTH    = 0,
    // bias precision
    parameter BIAS_WIDTH      = 16,
    parameter BIAS_FRAC_WIDTH = 0,

    // input feature map dimension
    parameter IN_WIDTH    = 4,
    parameter IN_HEIGHT   = 2,
    parameter IN_CHANNELS = 2,

    // kernel size and output feature map dimension
    parameter KERNEL_WIDTH  = 3,
    parameter KERNEL_HEIGHT = 2,
    parameter OUT_CHANNELS  = 3,

    // data dimension consumed by the engine
    parameter IN_SIZE = 2,  // IN_SIZE is for fold across channel only
    parameter W_SIZE = 4,
    parameter OUT_SIZE = OUT_CHANNELS,

    parameter SLIDING_SIZE = 6,  // out_width * out_height = 2 * 3 = 6

    parameter BIAS_SIZE = OUT_SIZE,
    parameter STRIDE    = 2,

    parameter PADDING_WIDTH  = 2,
    parameter PADDING_HEIGHT = 1
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

    output [UNCAST_OUT_WIDTH - 1:0] data_out      [OUT_SIZE - 1:0],
    output                          data_out_valid,
    input                           data_out_ready

);
  logic [DATA_WIDTH * IN_SIZE - 1:0] packed_kernel[KERNEL_HEIGHT * KERNEL_WIDTH - 1:0];
  logic [DATA_WIDTH - 1:0] kernel[KERNEL_HEIGHT * KERNEL_WIDTH * IN_SIZE - 1:0];
  logic kernel_valid;
  logic kernel_ready;
  // Here only DATA_WIDTH is taken into account because 1 bit * N bits = N bits 
  // + 1 sign bit because the fmm_data_out is a signed number (the instance contains popcount)
  // + 1 for bias
  localparam UNCAST_OUT_WIDTH = DATA_WIDTH + $clog2(
      KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS
  ) + 1 + 1;
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
      .DATA_WIDTH(DATA_WIDTH),   // 1
      .NUM       (KERNEL_SIZE),  //KERNEL_HEIGHT * KERNEL_WIDTH * IN_SIZE;
      .IN_SIZE   (IN_SIZE),      //Channel IN_SIZE
      .ROLL_NUM  (W_SIZE)        //Channel W_SIZE
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

  logic [BIAS_WIDTH-1:0] ib_bias[BIAS_SIZE -1:0];
  logic ib_bias_valid, ib_bias_ready;
  input_buffer #(
      .IN_WIDTH(BIAS_WIDTH),
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
  binary_activation_binary_linear #(
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
      .data_in(rolled_k),  // DATA_WIDTH
      .data_in_valid(rolled_k_valid),
      .data_in_ready(rolled_k_ready),
      .weight(ib_weight),  // W_WIDTH = 1 bit
      .weight_valid(ib_weight_valid),
      .weight_ready(ib_weight_ready),
      .bias(ib_bias),  // BIAS_WIDTH
      .bias_valid(ib_bias_valid),
      .bias_ready(ib_bias_ready), // UNCAST_OUT_WIDTH                                             = DATA_WIDTH + $clog2(KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS) + 1;
      .data_out(uncast_data_out), // IN_WIDTH + 1 + $clog2(IN_SIZE) + $clog2(IN_DEPTH) + HAS_BIAS = DATA_WIDTH + 1 + $clog2(W_SIZE) + $clog2(KERNEL_HEIGHT * KERNEL_WIDTH * IN_CHANNELS / W_SIZE) + 1
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready),
      .*
  );
  assign data_out = uncast_data_out;

  // Disable in layer casting
  // if (DATA_FRAC_WIDTH != 0) begin
  //     fixed_cast #(
  //         .IN_SIZE(OUT_SIZE),
  //         .IN_WIDTH(UNCAST_OUT_WIDTH),
  //         .IN_FRAC_WIDTH(UNCAST_OUT_FRAC_WIDTH),
  //         .OUT_WIDTH(DATA_WIDTH),
  //         .OUT_FRAC_WIDTH(DATA_FRAC_WIDTH)
  //     ) inst_cast (
  //         .data_in (uncast_data_out),
  //         .data_out(data_out)
  //     );
  // end else begin
  //       integer_cast #(
  //         .IN_SIZE(OUT_SIZE),
  //         .IN_WIDTH(UNCAST_OUT_WIDTH),
  //         .IN_FRAC_WIDTH(UNCAST_OUT_FRAC_WIDTH),
  //         .OUT_WIDTH(DATA_WIDTH),
  //         .OUT_FRAC_WIDTH(DATA_FRAC_WIDTH)
  //     ) inst_cast (
  //         .data_in (uncast_data_out),
  //         .data_out(data_out)
  //     );
  // end


endmodule
