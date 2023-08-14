
// =====================================
//     Mase Hardware
//     Model: top
//     05/08/2023 02:19:08
// =====================================
`timescale 1ns / 1ps
module top #(
    parameter fc1_IN_0_WIDTH = 32,
    parameter fc1_IN_0_FRAC_WIDTH = 0,
    parameter fc1_WEIGHT_WIDTH = 1,
    parameter fc1_WEIGHT_FRAC_WIDTH = 0,
    parameter fc1_BIAS_WIDTH = 1,
    parameter fc1_BIAS_FRAC_WIDTH = 0,
    parameter fc1_OUT_0_WIDTH = 32,
    parameter fc1_OUT_0_FRAC_WIDTH = 0,
    parameter relu_IN_0_WIDTH = 32,
    parameter relu_IN_0_FRAC_WIDTH = 0,
    parameter relu_OUT_0_WIDTH = 32,
    parameter relu_OUT_0_FRAC_WIDTH = 0,
    parameter fc2_IN_0_WIDTH = 32,
    parameter fc2_IN_0_FRAC_WIDTH = 0,
    parameter fc2_WEIGHT_WIDTH = 1,
    parameter fc2_WEIGHT_FRAC_WIDTH = 0,
    parameter fc2_BIAS_WIDTH = 1,
    parameter fc2_BIAS_FRAC_WIDTH = 0,
    parameter fc2_OUT_0_WIDTH = 32,
    parameter fc2_OUT_0_FRAC_WIDTH = 0,
    parameter fc1_HAS_BIAS = 1,
    parameter fc1_IN_0_SIZE = 1,
    parameter fc1_IN_0_DEPTH = 784,
    parameter fc1_PARALLELISM = 784,
    parameter fc1_WEIGHT_SIZE = 784,
    parameter fc1_OUT_0_SIZE = 784,
    parameter fc1_BIAS_SIZE = 784,
    parameter relu_IN_0_SIZE = 1,
    parameter relu_OUT_0_SIZE = 1,
    parameter fc2_HAS_BIAS = 1,
    parameter fc2_IN_0_SIZE = 1,
    parameter fc2_IN_0_DEPTH = 784,
    parameter fc2_PARALLELISM = 3136,
    parameter fc2_WEIGHT_SIZE = 3136,
    parameter fc2_OUT_0_SIZE = 3136,
    parameter fc2_BIAS_SIZE = 3136,

    parameter IN_WIDTH  = fc1_IN_0_WIDTH,
    parameter OUT_WIDTH = fc2_OUT_0_WIDTH,
    parameter IN_SIZE   = fc1_IN_0_SIZE,
    parameter OUT_SIZE  = fc2_OUT_0_SIZE
) (
    input clk,
    input rst,

    input [IN_WIDTH-1:0] data_in[IN_SIZE-1:0],
    input data_in_valid,
    output data_in_ready,

    // [1][784]
    input  [fc1_WEIGHT_WIDTH-1:0] fc1_weight      [fc1_WEIGHT_SIZE-1:0],
    input                         fc1_weight_valid,
    output                        fc1_weight_ready,
    // [1][784]
    input  [  fc1_BIAS_WIDTH-1:0] fc1_bias        [  fc1_BIAS_SIZE-1:0],
    input                         fc1_bias_valid,
    output                        fc1_bias_ready,

    // [1][3136]
    input  [fc2_WEIGHT_WIDTH-1:0] fc2_weight      [fc2_WEIGHT_SIZE-1:0],
    input                         fc2_weight_valid,
    output                        fc2_weight_ready,
    // [1][3136]
    input  [  fc2_BIAS_WIDTH-1:0] fc2_bias        [  fc2_BIAS_SIZE-1:0],
    input                         fc2_bias_valid,
    output                        fc2_bias_ready,

    output [OUT_WIDTH-1:0] data_out[OUT_SIZE-1:0],
    output data_out_valid,
    input data_out_ready
);

  // --------------------------
  //   fc1 signals
  // --------------------------
  // [32][1]
  logic [  fc1_IN_0_WIDTH-1:0] fc1_data_in_0         [  fc1_IN_0_SIZE-1:0];
  logic                        fc1_data_in_0_valid;
  logic                        fc1_data_in_0_ready;
  // // [1][784]
  // logic [fc1_WEIGHT_WIDTH-1:0]  fc1_weight        [fc1_WEIGHT_SIZE-1:0];
  // logic                             fc1_weight_valid;
  // logic                             fc1_weight_ready;
  // // [1][784]
  // logic [fc1_BIAS_WIDTH-1:0]  fc1_bias        [fc1_BIAS_SIZE-1:0];
  // logic                             fc1_bias_valid;
  // logic                             fc1_bias_ready;
  // [32][784]
  logic [ fc1_OUT_0_WIDTH-1:0] fc1_data_out_0        [ fc1_OUT_0_SIZE-1:0];
  logic                        fc1_data_out_0_valid;
  logic                        fc1_data_out_0_ready;

  // --------------------------
  //   relu signals
  // --------------------------
  // [32][1]
  logic [ relu_IN_0_WIDTH-1:0] relu_data_in_0        [ relu_IN_0_SIZE-1:0];
  logic                        relu_data_in_0_valid;
  logic                        relu_data_in_0_ready;
  // [32][1]
  logic [relu_OUT_0_WIDTH-1:0] relu_data_out_0       [relu_OUT_0_SIZE-1:0];
  logic                        relu_data_out_0_valid;
  logic                        relu_data_out_0_ready;

  // --------------------------
  //   fc2 signals
  // --------------------------
  // [32][1]
  logic [  fc2_IN_0_WIDTH-1:0] fc2_data_in_0         [  fc2_IN_0_SIZE-1:0];
  logic                        fc2_data_in_0_valid;
  logic                        fc2_data_in_0_ready;
  // // [1][3136]
  // logic [fc2_WEIGHT_WIDTH-1:0]  fc2_weight        [fc2_WEIGHT_SIZE-1:0];
  // logic                             fc2_weight_valid;
  // logic                             fc2_weight_ready;
  // // [1][3136]
  // logic [fc2_BIAS_WIDTH-1:0]  fc2_bias        [fc2_BIAS_SIZE-1:0];
  // logic                             fc2_bias_valid;
  // logic                             fc2_bias_ready;
  // [32][3136]
  logic [ fc2_OUT_0_WIDTH-1:0] fc2_data_out_0        [ fc2_OUT_0_SIZE-1:0];
  logic                        fc2_data_out_0_valid;
  logic                        fc2_data_out_0_ready;


  // --------------------------
  //   Kernel instantiation
  // --------------------------

  // fc1
  fixed_activation_binary_linear #(
      .HAS_BIAS(fc1_HAS_BIAS),  // = 1
      .IN_0_SIZE(fc1_IN_0_SIZE),  // = 1
      .IN_0_DEPTH(fc1_IN_0_DEPTH),  // = 784
      .PARALLELISM(fc1_PARALLELISM),  // = 784
      .IN_0_WIDTH(fc1_IN_0_WIDTH),  // = 32
      .IN_0_FRAC_WIDTH(fc1_IN_0_FRAC_WIDTH),  // = 0
      .WEIGHT_WIDTH(fc1_WEIGHT_WIDTH),  // = 1
      .WEIGHT_FRAC_WIDTH(fc1_WEIGHT_FRAC_WIDTH),  // = 0
      .BIAS_WIDTH(fc1_BIAS_WIDTH),  // = 1
      .BIAS_FRAC_WIDTH(fc1_BIAS_FRAC_WIDTH),  // = 0
      .WEIGHT_SIZE(fc1_WEIGHT_SIZE),  // = 784
      .OUT_0_SIZE(fc1_OUT_0_SIZE),  // = 784
      .BIAS_SIZE(fc1_BIAS_SIZE)
  ) fc1_inst (
      .clk(clk),
      .rst(rst),

      .data_in_0(fc1_data_in_0),  // [32][1]
      .data_in_0_valid(fc1_data_in_0_valid),
      .data_in_0_ready(fc1_data_in_0_ready),

      .weight(fc1_weight),  // [1][784]
      .weight_valid(fc1_weight_valid),
      .weight_ready(fc1_weight_ready),

      .bias(fc1_bias),  // [1][784]
      .bias_valid(fc1_bias_valid),
      .bias_ready(fc1_bias_ready),

      .data_out_0(fc1_data_out_0),  // [32][784]
      .data_out_0_valid(fc1_data_out_0_valid),
      .data_out_0_ready(fc1_data_out_0_ready)
  );

  // fc1_weight_source #(
  // .OUT_DEPTH(fc1_IN_0_DEPTH), // = 784
  // .OUT_WIDTH(fc1_WEIGHT_WIDTH), // = 1
  // .OUT_SIZE(fc1_WEIGHT_SIZE) // = 784
  // ) fc1_weight_source_0 (
  // .clk(clk),
  // .rst(rst),
  // .data_out(fc1_weight), // [1][784]
  // .data_out_ready(fc1_weight_ready),
  // .data_out_valid(fc1_weight_valid)
  // );

  // fc1_bias_source #(
  // .OUT_DEPTH(1), // = 1
  // .OUT_WIDTH(fc1_BIAS_WIDTH), // = 1
  // .OUT_SIZE(fc1_BIAS_SIZE) // = 784
  // ) fc1_bias_source_0 (
  // .clk(clk),
  // .rst(rst),
  // .data_out(fc1_bias), // [1][784]
  // .data_out_ready(fc1_bias_ready),
  // .data_out_valid(fc1_bias_valid)
  // );

  // relu
  fixed_relu #(
      .IN_0_SIZE(relu_IN_0_SIZE),  // = 1
      .IN_0_WIDTH(relu_IN_0_WIDTH),  // = 32
      .IN_0_FRAC_WIDTH(relu_IN_0_FRAC_WIDTH),  // = 0
      .OUT_0_SIZE(relu_OUT_0_SIZE)
  ) relu_inst (
      .clk(clk),
      .rst(rst),

      .data_in_0(relu_data_in_0),  // [32][1]
      .data_in_0_valid(relu_data_in_0_valid),
      .data_in_0_ready(relu_data_in_0_ready),

      .data_out_0(relu_data_out_0),  // [32][1]
      .data_out_0_valid(relu_data_out_0_valid),
      .data_out_0_ready(relu_data_out_0_ready)
  );

  // fc2
  fixed_activation_binary_linear #(
      .HAS_BIAS(fc2_HAS_BIAS),  // = 1
      .IN_0_SIZE(fc2_IN_0_SIZE),  // = 1
      .IN_0_DEPTH(fc2_IN_0_DEPTH),  // = 784
      .PARALLELISM(fc2_PARALLELISM),  // = 3136
      .IN_0_WIDTH(fc2_IN_0_WIDTH),  // = 32
      .IN_0_FRAC_WIDTH(fc2_IN_0_FRAC_WIDTH),  // = 0
      .WEIGHT_WIDTH(fc2_WEIGHT_WIDTH),  // = 1
      .WEIGHT_FRAC_WIDTH(fc2_WEIGHT_FRAC_WIDTH),  // = 0
      .BIAS_WIDTH(fc2_BIAS_WIDTH),  // = 1
      .BIAS_FRAC_WIDTH(fc2_BIAS_FRAC_WIDTH),  // = 0
      .WEIGHT_SIZE(fc2_WEIGHT_SIZE),  // = 3136
      .OUT_0_SIZE(fc2_OUT_0_SIZE),  // = 3136
      .BIAS_SIZE(fc2_BIAS_SIZE)
  ) fc2_inst (
      .clk(clk),
      .rst(rst),

      .data_in_0(fc2_data_in_0),  // [32][1]
      .data_in_0_valid(fc2_data_in_0_valid),
      .data_in_0_ready(fc2_data_in_0_ready),

      .weight(fc2_weight),  // [1][3136]
      .weight_valid(fc2_weight_valid),
      .weight_ready(fc2_weight_ready),

      .bias(fc2_bias),  // [1][3136]
      .bias_valid(fc2_bias_valid),
      .bias_ready(fc2_bias_ready),

      .data_out_0(fc2_data_out_0),  // [32][3136]
      .data_out_0_valid(fc2_data_out_0_valid),
      .data_out_0_ready(fc2_data_out_0_ready)
  );

  // fc2_weight_source #(
  // .OUT_DEPTH(fc2_IN_0_DEPTH), // = 784
  // .OUT_WIDTH(fc2_WEIGHT_WIDTH), // = 1
  // .OUT_SIZE(fc2_WEIGHT_SIZE) // = 3136
  // ) fc2_weight_source_0 (
  // .clk(clk),
  // .rst(rst),
  // .data_out(fc2_weight), // [1][3136]
  // .data_out_ready(fc2_weight_ready),
  // .data_out_valid(fc2_weight_valid)
  // );

  // fc2_bias_source #(
  // .OUT_DEPTH(1), // = 1
  // .OUT_WIDTH(fc2_BIAS_WIDTH), // = 1
  // .OUT_SIZE(fc2_BIAS_SIZE) // = 3136
  // ) fc2_bias_source_0 (
  // .clk(clk),
  // .rst(rst),
  // .data_out(fc2_bias), // [1][3136]
  // .data_out_ready(fc2_bias_ready),
  // .data_out_valid(fc2_bias_valid)
  // );


  // --------------------------
  //   Interconnections
  // --------------------------


  assign data_in_ready  = fc1_data_in_0_ready;
  assign fc1_data_in_0_valid    = data_in_valid;
  assign fc1_data_in_0 = data_in;


  assign fc2_data_out_0_ready  = data_out_ready;
  assign data_out_valid    = fc2_data_out_0_valid;
  assign data_out = fc2_data_out_0;

  logic [relu_IN_0_WIDTH-1:0] fc1_data_out_0_cast[fc1_OUT_0_SIZE-1:0];  //TODO

  fixed_cast #(  // TODO
      .IN_SIZE(fc1_OUT_0_SIZE),
      .IN_WIDTH(fc1_OUT_0_WIDTH),
      .IN_FRAC_WIDTH(fc1_IN_0_FRAC_WIDTH + fc1_WEIGHT_FRAC_WIDTH),
      .OUT_FRAC_WIDTH(relu_IN_0_FRAC_WIDTH),
      .OUT_WIDTH(relu_IN_0_WIDTH)
  ) fc1_data_out_0_relu_data_in_0_cast (
      .data_in(fc1_data_out_0),  // [1][1] frac_width = 0 
      .data_out(fc1_data_out_0_cast)  // [32][784]
  );

  assign fc1_data_out_0_ready  = relu_data_in_0_ready;
  assign relu_data_in_0_valid    = fc1_data_out_0_valid;
  assign relu_data_in_0 = fc1_data_out_0_cast;

  roller #(
      .DATA_WIDTH(relu_OUT_0_WIDTH),
      .NUM(relu_OUT_0_SIZE),
      .IN_SIZE(1),
      .ROLL_NUM(fc2_IN_0_SIZE)
  ) roller_inst (
      .data_in(relu_data_out_0),
      .data_in_valid(relu_data_out_0_valid),
      .data_in_ready(relu_data_out_0_ready),
      .data_out(fc2_data_in_0),
      .data_out_valid(fc2_data_in_0_valid),
      .data_out_ready(fc2_data_in_0_ready),
      .*
  );

  // assign relu_data_out_0_ready  = fc2_data_in_0_ready;
  // assign fc2_data_in_0_valid    = relu_data_out_0_valid;
  // assign fc2_data_in_0 = relu_data_out_0;

endmodule
