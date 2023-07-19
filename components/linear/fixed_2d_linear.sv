`timescale 1ns / 1ps
module fixed_2d_linear #(
    // input 
    parameter IN_WIDTH = 32,
    parameter IN_FRAC_WIDTH = 8,
    parameter WEIGHT_WIDTH = 16,
    parameter WEIGHT_FRAC_WIDTH = 8,

    parameter HAS_BIAS = 1,
    parameter BIAS_WIDTH = 16,
    parameter BIAS_FRAC_WIDTH = 4,
    //output 
    parameter OUT_WIDTH = 32,
    parameter OUT_FRAC_WIDTH = 8,
    // define as nm * mk
    parameter IN_PARALLELISM = 4,
    parameter IN_NUM_PARALLELISM = 2,
    parameter IN_SIZE = 4,
    parameter IN_DEPTH = 3,

    parameter W_PARALLELISM = 4,
    parameter W_NUM_PARALLELISM = 2
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[IN_PARALLELISM * IN_SIZE - 1:0],
    input data_in_valid,
    output data_in_ready,
    //input weight
    input [WEIGHT_WIDTH-1:0] weight[W_PARALLELISM * IN_SIZE - 1:0],
    input weight_valid,
    output weight_ready,
    //input bias
    input [BIAS_WIDTH-1:0] bias[W_PARALLELISM - 1:0],
    input bias_valid,
    output bias_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[IN_PARALLELISM * W_PARALLELISM - 1:0],
    output data_out_valid,
    input data_out_ready

);
  logic [BIAS_WIDTH-1:0] bias_extend[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic [BIAS_WIDTH-1:0] ib_bias[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic ib_bias_valid, ib_bias_ready;
  for (genvar i = 0; i < IN_PARALLELISM; i++)
    assign bias_extend[i*W_PARALLELISM+W_PARALLELISM-1 : i*W_PARALLELISM] = bias;

  input_buffer #(
      .IN_WIDTH(BIAS_WIDTH),
      .IN_PARALLELISM(IN_PARALLELISM),
      .IN_SIZE(W_PARALLELISM),
      .BUFFER_SIZE(W_NUM_PARALLELISM),
      .REPEAT(IN_NUM_PARALLELISM)
  ) bias_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(bias_extend),
      .data_in_valid(bias_valid),
      .data_in_ready(bias_ready),
      .data_out(ib_bias),
      .data_out_valid(ib_bias_valid),
      .data_out_ready(ib_bias_ready)
  );
  fixed_matmul #(
      .IN1_WIDTH(IN_WIDTH),
      .IN1_FRAC_WIDTH(IN_FRAC_WIDTH),
      .IN2_WIDTH(WEIGHT_WIDTH),
      .IN2_FRAC_WIDTH(WEIGHT_FRAC_WIDTH),
      .HAS_BIAS(HAS_BIAS),
      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN1_PARALLELISM(IN_PARALLELISM),
      .IN1_NUM_PARALLELISM(IN_NUM_PARALLELISM),
      .IN_SIZE(IN_SIZE),
      .IN2_PARALLELISM(W_PARALLELISM),
      .IN2_NUM_PARALLELISM(W_NUM_PARALLELISM),
      .IN_DEPTH(IN_DEPTH)
  ) inst_fmmc (
      .data_in1(data_in),
      .data_in1_valid(data_in_valid),
      .data_in1_ready(data_in_ready),
      .data_in2(weight),
      .data_in2_valid(weight_valid),
      .data_in2_ready(weight_ready),
      .bias(ib_bias),
      .bias_valid(ib_bias_valid),
      .bias_ready(ib_bias_ready),
      .*
  );

endmodule
