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
    parameter IN_Y = 8,
    parameter UNROLL_IN_Y = 4,
    parameter ITER_IN_Y = IN_Y / UNROLL_IN_Y,
    parameter IN_X = 12,
    parameter UNROLL_IN_X = 4,

    parameter W_Y = 8,
    parameter UNROLL_W_Y = 4,
    parameter ITER_W_Y = W_Y / UNROLL_W_Y
) (
    input clk,
    input rst,
    //input data
    input [IN_WIDTH-1:0] data_in[UNROLL_IN_Y * UNROLL_IN_X - 1:0],
    input data_in_valid,
    output data_in_ready,
    //input weight
    input [WEIGHT_WIDTH-1:0] weight[UNROLL_W_Y * UNROLL_IN_X - 1:0],
    input weight_valid,
    output weight_ready,
    //input bias
    input [BIAS_WIDTH-1:0] bias[UNROLL_W_Y - 1:0],
    input bias_valid,
    output bias_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[UNROLL_IN_Y * UNROLL_W_Y - 1:0],
    output data_out_valid,
    input data_out_ready

);
  logic [BIAS_WIDTH-1:0] bias_extend[UNROLL_IN_Y * UNROLL_W_Y - 1:0];
  logic [BIAS_WIDTH-1:0] ib_bias[UNROLL_IN_Y * UNROLL_W_Y - 1:0];
  logic ib_bias_valid, ib_bias_ready;
  for (genvar i = 0; i < UNROLL_IN_Y; i++)
    assign bias_extend[i*UNROLL_W_Y+UNROLL_W_Y-1 : i*UNROLL_W_Y] = bias;

  input_buffer #(
      .IN_WIDTH(BIAS_WIDTH),
      .IN_PARALLELISM(UNROLL_IN_Y),
      .IN_SIZE(UNROLL_W_Y),
      .BUFFER_SIZE(ITER_W_Y),
      .REPEAT(ITER_IN_Y)
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
      .IN1_Y(IN_Y),
      .UNROLL_IN1_Y(UNROLL_IN_Y),
      .IN1_X(IN_X),
      .UNROLL_IN1_X(UNROLL_IN_X),
      .IN2_Y(W_Y),
      .UNROLL_IN2_Y(UNROLL_W_Y)
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
