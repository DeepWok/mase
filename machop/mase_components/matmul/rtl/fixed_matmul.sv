`timescale 1ns / 1ps
module fixed_matmul #(
    // input 
    parameter IN1_WIDTH = 32,
    parameter IN1_FRAC_WIDTH = 8,
    parameter IN2_WIDTH = 16,
    parameter IN2_FRAC_WIDTH = 8,

    parameter HAS_BIAS = 0,
    parameter BIAS_WIDTH = 16,
    parameter BIAS_FRAC_WIDTH = 4,
    //output 
    parameter OUT_WIDTH = 32,
    parameter OUT_FRAC_WIDTH = 8,
    // define as nm * mk
    // rows refers to n, columns refers to m

    // this module input 
    // data_in1[UNROLL_IN1_Y * ITER_IN1_Y][UNROLL_IN1_X * ITER_IN1_X]
    // data_in2[UNROLL_IN2_Y * ITER_IN2_Y][UNROLL_IN1_X * ITER_IN1_X]
    // get output np.matmul(data_in1, data_in2.T)
    parameter IN1_Y = 8,
    parameter UNROLL_IN1_Y = 4,
    parameter ITER_IN1_Y = IN1_Y / UNROLL_IN1_Y,
    parameter IN1_X = 12,
    parameter UNROLL_IN1_X = 4,
    parameter ITER_IN1_X = IN1_X / UNROLL_IN1_X,

    parameter IN2_Y = 8,
    parameter UNROLL_IN2_Y = 4,
    parameter ITER_IN2_Y = IN2_Y / UNROLL_IN2_Y
) (
    input clk,
    input rst,
    //input data
    input [IN1_WIDTH-1:0] data_in1[UNROLL_IN1_Y * UNROLL_IN1_X - 1:0],
    input data_in1_valid,
    output data_in1_ready,
    //input weight
    input [IN2_WIDTH-1:0] data_in2[UNROLL_IN2_Y * UNROLL_IN1_X - 1:0],
    input data_in2_valid,
    output data_in2_ready,
    //input bias
    input [BIAS_WIDTH-1:0] bias[UNROLL_IN1_Y * UNROLL_IN2_Y - 1:0],
    input bias_valid,
    output bias_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[UNROLL_IN1_Y * UNROLL_IN2_Y - 1:0],
    output data_out_valid,
    input data_out_ready

);
  //input buffer
  logic [IN1_WIDTH-1:0] ib_data_in[UNROLL_IN1_Y * UNROLL_IN1_X - 1:0];
  logic ib_data_in_valid, ib_data_in_ready;
  input_buffer #(
      .IN_WIDTH(IN1_WIDTH),
      .IN_PARALLELISM(UNROLL_IN1_Y),
      .IN_SIZE(UNROLL_IN1_X),
      .BUFFER_SIZE(ITER_IN1_X),
      .REPEAT(ITER_IN2_Y)
  ) data_in_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in1),
      .data_in_valid(data_in1_valid),
      .data_in_ready(data_in1_ready),
      .data_out(ib_data_in),
      .data_out_valid(ib_data_in_valid),
      .data_out_ready(ib_data_in_ready)
  );

  logic [IN2_WIDTH-1:0] ib_weight[UNROLL_IN2_Y * UNROLL_IN1_X - 1:0];
  logic ib_weight_valid, ib_weight_ready;
  input_buffer #(
      .IN_WIDTH(IN2_WIDTH),
      .IN_PARALLELISM(UNROLL_IN2_Y),
      .IN_SIZE(UNROLL_IN1_X),
      .BUFFER_SIZE(ITER_IN1_X * ITER_IN2_Y),
      .REPEAT(ITER_IN1_Y)
  ) weight_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(data_in2),
      .data_in_valid(data_in2_valid),
      .data_in_ready(data_in2_ready),
      .data_out(ib_weight),
      .data_out_valid(ib_weight_valid),
      .data_out_ready(ib_weight_ready)
  );

  fixed_matmul_core #(
      .IN1_WIDTH(IN1_WIDTH),
      .IN1_FRAC_WIDTH(IN1_FRAC_WIDTH),
      .IN2_WIDTH(IN2_WIDTH),
      .IN2_FRAC_WIDTH(IN2_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN1_PARALLELISM(UNROLL_IN1_Y),
      .IN_SIZE(UNROLL_IN1_X),
      .IN2_PARALLELISM(UNROLL_IN2_Y),
      .IN_DEPTH(ITER_IN1_X),
      .HAS_BIAS(HAS_BIAS)
  ) inst_fmmc (
      .clk(clk),
      .rst(rst),
      .data_in1(ib_data_in),
      .data_in1_valid(ib_data_in_valid),
      .data_in1_ready(ib_data_in_ready),
      .data_in2(ib_weight),
      .data_in2_valid(ib_weight_valid),
      .data_in2_ready(ib_weight_ready),
      .bias(bias),
      .bias_valid(bias_valid),
      .bias_ready(bias_ready),
      .data_out(data_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
endmodule
