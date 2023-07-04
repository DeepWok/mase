`timescale 1ns / 1ps
module fixed_matmul #(
    // input 
    parameter IN1_WIDTH = 32,
    parameter IN1_FRAC_WIDTH = 8,
    parameter IN2_WIDTH = 16,
    parameter IN2_FRAC_WIDTH = 8,
    //output 
    parameter OUT_WIDTH = 32,
    parameter OUT_FRAC_WIDTH = 8,
    // define as nm * mk
    // rows refers to n, columns refers to m

    // this module input 
    // data_in1[IN1_PARALLELISM * IN1_NUM_PARALLELISM][IN_SIZE * IN_DEPTH]
    // data_in2[IN2_PARALLELISM * IN2_NUM_PARALLELISM][IN_SIZE * IN_DEPTH]
    // get output np.matmul(data_in1, data_in2.T)
    parameter IN1_PARALLELISM = 4,
    parameter IN1_NUM_PARALLELISM = 2,
    parameter IN_SIZE = 4,

    parameter IN2_PARALLELISM = 4,
    parameter IN2_NUM_PARALLELISM = 2,
    //defines the dataflow parameter, used for linear layer
    parameter IN_DEPTH = 3
) (
    input clk,
    input rst,
    //input data
    input [IN1_WIDTH-1:0] data_in1[IN1_PARALLELISM * IN_SIZE - 1:0],
    input data_in1_valid,
    output data_in1_ready,
    //input weight
    input [IN2_WIDTH-1:0] data_in2[IN2_PARALLELISM * IN_SIZE - 1:0],
    input data_in2_valid,
    output data_in2_ready,
    //output data
    output [OUT_WIDTH-1:0] data_out[IN1_PARALLELISM * IN2_PARALLELISM - 1:0],
    output data_out_valid,
    input data_out_ready

);
  //input buffer
  logic [IN1_WIDTH-1:0] ib_data_in[IN1_PARALLELISM * IN_SIZE - 1:0];
  logic ib_data_in_valid, ib_data_in_ready;
  input_buffer #(
      .IN_WIDTH(IN1_WIDTH),
      .IN_PARALLELISM(IN1_PARALLELISM),
      .IN_SIZE(IN_SIZE),
      .BUFFER_SIZE(IN_DEPTH),
      .REPEAT(IN2_NUM_PARALLELISM)
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

  logic [IN2_WIDTH-1:0] ib_weight[IN2_PARALLELISM * IN_SIZE - 1:0];
  logic ib_weight_valid, ib_weight_ready;
  input_buffer #(
      .IN_WIDTH(IN2_WIDTH),
      .IN_PARALLELISM(IN2_PARALLELISM),
      .IN_SIZE(IN_SIZE),
      .BUFFER_SIZE(IN_DEPTH * IN2_NUM_PARALLELISM),
      .REPEAT(IN1_NUM_PARALLELISM)
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
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN1_PARALLELISM(IN1_PARALLELISM),
      .IN_SIZE(IN_SIZE),
      .IN2_PARALLELISM(IN2_PARALLELISM),
      .IN_DEPTH(IN_DEPTH)
  ) inst_fmmc (
      .clk(clk),
      .rst(rst),
      .data_in1(ib_data_in),
      .data_in1_valid(ib_data_in_valid),
      .data_in1_ready(ib_data_in_ready),
      .data_in2(ib_weight),
      .data_in2_valid(ib_weight_valid),
      .data_in2_ready(ib_weight_ready),
      .data_out(data_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
endmodule
