`timescale 1ns / 1ps
module fixed_self_att #(
    parameter DATA_WIDTH = 8,
    parameter DATA_FRAC_WIDTH = 1,
    parameter WEIGHT_WIDTH = 8,
    parameter W_FRAC_WIDTH = 1,
    parameter BIAS_WIDTH = 8,
    parameter BIAS_FRAC_WIDTH = 1,


    parameter IN_PARALLELISM = 3,
    parameter IN_NUM_PARALLELISM = 2,

    parameter IN_SIZE  = 3,
    //define for matrix multilication
    parameter IN_DEPTH = 3,

    parameter W_PARALLELISM = 3,
    parameter W_NUM_PARALLELISM = 2,
    parameter W_SIZE = IN_SIZE,


    parameter OUT_PARALLELISM = IN_PARALLELISM,
    parameter OUT_SIZE = W_PARALLELISM
) (
    input clk,
    input rst,

    input [WEIGHT_WIDTH - 1:0] weight_q[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WEIGHT_WIDTH - 1:0] weight_k[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WEIGHT_WIDTH - 1:0] weight_v[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [BIAS_WIDTH - 1:0] bias_q[W_PARALLELISM -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BIAS_WIDTH - 1:0] bias_k[W_PARALLELISM -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BIAS_WIDTH - 1:0] bias_v[W_PARALLELISM -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [DATA_WIDTH -1:0] data_in[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [DATA_WIDTH -1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);
  logic data_in_q_ready, data_in_k_ready, data_in_v_ready;
  logic data_in_q_valid, data_in_k_valid, data_in_v_valid;

  assign data_in_q_valid = data_in_v_ready && data_in_k_ready && data_in_valid;
  assign data_in_k_valid = data_in_q_ready && data_in_v_ready && data_in_valid;
  assign data_in_v_valid = data_in_q_ready && data_in_k_ready && data_in_valid;
  assign data_in_ready   = data_in_q_ready && data_in_k_ready && data_in_v_ready;
  fixed_att #(
      .DATA_WIDTH(DATA_WIDTH),
      .DATA_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .WEIGHT_WIDTH(WEIGHT_WIDTH),
      .W_FRAC_WIDTH(W_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .IN_PARALLELISM(IN_PARALLELISM),
      .IN_NUM_PARALLELISM(IN_NUM_PARALLELISM),
      .IN_SIZE(IN_SIZE),
      .IN_DEPTH(IN_DEPTH),
      .W_PARALLELISM(W_PARALLELISM),
      .W_NUM_PARALLELISM(W_NUM_PARALLELISM)
  ) att_inst (
      .data_in_q(data_in),
      .data_in_k(data_in),
      .data_in_v(data_in),
      .*
  );
endmodule
