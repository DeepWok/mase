`timescale 1ns / 1ps
module fixed_self_att #(
    parameter DATA_WIDTH = 8,
    parameter DATA_FRAC_WIDTH = 1,

    parameter WQ_WIDTH = 8,
    parameter WQ_FRAC_WIDTH = 1,
    parameter WK_WIDTH = 8,
    parameter WK_FRAC_WIDTH = 1,
    parameter WV_WIDTH = 8,
    parameter WV_FRAC_WIDTH = 1,

    parameter BQ_WIDTH = 8,
    parameter BQ_FRAC_WIDTH = 1,
    parameter BK_WIDTH = 8,
    parameter BK_FRAC_WIDTH = 1,
    parameter BV_WIDTH = 8,
    parameter BV_FRAC_WIDTH = 1,

    parameter DQ_WIDTH = 8,
    parameter DQ_FRAC_WIDTH = 1,
    parameter DK_WIDTH = 8,
    parameter DK_FRAC_WIDTH = 1,
    parameter DV_WIDTH = 8,
    parameter DV_FRAC_WIDTH = 1,

    parameter DS_WIDTH = 8,
    parameter DS_FRAC_WIDTH = 1,
    parameter EXP_WIDTH = 8,
    parameter EXP_FRAC_WIDTH = 4,
    parameter DIV_WIDTH = 10,
    parameter DS_SOFTMAX_WIDTH = 8,
    parameter DS_SOFTMAX_FRAC_WIDTH = 7,

    parameter DZ_WIDTH = 8,
    parameter DZ_FRAC_WIDTH = 1,

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

    input [WQ_WIDTH - 1:0] weight_q[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WK_WIDTH - 1:0] weight_k[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WV_WIDTH - 1:0] weight_v[W_PARALLELISM * W_SIZE -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [BQ_WIDTH - 1:0] bias_q[W_PARALLELISM -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BK_WIDTH - 1:0] bias_k[W_PARALLELISM -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BV_WIDTH - 1:0] bias_v[W_PARALLELISM -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [DATA_WIDTH -1:0] data_in[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [DZ_WIDTH -1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
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
      .DQIN_WIDTH(DATA_WIDTH),
      .DQIN_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .DKIN_WIDTH(DATA_WIDTH),
      .DKIN_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .DVIN_WIDTH(DATA_WIDTH),
      .DVIN_FRAC_WIDTH(DATA_FRAC_WIDTH),

      .WQ_WIDTH(WQ_WIDTH),
      .WQ_FRAC_WIDTH(WQ_FRAC_WIDTH),
      .WK_WIDTH(WK_WIDTH),
      .WK_FRAC_WIDTH(WK_FRAC_WIDTH),
      .WV_WIDTH(WV_WIDTH),
      .WV_FRAC_WIDTH(WV_FRAC_WIDTH),

      .BQ_WIDTH(BQ_WIDTH),
      .BQ_FRAC_WIDTH(BQ_FRAC_WIDTH),
      .BK_WIDTH(BK_WIDTH),
      .BK_FRAC_WIDTH(BK_FRAC_WIDTH),
      .BV_WIDTH(BV_WIDTH),
      .BV_FRAC_WIDTH(BV_FRAC_WIDTH),

      .DQ_WIDTH(DQ_WIDTH),
      .DQ_FRAC_WIDTH(DQ_FRAC_WIDTH),
      .DK_WIDTH(DK_WIDTH),
      .DK_FRAC_WIDTH(DK_FRAC_WIDTH),
      .DV_WIDTH(DV_WIDTH),
      .DV_FRAC_WIDTH(DV_FRAC_WIDTH),

      .DS_WIDTH(DS_WIDTH),
      .DS_FRAC_WIDTH(DS_FRAC_WIDTH),
      .EXP_WIDTH(EXP_WIDTH),
      .EXP_FRAC_WIDTH(EXP_FRAC_WIDTH),
      .DIV_WIDTH(DIV_WIDTH),
      .DS_SOFTMAX_WIDTH(DS_SOFTMAX_WIDTH),
      .DS_SOFTMAX_FRAC_WIDTH(DS_SOFTMAX_FRAC_WIDTH),

      .DZ_WIDTH(DZ_WIDTH),
      .DZ_FRAC_WIDTH(DZ_FRAC_WIDTH),
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
