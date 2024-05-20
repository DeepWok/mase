`timescale 1ns / 1ps
module fixed_self_att #(
    parameter DATA_WIDTH = 8,
    parameter DATA_FRAC_WIDTH = 1,

    parameter WEIGHT_Q_WIDTH = 8,
    parameter WEIGHT_Q_FRAC_WIDTH = 1,
    parameter WEIGHT_K_WIDTH = 8,
    parameter WEIGHT_K_FRAC_WIDTH = 1,
    parameter WEIGHT_V_WIDTH = 8,
    parameter WEIGHT_V_FRAC_WIDTH = 1,

    parameter BIAS_Q_WIDTH = 8,
    parameter BIAS_Q_FRAC_WIDTH = 1,
    parameter BIAS_K_WIDTH = 8,
    parameter BIAS_K_FRAC_WIDTH = 1,
    parameter BIAS_V_WIDTH = 8,
    parameter BIAS_V_FRAC_WIDTH = 1,

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

    parameter BIAS_Q_SIZE   = 3,
    parameter BIAS_K_SIZE   = 3,
    parameter BIAS_V_SIZE   = 3,
    parameter WEIGHT_Q_SIZE = 9,
    parameter WEIGHT_K_SIZE = 9,
    parameter WEIGHT_V_SIZE = 9,

    parameter OUT_SIZE  = OUT_PARALLELISM * OUT_SIZE,
    parameter OUT_WIDTH = DZ_WIDTH
) (
    input clk,
    input rst,

    input [WEIGHT_Q_WIDTH - 1:0] weight_q[WEIGHT_Q_SIZE -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WEIGHT_K_WIDTH - 1:0] weight_k[WEIGHT_K_SIZE -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WEIGHT_V_WIDTH - 1:0] weight_v[WEIGHT_V_SIZE -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [BIAS_Q_WIDTH - 1:0] bias_q[BIAS_Q_SIZE -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BIAS_K_WIDTH - 1:0] bias_k[BIAS_K_SIZE -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BIAS_V_WIDTH - 1:0] bias_v[BIAS_V_SIZE -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [DATA_WIDTH -1:0] data_in_0[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_0_valid,
    output data_in_0_ready,

    output [OUT_WIDTH -1:0] data_out_0[OUT_SIZE - 1:0],
    output data_out_0_valid,
    input data_out_0_ready
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

      .WQ_WIDTH(WEIGHT_Q_WIDTH),
      .WQ_FRAC_WIDTH(WEIGHT_Q_FRAC_WIDTH),
      .WK_WIDTH(WEIGHT_K_WIDTH),
      .WL_FRAC_WIDTH(WEIGHT_K_FRAC_WIDTH),
      .WV_WIDTH(WEIGHT_V_WIDTH),
      .WV_FRAC_WIDTH(WEIGHT_V_FRAC_WIDTH),

      .BQ_WIDTH(BIAS_Q_WIDTH),
      .BQ_FRAC_WIDTH(BIAS_Q_FRAC_WIDTH),
      .BK_WIDTH(BIAS_K_WIDTH),
      .BK_FRAC_WIDTH(BIAS_K_FRAC_WIDTH),
      .BV_WIDTH(BIAS_V_WIDTH),
      .BV_FRAC_WIDTH(BIAS_V_FRAC_WIDTH),

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
      .data_in_q(data_in_0),
      .data_in_k(data_in_0),
      .data_in_v(data_in_0),
      .*
  );
endmodule
