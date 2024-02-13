`timescale 1ns / 1ps
module fixed_ViT #(
    parameter IN_WIDTH            = 6,
    parameter IN_FRAC_WIDTH       = 1,
    //patch
    parameter WC_WIDTH            = 4,
    parameter WC_FRAC_WIDTH       = 1,
    parameter BC_WIDTH            = 4,
    parameter BC_FRAC_WIDTH       = 1,
    parameter CONV_OUT_WIDTH      = 6,
    parameter CONV_OUT_FRAC_WIDTH = 1,
    //msa
    parameter WQ_WIDTH            = 4,
    parameter WQ_FRAC_WIDTH       = 1,
    parameter WK_WIDTH            = 4,
    parameter WK_FRAC_WIDTH       = 1,
    parameter WV_WIDTH            = 4,
    parameter WV_FRAC_WIDTH       = 1,

    parameter BQ_WIDTH = 4,
    parameter BQ_FRAC_WIDTH = 1,
    parameter BK_WIDTH = 4,
    parameter BK_FRAC_WIDTH = 1,
    parameter BV_WIDTH = 4,
    parameter BV_FRAC_WIDTH = 1,

    parameter WP_WIDTH = 4,
    parameter WP_FRAC_WIDTH = 1,
    parameter BP_WIDTH = 4,
    parameter BP_FRAC_WIDTH = 1,

    parameter DQ_WIDTH = 6,
    parameter DQ_FRAC_WIDTH = 1,
    parameter DK_WIDTH = 6,
    parameter DK_FRAC_WIDTH = 1,
    parameter DV_WIDTH = 6,
    parameter DV_FRAC_WIDTH = 1,

    parameter DS_WIDTH = 6,
    parameter DS_FRAC_WIDTH = 1,
    parameter DZ_WIDTH = 6,
    parameter DZ_FRAC_WIDTH = 1,

    parameter MSA_OUT_WIDTH = 6,
    parameter MSA_OUT_FRAC_WIDTH = 1,
    // mlp

    parameter WEIGHT_I2H_WIDTH = 4,
    parameter WEIGHT_I2H_FRAC_WIDTH = 1,
    parameter WEIGHT_H2O_WIDTH = 4,
    parameter WEIGHT_H2O_FRAC_WIDTH = 1,
    parameter MLP_HAS_BIAS = 1,
    parameter BIAS_I2H_WIDTH = 4,
    parameter BIAS_I2H_FRAC_WIDTH = 1,
    parameter BIAS_H2O_WIDTH = 4,
    parameter BIAS_H2O_FRAC_WIDTH = 1,

    parameter HIDDEN_WIDTH = 6,
    parameter HIDDEN_FRAC_WIDTH = 1,

    parameter OUT_WIDTH = 6,
    parameter OUT_FRAC_WIDTH = 1,
    // conv
    parameter IN_C = 3,
    parameter IN_Y = 16,
    parameter IN_X = 16,

    parameter OUT_C = 4,
    parameter KERNEL_C = IN_C,
    parameter KERNEL_SIZE = 2,
    parameter KERNEL_Y = KERNEL_SIZE,
    parameter KERNEL_X = KERNEL_SIZE,

    parameter PADDING_Y = KERNEL_Y / 2,
    parameter PADDING_X = KERNEL_Y / 2,

    parameter UNROLL_KERNEL_OUT = 2,
    parameter UNROLL_OUT_C = 2,
    parameter UNROLL_IN_C = 2,

    parameter SLIDING_NUM = 73,

    parameter STRIDE = KERNEL_SIZE,
    // patch embedding
    // TODO: IN_NUM = SLLIDING_NUM needs to be discussed
    parameter IN_NUM = SLIDING_NUM,
    parameter UNROLL_IN_NUM = 1,

    parameter IN_DIM = OUT_C,
    parameter UNROLL_IN_DIM = UNROLL_IN_C,
    // num_heads * wqkv_dim = IN_DIM
    parameter NUM_HEADS = 2,
    parameter WQKV_DIM = IN_DIM / NUM_HEADS,
    parameter UNROLL_WQKV_DIM = 1,
    parameter WP_DIM = IN_DIM,

    // set it with unroll_in_dim for residual matching
    parameter UNROLL_WP_DIM = UNROLL_IN_DIM,

    parameter OUT_NUM = IN_NUM,
    parameter OUT_DIM = IN_DIM,
    parameter UNROLL_OUT_NUM = UNROLL_IN_NUM,
    parameter UNROLL_OUT_DIM = UNROLL_WP_DIM,
    // get attention output after 4 * 3 cycles 
    // mlp
    parameter IN_FEATURES = IN_DIM,
    parameter HIDDEN_FEATURES = 2 * IN_FEATURES,
    parameter OUT_FEATURES = IN_FEATURES,

    parameter UNROLL_IN_FEATURES = UNROLL_IN_DIM,
    parameter UNROLL_HIDDEN_FEATURES = 3,
    parameter UNROLL_OUT_FEATURES = UNROLL_IN_DIM
) (
    input clk,
    input rst,

    input  [WC_WIDTH-1:0] weight_c      [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0],
    input                 weight_c_valid,
    output                weight_c_ready,

    input  [BC_WIDTH-1:0] bias_c      [UNROLL_OUT_C-1:0],
    input                 bias_c_valid,
    output                bias_c_ready,

    input [WQ_WIDTH - 1:0] weight_q[NUM_HEADS * UNROLL_WQKV_DIM * UNROLL_IN_DIM -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WK_WIDTH - 1:0] weight_k[NUM_HEADS * UNROLL_WQKV_DIM * UNROLL_IN_DIM -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WV_WIDTH - 1:0] weight_v[NUM_HEADS * UNROLL_WQKV_DIM * UNROLL_IN_DIM -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [WP_WIDTH - 1:0] weight_p[UNROLL_WP_DIM * NUM_HEADS * UNROLL_WQKV_DIM -1 : 0],
    input weight_p_valid,
    output weight_p_ready,

    input [BQ_WIDTH - 1:0] bias_q[NUM_HEADS * UNROLL_WQKV_DIM -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BK_WIDTH - 1:0] bias_k[NUM_HEADS * UNROLL_WQKV_DIM -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BV_WIDTH - 1:0] bias_v[NUM_HEADS * UNROLL_WQKV_DIM -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [BP_WIDTH - 1:0] bias_p[UNROLL_WP_DIM -1 : 0],
    input bias_p_valid,
    output bias_p_ready,

    input [WEIGHT_I2H_WIDTH-1:0] weight_in2hidden[UNROLL_HIDDEN_FEATURES * UNROLL_IN_FEATURES - 1:0],
    input weight_in2hidden_valid,
    output weight_in2hidden_ready,

    input [WEIGHT_H2O_WIDTH-1:0] weight_hidden2out[UNROLL_OUT_FEATURES * UNROLL_HIDDEN_FEATURES - 1:0],
    input weight_hidden2out_valid,
    output weight_hidden2out_ready,
    //input bias
    input [BIAS_I2H_WIDTH-1:0] bias_in2hidden[UNROLL_HIDDEN_FEATURES - 1:0],
    input bias_in2hidden_valid,
    output bias_in2hidden_ready,

    input [BIAS_H2O_WIDTH-1:0] bias_hidden2out[UNROLL_OUT_FEATURES - 1:0],
    input bias_hidden2out_valid,
    output bias_hidden2out_ready,

    input [IN_WIDTH -1:0] data_in[UNROLL_IN_NUM * UNROLL_IN_DIM - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [OUT_WIDTH -1:0] data_out[UNROLL_OUT_NUM * UNROLL_OUT_FEATURES - 1:0],
    output data_out_valid,
    input data_out_ready
);
  logic [CONV_OUT_WIDTH - 1:0] conv_out[UNROLL_OUT_C - 1:0];
  logic conv_out_valid, conv_out_ready;
  fixed_patch_embed #(
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .W_WIDTH(WC_WIDTH),
      .W_FRAC_WIDTH(WC_FRAC_WIDTH),
      .BIAS_WIDTH(BC_WIDTH),
      .BIAS_FRAC_WIDTH(BC_FRAC_WIDTH),
      .OUT_WIDTH(CONV_OUT_WIDTH),
      .OUT_FRAC_WIDTH(CONV_OUT_FRAC_WIDTH),
      .IN_C(IN_C),
      .IN_Y(IN_Y),
      .IN_X(IN_X),
      .OUT_C(OUT_C),
      .KERNEL_SIZE(KERNEL_SIZE),
      .UNROLL_KERNEL_OUT(UNROLL_KERNEL_OUT),
      .UNROLL_OUT_C(UNROLL_OUT_C),
      .UNROLL_IN_C(UNROLL_IN_C),
      .SLIDING_NUM(SLIDING_NUM)
  ) patemb_inst (
      .weight(weight_c),
      .weight_valid(weight_c_valid),
      .weight_ready(weight_c_ready),

      .bias(bias_c),
      .bias_valid(bias_c_valid),
      .bias_ready(bias_c_ready),

      .data_out(conv_out),
      .data_out_valid(conv_out_valid),
      .data_out_ready(conv_out_ready),
      .*
  );

  fixed_block #(
      .IN_WIDTH(CONV_OUT_WIDTH),
      .IN_FRAC_WIDTH(CONV_OUT_FRAC_WIDTH),
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

      .WP_WIDTH(WP_WIDTH),
      .WP_FRAC_WIDTH(WP_FRAC_WIDTH),
      .BP_WIDTH(BP_WIDTH),
      .BP_FRAC_WIDTH(BP_FRAC_WIDTH),

      .DQ_WIDTH(DQ_WIDTH),
      .DQ_FRAC_WIDTH(DQ_FRAC_WIDTH),
      .DK_WIDTH(DK_WIDTH),
      .DK_FRAC_WIDTH(DK_FRAC_WIDTH),
      .DV_WIDTH(DV_WIDTH),
      .DV_FRAC_WIDTH(DV_FRAC_WIDTH),

      .DS_WIDTH(DS_WIDTH),
      .DS_FRAC_WIDTH(DS_FRAC_WIDTH),
      .DZ_WIDTH(DZ_WIDTH),
      .DZ_FRAC_WIDTH(DZ_FRAC_WIDTH),

      .MSA_OUT_WIDTH(MSA_OUT_WIDTH),
      .MSA_OUT_FRAC_WIDTH(MSA_OUT_FRAC_WIDTH),

      .WEIGHT_I2H_WIDTH(WEIGHT_I2H_WIDTH),
      .WEIGHT_I2H_FRAC_WIDTH(WEIGHT_I2H_FRAC_WIDTH),
      .WEIGHT_H2O_WIDTH(WEIGHT_H2O_WIDTH),
      .WEIGHT_H2O_FRAC_WIDTH(WEIGHT_H2O_FRAC_WIDTH),
      .MLP_HAS_BIAS(MLP_HAS_BIAS),
      .BIAS_I2H_WIDTH(BIAS_I2H_WIDTH),
      .BIAS_I2H_FRAC_WIDTH(BIAS_I2H_FRAC_WIDTH),
      .BIAS_H2O_WIDTH(BIAS_H2O_WIDTH),
      .BIAS_H2O_FRAC_WIDTH(BIAS_H2O_FRAC_WIDTH),

      .HIDDEN_WIDTH(HIDDEN_WIDTH),
      .HIDDEN_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),

      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),

      .IN_NUM(IN_NUM),
      .IN_DIM(IN_DIM),
      .NUM_HEADS(NUM_HEADS),
      .UNROLL_IN_NUM(UNROLL_IN_NUM),
      .UNROLL_IN_DIM(UNROLL_IN_DIM),
      .UNROLL_WQKV_DIM(UNROLL_WQKV_DIM),
      .UNROLL_HIDDEN_FEATURES(UNROLL_HIDDEN_FEATURES)
  ) block_inst (
      .data_in(conv_out),
      .data_in_valid(conv_out_valid),
      .data_in_ready(conv_out_ready),
      .*
  );
endmodule
