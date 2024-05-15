`timescale 1ns / 1ps
module fixed_pvt #(
    parameter IN_WIDTH                   = 8,
    parameter IN_FRAC_WIDTH              = 3,
    parameter OUT_WIDTH                  = 8,
    parameter OUT_FRAC_WIDTH             = 3,
    //patch has bias
    parameter PATCH_EMBED_W_WIDTH_3      = 8,
    parameter PATCH_EMBED_W_FRAC_WIDTH_3 = 6,
    parameter PATCH_EMBED_B_WIDTH_3      = 8,
    parameter PATCH_EMBED_B_FRAC_WIDTH_3 = 5,
    parameter POS_ADD_IN_WIDTH_3         = 8,
    parameter POS_ADD_IN_FRAC_WIDTH_3    = 5,

    // block has bias
    parameter BLOCK_IN_WIDTH = 8,
    parameter BLOCK_IN_FRAC_WIDTH = 3,

    parameter BLOCK_AF_MSA_ADD_WIDTH = 8,
    parameter BLOCK_AF_MSA_ADD_FRAC_WIDTH = 3,
    parameter BLOCK_MSA_IN_WIDTH = 8,
    parameter BLOCK_MSA_IN_FRAC_WIDTH = 3,

    parameter BLOCK_WQ_WIDTH = 6,
    parameter BLOCK_WQ_FRAC_WIDTH = 4,
    parameter BLOCK_WK_WIDTH = 6,
    parameter BLOCK_WK_FRAC_WIDTH = 4,
    parameter BLOCK_WV_WIDTH = 6,
    parameter BLOCK_WV_FRAC_WIDTH = 4,

    parameter BLOCK_BQ_WIDTH = 6,
    parameter BLOCK_BQ_FRAC_WIDTH = 4,
    parameter BLOCK_BK_WIDTH = 6,
    parameter BLOCK_BK_FRAC_WIDTH = 4,
    parameter BLOCK_BV_WIDTH = 6,
    parameter BLOCK_BV_FRAC_WIDTH = 4,

    parameter BLOCK_WP_WIDTH = 6,
    parameter BLOCK_WP_FRAC_WIDTH = 4,
    parameter BLOCK_BP_WIDTH = 6,
    parameter BLOCK_BP_FRAC_WIDTH = 4,

    parameter BLOCK_DQ_WIDTH = 8,
    parameter BLOCK_DQ_FRAC_WIDTH = 3,
    parameter BLOCK_DK_WIDTH = 8,
    parameter BLOCK_DK_FRAC_WIDTH = 3,
    parameter BLOCK_DV_WIDTH = 8,
    parameter BLOCK_DV_FRAC_WIDTH = 3,

    parameter BLOCK_DS_WIDTH = 8,
    parameter BLOCK_DS_FRAC_WIDTH = 3,
    parameter BLOCK_EXP_WIDTH = 8,
    parameter BLOCK_EXP_FRAC_WIDTH = 5,
    parameter BLOCK_DIV_WIDTH = 9,
    parameter BLOCK_DS_SOFTMAX_WIDTH = 8,
    parameter BLOCK_DS_SOFTMAX_FRAC_WIDTH = 3,
    parameter BLOCK_DZ_WIDTH = 8,
    parameter BLOCK_DZ_FRAC_WIDTH = 3,

    parameter BLOCK_AF_MLP_IN_WIDTH = 9,
    parameter BLOCK_AF_MLP_IN_FRAC_WIDTH = 3,
    parameter BLOCK_AF_MLP_ADD_WIDTH = 8,
    parameter BLOCK_AF_MLP_ADD_FRAC_WIDTH = 3,

    parameter BLOCK_MLP_IN_WIDTH = 8,
    parameter BLOCK_MLP_IN_FRAC_WIDTH = 3,

    parameter BLOCK_WEIGHT_I2H_WIDTH      = 6,
    parameter BLOCK_WEIGHT_I2H_FRAC_WIDTH = 4,
    parameter BLOCK_WEIGHT_H2O_WIDTH      = 6,
    parameter BLOCK_WEIGHT_H2O_FRAC_WIDTH = 4,
    parameter BLOCK_MLP_HIDDEN_WIDTH      = 8,
    parameter BLOCK_MLP_HIDDEN_FRAC_WIDTH = 3,
    parameter BLOCK_MLP_HAS_BIAS          = 1,
    parameter BLOCK_BIAS_I2H_WIDTH        = 6,
    parameter BLOCK_BIAS_I2H_FRAC_WIDTH   = 4,
    parameter BLOCK_BIAS_H2O_WIDTH        = 6,
    parameter BLOCK_BIAS_H2O_FRAC_WIDTH   = 4,
    //head has bias
    parameter HEAD_IN_WIDTH               = 8,
    parameter HEAD_IN_FRAC_WIDTH          = 3,
    parameter HEAD_W_WIDTH                = 8,
    parameter HEAD_W_FRAC_WIDTH           = 4,
    parameter HEAD_B_WIDTH                = 8,
    parameter HEAD_B_FRAC_WIDTH           = 4,

    // patch embedding
    // INPUT = IN_C * IN_Y * IN_X
    // output = OUT_Y * OUT_X( OUT_Y is NUM_PATCH, OUT_X is EMBED_DIM)
    parameter PATCH_EMBED_IN_C_3 = 3,
    parameter PATCH_EMBED_IN_Y_3 = 224,
    parameter PATCH_EMEBD_IN_X_3 = 224,
    parameter PATCH_SIZE_3 = 16,
    parameter PATCH_EMEBD_NUM_PATCH_3 = PATCH_EMBED_IN_Y_3*PATCH_EMEBD_IN_X_3/(PATCH_SIZE_3*PATCH_SIZE_3),
    parameter PATCH_EMBED_EMBED_DIM_3 = 384,

    parameter NUM_HEADS   = 6,
    parameter MLP_RATIO   = 2,
    parameter NUM_CLASSES = 10,

    parameter PATCH_EMEBD_UNROLL_KERNEL_OUT_3 = 24,
    parameter PATCH_EMEBD_UNROLL_IN_C_3 = 3,
    parameter PATCH_EMBED_UNROLL_EMBED_DIM_3 = 8,
    parameter BLOCK_UNROLL_WQKV_DIM = 2,
    parameter BLOCK_UNROLL_HIDDEN_FEATURES = 2,
    parameter HEAD_UNROLL_OUT_X = 1,

    parameter BLOCK_IN_NUM = PATCH_EMEBD_NUM_PATCH_3 + 1,  // cls token
    parameter BLOCK_IN_DIM = PATCH_EMBED_EMBED_DIM_3,
    // num_heads * wqkv_dim = IN_DIM

    parameter BLOCK_UNROLL_IN_NUM = 1,
    parameter BLOCK_UNROLL_IN_DIM = PATCH_EMBED_UNROLL_EMBED_DIM_3,
    // head
    parameter HEAD_IN_Y = BLOCK_IN_NUM,
    parameter HEAD_IN_X = BLOCK_IN_DIM,
    parameter HEAD_OUT_X = NUM_CLASSES,
    parameter HEAD_UNROLL_IN_Y = 1,
    parameter HEAD_UNROLL_IN_X = BLOCK_UNROLL_IN_DIM

) (
    input clk,
    input rst,
    // patch embedding
    input  [PATCH_EMBED_W_WIDTH_3-1:0] patch_embed_weight_3      [PATCH_EMEBD_UNROLL_KERNEL_OUT_3 * PATCH_EMBED_UNROLL_EMBED_DIM_3 -1:0],
    input patch_embed_weight_3_valid,
    output patch_embed_weight_3_ready,

    input  [PATCH_EMBED_B_WIDTH_3-1:0] patch_embed_bias_3      [PATCH_EMBED_UNROLL_EMBED_DIM_3-1:0],
    input                              patch_embed_bias_3_valid,
    output                             patch_embed_bias_3_ready,

    // position embedding 
    input [POS_ADD_IN_WIDTH_3-1:0] cls_token[PATCH_EMBED_UNROLL_EMBED_DIM_3-1:0],
    input cls_token_valid,
    output cls_token_ready,

    input [POS_ADD_IN_WIDTH_3-1:0] pos_embed_in[PATCH_EMBED_UNROLL_EMBED_DIM_3-1:0],
    input pos_embed_in_valid,
    output pos_embed_in_ready,
    //msa
    input [BLOCK_IN_WIDTH - 1:0] af_msa_weight[BLOCK_UNROLL_IN_NUM * BLOCK_UNROLL_IN_DIM - 1:0],
    input af_msa_weight_valid,
    output af_msa_weight_ready,
    input [BLOCK_AF_MSA_ADD_WIDTH - 1:0] af_msa_bias [BLOCK_UNROLL_IN_NUM * BLOCK_UNROLL_IN_DIM - 1:0],
    input af_msa_bias_valid,
    output af_msa_bias_ready,

    input [BLOCK_WQ_WIDTH - 1:0] weight_q[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM * BLOCK_UNROLL_IN_DIM -1 : 0],
    input weight_q_valid,
    output weight_q_ready,
    input [BLOCK_WK_WIDTH - 1:0] weight_k[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM * BLOCK_UNROLL_IN_DIM -1 : 0],
    input weight_k_valid,
    output weight_k_ready,
    input [BLOCK_WV_WIDTH - 1:0] weight_v[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM * BLOCK_UNROLL_IN_DIM -1 : 0],
    input weight_v_valid,
    output weight_v_ready,
    input [BLOCK_WP_WIDTH - 1:0] weight_p[BLOCK_UNROLL_IN_DIM * NUM_HEADS * BLOCK_UNROLL_WQKV_DIM -1 : 0],
    input weight_p_valid,
    output weight_p_ready,
    input [BLOCK_BQ_WIDTH - 1:0] bias_q[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM -1 : 0],
    input bias_q_valid,
    output bias_q_ready,
    input [BLOCK_BK_WIDTH - 1:0] bias_k[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM -1 : 0],
    input bias_k_valid,
    output bias_k_ready,
    input [BLOCK_BV_WIDTH - 1:0] bias_v[NUM_HEADS * BLOCK_UNROLL_WQKV_DIM -1 : 0],
    input bias_v_valid,
    output bias_v_ready,
    input [BLOCK_BP_WIDTH - 1:0] bias_p[BLOCK_UNROLL_IN_DIM -1 : 0],
    input bias_p_valid,
    output bias_p_ready,
    //mlp
    input [BLOCK_AF_MLP_IN_WIDTH - 1:0] af_mlp_weight [BLOCK_UNROLL_IN_NUM * BLOCK_UNROLL_IN_DIM - 1:0],
    input af_mlp_weight_valid,
    output af_mlp_weight_ready,
    input [BLOCK_AF_MLP_ADD_WIDTH - 1:0] af_mlp_bias [BLOCK_UNROLL_IN_NUM * BLOCK_UNROLL_IN_DIM - 1:0],
    input af_mlp_bias_valid,
    output af_mlp_bias_ready,

    input [BLOCK_WEIGHT_I2H_WIDTH-1:0] weight_in2hidden[BLOCK_UNROLL_HIDDEN_FEATURES * BLOCK_UNROLL_IN_DIM - 1:0],
    input weight_in2hidden_valid,
    output weight_in2hidden_ready,
    input [BLOCK_WEIGHT_H2O_WIDTH-1:0] weight_hidden2out[BLOCK_UNROLL_IN_DIM * BLOCK_UNROLL_HIDDEN_FEATURES - 1:0],
    input weight_hidden2out_valid,
    output weight_hidden2out_ready,
    input [BLOCK_BIAS_I2H_WIDTH-1:0] bias_in2hidden[BLOCK_UNROLL_HIDDEN_FEATURES - 1:0],
    input bias_in2hidden_valid,
    output bias_in2hidden_ready,
    input [BLOCK_BIAS_H2O_WIDTH-1:0] bias_hidden2out[BLOCK_UNROLL_IN_DIM - 1:0],
    input bias_hidden2out_valid,
    output bias_hidden2out_ready,
    // head
    input [HEAD_W_WIDTH-1:0] head_weight[HEAD_UNROLL_OUT_X * HEAD_UNROLL_IN_X - 1:0],
    input head_weight_valid,
    output head_weight_ready,
    input [HEAD_B_WIDTH-1:0] head_bias[HEAD_UNROLL_OUT_X - 1:0],
    input head_bias_valid,
    output head_bias_ready,


    input [IN_WIDTH -1:0] data_in[PATCH_EMEBD_UNROLL_IN_C_3 - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [OUT_WIDTH -1:0] data_out[HEAD_UNROLL_OUT_X - 1:0],
    output data_out_valid,
    input data_out_ready
);
  logic [POS_ADD_IN_WIDTH_3 - 1:0] patch_embed_out_3[PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];
  logic patch_embed_out_3_valid, patch_embed_out_3_ready;
  fixed_patch_embed #(
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .W_WIDTH(PATCH_EMBED_W_WIDTH_3),
      .W_FRAC_WIDTH(PATCH_EMBED_W_FRAC_WIDTH_3),
      .BIAS_WIDTH(PATCH_EMBED_B_WIDTH_3),
      .BIAS_FRAC_WIDTH(PATCH_EMBED_B_FRAC_WIDTH_3),
      .OUT_WIDTH(POS_ADD_IN_WIDTH_3),
      .OUT_FRAC_WIDTH(POS_ADD_IN_FRAC_WIDTH_3),
      .IN_C(PATCH_EMBED_IN_C_3),
      .IN_Y(PATCH_EMBED_IN_Y_3),
      .IN_X(PATCH_EMEBD_IN_X_3),
      .OUT_C(PATCH_EMBED_EMBED_DIM_3),
      .KERNEL_SIZE(PATCH_SIZE_3),
      .UNROLL_KERNEL_OUT(PATCH_EMEBD_UNROLL_KERNEL_OUT_3),
      .UNROLL_OUT_C(PATCH_EMBED_UNROLL_EMBED_DIM_3),
      .UNROLL_IN_C(PATCH_EMEBD_UNROLL_IN_C_3),
      .SLIDING_NUM(PATCH_EMEBD_NUM_PATCH_3)
  ) patemb_inst (
      .weight(patch_embed_weight_3),
      .weight_valid(patch_embed_weight_3_valid),
      .weight_ready(patch_embed_weight_3_ready),

      .bias(patch_embed_bias_3),
      .bias_valid(patch_embed_bias_3_valid),
      .bias_ready(patch_embed_bias_3_ready),

      .data_out(patch_embed_out_3),
      .data_out_valid(patch_embed_out_3_valid),
      .data_out_ready(patch_embed_out_3_ready),
      .*
  );
  logic [POS_ADD_IN_WIDTH_3 - 1:0] pos_data_in[PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];
  logic pos_data_in_valid, pos_data_in_ready;
  // cls token
  wrap_data #(
      .IN_WIDTH(POS_ADD_IN_WIDTH_3),
      .WRAP_Y(1),
      .IN_Y(PATCH_EMEBD_NUM_PATCH_3),
      .IN_X(PATCH_EMBED_EMBED_DIM_3),
      .UNROLL_IN_X(PATCH_EMBED_UNROLL_EMBED_DIM_3)
  ) cls_inst (
      .wrap_in(cls_token),
      .wrap_in_valid(cls_token_valid),
      .wrap_in_ready(cls_token_ready),
      .data_in(patch_embed_out_3),
      .data_in_valid(patch_embed_out_3_valid),
      .data_in_ready(patch_embed_out_3_ready),
      .data_out(pos_data_in),
      .data_out_valid(pos_data_in_valid),
      .data_out_ready(pos_data_in_ready),
      .*
  );
  // position embedding
  logic [POS_ADD_IN_WIDTH_3 + 1 - 1:0] pos_embed_out[PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];
  for (genvar i = 0; i < PATCH_EMBED_UNROLL_EMBED_DIM_3; i++)
    assign pos_embed_out[i] = {pos_embed_in[i][IN_WIDTH-1], pos_embed_in[i]} + {pos_data_in[i][IN_WIDTH-1], pos_data_in[i]};
  logic pos_embed_out_valid, pos_embed_out_ready;

  join2 #() fmm_join_inst (
      .data_in_ready ({pos_data_in_ready, pos_embed_in_ready}),
      .data_in_valid ({pos_data_in_valid, pos_embed_in_valid}),
      .data_out_valid(pos_embed_out_valid),
      .data_out_ready(pos_embed_out_ready)
  );

  logic [BLOCK_IN_WIDTH - 1:0] block_in [PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];

  logic [ HEAD_IN_WIDTH - 1:0] block_out[PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];
  logic block_out_valid, block_out_ready;
  fixed_rounding #(
      .IN_SIZE(PATCH_EMBED_UNROLL_EMBED_DIM_3),
      .IN_WIDTH(POS_ADD_IN_WIDTH_3 + 1),
      .IN_FRAC_WIDTH(POS_ADD_IN_FRAC_WIDTH_3),
      .OUT_WIDTH(BLOCK_IN_WIDTH),
      .OUT_FRAC_WIDTH(BLOCK_IN_FRAC_WIDTH)
  ) head_cast (
      .data_in (pos_embed_out),
      .data_out(block_in)
  );
  fixed_block #(
      .IN_WIDTH              (BLOCK_IN_WIDTH),
      .IN_FRAC_WIDTH         (BLOCK_IN_FRAC_WIDTH),
      .AF_MSA_ADD_WIDTH      (BLOCK_AF_MSA_ADD_WIDTH),
      .AF_MSA_ADD_FRAC_WIDTH (BLOCK_AF_MSA_ADD_FRAC_WIDTH),
      .MSA_IN_WIDTH          (BLOCK_MSA_IN_WIDTH),
      .MSA_IN_FRAC_WIDTH     (BLOCK_MSA_IN_FRAC_WIDTH),
      .WQ_WIDTH              (BLOCK_WQ_WIDTH),
      .WQ_FRAC_WIDTH         (BLOCK_WQ_FRAC_WIDTH),
      .WK_WIDTH              (BLOCK_WK_WIDTH),
      .WK_FRAC_WIDTH         (BLOCK_WK_FRAC_WIDTH),
      .WV_WIDTH              (BLOCK_WV_WIDTH),
      .WV_FRAC_WIDTH         (BLOCK_WV_FRAC_WIDTH),
      .BQ_WIDTH              (BLOCK_BQ_WIDTH),
      .BQ_FRAC_WIDTH         (BLOCK_BQ_FRAC_WIDTH),
      .BK_WIDTH              (BLOCK_BK_WIDTH),
      .BK_FRAC_WIDTH         (BLOCK_BK_FRAC_WIDTH),
      .BV_WIDTH              (BLOCK_BV_WIDTH),
      .BV_FRAC_WIDTH         (BLOCK_BV_FRAC_WIDTH),
      .WP_WIDTH              (BLOCK_WP_WIDTH),
      .WP_FRAC_WIDTH         (BLOCK_WP_FRAC_WIDTH),
      .BP_WIDTH              (BLOCK_BP_WIDTH),
      .BP_FRAC_WIDTH         (BLOCK_BP_FRAC_WIDTH),
      .DQ_WIDTH              (BLOCK_DQ_WIDTH),
      .DQ_FRAC_WIDTH         (BLOCK_DQ_FRAC_WIDTH),
      .DK_WIDTH              (BLOCK_DK_WIDTH),
      .DK_FRAC_WIDTH         (BLOCK_DK_FRAC_WIDTH),
      .DV_WIDTH              (BLOCK_DV_WIDTH),
      .DV_FRAC_WIDTH         (BLOCK_DV_FRAC_WIDTH),
      .DS_WIDTH              (BLOCK_DS_WIDTH),
      .DS_FRAC_WIDTH         (BLOCK_DS_FRAC_WIDTH),
      .EXP_WIDTH             (BLOCK_EXP_WIDTH),
      .EXP_FRAC_WIDTH        (BLOCK_EXP_FRAC_WIDTH),
      .DIV_WIDTH             (BLOCK_DIV_WIDTH),
      .DS_SOFTMAX_WIDTH      (BLOCK_DS_SOFTMAX_WIDTH),
      .DS_SOFTMAX_FRAC_WIDTH (BLOCK_DS_SOFTMAX_FRAC_WIDTH),
      .DZ_WIDTH              (BLOCK_DZ_WIDTH),
      .DZ_FRAC_WIDTH         (BLOCK_DZ_FRAC_WIDTH),
      .AF_MLP_IN_WIDTH       (BLOCK_AF_MLP_IN_WIDTH),
      .AF_MLP_IN_FRAC_WIDTH  (BLOCK_AF_MLP_IN_FRAC_WIDTH),
      .AF_MLP_ADD_WIDTH      (BLOCK_AF_MLP_ADD_WIDTH),
      .AF_MLP_ADD_FRAC_WIDTH (BLOCK_AF_MLP_ADD_FRAC_WIDTH),
      .MLP_IN_WIDTH          (BLOCK_MLP_IN_WIDTH),
      .MLP_IN_FRAC_WIDTH     (BLOCK_MLP_IN_FRAC_WIDTH),
      .WEIGHT_I2H_WIDTH      (BLOCK_WEIGHT_I2H_WIDTH),
      .WEIGHT_I2H_FRAC_WIDTH (BLOCK_WEIGHT_I2H_FRAC_WIDTH),
      .WEIGHT_H2O_WIDTH      (BLOCK_WEIGHT_H2O_WIDTH),
      .WEIGHT_H2O_FRAC_WIDTH (BLOCK_WEIGHT_H2O_FRAC_WIDTH),
      .MLP_HAS_BIAS          (BLOCK_MLP_HAS_BIAS),
      .BIAS_I2H_WIDTH        (BLOCK_BIAS_I2H_WIDTH),
      .BIAS_I2H_FRAC_WIDTH   (BLOCK_BIAS_I2H_FRAC_WIDTH),
      .BIAS_H2O_WIDTH        (BLOCK_BIAS_H2O_WIDTH),
      .BIAS_H2O_FRAC_WIDTH   (BLOCK_BIAS_H2O_FRAC_WIDTH),
      .HIDDEN_WIDTH          (BLOCK_MLP_HIDDEN_WIDTH),
      .HIDDEN_FRAC_WIDTH     (BLOCK_MLP_HIDDEN_FRAC_WIDTH),
      .OUT_WIDTH             (HEAD_IN_WIDTH),
      .OUT_FRAC_WIDTH        (HEAD_IN_FRAC_WIDTH),
      .IN_NUM                (BLOCK_IN_NUM),
      .IN_DIM                (BLOCK_IN_DIM),
      .NUM_HEADS             (NUM_HEADS),
      .MLP_RATIO             (MLP_RATIO),
      .UNROLL_IN_NUM         (BLOCK_UNROLL_IN_NUM),
      .UNROLL_IN_DIM         (BLOCK_UNROLL_IN_DIM),
      .UNROLL_WQKV_DIM       (BLOCK_UNROLL_WQKV_DIM),
      .UNROLL_HIDDEN_FEATURES(BLOCK_UNROLL_HIDDEN_FEATURES)
  ) block_inst (
      .data_in(block_in),
      .data_in_valid(pos_embed_out_valid),
      .data_in_ready(pos_embed_out_ready),
      .data_out(block_out),
      .data_out_valid(block_out_valid),
      .data_out_ready(block_out_ready),
      .*
  );

  logic [HEAD_IN_WIDTH - 1:0] head_in[PATCH_EMBED_UNROLL_EMBED_DIM_3 - 1:0];
  logic head_in_valid, head_in_ready;
  cut_data #(
      .IN_WIDTH(HEAD_IN_WIDTH),
      .IN_Y(HEAD_IN_Y),
      .IN_X(HEAD_IN_X),
      .UNROLL_IN_X(HEAD_UNROLL_IN_X)
  ) cut_inst (
      .data_in(block_out),
      .data_in_valid(block_out_valid),
      .data_in_ready(block_out_ready),
      .data_out(head_in),
      .data_out_valid(head_in_valid),
      .data_out_ready(head_in_ready),
      .*
  );
  fixed_2d_linear #(
      .IN_WIDTH(HEAD_IN_WIDTH),
      .IN_FRAC_WIDTH(HEAD_IN_FRAC_WIDTH),
      .WEIGHT_WIDTH(HEAD_W_WIDTH),
      .WEIGHT_FRAC_WIDTH(HEAD_W_FRAC_WIDTH),
      .HAS_BIAS(1),
      .BIAS_WIDTH(HEAD_B_WIDTH),
      .BIAS_FRAC_WIDTH(HEAD_B_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),
      .IN_Y(1),
      .UNROLL_IN_Y(1),
      .IN_X(HEAD_IN_X),
      .UNROLL_IN_X(HEAD_UNROLL_IN_X),
      .W_Y(NUM_CLASSES),
      .UNROLL_W_Y(HEAD_UNROLL_OUT_X)
  ) head_inst (
      .data_in(head_in),
      .data_in_valid(head_in_valid),
      .data_in_ready(head_in_ready),
      .weight(head_weight),
      .weight_valid(head_weight_valid),
      .weight_ready(head_weight_ready),
      .bias(head_bias),
      .bias_valid(head_bias_valid),
      .bias_ready(head_bias_ready),
      .*
  );
endmodule
