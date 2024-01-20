`timescale 1ns / 1ps
module fixed_block #(
    parameter IN_WIDTH = 6,
    parameter IN_FRAC_WIDTH = 1,

    parameter AF_MSA_ADD_WIDTH = 6,
    parameter AF_MSA_ADD_FRAC_WIDTH = 1,
    parameter MSA_IN_WIDTH = 6,
    parameter MSA_IN_FRAC_WIDTH = 6,

    parameter WQ_WIDTH = 4,
    parameter WQ_FRAC_WIDTH = 1,
    parameter WK_WIDTH = 4,
    parameter WK_FRAC_WIDTH = 1,
    parameter WV_WIDTH = 4,
    parameter WV_FRAC_WIDTH = 1,

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
    parameter EXP_WIDTH = 8,
    parameter EXP_FRAC_WIDTH = 4,
    parameter DIV_WIDTH = 10,
    parameter DS_SOFTMAX_WIDTH = 8,
    parameter DS_SOFTMAX_FRAC_WIDTH = 7,
    parameter DZ_WIDTH = 6,
    parameter DZ_FRAC_WIDTH = 1,


    parameter AF_MLP_IN_WIDTH = 7,
    parameter AF_MLP_IN_FRAC_WIDTH = 1,
    parameter AF_MLP_ADD_WIDTH = 6,
    parameter AF_MLP_ADD_FRAC_WIDTH = 1,

    parameter MLP_IN_WIDTH = 6,
    parameter MLP_IN_FRAC_WIDTH = 6,

    parameter MSA_OUT_WIDTH = AF_MLP_IN_WIDTH - 1,
    parameter MSA_OUT_FRAC_WIDTH = AF_MLP_IN_FRAC_WIDTH,
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

    parameter IN_NUM = 16,
    parameter IN_DIM = 6,
    // num_heads * wqkv_dim = IN_DIM
    parameter NUM_HEADS = 2,
    parameter WQKV_DIM = IN_DIM / NUM_HEADS,
    parameter WP_DIM = IN_DIM,
    parameter MLP_RATIO = 2,
    parameter UNROLL_IN_NUM = 2,
    parameter UNROLL_IN_DIM = 3,
    parameter UNROLL_WQKV_DIM = 3,
    // set it with unroll_in_dim for residual matching
    parameter UNROLL_WP_DIM = UNROLL_IN_DIM,

    parameter OUT_NUM = IN_NUM,
    parameter OUT_DIM = IN_DIM,
    parameter UNROLL_OUT_NUM = UNROLL_IN_NUM,
    parameter UNROLL_OUT_DIM = UNROLL_WP_DIM,
    // get attention output after 4 * 3 cycles 
    // mlp
    parameter IN_FEATURES = IN_DIM,
    parameter HIDDEN_FEATURES = MLP_RATIO * IN_FEATURES,
    parameter OUT_FEATURES = IN_FEATURES,

    parameter UNROLL_IN_FEATURES = UNROLL_IN_DIM,
    parameter UNROLL_HIDDEN_FEATURES = 3,
    parameter UNROLL_OUT_FEATURES = UNROLL_IN_DIM
) (
    input clk,
    input rst,
    //msa
    input [IN_WIDTH - 1:0] af_msa_weight[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0],
    input af_msa_weight_valid,
    output af_msa_weight_ready,
    input [AF_MSA_ADD_WIDTH - 1:0] af_msa_bias[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0],
    input af_msa_bias_valid,
    output af_msa_bias_ready,

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
    //mlp
    input [AF_MLP_IN_WIDTH - 1:0] af_mlp_weight[UNROLL_IN_NUM * UNROLL_IN_FEATURES - 1:0],
    input af_mlp_weight_valid,
    output af_mlp_weight_ready,
    input [AF_MLP_ADD_WIDTH - 1:0] af_mlp_bias[UNROLL_IN_NUM * UNROLL_IN_FEATURES - 1:0],
    input af_mlp_bias_valid,
    output af_mlp_bias_ready,

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
  logic [MSA_IN_WIDTH - 1:0] af_msa_out[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0];
  logic af_msa_out_valid, af_msa_out_ready;
  logic [MSA_OUT_WIDTH - 1:0] msa_out[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0];
  logic msa_out_valid, msa_out_ready;
  logic [AF_MLP_IN_WIDTH - 1:0] res_msa[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0];
  logic res_msa_valid, res_msa_ready;
  //msa
  logic ff_in_valid, ra_msa_in_valid;
  logic ff_in_ready, ra_msa_in_ready;
  logic [IN_WIDTH -1:0] ff_data_in[UNROLL_IN_NUM * UNROLL_IN_DIM - 1 : 0];
  logic ff_data_in_valid, ff_data_in_ready;
  split2 split2_inst (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({ff_in_valid, ra_msa_in_valid}),
      .data_out_ready({ff_in_ready, ra_msa_in_ready})
  );
  unpacked_fifo #(
      .DEPTH(IN_NUM * IN_DIM / (UNROLL_IN_DIM * UNROLL_IN_NUM)),
      .DATA_WIDTH(IN_WIDTH),
      .IN_NUM(UNROLL_IN_NUM * UNROLL_IN_DIM)
  ) fifo_in_inst (
      .data_out(ff_data_in),
      .data_out_valid(ff_data_in_valid),
      .data_out_ready(ff_data_in_ready),
      .data_in_valid(ff_in_valid),
      .data_in_ready(ff_in_ready),
      .*
  );
  affine_layernorm #(
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .OUT_WIDTH(MSA_IN_WIDTH),
      .OUT_FRAC_WIDTH(MSA_IN_FRAC_WIDTH),
      .BIAS_WIDTH(AF_MSA_ADD_WIDTH),
      .BIAS_FRAC_WIDTH(AF_MSA_ADD_FRAC_WIDTH),
      .IN_SIZE(UNROLL_IN_NUM * UNROLL_IN_DIM)
  ) aff_att (
      .weight(af_msa_weight),
      .weight_valid(af_msa_weight_valid),
      .weight_ready(af_msa_weight_ready),
      .bias(af_msa_bias),
      .bias_valid(af_msa_bias_valid),
      .bias_ready(af_msa_bias_ready),
      .data_in(ff_data_in),
      .data_in_valid(ff_data_in_valid),
      .data_in_ready(ff_data_in_ready),
      .data_out(af_msa_out),
      .data_out_valid(af_msa_out_valid),
      .data_out_ready(af_msa_out_ready),
      .*
  );

  //TODO: NORM here
  // msa
  fixed_msa #(
      .IN_WIDTH(MSA_IN_WIDTH),
      .IN_FRAC_WIDTH(MSA_IN_FRAC_WIDTH),

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

      .EXP_WIDTH(EXP_WIDTH),
      .EXP_FRAC_WIDTH(EXP_FRAC_WIDTH),
      .DIV_WIDTH(DIV_WIDTH),
      .DS_SOFTMAX_WIDTH(DS_SOFTMAX_WIDTH),
      .DS_SOFTMAX_FRAC_WIDTH(DS_SOFTMAX_FRAC_WIDTH),
      .DZ_WIDTH(DZ_WIDTH),
      .DZ_FRAC_WIDTH(DZ_FRAC_WIDTH),

      .OUT_WIDTH(MSA_OUT_WIDTH),
      .OUT_FRAC_WIDTH(MSA_OUT_FRAC_WIDTH),

      .IN_Y(IN_NUM),
      .IN_X(IN_DIM),
      .NUM_HEADS(NUM_HEADS),
      .UNROLL_IN_Y(UNROLL_IN_NUM),
      .UNROLL_IN_X(UNROLL_IN_DIM),
      .UNROLL_WQKV_Y(UNROLL_WQKV_DIM),
      .WP_Y(WP_DIM),
      .UNROLL_WP_Y(UNROLL_WP_DIM)
  ) msa_inst (
      .data_in(af_msa_out),
      .data_in_valid(af_msa_out_valid),
      .data_in_ready(af_msa_out_ready),
      .data_out(msa_out),
      .data_out_valid(msa_out_valid),
      .data_out_ready(msa_out_ready),
      .*
  );
  res_add #(
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .MODULE_WIDTH(MSA_OUT_WIDTH),
      .MODULE_FRAC_WIDTH(MSA_OUT_FRAC_WIDTH),
      .IN_SIZE(IN_NUM * IN_DIM),
      .UNROLL_IN_SIZE(UNROLL_IN_NUM * UNROLL_IN_DIM)
  ) ra_msa_inst (
      .data_in(data_in),
      .data_in_valid(ra_msa_in_valid),
      .data_in_ready(ra_msa_in_ready),
      .module_in(msa_out),
      .module_in_valid(msa_out_valid),
      .module_in_ready(msa_out_ready),
      .data_out(res_msa),
      .data_out_valid(res_msa_valid),
      .data_out_ready(res_msa_ready),
      .*
  );

  logic [MLP_IN_WIDTH - 1:0] af_mlp_out[UNROLL_IN_NUM * UNROLL_IN_FEATURES - 1:0];
  logic af_mlp_out_valid, af_mlp_out_ready;
  logic ra_mlp_in_valid;
  assign ra_mlp_in_valid = res_msa_ready && res_msa_valid;
  localparam MLP_OUT_WIDTH = OUT_WIDTH - 1;
  localparam MLP_OUT_FRAC_WIDTH = OUT_FRAC_WIDTH;
  logic [MLP_OUT_WIDTH-1:0] mlp_out[UNROLL_IN_NUM * UNROLL_IN_DIM - 1:0];
  logic mlp_out_valid, mlp_out_ready;
  // mlp  
  affine_layernorm #(
      .IN_WIDTH(AF_MLP_IN_WIDTH),
      .IN_FRAC_WIDTH(AF_MLP_IN_FRAC_WIDTH),
      .OUT_WIDTH(MLP_IN_WIDTH),
      .OUT_FRAC_WIDTH(MLP_IN_FRAC_WIDTH),
      .BIAS_WIDTH(AF_MLP_ADD_WIDTH),
      .BIAS_FRAC_WIDTH(AF_MLP_ADD_FRAC_WIDTH),
      .IN_SIZE(UNROLL_IN_NUM * UNROLL_IN_FEATURES)
  ) aff_mlp (
      .data_in(res_msa),
      .data_in_valid(res_msa_valid),
      .data_in_ready(res_msa_ready),
      .weight(af_mlp_weight),
      .weight_valid(af_mlp_weight_valid),
      .weight_ready(af_mlp_weight_ready),
      .bias(af_mlp_bias),
      .bias_valid(af_mlp_bias_valid),
      .bias_ready(af_mlp_bias_ready),
      .data_out(af_mlp_out),
      .data_out_valid(af_mlp_out_valid),
      .data_out_ready(af_mlp_out_ready),
      .*
  );
  fixed_mlp #(
      .IN_WIDTH(MLP_IN_WIDTH),
      .IN_FRAC_WIDTH(MLP_IN_FRAC_WIDTH),
      .HIDDEN_WIDTH(HIDDEN_WIDTH),
      .HIDDEN_FRAC_WIDTH(HIDDEN_FRAC_WIDTH),
      .OUT_WIDTH(MLP_OUT_WIDTH),
      .OUT_FRAC_WIDTH(MLP_OUT_FRAC_WIDTH),

      .WEIGHT_I2H_WIDTH(WEIGHT_I2H_WIDTH),
      .WEIGHT_I2H_FRAC_WIDTH(WEIGHT_I2H_FRAC_WIDTH),
      .BIAS_I2H_WIDTH(BIAS_I2H_WIDTH),
      .BIAS_I2H_FRAC_WIDTH(BIAS_I2H_FRAC_WIDTH),

      .WEIGHT_H2O_WIDTH(WEIGHT_H2O_WIDTH),
      .WEIGHT_H2O_FRAC_WIDTH(WEIGHT_H2O_FRAC_WIDTH),
      .BIAS_H2O_WIDTH(BIAS_H2O_WIDTH),
      .BIAS_H2O_FRAC_WIDTH(BIAS_H2O_FRAC_WIDTH),

      .IN_NUM(IN_NUM),
      .IN_FEATURES(IN_FEATURES),
      .HIDDEN_FEATURES(HIDDEN_FEATURES),
      .UNROLL_IN_NUM(UNROLL_IN_NUM),
      .UNROLL_IN_FEATURES(UNROLL_IN_FEATURES),
      .UNROLL_HIDDEN_FEATURES(UNROLL_HIDDEN_FEATURES),
      .UNROLL_OUT_FEATURES(UNROLL_OUT_FEATURES)
  ) mlp_inst (
      .data_in(af_mlp_out),
      .data_in_valid(af_mlp_out_valid),
      .data_in_ready(af_mlp_out_ready),
      .data_out(mlp_out),
      .data_out_valid(mlp_out_valid),
      .data_out_ready(mlp_out_ready),
      .*

  );

  res_add #(
      .IN_WIDTH(AF_MLP_IN_WIDTH),
      .IN_FRAC_WIDTH(AF_MLP_IN_FRAC_WIDTH),
      .MODULE_WIDTH(MLP_OUT_WIDTH),
      .MODULE_FRAC_WIDTH(MLP_OUT_FRAC_WIDTH),
      .IN_SIZE(IN_NUM * IN_DIM),
      .UNROLL_IN_SIZE(UNROLL_IN_NUM * UNROLL_IN_DIM)
  ) ra_mlp_inst (
      .data_in(res_msa),
      .data_in_valid(ra_mlp_in_valid),
      .data_in_ready(),
      .module_in(mlp_out),
      .module_in_valid(mlp_out_valid),
      .module_in_ready(mlp_out_ready),
      .*
  );
endmodule

module res_add #(
    parameter IN_WIDTH = 32,
    parameter IN_FRAC_WIDTH = 1,
    parameter MODULE_WIDTH = 16,
    parameter MODULE_FRAC_WIDTH = 1,
    parameter OUT_WIDTH = MODULE_WIDTH + 1,
    parameter UNROLL_IN_SIZE = 3,
    parameter IN_SIZE = 3
) (
    input logic clk,
    input logic rst,
    input logic [IN_WIDTH-1:0] data_in[UNROLL_IN_SIZE - 1:0],
    input logic data_in_valid,
    output logic data_in_ready,
    input logic [MODULE_WIDTH-1:0] module_in[UNROLL_IN_SIZE - 1:0],
    input logic module_in_valid,
    output logic module_in_ready,
    output logic [OUT_WIDTH-1:0] data_out[UNROLL_IN_SIZE-1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  logic [OUT_WIDTH-1:0] reg_in[UNROLL_IN_SIZE - 1:0];
  logic reg_in_valid, reg_in_ready;
  unpacked_skid_buffer #(
      .DATA_WIDTH(OUT_WIDTH),
      .IN_NUM(UNROLL_IN_SIZE)
  ) reg_inst (
      .data_in(reg_in),
      .data_in_valid(reg_in_valid),
      .data_in_ready(reg_in_ready),
      .*
  );
  logic [IN_WIDTH-1:0] ff_data_in[UNROLL_IN_SIZE - 1:0];
  logic ff_data_in_valid, ff_data_in_ready;
  unpacked_fifo #(
      .DEPTH(IN_SIZE),
      .DATA_WIDTH(IN_WIDTH),
      .IN_NUM(UNROLL_IN_SIZE)
  ) fifo_in_inst (
      .data_out(ff_data_in),
      .data_out_valid(ff_data_in_valid),
      .data_out_ready(ff_data_in_ready),
      .*
  );
  logic [MODULE_WIDTH - 1:0] cast_in[UNROLL_IN_SIZE - 1:0];
  logic cast_in_valid, cast_in_ready;
  fixed_rounding #(
      .IN_SIZE(UNROLL_IN_SIZE),
      .IN_WIDTH(IN_WIDTH),
      .IN_FRAC_WIDTH(IN_FRAC_WIDTH),
      .OUT_WIDTH(MODULE_WIDTH),
      .OUT_FRAC_WIDTH(MODULE_FRAC_WIDTH)
  ) msa_in_cast (
      .data_in (ff_data_in),
      .data_out(cast_in)
  );
  for (genvar i = 0; i < UNROLL_IN_SIZE; i++)
    assign reg_in[i] = {cast_in[i][MODULE_WIDTH-1],cast_in[i]}+ {module_in[i][MODULE_WIDTH-1],module_in[i]};

  join2 #() resadd_msa_join_inst (
      .data_in_ready ({module_in_ready, ff_data_in_ready}),
      .data_in_valid ({module_in_valid, ff_data_in_valid}),
      .data_out_valid(reg_in_valid),
      .data_out_ready(reg_in_ready)
  );
endmodule

