`timescale 1ns / 1ps
module fixed_msa #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 1,

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

    parameter WP_WIDTH = 8,
    parameter WP_FRAC_WIDTH = 1,
    parameter BP_WIDTH = 8,
    parameter BP_FRAC_WIDTH = 1,

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

    parameter OUT_WIDTH = 8,
    parameter OUT_FRAC_WIDTH = 1,

    parameter IN_Y = 6,
    parameter UNROLL_IN_Y = 1,
    parameter ITER_IN_Y = IN_Y / UNROLL_IN_Y,

    parameter IN_X = 12,
    parameter UNROLL_IN_X = 2,
    parameter ITER_IN_X = IN_X / UNROLL_IN_X,

    // make sure NUM_HEADS * WQKV_Y = IN_X
    parameter NUM_HEADS = 1,
    parameter WQKV_Y = IN_X / NUM_HEADS,
    parameter UNROLL_WQKV_Y = 1,
    parameter ITER_WQKV_Y = WQKV_Y / UNROLL_WQKV_Y,

    // WP_Y = IN_X
    parameter WP_Y = 12,
    parameter UNROLL_WP_Y = 2,
    parameter WP_SIZE = NUM_HEADS * UNROLL_WQKV_Y,
    // and WH_PARALLEL
    parameter WQKV_SIZE = UNROLL_IN_X,
    parameter OUT_PARALLELISM = UNROLL_IN_Y,
    parameter OUT_SIZE = UNROLL_WP_Y
) (
    input clk,
    input rst,

    input [WQ_WIDTH - 1:0] weight_q[NUM_HEADS * UNROLL_WQKV_Y * WQKV_SIZE -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WK_WIDTH - 1:0] weight_k[NUM_HEADS * UNROLL_WQKV_Y * WQKV_SIZE -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WV_WIDTH - 1:0] weight_v[NUM_HEADS * UNROLL_WQKV_Y * WQKV_SIZE -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [WP_WIDTH - 1:0] weight_p[UNROLL_WP_Y * WP_SIZE -1 : 0],
    input weight_p_valid,
    output weight_p_ready,

    input [BQ_WIDTH - 1:0] bias_q[NUM_HEADS * UNROLL_WQKV_Y -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BK_WIDTH - 1:0] bias_k[NUM_HEADS * UNROLL_WQKV_Y -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BV_WIDTH - 1:0] bias_v[NUM_HEADS * UNROLL_WQKV_Y -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [BP_WIDTH - 1:0] bias_p[UNROLL_WP_Y -1 : 0],
    input bias_p_valid,
    output bias_p_ready,


    input [IN_WIDTH -1:0] data_in[UNROLL_IN_Y * UNROLL_IN_X - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [OUT_WIDTH -1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);
  // define head in size
  localparam H_IN_SIZE = UNROLL_WQKV_Y * WQKV_SIZE;
  logic [DZ_WIDTH - 1:0] sa_out[UNROLL_IN_Y * NUM_HEADS * UNROLL_WQKV_Y - 1:0];
  logic sa_out_valid, sa_out_ready;
  for (genvar i = 0; i < NUM_HEADS; i++) begin : head
    /* verilator lint_off UNUSEDSIGNAL */
    // define each head data_out
    logic [DZ_WIDTH - 1:0] h_sa_out[UNROLL_IN_Y * UNROLL_WQKV_Y - 1:0];
    logic h_sa_out_valid;
    logic h_sa_out_ready;
    assign h_sa_out_ready = sa_out_ready;
    // define each head data_in
    logic [IN_WIDTH - 1:0] h_data_in[UNROLL_IN_Y * UNROLL_IN_X - 1:0];
    logic h_data_in_valid, h_data_in_ready;
    assign h_data_in = data_in;
    assign h_data_in_valid = data_in_valid;
    // define each head weight
    logic [WQ_WIDTH - 1:0] h_weight_q[UNROLL_WQKV_Y * WQKV_SIZE - 1:0];
    logic h_weight_q_valid, h_weight_q_ready;
    logic [WK_WIDTH - 1:0] h_weight_k[UNROLL_WQKV_Y * WQKV_SIZE - 1:0];
    logic h_weight_k_valid, h_weight_k_ready;
    logic [WV_WIDTH - 1:0] h_weight_v[UNROLL_WQKV_Y * WQKV_SIZE - 1:0];
    logic h_weight_v_valid, h_weight_v_ready;

    logic [BQ_WIDTH-1:0] h_bias_q[UNROLL_WQKV_Y - 1:0];
    logic h_bias_q_valid, h_bias_q_ready;
    logic [BK_WIDTH-1:0] h_bias_k[UNROLL_WQKV_Y - 1:0];
    logic h_bias_k_valid, h_bias_k_ready;
    logic [BV_WIDTH-1:0] h_bias_v[UNROLL_WQKV_Y - 1:0];
    logic h_bias_v_valid, h_bias_v_ready;


    assign h_weight_q = weight_q[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_k = weight_k[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_v = weight_v[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_q_valid = weight_q_valid;
    assign h_weight_k_valid = weight_k_valid;
    assign h_weight_v_valid = weight_v_valid;

    assign h_bias_q = bias_q[UNROLL_WQKV_Y*i+UNROLL_WQKV_Y-1:UNROLL_WQKV_Y*i];
    assign h_bias_k = bias_k[UNROLL_WQKV_Y*i+UNROLL_WQKV_Y-1:UNROLL_WQKV_Y*i];
    assign h_bias_v = bias_v[UNROLL_WQKV_Y*i+UNROLL_WQKV_Y-1:UNROLL_WQKV_Y*i];
    assign h_bias_q_valid = bias_q_valid;
    assign h_bias_k_valid = bias_k_valid;
    assign h_bias_v_valid = bias_v_valid;

    fixed_self_att #(
        .DATA_WIDTH(IN_WIDTH),
        .DATA_FRAC_WIDTH(IN_FRAC_WIDTH),

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
        .IN_PARALLELISM(UNROLL_IN_Y),
        .IN_NUM_PARALLELISM(ITER_IN_Y),
        .IN_SIZE(UNROLL_IN_X),
        .IN_DEPTH(ITER_IN_X),
        .W_PARALLELISM(UNROLL_WQKV_Y),
        .W_NUM_PARALLELISM(ITER_WQKV_Y)
    ) satt_inst (
        .data_in(h_data_in),
        .data_in_valid(h_data_in_valid),
        .data_in_ready(h_data_in_ready),
        .weight_q(h_weight_q),
        .weight_q_valid(h_weight_q_valid),
        .weight_q_ready(h_weight_q_ready),
        .weight_k(h_weight_k),
        .weight_k_valid(h_weight_k_valid),
        .weight_k_ready(h_weight_k_ready),
        .weight_v(h_weight_v),
        .weight_v_valid(h_weight_v_valid),
        .weight_v_ready(h_weight_v_ready),
        .bias_q(h_bias_q),
        .bias_q_valid(h_bias_q_valid),
        .bias_q_ready(h_bias_q_ready),
        .bias_k(h_bias_k),
        .bias_k_valid(h_bias_k_valid),
        .bias_k_ready(h_bias_k_ready),
        .bias_v(h_bias_v),
        .bias_v_valid(h_bias_v_valid),
        .bias_v_ready(h_bias_v_ready),
        .data_out(h_sa_out),
        .data_out_valid(h_sa_out_valid),
        .data_out_ready(h_sa_out_ready),
        .*
    );
  end
  assign weight_q_ready = head[0].h_weight_q_ready;
  assign weight_k_ready = head[0].h_weight_k_ready;
  assign weight_v_ready = head[0].h_weight_v_ready;
  assign bias_q_ready   = head[0].h_bias_q_ready;
  assign bias_k_ready   = head[0].h_bias_k_ready;
  assign bias_v_ready   = head[0].h_bias_v_ready;
  assign data_in_ready  = head[0].h_data_in_ready;

  //transpose here
  for (genvar i = 0; i < UNROLL_IN_Y; i++)
  for (genvar j = 0; j < NUM_HEADS; j++)
  for (genvar k = 0; k < UNROLL_WQKV_Y; k++)
    assign sa_out[(i*NUM_HEADS+j)*UNROLL_WQKV_Y+k] = head[j].h_sa_out[i*UNROLL_WQKV_Y+k];

  assign sa_out_valid = head[0].h_sa_out_valid;
  fixed_2d_linear #(
      .IN_WIDTH(DZ_WIDTH),
      .IN_FRAC_WIDTH(DZ_FRAC_WIDTH),
      .WEIGHT_WIDTH(WP_WIDTH),
      .WEIGHT_FRAC_WIDTH(WP_FRAC_WIDTH),
      .BIAS_WIDTH(BP_WIDTH),
      .BIAS_FRAC_WIDTH(BP_FRAC_WIDTH),
      .OUT_WIDTH(OUT_WIDTH),
      .OUT_FRAC_WIDTH(OUT_FRAC_WIDTH),

      .IN_Y(IN_Y),
      .UNROLL_IN_Y(UNROLL_IN_Y),
      .IN_X(NUM_HEADS * WQKV_Y),
      .UNROLL_IN_X(NUM_HEADS * UNROLL_WQKV_Y),
      .W_Y(WP_Y),
      .UNROLL_W_Y(UNROLL_WP_Y)
  ) inst_fmmc_k (
      .clk(clk),
      .rst(rst),
      .data_in(sa_out),
      .data_in_valid(sa_out_valid),
      .data_in_ready(sa_out_ready),
      .weight(weight_p),
      .weight_valid(weight_p_valid),
      .weight_ready(weight_p_ready),
      .bias(bias_p),
      .bias_valid(bias_p_valid),
      .bias_ready(bias_p_ready),
      .data_out(data_out),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
endmodule

