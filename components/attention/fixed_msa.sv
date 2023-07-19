`timescale 1ns / 1ps
module fixed_msa #(
    parameter DATA_WIDTH = 8,
    parameter DATA_FRAC_WIDTH = 1,
    parameter WEIGHT_WIDTH = 8,
    parameter W_FRAC_WIDTH = 1,
    parameter BIAS_WIDTH = 8,
    parameter BIAS_FRAC_WIDTH = 1,

    parameter IN_PARALLELISM = 3,
    parameter IN_NUM_PARALLELISM = 2,

    parameter IN_SIZE  = 4,
    //define for matrix multilication
    parameter IN_DEPTH = 3,

    // make sure NUM_HEADS * WQKV_PARALLELISM * WQKV_NUM_PARALLELISM = IN_SIZE * IN_DEPTH
    // and WP_PARALLELISM * WP_NUM_PARALLELISM = IN_SIZE * IN_DEPTH
    parameter NUM_HEADS = 2,
    parameter WQKV_PARALLELISM = 3,
    parameter WQKV_NUM_PARALLELISM = 2,

    parameter WP_PARALLELISM = 4,
    parameter WP_NUM_PARALLELISM = 3,
    parameter WP_SIZE = NUM_HEADS * WQKV_PARALLELISM,
    // and WH_PARALLEL
    parameter WQKV_SIZE = IN_SIZE,


    parameter OUT_PARALLELISM = IN_PARALLELISM,
    parameter OUT_SIZE = WP_PARALLELISM
) (
    input clk,
    input rst,

    input [WEIGHT_WIDTH - 1:0] weight_q[NUM_HEADS * WQKV_PARALLELISM * WQKV_SIZE -1 : 0],
    input weight_q_valid,
    output weight_q_ready,

    input [WEIGHT_WIDTH - 1:0] weight_k[NUM_HEADS * WQKV_PARALLELISM * WQKV_SIZE -1 : 0],
    input weight_k_valid,
    output weight_k_ready,

    input [WEIGHT_WIDTH - 1:0] weight_v[NUM_HEADS * WQKV_PARALLELISM * WQKV_SIZE -1 : 0],
    input weight_v_valid,
    output weight_v_ready,

    input [WEIGHT_WIDTH - 1:0] weight_p[WP_PARALLELISM * WP_SIZE -1 : 0],
    input weight_p_valid,
    output weight_p_ready,

    input [BIAS_WIDTH - 1:0] bias_q[NUM_HEADS * WQKV_PARALLELISM -1 : 0],
    input bias_q_valid,
    output bias_q_ready,

    input [BIAS_WIDTH - 1:0] bias_k[NUM_HEADS * WQKV_PARALLELISM -1 : 0],
    input bias_k_valid,
    output bias_k_ready,

    input [BIAS_WIDTH - 1:0] bias_v[NUM_HEADS * WQKV_PARALLELISM -1 : 0],
    input bias_v_valid,
    output bias_v_ready,

    input [BIAS_WIDTH - 1:0] bias_p[WP_PARALLELISM -1 : 0],
    input bias_p_valid,
    output bias_p_ready,


    input [DATA_WIDTH -1:0] data_in[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_valid,
    output data_in_ready,

    output [DATA_WIDTH -1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);
  // define head in size
  localparam H_IN_SIZE = WQKV_PARALLELISM * WQKV_SIZE;
  logic [DATA_WIDTH - 1:0] sa_out[IN_PARALLELISM * NUM_HEADS * WQKV_PARALLELISM - 1:0];
  logic sa_out_valid, sa_out_ready;
  for (genvar i = 0; i < NUM_HEADS; i++) begin : head
    /* verilator lint_off UNUSEDSIGNAL */
    // define each head data_out
    logic [DATA_WIDTH - 1:0] h_sa_out[IN_PARALLELISM * WQKV_PARALLELISM - 1:0];
    logic h_sa_out_valid;
    logic h_sa_out_ready;
    assign h_sa_out_ready = sa_out_ready;
    // define each head data_in
    logic [DATA_WIDTH - 1:0] h_data_in[IN_PARALLELISM * IN_SIZE - 1:0];
    logic h_data_in_valid, h_data_in_ready;
    assign h_data_in = data_in;
    assign h_data_in_valid = data_in_valid;
    // define each head weight
    logic [WEIGHT_WIDTH - 1:0] h_weight_q[WQKV_PARALLELISM * WQKV_SIZE - 1:0];
    logic h_weight_q_valid, h_weight_q_ready;
    logic [WEIGHT_WIDTH - 1:0] h_weight_k[WQKV_PARALLELISM * WQKV_SIZE - 1:0];
    logic h_weight_k_valid, h_weight_k_ready;
    logic [WEIGHT_WIDTH - 1:0] h_weight_v[WQKV_PARALLELISM * WQKV_SIZE - 1:0];
    logic h_weight_v_valid, h_weight_v_ready;

    logic [WEIGHT_WIDTH-1:0] h_bias_q[WQKV_PARALLELISM - 1:0];
    logic h_bias_q_valid, h_bias_q_ready;
    logic [WEIGHT_WIDTH-1:0] h_bias_k[WQKV_PARALLELISM - 1:0];
    logic h_bias_k_valid, h_bias_k_ready;
    logic [WEIGHT_WIDTH-1:0] h_bias_v[WQKV_PARALLELISM - 1:0];
    logic h_bias_v_valid, h_bias_v_ready;


    assign h_weight_q = weight_q[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_k = weight_k[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_v = weight_v[H_IN_SIZE*i+H_IN_SIZE-1:H_IN_SIZE*i];
    assign h_weight_q_valid = weight_q_valid;
    assign h_weight_k_valid = weight_k_valid;
    assign h_weight_v_valid = weight_v_valid;

    assign h_bias_q = bias_q[WQKV_PARALLELISM*i+WQKV_PARALLELISM-1:WQKV_PARALLELISM*i];
    assign h_bias_k = bias_k[WQKV_PARALLELISM*i+WQKV_PARALLELISM-1:WQKV_PARALLELISM*i];
    assign h_bias_v = bias_v[WQKV_PARALLELISM*i+WQKV_PARALLELISM-1:WQKV_PARALLELISM*i];
    assign h_bias_q_valid = bias_q_valid;
    assign h_bias_k_valid = bias_k_valid;
    assign h_bias_v_valid = bias_v_valid;

    fixed_self_att #(
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
        .W_PARALLELISM(WQKV_PARALLELISM),
        .W_NUM_PARALLELISM(WQKV_NUM_PARALLELISM)
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

  logic [WEIGHT_WIDTH - 1:0] test_weight[WQKV_PARALLELISM * WQKV_SIZE - 1:0];
  assign test_weight = weight_v[H_IN_SIZE-1:0];
  //transpose here
  for (genvar i = 0; i < IN_PARALLELISM; i++)
  for (genvar j = 0; j < NUM_HEADS; j++)
  for (genvar k = 0; k < WQKV_PARALLELISM; k++)
    assign sa_out[(i*NUM_HEADS+j)*WQKV_PARALLELISM+k] = head[j].h_sa_out[i*WQKV_PARALLELISM+k];

  assign sa_out_valid = head[0].h_sa_out_valid;
  fixed_2d_linear #(
      .IN_WIDTH(DATA_WIDTH),
      .IN_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .WEIGHT_WIDTH(WEIGHT_WIDTH),
      .WEIGHT_FRAC_WIDTH(W_FRAC_WIDTH),
      .BIAS_WIDTH(BIAS_WIDTH),
      .BIAS_FRAC_WIDTH(BIAS_FRAC_WIDTH),
      .OUT_WIDTH(DATA_WIDTH),
      .OUT_FRAC_WIDTH(DATA_FRAC_WIDTH),
      .IN_PARALLELISM(IN_PARALLELISM),
      .IN_NUM_PARALLELISM(IN_NUM_PARALLELISM),
      .IN_SIZE(NUM_HEADS * WQKV_PARALLELISM),
      .IN_DEPTH(WQKV_NUM_PARALLELISM),
      .W_PARALLELISM(WP_PARALLELISM),
      .W_NUM_PARALLELISM(WP_NUM_PARALLELISM)
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

