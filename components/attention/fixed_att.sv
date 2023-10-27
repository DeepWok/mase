`timescale 1ns / 1ps
/* 
The first version without softmax 
noted in order to make output dimension match
make w_parallelism = w_size
    w_num_parallelism = in_depth
    but the dimension constraint, shown in the testbench part but not here
    data_in [IN_PARALLELISM * IN_NUM_PARALLELISM][IN_SIZE * IN_DEPTH]
    weight_q[W_PARALLELISM * W_NUM_PARALLELISM][W_SIZE * IN_DEPTH]
    weight_k[W_PARALLELISM * W_NUM_PARALLELISM][W_SIZE * IN_DEPTH]
    weight_v[W_PARALLELISM * W_NUM_PARALLELISM][W_SIZE * IN_DEPTH]

    data_q  [W_PARALLELISM * W_NUM_PARALLELISM][IN_PARALLELISM * IN_NUM_PARALLELISM]
    data_k  [W_PARALLELISM * W_NUM_PARALLELISM][IN_PARALLELISM * IN_NUM_PARALLELISM]

    data_v_t[IN_PARALLELISM * IN_NUM_PARALLELISM][W_PARALLELISM * W_NUM_PARALLELISM]

    data_s  [W_PARALLELISM * W_NUM_PARALLELISM][W_PARALLELISM * W_NUM_PARALLELISM]
    
    data_z  [W_PARALLELISM * W_NUM_PARALLELISM][IN_PARALLELISM * IN_NUM_PARALLELISM]
    data_out[IN_PARALLELISM][W_PARALLELISM]

    realize the function
    data_z = att(data_in)
*/
module fixed_att #(
    parameter DQIN_WIDTH = 8,
    parameter DQIN_FRAC_WIDTH = 1,
    parameter DKIN_WIDTH = 8,
    parameter DKIN_FRAC_WIDTH = 1,
    parameter DVIN_WIDTH = 8,
    parameter DVIN_FRAC_WIDTH = 1,

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

    parameter W_PARALLELISM = 2,
    parameter W_NUM_PARALLELISM = 3,
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

    input [DQIN_WIDTH -1:0] data_in_q[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_q_valid,
    output data_in_q_ready,

    input [DKIN_WIDTH -1:0] data_in_k[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_k_valid,
    output data_in_k_ready,

    input [DVIN_WIDTH -1:0] data_in_v[IN_PARALLELISM * IN_SIZE - 1 : 0],
    input data_in_v_valid,
    output data_in_v_ready,

    output [DZ_WIDTH -1:0] data_out[OUT_PARALLELISM * OUT_SIZE - 1:0],
    output data_out_valid,
    input data_out_ready
);

  logic [DQIN_WIDTH-1:0] ff_data_in_q[IN_PARALLELISM * IN_SIZE - 1:0];
  logic [DKIN_WIDTH-1:0] ff_data_in_k[IN_PARALLELISM * IN_SIZE - 1:0];
  logic [DVIN_WIDTH-1:0] ff_data_in_v[IN_PARALLELISM * IN_SIZE - 1:0];
  logic ff_data_in_q_valid, ff_data_in_k_valid, ff_data_in_v_valid;
  logic ff_data_in_q_ready, ff_data_in_k_ready, ff_data_in_v_ready;
  //   assign ff_data_in_qk_ready = ff_data_in_q_ready&&ff_data_in_k_ready;
  // fifo for qk
  unpacked_fifo #(
      .DEPTH(IN_DEPTH * IN_NUM_PARALLELISM),
      .DATA_WIDTH(DQIN_WIDTH),
      .IN_NUM(IN_PARALLELISM * IN_SIZE)
  ) fifo_q (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_q),
      .data_in_valid(data_in_q_valid),
      .data_in_ready(data_in_q_ready),
      .data_out(ff_data_in_q),
      .data_out_valid(ff_data_in_q_valid),
      .data_out_ready(ff_data_in_q_ready)
  );

  unpacked_fifo #(
      .DEPTH(IN_DEPTH * IN_NUM_PARALLELISM),
      .DATA_WIDTH(DKIN_WIDTH),
      .IN_NUM(IN_PARALLELISM * IN_SIZE)
  ) fifo_k (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_k),
      .data_in_valid(data_in_k_valid),
      .data_in_ready(data_in_k_ready),
      .data_out(ff_data_in_k),
      .data_out_valid(ff_data_in_k_valid),
      .data_out_ready(ff_data_in_k_ready)
  );

  logic [DQ_WIDTH - 1 : 0] data_q[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic data_q_valid, data_q_ready;
  logic [DK_WIDTH - 1 : 0] data_k[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic data_k_valid, data_k_ready;
  //matmul qk
  fixed_2d_linear #(
      .IN_WIDTH(DQIN_WIDTH),
      .IN_FRAC_WIDTH(DQIN_FRAC_WIDTH),
      .WEIGHT_WIDTH(WQ_WIDTH),
      .WEIGHT_FRAC_WIDTH(WQ_FRAC_WIDTH),
      .BIAS_WIDTH(BQ_WIDTH),
      .BIAS_FRAC_WIDTH(BQ_FRAC_WIDTH),
      .OUT_WIDTH(DQ_WIDTH),
      .OUT_FRAC_WIDTH(DQ_FRAC_WIDTH),
      .IN_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN_Y(IN_PARALLELISM),
      .IN_X(IN_DEPTH * IN_SIZE),
      .UNROLL_IN_X(IN_SIZE),
      .W_Y(W_PARALLELISM * W_NUM_PARALLELISM),
      .UNROLL_W_Y(W_PARALLELISM)
  ) inst_fmmc_q (
      .clk(clk),
      .rst(rst),
      .data_in(ff_data_in_q),
      .data_in_valid(ff_data_in_q_valid),
      .data_in_ready(ff_data_in_q_ready),
      .weight(weight_q),
      .weight_valid(weight_q_valid),
      .weight_ready(weight_q_ready),
      .bias(bias_q),
      .bias_valid(bias_q_valid),
      .bias_ready(bias_q_ready),
      .data_out(data_q),
      .data_out_valid(data_q_valid),
      .data_out_ready(data_q_ready)
  );
  fixed_2d_linear #(
      .IN_WIDTH(DKIN_WIDTH),
      .IN_FRAC_WIDTH(DKIN_FRAC_WIDTH),
      .WEIGHT_WIDTH(WK_WIDTH),
      .WEIGHT_FRAC_WIDTH(WK_FRAC_WIDTH),
      .BIAS_WIDTH(BK_WIDTH),
      .BIAS_FRAC_WIDTH(BK_FRAC_WIDTH),
      .OUT_WIDTH(DK_WIDTH),
      .OUT_FRAC_WIDTH(DK_FRAC_WIDTH),
      .IN_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN_Y(IN_PARALLELISM),
      .IN_X(IN_DEPTH * IN_SIZE),
      .UNROLL_IN_X(IN_SIZE),
      .W_Y(W_PARALLELISM * W_NUM_PARALLELISM),
      .UNROLL_W_Y(W_PARALLELISM)
  ) inst_fmmc_k (
      .clk(clk),
      .rst(rst),
      .data_in(ff_data_in_k),
      .data_in_valid(ff_data_in_k_valid),
      .data_in_ready(ff_data_in_k_ready),
      .weight(weight_k),
      .weight_valid(weight_k_valid),
      .weight_ready(weight_k_ready),
      .bias(bias_k),
      .bias_valid(bias_k_valid),
      .bias_ready(bias_k_ready),
      .data_out(data_k),
      .data_out_valid(data_k_valid),
      .data_out_ready(data_k_ready)
  );
  logic [DS_WIDTH - 1 : 0] data_s[IN_PARALLELISM * IN_PARALLELISM - 1:0];
  logic data_s_valid, data_s_ready;
  // matmul s
  /* verilator lint_off PINMISSING */
  fixed_matmul #(
      .IN1_WIDTH(DQ_WIDTH),
      .IN1_FRAC_WIDTH(DQ_FRAC_WIDTH),
      .IN2_WIDTH(DK_WIDTH),
      .IN2_FRAC_WIDTH(DK_FRAC_WIDTH),
      .OUT_WIDTH(DS_WIDTH),
      .OUT_FRAC_WIDTH(DS_FRAC_WIDTH),
      .IN1_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN1_Y(IN_PARALLELISM),
      .IN1_X(W_PARALLELISM * W_NUM_PARALLELISM),
      .UNROLL_IN1_X(W_PARALLELISM),
      .IN2_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN2_Y(IN_PARALLELISM)
  ) inst_fmmc_s (
      .clk(clk),
      .rst(rst),
      .data_in1(data_q),
      .data_in1_valid(data_q_valid),
      .data_in1_ready(data_q_ready),
      .data_in2(data_k),
      .data_in2_valid(data_k_valid),
      .data_in2_ready(data_k_ready),
      .data_out(data_s),
      .data_out_valid(data_s_valid),
      .data_out_ready(data_s_ready)
  );
  logic [DS_SOFTMAX_WIDTH - 1:0] softmax_s[IN_PARALLELISM * IN_PARALLELISM - 1:0];
  logic softmax_s_valid, softmax_s_ready;

  hash_softmax #(
      .IN_WIDTH(DS_WIDTH),
      .IN_FRAC_WIDTH(DS_FRAC_WIDTH),
      .EXP_WIDTH(EXP_WIDTH),
      .EXP_FRAC_WIDTH(EXP_FRAC_WIDTH),
      .DIV_WIDTH(DIV_WIDTH),
      .OUT_WIDTH(DS_SOFTMAX_WIDTH),
      .OUT_FRAC_WIDTH(DS_SOFTMAX_FRAC_WIDTH),
      .IN_SIZE(IN_PARALLELISM * IN_PARALLELISM),
      .OUT_SIZE(IN_PARALLELISM * IN_PARALLELISM),
      .IN_DEPTH(IN_NUM_PARALLELISM)
  ) softmax_inst (
      .data_in(data_s),
      .data_in_valid(data_s_valid),
      .data_in_ready(data_s_ready),
      .data_out(softmax_s),
      .data_out_valid(softmax_s_valid),
      .data_out_ready(softmax_s_ready),
      .*
  );

  logic [BV_WIDTH-1:0] bias_v_extend[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic [BV_WIDTH-1:0] ib_bias_v[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic ib_bias_v_valid, ib_bias_v_ready;
  // bias_v require transpose here
  for (genvar i = 0; i < W_PARALLELISM; i++)
  for (genvar j = 0; j < IN_PARALLELISM; j++) assign bias_v_extend[i*IN_PARALLELISM+j] = bias_v[i];

  input_buffer #(
      .IN_WIDTH(BV_WIDTH),
      .IN_PARALLELISM(W_PARALLELISM),
      .IN_SIZE(IN_PARALLELISM),
      .BUFFER_SIZE(1),
      .REPEAT(IN_NUM_PARALLELISM)
  ) bias_v_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(bias_v_extend),
      .data_in_valid(bias_v_valid),
      .data_in_ready(bias_v_ready),
      .data_out(ib_bias_v),
      .data_out_valid(ib_bias_v_valid),
      .data_out_ready(ib_bias_v_ready)
  );
  unpacked_fifo #(
      .DEPTH(IN_DEPTH * IN_NUM_PARALLELISM),
      .DATA_WIDTH(DVIN_WIDTH),
      .IN_NUM(IN_PARALLELISM * IN_SIZE)
  ) fifo_v (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_v),
      .data_in_valid(data_in_v_valid),
      .data_in_ready(data_in_v_ready),
      .data_out(ff_data_in_v),
      .data_out_valid(ff_data_in_v_valid),
      .data_out_ready(ff_data_in_v_ready)
  );
  //matmul_v
  logic [DV_WIDTH - 1 : 0] data_v_t[W_PARALLELISM * IN_PARALLELISM - 1:0];
  logic data_v_t_valid, data_v_t_ready;
  fixed_matmul #(
      .IN1_WIDTH(WV_WIDTH),
      .IN1_FRAC_WIDTH(WV_FRAC_WIDTH),
      .IN2_WIDTH(DVIN_WIDTH),
      .IN2_FRAC_WIDTH(DVIN_FRAC_WIDTH),
      .HAS_BIAS(1),
      .BIAS_WIDTH(BV_WIDTH),
      .BIAS_FRAC_WIDTH(BV_FRAC_WIDTH),
      .OUT_WIDTH(DV_WIDTH),
      .OUT_FRAC_WIDTH(DV_FRAC_WIDTH),
      .IN1_Y(W_PARALLELISM * W_NUM_PARALLELISM),
      .UNROLL_IN1_Y(W_PARALLELISM),
      .IN1_X(IN_SIZE * IN_DEPTH),
      .UNROLL_IN1_X(IN_SIZE),
      .IN2_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN2_Y(IN_PARALLELISM)
  ) inst_fmmc_v (
      .clk(clk),
      .rst(rst),
      .data_in1(weight_v),
      .data_in1_valid(weight_v_valid),
      .data_in1_ready(weight_v_ready),
      .data_in2(ff_data_in_v),
      .data_in2_valid(ff_data_in_v_valid),
      .data_in2_ready(ff_data_in_v_ready),
      .bias(ib_bias_v),
      .bias_valid(ib_bias_v_valid),
      .bias_ready(ib_bias_v_ready),
      .data_out(data_v_t),
      .data_out_valid(data_v_t_valid),
      .data_out_ready(data_v_t_ready)
  );

  logic [DZ_WIDTH - 1:0] data_z[IN_PARALLELISM * W_PARALLELISM - 1:0];
  logic data_z_valid, data_z_ready;
  //z = s*v_t
  always_ff @(posedge clk) $display("%b, %b, data_q", data_q_valid, data_q_ready);
  always_ff @(posedge clk) $display("%b, %b, data_k", data_k_valid, data_k_ready);
  always_ff @(posedge clk) $display("%b, %b, data_s", data_s_valid, data_s_ready);
  always_ff @(posedge clk) $display("%b, %b, data_in_q", data_in_q_valid, data_in_q_ready);
  always_ff @(posedge clk) $display("%b, %b, data_in_k", data_in_k_valid, data_in_k_ready);
  always_ff @(posedge clk) $display("%b, %b, ff_in_k", ff_data_in_k_valid, ff_data_in_k_ready);
  always_ff @(posedge clk) $display("%b, %b, ff_in_q", ff_data_in_q_valid, ff_data_in_q_ready);
  always_ff @(posedge clk) $display("");
  fixed_matmul #(
      .IN1_WIDTH(DS_SOFTMAX_WIDTH),
      .IN1_FRAC_WIDTH(DS_SOFTMAX_FRAC_WIDTH),
      .IN2_WIDTH(DV_WIDTH),
      .IN2_FRAC_WIDTH(DV_FRAC_WIDTH),
      .OUT_WIDTH(DZ_WIDTH),
      .OUT_FRAC_WIDTH(DZ_FRAC_WIDTH),

      .IN1_Y(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN1_Y(IN_PARALLELISM),
      .IN1_X(IN_PARALLELISM * IN_NUM_PARALLELISM),
      .UNROLL_IN1_X(IN_PARALLELISM),

      .IN2_Y(W_PARALLELISM * W_NUM_PARALLELISM),
      .UNROLL_IN2_Y(W_PARALLELISM)
  ) inst_fmmc_z (
      .clk(clk),
      .rst(rst),
      .data_in1(softmax_s),
      .data_in1_valid(softmax_s_valid),
      .data_in1_ready(softmax_s_ready),
      .data_in2(data_v_t),
      .data_in2_valid(data_v_t_valid),
      .data_in2_ready(data_v_t_ready),
      .data_out(data_z),
      .data_out_valid(data_z_valid),
      .data_out_ready(data_z_ready)
  );
  assign data_out = data_z;
  assign data_out_valid = data_z_valid;
  assign data_z_ready = data_out_ready;


endmodule

