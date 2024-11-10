`timescale 1ns / 1ps
/*
  Currently, we dont' want to support parallelism
  Cause in attention, it's actually not in parallel
*/
module mxint_softmax #(
    /* verilator lint_off UNUSEDPARAM */

    parameter DATA_IN_0_PRECISION_0 = 4,
    parameter DATA_IN_0_PRECISION_1 = 8,
    parameter DATA_IN_0_DIM = 8,  // input vector size
    parameter DATA_IN_0_PARALLELISM = 1,  // batch size

    parameter IN_0_DEPTH = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PRECISION_0 = 4,
    parameter DATA_OUT_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_DIM = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PARALLELISM = DATA_IN_0_PARALLELISM,

    parameter DATA_N_PRECISION_0 = DATA_OUT_0_PRECISION_1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] mdata_in_0[DATA_IN_0_PARALLELISM-1:0],
    input logic [DATA_IN_0_PRECISION_1-1:0] edata_in_0,
    output logic [DATA_OUT_0_PRECISION_0-1:0] mdata_out_0[DATA_OUT_0_PARALLELISM-1:0],
    output logic [DATA_OUT_0_PRECISION_1-1:0] edata_out_0,

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  // softmax over a vector
  // each vector might be split into block of elements
  // Can handle multiple batches at once
  // each iteration recieves a batch of blocks

  // Constants
  localparam DATA_R_PRECISION_0 = 9;
  localparam DATA_R_PRECISION_1 = 7;
  // The current version only support precision of taylor_exp output to be the same with data_out_r
  localparam DATA_EXP_0_PRECISION_0 = DATA_R_PRECISION_0;
  localparam DATA_EXP_0_PRECISION_1 = DATA_R_PRECISION_1;
  // The target is to make the mantissa of output of softmax to be ?.DATA_OUT_0_PRECISION_0 - 1
  localparam EXTENDED_DIVIDEND_WIDTH = DATA_EXP_0_PRECISION_0 + DATA_OUT_0_PRECISION_0 - 1;

  localparam ACC_WIDTH = $clog2(IN_0_DEPTH) + DATA_EXP_0_PRECISION_0;

  localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM; 
  initial begin
    assert (BLOCK_SIZE == 1) else $fatal("Currently only BLOCK_SIZE of 1 is supported.");
  end

  // Range reduction and exp signals
  // data_out_r - fixed point representation
  // data_out_n - integer representation
  logic [DATA_R_PRECISION_0-1:0] data_out_r [BLOCK_SIZE - 1:0];
  logic data_out_r_valid, data_out_r_ready;
  logic [DATA_N_PRECISION_0-1:0] data_out_n[BLOCK_SIZE - 1:0];
  logic data_out_n_valid, data_out_n_ready;

  logic [DATA_EXP_0_PRECISION_0-1:0] taylor_exp;
  logic taylor_exp_valid, taylor_exp_ready;

  // Mxint exp signals
  logic [DATA_EXP_0_PRECISION_0-1:0] mxint_mexp[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_N_PRECISION_0-1:0] mxint_eexp;
  logic mxint_exp_valid, mxint_exp_ready;

  // Split2 and FF signals for exp path
  logic [DATA_EXP_0_PRECISION_0-1:0] ff_exp_mdata_out[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_N_PRECISION_0-1:0] ff_exp_edata_out;
  logic ff_exp_data_valid, ff_exp_data_ready;
  
  // Straight path signals
  logic [DATA_EXP_0_PRECISION_0-1:0] straight_exp_mdata_out[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_N_PRECISION_0-1:0] straight_exp_edata_out;
  logic straight_exp_data_valid, straight_exp_data_ready;

  // Accumulator signals
  logic [ACC_WIDTH-1:0] acc_mdata_out[BLOCK_SIZE-1:0];
  logic [DATA_N_PRECISION_0-1:0] acc_edata_out;
  logic acc_data_out_valid, acc_data_out_ready;

  // Circular buffer signals
  logic [ACC_WIDTH-1:0] circ_mdata_out[DATA_OUT_0_PARALLELISM-1:0];
  logic [DATA_N_PRECISION_0-1:0] circ_edata_out;
  logic circ_data_out_valid, circ_data_out_ready;

  // split valid and ready signals
  logic ff_exp_mdata_valid, ff_exp_mdata_ready;
  logic ff_exp_edata_valid, ff_exp_edata_ready;
  logic [ACC_WIDTH-1:0] circ_mdata_out_ff[DATA_OUT_0_PARALLELISM-1:0];
  logic [DATA_N_PRECISION_0-1:0] circ_edata_out_straight;
  logic circ_mdata_valid, circ_mdata_ready;
  logic circ_edata_valid, circ_edata_ready;

  // Division signals
  logic [EXTENDED_DIVIDEND_WIDTH - 1:0] extended_dividend[0:0];
  logic [EXTENDED_DIVIDEND_WIDTH - 1:0] mquotient_data[0:0];
  logic mquotient_data_valid, mquotient_data_ready;

  // Exponent quotient signals
  logic [DATA_OUT_0_PRECISION_1-1:0] equotient_data;
  logic equotient_data_valid, equotient_data_ready;
  logic [DATA_OUT_0_PRECISION_1-1:0] ff_equotient_data;
  logic ff_equotient_valid, ff_equotient_ready;

  // Final join and cast signals
  logic quotient_joined_valid, quotient_joined_ready;

  // Add signals for skid buffer
  logic [DATA_N_PRECISION_0-1:0] ff_exp_edata_skid;
  logic ff_exp_edata_skid_valid, ff_exp_edata_skid_ready;

  // Add signals for mantissa skid buffer
  logic [DATA_EXP_0_PRECISION_0-1:0] ff_exp_mdata_skid[DATA_IN_0_PARALLELISM-1:0];
  logic ff_exp_mdata_skid_valid, ff_exp_mdata_skid_ready;

  // generate r and n
  mxint_range_reduction #(
      .DATA_IN_MAN_WIDTH(DATA_IN_0_PRECISION_0),
      .DATA_IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .BLOCK_SIZE(DATA_IN_0_PARALLELISM),
      .DATA_OUT_N_WIDTH(DATA_N_PRECISION_0)
  ) mxint_range_reduction_inst (
      .rst(rst),
      .clk(clk),
      .mdata_in_0(mdata_in_0),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      .data_out_n(data_out_n),
      .data_out_n_valid(data_out_n_valid),
      .data_out_n_ready(data_out_n_ready),
      .data_out_r(data_out_r),
      .data_out_r_valid(data_out_r_valid),
      .data_out_r_ready(data_out_r_ready)
  );

  // currently we only support BLOCK_SIZE = 1
  fixed_taylor_exp #(
      .DATA_IN_WIDTH(DATA_R_PRECISION_0),
      .DATA_IN_FRAC_WIDTH(DATA_R_PRECISION_1),
      .DATA_OUT_WIDTH(DATA_EXP_0_PRECISION_0),
      .DATA_OUT_FRAC_WIDTH(DATA_EXP_0_PRECISION_1)
  ) fixed_taylor_exp_inst (
      .rst(rst),
      .clk(clk),
      .data_in_0(data_out_r[0]),
      .data_in_0_valid(data_out_r_valid),
      .data_in_0_ready(data_out_r_ready),
      .data_out_0(taylor_exp),
      .data_out_0_valid(taylor_exp_valid),
      .data_out_0_ready(taylor_exp_ready)
  );

  // After fixed_taylor_exp instance, add join2
  join2 #() join_exp_n (
      .data_in_valid({taylor_exp_valid, data_out_n_valid}),
      .data_in_ready({taylor_exp_ready, data_out_n_ready}),
      .data_out_valid(mxint_exp_valid),
      .data_out_ready(mxint_exp_ready)
  );
  assign mxint_mexp[0] = taylor_exp[0];
  assign mxint_eexp = data_out_n[0];

  unpacked_mx_split2_with_data #(
      .DEPTH(DATA_IN_0_DIM*2),
      .MAN_WIDTH(DATA_EXP_0_PRECISION_0),
      .EXP_WIDTH(DATA_N_PRECISION_0),
      .IN_SIZE(DATA_IN_0_PARALLELISM)
  ) split2_mxint_exp_inst (
      .clk(clk),
      .rst(rst),
      // Input from mxint exp
      .mdata_in(mxint_mexp),
      .edata_in(mxint_eexp),
      .data_in_valid(mxint_exp_valid),
      .data_in_ready(mxint_exp_ready),
      // FIFO output path 
      .fifo_mdata_out(ff_exp_mdata_out),
      .fifo_edata_out(ff_exp_edata_out),  // Not used
      .fifo_data_out_valid(ff_exp_data_valid),
      .fifo_data_out_ready(ff_exp_data_ready),
      // Straight output path
      .straight_mdata_out(straight_exp_mdata_out),
      .straight_edata_out(straight_exp_edata_out),
      .straight_data_out_valid(straight_exp_data_out_valid),
      .straight_data_out_ready(straight_exp_data_out_ready)
  );

  mxint_accumulator #(
      .DATA_IN_0_PRECISION_0(DATA_EXP_0_PRECISION_0),
      .DATA_IN_0_PRECISION_1(DATA_N_PRECISION_0),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM),
      .IN_DEPTH(IN_0_DEPTH)
  ) mxint_accumulator_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in_0(straight_exp_mdata_out),     // From split2 straight output
      .edata_in_0(straight_exp_edata_out),     // From split2 straight output
      .data_in_0_valid(straight_exp_data_out_valid),
      .data_in_0_ready(straight_exp_data_out_ready),
      .mdata_out_0(acc_mdata_out),
      .edata_out_0(acc_edata_out),
      .data_out_0_valid(acc_data_out_valid),
      .data_out_0_ready(acc_data_out_ready)
  );

  // Replace existing signals
  // Replace input_buffer with mxint_circular
  mxint_circular #(
      .DATA_PRECISION_0(ACC_WIDTH),
      .DATA_PRECISION_1(DATA_N_PRECISION_0),
      .IN_NUM(DATA_OUT_0_PARALLELISM),
      .REPEAT(IN_0_DEPTH),
      .BUFFER_SIZE(1)
  ) acc_circular (
      .clk(clk),
      .rst(rst),
      .mdata_in(acc_mdata_out),
      .edata_in(acc_edata_out),
      .data_in_valid(acc_data_out_valid),
      .data_in_ready(acc_data_out_ready),
      .mdata_out(circ_mdata_out),
      .edata_out(circ_edata_out),
      .data_out_valid(circ_data_out_valid),
      .data_out_ready(circ_data_out_ready)
  );

  // Replace split2 for circular buffer outputs with unpacked_mx_split2_with_data
  unpacked_mx_split2_with_data #(
      .DEPTH(DATA_IN_0_DIM),
      .MAN_WIDTH(ACC_WIDTH),
      .EXP_WIDTH(DATA_N_PRECISION_0),
      .IN_SIZE(DATA_OUT_0_PARALLELISM)
  ) split2_circ (
      .clk(clk),
      .rst(rst),
      // Input from circular buffer
      .mdata_in(circ_mdata_out),
      .edata_in(circ_edata_out),
      .data_in_valid(circ_data_out_valid),
      .data_in_ready(circ_data_out_ready),
      // FIFO output path (not used)
      .fifo_mdata_out(circ_mdata_out_ff),  
      .fifo_edata_out(),
      .fifo_data_out_valid(circ_mdata_valid),
      .fifo_data_out_ready(circ_mdata_ready),
      // Straight output path
      .straight_mdata_out(),  // Connect to the same signals previously used
      .straight_edata_out(circ_edata_out_straight),
      .straight_data_out_valid(circ_edata_valid),
      .straight_data_out_ready(circ_edata_ready)
  );

  // Add split2 for ff_exp outputs
  split2 #(
  ) split2_ff_exp (
      .data_in_valid(ff_exp_data_valid),
      .data_in_ready(ff_exp_data_ready),
      .data_out_valid({ff_exp_mdata_valid, ff_exp_edata_valid}),
      .data_out_ready({ff_exp_mdata_ready, ff_exp_edata_ready})
  );

  // Add skid buffer for breaking the combinational logic connection between split2 and join2 exponent path
  unpacked_skid_buffer #(
      .DATA_WIDTH(DATA_N_PRECISION_0),
      .IN_NUM(1)
  ) ff_exp_edata_skid_buffer (
      .clk(clk),
      .rst(rst),
      .data_in({ff_exp_edata_out}),
      .data_in_valid(ff_exp_edata_valid),
      .data_in_ready(ff_exp_edata_ready),
      .data_out({ff_exp_edata_skid}),
      .data_out_valid(ff_exp_edata_skid_valid),
      .data_out_ready(ff_exp_edata_skid_ready)
  );

  // Add skid buffer for breaking the combinational logic connection between split2 and int_div mantissa path
  unpacked_skid_buffer #(
      .DATA_WIDTH(DATA_EXP_0_PRECISION_0),
      .IN_NUM(DATA_IN_0_PARALLELISM)
  ) ff_exp_mdata_skid_buffer (
      .clk(clk),
      .rst(rst),
      .data_in(ff_exp_mdata_out),
      .data_in_valid(ff_exp_mdata_valid),
      .data_in_ready(ff_exp_mdata_ready),
      .data_out(ff_exp_mdata_skid),
      .data_out_valid(ff_exp_mdata_skid_valid),
      .data_out_ready(ff_exp_mdata_skid_ready)
  );

  // Update the downstream logic to use circ outputs instead of ff_accumulated signals
  always_comb begin
    extended_dividend[0] = ff_exp_mdata_skid[0] <<< (DATA_OUT_0_PRECISION_0 - 1);
  end

  int_div #(
      .IN_NUM(DATA_OUT_0_PARALLELISM),
      .DIVIDEND_WIDTH(EXTENDED_DIVIDEND_WIDTH),
      .DIVISOR_WIDTH(ACC_WIDTH),
      .QUOTIENT_WIDTH(EXTENDED_DIVIDEND_WIDTH)
  ) div_inst (
      .clk(clk),
      .rst(rst),
      .dividend_data(extended_dividend),
      .dividend_data_valid(ff_exp_mdata_skid_valid),  // Updated to use skid buffer
      .dividend_data_ready(ff_exp_mdata_skid_ready),  // Updated to use skid buffer
      .divisor_data(circ_mdata_out_ff),
      .divisor_data_valid(circ_mdata_valid),
      .divisor_data_ready(circ_mdata_ready),
      .quotient_data(mquotient_data),
      .quotient_data_valid(mquotient_data_valid),
      .quotient_data_ready(mquotient_data_ready)
  );
  // Update equotient_data assignment to use skid buffer output
  assign equotient_data = $signed(ff_exp_edata_skid) - $signed(circ_edata_out_straight);
  // Update join2 connection to use skid buffer output
  join2 #() join_equotient_2 (
      .data_in_valid({ff_exp_edata_skid_valid, circ_edata_valid}),
      .data_in_ready({ff_exp_edata_skid_ready, circ_edata_ready}),
      .data_out_valid(equotient_data_valid),
      .data_out_ready(equotient_data_ready)
  );

  // Add FIFO for equotient_data
  fifo #(
      .DEPTH(DATA_OUT_0_DIM / DATA_OUT_0_PARALLELISM),
      .DATA_WIDTH(DATA_OUT_0_PRECISION_1)
  ) ff_inst (
      .clk(clk),
      .rst(rst),
      .in_data(equotient_data),
      .in_valid(equotient_data_valid),
      .in_ready(equotient_data_ready),
      .out_data(ff_equotient_data),
      .out_valid(ff_equotient_valid),
      .out_ready(ff_equotient_ready),
      .empty(),
      .full()
  );

  // Add join2 to combine quotient data paths

  join2 #(
  ) join_quotient (
      .data_in_valid({ff_equotient_valid, mquotient_data_valid}),
      .data_in_ready({ff_equotient_ready, mquotient_data_ready}),
      .data_out_valid(quotient_joined_valid),
      .data_out_ready(quotient_joined_ready)
  );

  // Add mxint_cast instance
  mxint_cast #(
      .IN_MAN_WIDTH(EXTENDED_DIVIDEND_WIDTH),
      .IN_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM)
  ) cast_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(mquotient_data),
      .edata_in(ff_equotient_data),
      .data_in_valid(quotient_joined_valid),
      .data_in_ready(quotient_joined_ready),
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
