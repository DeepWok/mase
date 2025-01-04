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
    parameter DATA_R_WIDTH = 2,

    parameter IN_0_DEPTH = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PRECISION_0 = 4,
    parameter DATA_OUT_0_PRECISION_1 = 8,
    parameter DATA_OUT_0_DIM = DATA_IN_0_DIM,
    parameter DATA_OUT_0_PARALLELISM = DATA_IN_0_PARALLELISM,
    parameter EXP_SUM_UNDERFLOW_BITS = 4,
    parameter DIVISION_UNDERFLOW_BITS = 4
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

  // The current version only support precision of taylor_exp output to be the same with data_out_r
  localparam DATA_EXP_0_PRECISION_0 = DATA_IN_0_PRECISION_0;
  localparam DATA_EXP_0_FRAC_WIDTH = DATA_EXP_0_PRECISION_0 - 2;
  localparam DATA_EXP_0_PRECISION_1 = DATA_IN_0_PRECISION_1;

  localparam ACC_WIDTH = $clog2(IN_0_DEPTH) + DATA_EXP_0_PRECISION_0 + EXP_SUM_UNDERFLOW_BITS;
  localparam ACC_FRAC_WIDTH = DATA_EXP_0_FRAC_WIDTH + EXP_SUM_UNDERFLOW_BITS;

  localparam DATA_DIVIDEND_PRECISION_0 = DATA_EXP_0_PRECISION_0 + EXP_SUM_UNDERFLOW_BITS + DIVISION_UNDERFLOW_BITS;
  localparam DATA_DIVIDEND_PRECISION_1 = DATA_EXP_0_PRECISION_1;
  localparam DATA_DIVISOR_PRECISION_0 = ACC_WIDTH;
  localparam DATA_DIVISOR_PRECISION_1 = DATA_EXP_0_PRECISION_1;
  localparam DATA_QUOTIENT_PRECISION_0 = DATA_DIVIDEND_PRECISION_0;
  localparam DATA_QUOTIENT_FRAC_WIDTH = DIVISION_UNDERFLOW_BITS;
  localparam DATA_QUOTIENT_PRECISION_1 =DATA_EXP_0_PRECISION_1 + 1;


  localparam BLOCK_SIZE = DATA_IN_0_PARALLELISM;
  initial begin
    assert (BLOCK_SIZE == 1)
    else $fatal("Currently only BLOCK_SIZE of 1 is supported.");
  end

  // Add missing signals for mxint_exp interface
  logic [DATA_EXP_0_PRECISION_0-1:0] mdata_exp[BLOCK_SIZE - 1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] edata_exp[BLOCK_SIZE - 1:0];
  logic data_exp_valid, data_exp_ready;

  // Split2 and FF signals for exp path
  logic [DATA_EXP_0_PRECISION_0-1:0] ff_exp_mdata_out[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] ff_exp_edata_out;
  logic ff_exp_data_valid, ff_exp_data_ready;

  // Straight path signals
  logic [DATA_EXP_0_PRECISION_0-1:0] straight_exp_mdata_out[DATA_IN_0_PARALLELISM-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] straight_exp_edata_out;
  logic straight_exp_data_valid, straight_exp_data_ready;

  // Accumulator signals
  logic [ACC_WIDTH-1:0] acc_mdata_out[BLOCK_SIZE-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] acc_edata_out;
  logic acc_data_out_valid, acc_data_out_ready;

  // Circular buffer signals
  logic [ACC_WIDTH-1:0] circ_mdata_out[DATA_OUT_0_PARALLELISM-1:0];
  logic [DATA_EXP_0_PRECISION_1-1:0] circ_edata_out;
  logic circ_data_out_valid, circ_data_out_ready;

  logic [DATA_DIVIDEND_PRECISION_0 - 1:0] mdata_dividend [BLOCK_SIZE - 1:0];
  logic [DATA_DIVIDEND_PRECISION_1 - 1:0] edata_dividend;
  // Division signals
  logic [DATA_QUOTIENT_PRECISION_0 - 1:0] mquotient_data[BLOCK_SIZE - 1:0];
  logic [DATA_QUOTIENT_PRECISION_1 - 1:0] equotient_data;
  logic quotient_data_valid, quotient_data_ready;

  // Updated mxint_exp instantiation with all parameters and proper signal connections
  mxint_exp #(
      .DATA_IN_MAN_WIDTH(DATA_IN_0_PRECISION_0),
      .DATA_IN_EXP_WIDTH(DATA_IN_0_PRECISION_1),
      .BLOCK_SIZE(BLOCK_SIZE),
      .DATA_R_WIDTH(DATA_R_WIDTH),
      .DATA_OUT_MAN_WIDTH(DATA_EXP_0_PRECISION_0),
      .DATA_OUT_EXP_WIDTH(DATA_EXP_0_PRECISION_1)
  ) mxint_exp_inst (
      .rst(rst),
      .clk(clk),
      // Input interface
      .mdata_in_0(mdata_in_0),
      .edata_in_0(edata_in_0),
      .data_in_0_valid(data_in_0_valid),
      .data_in_0_ready(data_in_0_ready),
      // Output interface
      .mdata_out_0(mdata_exp),
      .edata_out_0(edata_exp),
      .data_out_0_valid(data_exp_valid),
      .data_out_0_ready(data_exp_ready)
  );

  unpacked_mx_split2_with_data #(
      .DEPTH(DATA_IN_0_DIM * 2),
      .MAN_WIDTH(DATA_EXP_0_PRECISION_0),
      .EXP_WIDTH(DATA_EXP_0_PRECISION_1),
      .IN_SIZE(DATA_IN_0_PARALLELISM)
  ) split2_mxint_exp_inst (
      .clk(clk),
      .rst(rst),
      // Input from mxint exp
      .mdata_in(mdata_exp),
      .edata_in(edata_exp[0]),
      .data_in_valid(data_exp_valid),
      .data_in_ready(data_exp_ready),
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
      .DATA_IN_0_PRECISION_1(DATA_EXP_0_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM),
      .IN_DEPTH(IN_0_DEPTH),
      .UNDERFLOW_BITS(EXP_SUM_UNDERFLOW_BITS)
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
      .DATA_PRECISION_1(DATA_EXP_0_PRECISION_1),
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

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : dividend
    assign mdata_dividend[i] = ff_exp_mdata_out[i] << EXP_SUM_UNDERFLOW_BITS + DIVISION_UNDERFLOW_BITS;
  end
    assign edata_dividend = ff_exp_edata_out;
  // Add after mxint_circular instance
  mxint_div #(
      .DATA_DIVIDEND_PRECISION_0(DATA_DIVIDEND_PRECISION_0),
      .DATA_DIVIDEND_PRECISION_1(DATA_DIVIDEND_PRECISION_1),
      .DATA_DIVISOR_PRECISION_0(DATA_DIVISOR_PRECISION_0),
      .DATA_DIVISOR_PRECISION_1(DATA_DIVISOR_PRECISION_1),
      .DATA_QUOTIENT_PRECISION_0(DATA_QUOTIENT_PRECISION_0),
      .DATA_QUOTIENT_PRECISION_1(DATA_QUOTIENT_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM)
  ) div_inst (
      .clk(clk),
      .rst(rst),
      // Connect dividend (ff_exp_data)
      .mdividend_data(mdata_dividend),
      .edividend_data(edata_dividend),
      .dividend_data_valid(ff_exp_data_valid),
      .dividend_data_ready(ff_exp_data_ready),
      // Connect divisor (circ_data)
      .mdivisor_data(circ_mdata_out),
      .edivisor_data(circ_edata_out),
      .divisor_data_valid(circ_data_out_valid),
      .divisor_data_ready(circ_data_out_ready),
      // Connect quotient output
      .mquotient_data(mquotient_data),
      .equotient_data(equotient_data),
      .quotient_data_valid(quotient_data_valid),
      .quotient_data_ready(quotient_data_ready)
  );


  // Add mxint_cast instance
  mxint_cast #(
      .IN_MAN_WIDTH(DATA_QUOTIENT_PRECISION_0),
      .IN_MAN_FRAC_WIDTH(DATA_QUOTIENT_FRAC_WIDTH),
      .IN_EXP_WIDTH(DATA_QUOTIENT_PRECISION_1),
      .OUT_MAN_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_EXP_WIDTH(DATA_OUT_0_PRECISION_1),
      .BLOCK_SIZE(DATA_OUT_0_PARALLELISM),
      .ROUND_BITS(4)
  ) cast_inst (
      .clk(clk),
      .rst(rst),
      .mdata_in(mquotient_data),
      .edata_in(equotient_data),
      .data_in_valid(quotient_data_valid),      // Updated
      .data_in_ready(quotient_data_ready),      // Updated
      .mdata_out(mdata_out_0),
      .edata_out(edata_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );

endmodule
