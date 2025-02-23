`timescale 1ns / 1ps
/*
Module      : Mxint cast
Description : MxInt Cast between Layers.
*/
module mxint_cast #(
    parameter IN_MAN_WIDTH = 1,
    parameter IN_MAN_FRAC_WIDTH = IN_MAN_WIDTH - 1,
    parameter IN_EXP_WIDTH = 1,
    parameter OUT_MAN_WIDTH = 1,
    parameter OUT_EXP_WIDTH = 1,
    parameter ROUND_BITS = 4,
    parameter BLOCK_SIZE = 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                     clk,
    input  logic                     rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_MAN_WIDTH-1:0] mdata_in      [BLOCK_SIZE-1:0],
    input  logic [ IN_EXP_WIDTH-1:0] edata_in,
    input  logic                     data_in_valid,
    output logic                     data_in_ready,
    output logic [OUT_MAN_WIDTH-1:0] mdata_out     [BLOCK_SIZE-1:0],
    output logic [OUT_EXP_WIDTH-1:0] edata_out,
    output logic                     data_out_valid,
    input  logic                     data_out_ready
);
  //get max_abs_value of input
  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;

  localparam LOSSLESSS_EDATA_WIDTH = 
    (LOG2_WIDTH > IN_EXP_WIDTH && LOG2_WIDTH > OUT_EXP_WIDTH) ? LOG2_WIDTH + 2 :
    (IN_EXP_WIDTH > OUT_EXP_WIDTH) ? IN_EXP_WIDTH + 2:
    OUT_EXP_WIDTH + 2;

  localparam SHIFT_WIDTH = (OUT_EXP_WIDTH > IN_EXP_WIDTH) ? OUT_EXP_WIDTH + 1 : IN_EXP_WIDTH + 1;
  localparam SHIFT_DATA_WIDTH = OUT_MAN_WIDTH + 1;

  localparam CAST_WIDTH = OUT_MAN_WIDTH + ROUND_BITS;

  logic [IN_MAN_WIDTH - 1:0] mdata_for_max [BLOCK_SIZE - 1:0];
  logic data_for_max_valid, data_for_max_ready;

  logic [IN_MAN_WIDTH-1:0] mdata_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] edata_for_out;
  logic data_for_out_valid, data_for_out_ready;

  logic [CAST_WIDTH-1:0] mdata_for_cast [BLOCK_SIZE-1:0];

  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  logic [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;
  logic [SHIFT_WIDTH - 1:0] shift_value;
  // we dont need to implement full shift here, because we'll clamp in the final.
  // in order to avoid shift loss, we set the shift_data_width = OUT_MAN_WIDTH + 1.

  logic [SHIFT_DATA_WIDTH - 1:0] shift_buffer_data_for_out[BLOCK_SIZE - 1:0];
  logic [SHIFT_DATA_WIDTH - 1:0] shift_data[BLOCK_SIZE - 1:0][SHIFT_DATA_WIDTH - 1:0];

    // Add intermediate signals
    logic [OUT_MAN_WIDTH-1:0] mdata_out_unreg [BLOCK_SIZE-1:0];
    logic [OUT_EXP_WIDTH-1:0] edata_out_unreg;
    logic data_out_unreg_valid;
    logic data_out_unreg_ready;
  unpacked_mx_split2_with_data #(
      .DEPTH($clog2(BLOCK_SIZE) + 1),
      .MAN_WIDTH(IN_MAN_WIDTH),
      .EXP_WIDTH(IN_EXP_WIDTH),
      .IN_SIZE(BLOCK_SIZE)
  ) data_in_0_unpacked_mx_split2_with_data_i (
      .clk(clk),
      .rst(rst),
      .mdata_in(mdata_in),
      .edata_in(edata_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .fifo_mdata_out(mdata_for_out),
      .fifo_edata_out(edata_for_out),
      .fifo_data_out_valid(data_for_out_valid),
      .fifo_data_out_ready(data_for_out_ready),
      .straight_mdata_out(mdata_for_max),
      .straight_edata_out(),
      .straight_data_out_valid(data_for_max_valid),
      .straight_data_out_ready(data_for_max_ready)
  );

  log2_max_abs #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(IN_MAN_WIDTH)
  ) max_bas_i (
      .clk,
      .rst,
      .data_in_0(mdata_for_max),
      .data_in_0_valid(data_for_max_valid),
      .data_in_0_ready(data_for_max_ready),
      .data_out_0(log2_max_value_unreg),
      .data_out_0_valid(log2_max_value_valid_unreg),
      .data_out_0_ready(log2_max_value_ready_unreg)
  );

  // Add register slice after log2_max_abs
  logic [LOG2_WIDTH-1:0] log2_max_value_unreg;
  logic log2_max_value_valid_unreg, log2_max_value_ready_unreg;

  skid_buffer #(
      .DATA_WIDTH(LOG2_WIDTH)
  ) log2_reg_slice (
      .clk(clk),
      .rst(rst),
      .data_in(log2_max_value_unreg),
      .data_in_valid(log2_max_value_valid_unreg),
      .data_in_ready(log2_max_value_ready_unreg),
      .data_out(log2_max_value),
      .data_out_valid(log2_max_value_valid),
      .data_out_ready(log2_max_value_ready)
  );

  join2 #() join_inst (
      .data_in_ready ({data_for_out_ready, log2_max_value_ready}),
      .data_in_valid ({data_for_out_valid, log2_max_value_valid}),
      .data_out_valid(data_out_unreg_valid),
      .data_out_ready(data_out_unreg_ready)
  );

  assign edata_out_full = $signed(
      log2_max_value
  ) + $signed(
      edata_for_out
  ) - IN_MAN_FRAC_WIDTH;

  // clamp 
  signed_clamp #(
      .IN_WIDTH (LOSSLESSS_EDATA_WIDTH),
      .OUT_WIDTH(OUT_EXP_WIDTH)
  ) exp_clamp (
      .in_data (edata_out_full),
      .out_data(edata_out_unreg)
  );

  optimized_right_shift #(
      .IN_WIDTH(IN_MAN_WIDTH),
      .SHIFT_WIDTH(SHIFT_WIDTH),
      .OUT_WIDTH(CAST_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) ovshift_inst (
      .data_in(mdata_for_out),
      .shift_value(shift_value),
      .data_out(mdata_for_cast)
  );

  assign shift_value = $signed(
      edata_out_unreg
  ) - $signed(
      edata_for_out
  ) + IN_MAN_FRAC_WIDTH - (CAST_WIDTH - 1);
    fixed_rounding #(
        .IN_SIZE(BLOCK_SIZE),
        .IN_WIDTH(CAST_WIDTH),
        .IN_FRAC_WIDTH(CAST_WIDTH - 1),
        .OUT_WIDTH(OUT_MAN_WIDTH),
        .OUT_FRAC_WIDTH(OUT_MAN_WIDTH - 1)
    ) fixed_cast_inst (
        .data_in(mdata_for_cast),
        .data_out(mdata_out_unreg)  // Changed to feed into skid buffer
    );

    // Add skid buffer at the end
    mxint_skid_buffer #(
        .DATA_PRECISION_0(OUT_MAN_WIDTH),
        .DATA_PRECISION_1(OUT_EXP_WIDTH),
        .IN_NUM(BLOCK_SIZE)
    ) output_skid_buffer (
        .clk(clk),
        .rst(rst),
        .mdata_in(mdata_out_unreg),
        .edata_in(edata_out_unreg),
        .data_in_valid(data_out_unreg_valid),
        .data_in_ready(data_out_unreg_ready),
        .mdata_out(mdata_out),
        .edata_out(edata_out),
        .data_out_valid(data_out_valid),
        .data_out_ready(data_out_ready)
    );

endmodule