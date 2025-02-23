`timescale 1ns / 1ps
/*
 This code actually input mxint and then output rounded integer n,
 In the first version, we just keep the width of n is 8
 which means like output n range from [-128:127]
*/
module mxint_exp #(
    parameter DATA_IN_MAN_WIDTH = 8,
    parameter DATA_IN_EXP_WIDTH = 3,
    parameter BLOCK_SIZE        = 16,
    parameter DATA_OUT_MAN_WIDTH= 10,
    parameter DATA_OUT_EXP_WIDTH= 4,
    parameter DATA_R_WIDTH      = 7
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_MAN_WIDTH-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_EXP_WIDTH-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_MAN_WIDTH-1:0] mdata_out_0[BLOCK_SIZE - 1 : 0],
    output logic [DATA_OUT_EXP_WIDTH-1:0] edata_out_0[BLOCK_SIZE - 1 : 0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam signed MLOG2_E = 8'd92;
  localparam signed ELOG2_E = 4'd1;

  localparam LOG2_E_MAN_WIDTH = 8;
  localparam LOG2_E_EXP_WIDTH = 4;
  localparam DATA_LOG2_E_MAN_WIDTH = DATA_IN_MAN_WIDTH + LOG2_E_MAN_WIDTH;
  localparam DATA_LOG2_E_MAN_FRAC_WIDTH = DATA_IN_MAN_WIDTH - 1 + LOG2_E_MAN_WIDTH - 1;
  localparam DATA_LOG2_E_EXP_WIDTH = DATA_IN_EXP_WIDTH;

  localparam SHIFT_WIDTH = $clog2(DATA_LOG2_E_MAN_WIDTH) + 2;
  localparam DATA_N_WIDTH = DATA_OUT_EXP_WIDTH;

  localparam CASTED_DATA_LOG2_E_WIDTH = DATA_N_WIDTH + DATA_R_WIDTH - 1;
  localparam CASTED_DATA_LOG2_E_FRAC_WIDTH = DATA_R_WIDTH - 1;

  logic [DATA_LOG2_E_MAN_WIDTH - 1:0] mdata_in_0_log2_e[BLOCK_SIZE - 1:0];
  logic [DATA_LOG2_E_EXP_WIDTH - 1:0] edata_in_0_log2_e;

  logic signed [SHIFT_WIDTH - 1:0] shift_value;
  logic [CASTED_DATA_LOG2_E_WIDTH - 1:0] casted_data_in_0_log2_e[BLOCK_SIZE - 1:0];

  logic [DATA_N_WIDTH - 1:0] temp_data_out_n[BLOCK_SIZE - 1 : 0];
  logic [DATA_R_WIDTH - 1:0] temp_data_out_r[BLOCK_SIZE - 1 : 0];

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    assign mdata_in_0_log2_e[i] = $signed(mdata_in_0[i]) * MLOG2_E;
  end
  assign edata_in_0_log2_e = $signed(edata_in_0) + ELOG2_E;

  // So basically, The input frac_width is DATA_LOG2_E_MAN_FRAC_WIDTH
  // We wish to make the output frac_width = CASTED_DATA_LOG2_E_FRAC_WIDTH
  // real_data = man * 2** exp this is left shift here
  assign shift_value = DATA_LOG2_E_MAN_FRAC_WIDTH - CASTED_DATA_LOG2_E_FRAC_WIDTH - $signed(edata_in_0_log2_e);

  optimized_right_shift #(
      .IN_WIDTH(DATA_LOG2_E_MAN_WIDTH),
      .SHIFT_WIDTH(SHIFT_WIDTH),
      .OUT_WIDTH(CASTED_DATA_LOG2_E_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) ovshift_inst (
      .data_in(mdata_in_0_log2_e),
      .shift_value(shift_value),
      .data_out(casted_data_in_0_log2_e)
  );

  // Then we need to extract the casted_data_in_0_log2_e to get the n and r


  logic [DATA_OUT_MAN_WIDTH - 1:0] mexp [BLOCK_SIZE - 1:0];
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin : power2_lut_inst
    assign temp_data_out_n[i] = casted_data_in_0_log2_e[i][CASTED_DATA_LOG2_E_WIDTH - 1: DATA_R_WIDTH - 1];
    assign temp_data_out_r[i] = {
      casted_data_in_0_log2_e[i][DATA_N_WIDTH + DATA_R_WIDTH - 1],
      casted_data_in_0_log2_e[i][DATA_R_WIDTH - 2: 0]};
    power2_lut #(
        .DATA_IN_0_PRECISION_0(DATA_R_WIDTH),
        .DATA_IN_0_PRECISION_1(DATA_R_WIDTH - 1),
        .DATA_OUT_0_PRECISION_0(DATA_OUT_MAN_WIDTH),
        .DATA_OUT_0_PRECISION_1(DATA_OUT_MAN_WIDTH - 2)
    ) power2_lut_inst (
        .data_in_0(temp_data_out_r[i]),
        .data_out_0(mexp[i])
    );
    assign mdata_out_0[i] = mexp[i];
    assign edata_out_0[i] = temp_data_out_n[i];
  end
  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready = data_out_0_ready;

endmodule

