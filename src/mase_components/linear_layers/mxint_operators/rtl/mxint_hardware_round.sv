`timescale 1ns / 1ps
/*
 This code actually input mxint and then output rounded integer n,
 In the first version, we just keep the width of n is 8
 which means like output n range from [-128:127]
*/
module mxint_hardware_round #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_MAN_WIDTH = 4,
    parameter DATA_IN_MAN_FRAC_WIDTH = 4,
    parameter DATA_IN_EXP_WIDTH = 8,
    parameter BLOCK_SIZE = 16,
    parameter DATA_OUT_WIDTH = 8
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_MAN_WIDTH-1:0] mdata_in_0[BLOCK_SIZE - 1:0],
    input logic [DATA_IN_EXP_WIDTH-1:0] edata_in_0,
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    output logic [DATA_OUT_WIDTH-1:0] data_out_0[BLOCK_SIZE - 1 : 0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam SHIFT_WIDTH = DATA_IN_EXP_WIDTH + 1;

  logic [SHIFT_WIDTH - 1:0] shift_value;
  logic [DATA_IN_MAN_WIDTH - 1:0] mid_n[BLOCK_SIZE - 1:0];
  logic [DATA_IN_MAN_WIDTH-1:0] shift_result[BLOCK_SIZE-1:0];

  assign shift_value = DATA_IN_MAN_FRAC_WIDTH - $signed(edata_in_0);
  optimized_right_shift #(
      .IN_WIDTH(DATA_IN_MAN_WIDTH),
      .SHIFT_WIDTH(SHIFT_WIDTH),
      .OUT_WIDTH(DATA_IN_MAN_WIDTH),
      .BLOCK_SIZE(BLOCK_SIZE)
  ) ovshift_inst (
      .data_in(mdata_in_0),
      .shift_value(shift_value),
      .data_out(shift_result)
  );

  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    always_comb begin
      if ($signed(shift_value) >= DATA_IN_MAN_FRAC_WIDTH) begin
        mid_n[i] = (mdata_in_0[i][DATA_IN_MAN_WIDTH-1]) ? -1 : 0;
      end else begin
        mid_n[i] = shift_result[i];
      end
    end
  end

  logic [DATA_OUT_WIDTH - 1:0] clamped_n[BLOCK_SIZE - 1:0];
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    signed_clamp #(
        .IN_WIDTH (DATA_IN_MAN_WIDTH),
        .OUT_WIDTH(DATA_OUT_WIDTH)
    ) n_clamp (
        .in_data (mid_n[i]),
        .out_data(clamped_n[i])
    );
  end
  unpacked_register_slice #(
      .DATA_WIDTH(DATA_OUT_WIDTH),
      .IN_SIZE   (BLOCK_SIZE)
  ) register_slice_i (
      .clk(clk),
      .rst(rst),

      .data_in(clamped_n),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),

      .data_out(data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );
endmodule

