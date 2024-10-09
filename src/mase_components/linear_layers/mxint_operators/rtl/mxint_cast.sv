`timescale 1ns / 1ps
/*
Module      : Mxint cast
Description : MxInt Cast between Layers.
*/
module mxint_cast #(
    parameter IN_MAN_WIDTH = 1,
    parameter IN_EXP_WIDTH = 1,
    parameter OUT_MAN_WIDTH = 1,
    parameter OUT_EXP_WIDTH = 1,
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
  logic data_for_max_valid, data_for_max_ready, data_for_out_valid, data_for_out_ready;
  split2 #() split_i (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({data_for_max_valid, data_for_out_valid}),
      .data_out_ready({data_for_max_ready, data_for_out_ready})
  );
  logic [IN_MAN_WIDTH-1:0] mbuffer_data_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] ebuffer_data_for_out;
  logic buffer_data_for_out_valid, buffer_data_for_out_ready;

  localparam LOG2_WIDTH = $clog2(IN_MAN_WIDTH) + 1;
  logic [LOG2_WIDTH - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  localparam EBIAS = 2 ** (OUT_EXP_WIDTH - 1);
  localparam LOSSLESSS_EDATA_WIDTH = max(LOG2_WIDTH, IN_EXP_WIDTH, OUT_EXP_WIDTH) + 2;
  localparam FIFO_DEPTH = $clog2(BLOCK_SIZE);
  logic [LOSSLESSS_EDATA_WIDTH - 1:0] edata_out_full;
  log2_max_abs #(
      .IN_SIZE (BLOCK_SIZE),
      .IN_WIDTH(IN_MAN_WIDTH),
  ) max_bas_i (
      .clk,
      .rst,
      .data_in(mdata_in),
      .data_in_valid(data_for_max_valid),
      .data_in_ready(data_for_max_ready),
      .data_out(log2_max_value),
      .data_out_valid(log2_max_value_valid),
      .data_out_ready(log2_max_value_ready)
  );

  if (FIFO_DEPTH == 0) begin
    always_comb begin
      mbuffer_data_for_out = mdata_in;
      ebuffer_data_for_out = edata_in;
      buffer_data_for_out_valid = data_for_out_valid;
      data_for_out_ready = buffer_data_for_out_ready;
    end
  end else begin
    unpacked_mx_fifo #(
        .DEPTH(FIFO_DEPTH),
        .MAN_WIDTH(IN_MAN_WIDTH),
        .EXP_WIDTH(IN_EXP_WIDTH),
        .IN_SIZE(BLOCK_SIZE)
    ) ff_inst (
        .clk(clk),
        .rst(rst),
        .mdata_in(mdata_in),
        .edata_in(edata_in),
        .data_in_valid(data_for_out_valid),
        .data_in_ready(data_for_out_ready),
        .mdata_out(mbuffer_data_for_out),
        .edata_out(ebuffer_data_for_out),
        .data_out_valid(buffer_data_for_out_valid),
        .data_out_ready(buffer_data_for_out_ready)
    );
  end
  join2 #() join_inst (
      .data_in_ready ({buffer_data_for_out_ready, log2_max_value_ready}),
      .data_in_valid ({buffer_data_for_out_valid, log2_max_value_valid}),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
  assign edata_out_full = $signed(log2_max_value) + $signed(ebuffer_data_for_out) - EBIAS;
  // clamp 
  signed_clamp #(
      .IN_WIDTH (LOSSLESSS_EDATA_WIDTH),
      .OUT_WIDTH(OUT_EXP_WIDTH)
  ) exp_clamp (
      .in_data (edata_out_full),
      .out_data(edata_out)
  );
  localparam SHIFT_WIDTH = max(OUT_EXP_WIDTH, IN_EXP_WIDTH, 0) + 1;
  logic [SHIFT_WIDTH - 1:0] shift_value;
  assign shift_value = $signed(edata_out) - $signed(ebuffer_data_for_out);
  logic [SHIFT_WIDTH - 1:0] abs_shift_value;
  assign abs_shift_value = (shift_value[SHIFT_WIDTH-1]) ? (~shift_value + 1) : shift_value;

  logic [IN_MAN_WIDTH + EBIAS - 1:0] shift_buffer_data_for_out[BLOCK_SIZE - 1:0];
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    for (genvar j = 0; j < 2 ** SHIFT_WIDTH; j++)
    always_comb
      if (abs_shift_value == j)
        shift_buffer_data_for_out[i] = (shift_value[SHIFT_WIDTH-1]) ? $signed(
            mbuffer_data_for_out[i]
        ) <<< j : $signed(
            mbuffer_data_for_out[i]
        ) >>> j;
    signed_clamp #(
        .IN_WIDTH (IN_MAN_WIDTH + EBIAS),
        .OUT_WIDTH(OUT_MAN_WIDTH)
    ) exp_clamp (
        .in_data (shift_buffer_data_for_out[i]),
        .out_data(mdata_out[i])
    );
  end
endmodule
function [31:0] max;
  input [31:0] x, y, z;
  begin
    if (x > y && x > z) max = x;
    else if (y > z) max = y;
    else max = z;
  end
endfunction
