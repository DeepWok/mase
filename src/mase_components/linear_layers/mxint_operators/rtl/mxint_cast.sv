`timescale 1ns / 1ps
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
  split_n #(
      .N(2)
  ) split_i (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({data_for_max_valid, data_for_out_valid}),
      .data_out_ready({data_for_max_ready, data_for_out_ready})
  );
  logic [IN_MAN_WIDTH-1:0] mbuffer_data_for_out [BLOCK_SIZE-1:0];
  logic [IN_EXP_WIDTH-1:0] ebuffer_data_for_out;
  logic buffer_data_for_out_valid, buffer_data_for_out_ready;

  logic [$clog2(IN_MAN_WIDTH) - 1:0] log2_max_value;
  logic log2_max_value_valid, log2_max_value_ready;

  localparam EBIAS = 2 ** (OUT_EXP_WIDTH - 1);
  localparam LOSSLESSS_EDATA_WIDTH = max($clog2(IN_MAN_WIDTH), IN_EXP_WIDTH, OUT_EXP_WIDTH) + 2;
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

  unpacked_mx_fifo #(
      .DEPTH($clog2(BLOCK_SIZE)),
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

  logic [IN_MAN_WIDTH + EBIAS - 1:0] shift_buffer_data_for_out[BLOCK_SIZE - 1:0];
  for (genvar i = 0; i < BLOCK_SIZE; i++) begin
    assign shift_buffer_data_for_out[i] = (shift_value[SHIFT_WIDTH - 1])? 
    mbuffer_data_for_out[i] <<< ~shift_value + 1 :
    mbuffer_data_for_out[i] >>> shift_value;
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
