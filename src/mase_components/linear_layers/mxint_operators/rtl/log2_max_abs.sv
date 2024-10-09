`timescale 1ns / 1ps
/*
Module      : log2_max_abs
Description : For any given input, this module will calculate ceil(log2(abs(input + 1e-9))).
              The 1e-9 is for hardware convenience, for example, if input = 4, this module will output ceil(log2(abs(4 + 1e-9)) = 3
*/
module log2_max_abs #(
    parameter IN_SIZE   = 2,
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_WIDTH) + 1
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic                 clk,
    input  logic                 rst,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [ IN_WIDTH-1:0] data_in       [IN_SIZE-1:0],
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    output logic [OUT_WIDTH-1:0] data_out,
    output logic                 data_out_valid,
    input  logic                 data_out_ready
);
  logic [IN_WIDTH - 1:0] or_result;
  logic [IN_WIDTH - 1:0] abs_data_in[IN_SIZE - 1:0];
  for (genvar i = 0; i < IN_SIZE; i++) begin
    abs #(
        .IN_WIDTH(IN_WIDTH)
    ) abs_i (
        .data_in (data_in[i]),
        .data_out(abs_data_in[i])
    );
  end
  or_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(IN_WIDTH),
  ) max_bas_i (
      .clk,
      .rst,
      .data_in(abs_data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .data_out(or_result),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
  log2_value #(
      .IN_WIDTH(IN_WIDTH),
  ) log2_i (
      .data_in (or_result),
      .data_out(data_out)
  );

endmodule

module log2_value #(
    /* verilator lint_off UNUSEDPARAM */
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = $clog2(IN_WIDTH) + 1
) (
    input logic [IN_WIDTH - 1:0] data_in,  // 32-bit input number
    output logic [OUT_WIDTH-1:0] data_out  // 5-bit output log2 result, since log2(32-bit) is max 31 (5-bit)
);
  integer i;
  logic [$clog2(IN_WIDTH) - 1:0] unsigned_log_out;
  always_comb begin
    for (i = IN_WIDTH - 1; i >= 0; i = i - 1) begin
      if (data_in[i] == 1) begin
        unsigned_log_out = i + 1;
        break;
      end
    end
  end
  assign data_out = {1'b0, unsigned_log_out};
endmodule

module abs #(
    parameter IN_WIDTH = 8  // Parameter for bit-width, can be adjusted
) (
    input  wire [IN_WIDTH-1:0] data_in,  // N-bit input number
    output wire [IN_WIDTH-1:0] data_out  // N-bit output representing 2's complement
);

  // 2's complement calculation
  assign data_out = data_in[IN_WIDTH-1] ? ~data_in + 1 : data_in;
endmodule
