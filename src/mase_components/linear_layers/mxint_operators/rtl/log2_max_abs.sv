`timescale 1ns / 1ps

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
  logic [IN_WIDTH - 1:0] max_abs_value;
  max_abs_tree #(
      .IN_SIZE (IN_SIZE),
      .IN_WIDTH(IN_WIDTH),
  ) max_bas_i (
      .clk,
      .rst,
      .data_in(data_in),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .data_out(max_abs_value),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready)
  );
  log2_value #(
      .IN_WIDTH(IN_WIDTH),
  ) log2_i (
      .data_in (max_abs_value),
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
      if ((data_in > (1 << i))) begin
        unsigned_log_out = i + 1;
        break;
      end
    end
  end
  assign data_out = {1'b0, unsigned_log_out};
endmodule
