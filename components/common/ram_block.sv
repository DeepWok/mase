`timescale 1 ns / 1 ps
module ram_block #(
    parameter DWIDTH   = 8,
    parameter AWIDTH   = 13,
    parameter MEM_SIZE = 3072
) (  /* verilator lint_off UNUSEDSIGNAL */
    input logic [AWIDTH:0] addr1,
    input ce1,
    input logic [DWIDTH-1:0] d1,
    input we1,
    output logic [DWIDTH-1:0] q1,
    input logic [AWIDTH:0] addr0,
    input ce0,
    input logic [DWIDTH-1:0] d0,
    input we0,
    output logic [DWIDTH-1:0] q0,
    input clk
);

  logic [DWIDTH-1:0] ram[0:MEM_SIZE-1];
  logic [DWIDTH-1:0] q0_t0;
  logic [DWIDTH-1:0] q0_t1;
  logic [DWIDTH-1:0] q1_t0;
  logic [DWIDTH-1:0] q1_t1;

  assign q1 = q1_t1;
  assign q0 = q0_t1;

  always_ff @(posedge clk) begin
    if (ce0) q0_t1 <= q0_t0;
    if (ce1) q1_t1 <= q1_t0;
  end
  /* verilator lint_off WIDTH */
  always_ff @(posedge clk) begin
    if (ce0) begin
      if (we0) ram[addr0] <= d0;
      q0_t0 <= ram[addr0];
    end
    if (ce1) begin
      if (we1) ram[addr1] <= d1;
      q1_t0 <= ram[addr1];
    end
  end
  /* verilator lint_on WIDTH */
endmodule
