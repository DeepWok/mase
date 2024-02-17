`timescale 1 ns / 1 ps
module bram_cast #(
    parameter IN_WIDTH   = 8,
    parameter ADDR_RANGE = 100,
    parameter ADDR_WIDTH = 7
) (
    input rst,
    input clk,
    input logic [ADDR_WIDTH - 1:0] address0,
    input ce0,
    output logic [IN_WIDTH - 1:0] q0,
    input logic [ADDR_WIDTH - 1:0] address1,
    input ce1,
    input we1,
    input logic [IN_WIDTH - 1:0] d1,
    // Consumer state
    output out_start,
    input out_ready,
    // Producer state
    input in_done,
    output in_ce
);

  logic we0;
  logic [IN_WIDTH - 1:0] d0;
  logic [IN_WIDTH - 1:0] q1;

  assign we0 = 0;

  // The state indicates the state of the bram.
  // 0: serve as sink for the producer
  // 1: hold valid data and wait for the consumer to be ready
  logic [1:0] state;
  always_ff @(posedge clk) begin
    if (rst) state <= 0;
    else begin
      if (in_done && state == 0) state <= 1;
      if (state == 1 && out_ready) state <= 0;
    end
  end

  assign in_ce = (state == 0);
  assign out_start = (state == 1);

  ram_block #(
      .DWIDTH  (IN_WIDTH),
      .AWIDTH  (ADDR_WIDTH),
      .MEM_SIZE(ADDR_RANGE)
  ) data_ram (
      .clk(clk),
      .addr1(address1),
      .ce1(ce1),
      .we1(we1),
      .d1(d1),
      .q1(q1),
      .addr0(address0),
      .ce0(ce0),
      .we0(we0),
      .d0(d0),
      .q0(q0)
  );

endmodule
