`timescale 1 ns / 1 ps
module bram2hs_cast #(
    parameter OUT_SIZE   = 8,
    parameter OUT_WIDTH  = 8,
    parameter ADDR_RANGE = 100,
    parameter ADDR_WIDTH = 7
) (
    input rst,
    input clk,

    input logic [ADDR_WIDTH - 1:0] address0,
    input ce0,
    input we0,
    input logic [OUT_WIDTH - 1:0] d0,

    output logic [OUT_WIDTH-1:0] data_out      [OUT_SIZE-1:0],
    output                       data_out_valid,
    output                       data_out_ready,

    // Producer state
    input in_done,
    output in_ce
);

  logic [OUT_WIDTH - 1:0] q0;
  logic ce1;
  logic we1;
  logic [ADDR_WIDTH - 1:0] address1;
  logic [OUT_WIDTH - 1:0] d1;
  logic [OUT_WIDTH - 1:0] q1;

  // 1-bit wider so OUT_DEPTH also fits.
  logic [ADDR_WIDTH-1:0] address_counter;
  // 1-bit wider so OUT_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(OUT_SIZE);
  logic [COUNTER_WIDTH-1:0] data_counter;
  logic [1:0] state;

  // Port 1 is for read only
  assign we1 = 0;
  assign ce1 = 1;

  // data_buff
  logic [OUT_WIDTH-1:0] data_buff[OUT_SIZE-1:0];
  always_ff @(posedge clk) if (state == 1) data_buff[data_counter] <= q1;
  assign data_out = data_buff;

  // address_counter
  always_ff @(posedge clk)
    if (rst) address_counter <= 0;
    else begin
      if (state == 1) address_counter <= address_counter + 1;
      // Ready for the next iteration 
      if (state == 0) address_counter <= 0;
    end

  // data_counter
  always_ff @(posedge clk)
    if (rst) data_counter <= 0;
    else begin
      if (state == 2) data_counter <= 0;
      if (state == 1) data_counter <= data_counter + 1;
    end

  assign address1 = address_counter;
  assign data_out_valid = (state == 2);
  assign in_ce = (state == 0);

  // The state indicates the state of the bram.
  // 0: waiting for the next input 
  // 1: preparing the output batch - which might take multiple cycles
  // 2: hold valid data and wait for the consumer to be ready
  always_ff @(posedge clk) begin
    if (rst) state <= 0;
    else begin
      if (in_done && state == 0) state <= 1;
      if (state == 1) begin
        // Data is full - early exit for imperfect partition
        /* verilator lint_off WIDTH */
        if (address_counter == ADDR_RANGE - 1 || data_counter == OUT_SIZE - 1) state <= 2;
        /* verilator lint_on WIDTH */
      end
      // Wait until the consummer is ready 
      if (state == 2 && data_out_ready) begin
        if (address_counter == ADDR_RANGE - 1) state <= 0;
        else state <= 1;
      end
    end
  end

  ram_block #(
      .DWIDTH  (OUT_WIDTH),
      .AWIDTH  (ADDR_WIDTH),
      .MEM_SIZE(ADDR_RANGE)
  ) data_ram (
      .clk(clk),
      .addr1(address1),
      .ce1(ce1),
      .we1(we1),
      .q1(q1),
      .d1(d1),
      .addr0(address0),
      .ce0(ce0),
      .we0(we0),
      .d0(d0),
      .q0(q0)
  );

endmodule
