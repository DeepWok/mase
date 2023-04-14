`timescale 1 ns / 1 ps
module hs2bram_cast #(
    parameter IN_SIZE = 8,
    parameter IN_WIDTH = 8,
    parameter ADDR_RANGE = 100,
    parameter ADDR_WIDTH = 7
) (
    input rst,
    input clk,

    input logic [ADDR_WIDTH - 1:0] address0,
    input ce0,
    output logic [IN_WIDTH - 1:0] q0,

    input  logic [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                       data_in_valid,
    output                      data_in_ready,

    // Consumer state
    output out_start,
    input  out_ready,
    input  out_done
);

  logic we0;
  logic [IN_WIDTH - 1:0] d0;
  logic ce1;
  logic we1;
  logic [ADDR_WIDTH - 1:0] address1;
  logic [IN_WIDTH - 1:0] d1;
  logic [IN_WIDTH - 1:0] q1;

  // 1-bit wider so IN_DEPTH also fits.
  logic [ADDR_WIDTH-1:0] address_counter;
  // 1-bit wider so IN_DEPTH also fits.
  localparam COUNTER_WIDTH = $clog2(IN_SIZE);
  logic [COUNTER_WIDTH-1:0] data_counter;
  logic [1:0] state;

  // Port 0 is for read only
  assign we0 = 0;

  // data_buff
  logic [IN_WIDTH-1:0] data_buff[IN_SIZE-1:0];
  always_ff @(posedge clk) if (state == 0 && data_in_valid) data_buff <= data_in;

  // address_counter
  always_ff @(posedge clk)
    if (rst) address_counter <= 0;
    else begin
      if (state == 1) address_counter <= address_counter + 1;
      // Ready for the next iteration 
      if (state == 3) address_counter <= 0;
    end

  // data_counter
  always_ff @(posedge clk)
    if (rst) data_counter <= 0;
    else begin
      if (state == 0 && data_in_valid) data_counter <= 0;
      if (state == 1) data_counter <= data_counter + 1;
    end

  assign we1 = (state == 1);
  assign ce1 = (state == 1);
  assign address1 = address_counter;
  assign d1 = data_buff[data_counter];
  assign data_in_ready = (state == 0);

  // The state indicates the state of the bram.
  // 0: waiting for the next input 
  // 1: processing the current input - which might take multiple cycles
  // 2: hold valid data and wait for the consumer to be ready
  // 3: serve as source for the consumer 
  always_ff @(posedge clk) begin
    if (rst) state <= 0;
    else begin
      if (data_in_valid && state == 0) state <= 1;
      if (state == 1) begin
        // Data is full - early exit for imperfect partition
        if (address_counter == ADDR_RANGE - 1) state <= 2;
        // Wait for the next input
        /* verilator lint_off WIDTH */
        else if (data_counter == IN_SIZE - 1) state <= 0;
        /* verilator lint_on WIDTH */
      end
      // Wait until the consummer is ready 
      if (state == 2 && out_ready) state <= 3;
      // Ready for the next iteration 
      if (state == 3 && out_done) state <= 0;
    end
  end

  assign out_start = (state == 2);

  ram_block #(
      .DWIDTH  (IN_WIDTH),
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
