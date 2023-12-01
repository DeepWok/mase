`timescale 1ns / 1ps
module roller #(
    parameter DATA_WIDTH = 16,
    parameter NUM        = 8,
    parameter ROLL_NUM   = 2
) (
    input logic clk,
    input logic rst,

    input logic [DATA_WIDTH -1:0] data_in[NUM - 1:0],
    input data_in_valid,
    output logic data_in_ready,

    output [DATA_WIDTH - 1:0] data_out[ROLL_NUM - 1:0],
    output data_out_valid,
    input data_out_ready

);
  /* verilator lint_off WIDTH */
  localparam CYCLES = NUM / ROLL_NUM;

  logic [$clog2(CYCLES):0] counter, counter_next;
  logic [DATA_WIDTH-1:0] shift_reg[CYCLES * ROLL_NUM - 1:0];
  logic [DATA_WIDTH-1:0] zeros[CYCLES * ROLL_NUM - 1:0];
  logic [DATA_WIDTH-1:0] shift_reg_next[(CYCLES) * ROLL_NUM - 1:0];

  logic data_in_ready_next;

  logic shift_reg_empty;
  assign shift_reg_empty = counter == 0;
  // Output is valid if we have something stored in buffer or in shift_reg
  assign data_out_valid = !shift_reg_empty;

  // If data_in_ready is true, then we either have CYCLES or 0 in counter depending on if we have  filled in from in port
  // Otherwise we decrease the counter if we will shift the registers.
  assign counter_next = data_in_ready ? (data_in_valid ? CYCLES : 0) : (data_out_ready&&data_out_valid ? counter - 1: counter);

  // We can take in value if we know shift_reg will be free next cycle.
  assign data_in_ready_next = counter_next == 0;
  // shift_reg_next = {shift_reg[NUM - ROLL_NUM : NUM - 1], 0}
  for (genvar i = 0; i < NUM - ROLL_NUM; i++) assign shift_reg_next[i] = shift_reg[i+ROLL_NUM];
  for (genvar i = NUM - ROLL_NUM; i < NUM; i++) begin
    assign shift_reg_next[i] = 0;
    assign data_out[i-(NUM-ROLL_NUM)] = shift_reg[i-(NUM-ROLL_NUM)];
  end
  for (genvar i = 0; i < NUM; i++) assign zeros[i] = 0;


  always_ff @(posedge clk)
    if (rst) begin
      counter <= 0;
      shift_reg <= zeros;
      data_in_ready <= 1'b1;
    end else begin
      counter <= counter_next;
      data_in_ready <= data_in_ready_next;

      // Don't consider the value data_in_valid when feeding into in_shaped
      // to avoid data_in_valid from faning-out to all shift registers.
      if (data_in_ready) shift_reg <= data_in;
      else if (data_out_ready) shift_reg <= shift_reg_next;
    end



endmodule
