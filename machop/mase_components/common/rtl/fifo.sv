`timescale 1ns / 1ps
module fifo #(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8
) (
    input clk,
    input rst,

    // Input stream
    input [DATA_WIDTH-1:0] in_data,
    input in_valid,
    output logic in_ready,

    // Output stream
    output logic [DATA_WIDTH-1:0] out_data,
    output logic out_valid,
    input out_ready,

    // Flags
    output logic full,
    output logic empty,
    output logic [$clog2(DEPTH):0] count
);
  logic w_en, r_en;
  reg [$clog2(DEPTH):0] w_ptr, r_ptr;
  reg [DATA_WIDTH-1:0] fifo_ram[0:DEPTH - 1];

  assign w_en = in_valid && in_ready;
  assign r_en = out_valid && out_ready;
  // Set Default values on reset.
  always @(posedge clk) begin
    if (rst) count <= 0;
    else begin
      case ({
        w_en, r_en
      })
        2'b00, 2'b11: count <= count;
        2'b01:
        if (empty) count <= 0;
        else count <= count - 1'b1;
        2'b10:
        if (full) count <= count;
        else count <= count + 1'b1;
      endcase
    end
  end

  // To write data to FIFO
  always @(posedge clk) begin
    if (rst) w_ptr <= 0;
    else if (w_en & !full) begin
      /* verilator lint_off WIDTH */
      fifo_ram[w_ptr] <= in_data;
      /* verilator lint_off UNSIGNED */
      if (w_ptr < DEPTH - 1) w_ptr <= w_ptr + 1;
      /* verilator lint_on WIDTH */
      /* verilator lint_on UNSIGNED */
      else
        w_ptr <= 0;
    end
  end


  // To read data from FIFO
  always @(posedge clk)
    if (rst) out_data <= 0;
    /* verilator lint_off WIDTH */
    else
      out_data <= fifo_ram[r_ptr];
  /* verilator lint_on WIDTH */

  always @(posedge clk) begin
    if (rst) begin
      r_ptr <= 0;
      out_valid <= 0;
    end
    if (empty) begin
      out_valid <= 0;
      r_ptr <= r_ptr;
    end else if (r_en) begin
      out_valid <= 0;
      /* verilator lint_off WIDTH */
      /* verilator lint_off UNSIGNED */
      if (r_ptr < DEPTH - 1) r_ptr <= r_ptr + 1;
      /* verilator lint_on WIDTH */
      /* verilator lint_on UNSIGNED */
      else
        r_ptr <= 0;
    end else begin
      out_valid <= 1;
      r_ptr <= r_ptr;
    end
  end
  /* verilator lint_off WIDTH */
  assign full = (count == DEPTH);
  /* verilator lint_on WIDTH */
  assign empty = (count == 0);
  assign in_ready = (!full);
endmodule
