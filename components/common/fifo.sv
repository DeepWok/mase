`timescale 1ns / 1ps
module fifo #(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8,
    parameter IN_NUM = 8
) (
    input clk,
    input rst,
    input [DATA_WIDTH-1:0] data_in[IN_NUM - 1:0],
    input data_in_valid,
    output logic data_in_ready,
    output logic [DATA_WIDTH-1:0] data_out[IN_NUM - 1:0],
    output logic data_out_valid,
    input data_out_ready
);
  //array reshape
  logic [DATA_WIDTH * IN_NUM - 1:0] data_in_flatten;
  logic [DATA_WIDTH * IN_NUM - 1:0] data_out_flatten;
  for (genvar i = 0; i < IN_NUM; i++) begin : reshape
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
  end

  for (genvar i = 0; i < IN_NUM; i++) begin : unreshape
    assign data_out[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
  logic w_en, r_en;
  logic full, empty;
  reg [$clog2(DEPTH):0] w_ptr, r_ptr;
  reg [DATA_WIDTH * IN_NUM-1:0] fifo_ram[0:DEPTH - 1];
  reg [$clog2(DEPTH):0] count;

  assign w_en = data_in_valid && data_in_ready;
  assign r_en = data_out_valid && data_out_ready;
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
      fifo_ram[w_ptr] <= data_in_flatten;
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
    if (rst) data_out_flatten <= 0;
    /* verilator lint_off WIDTH */
    else data_out_flatten <= fifo_ram[r_ptr];
    /* verilator lint_on WIDTH */

  always @(posedge clk) begin
    if (rst) begin
      r_ptr <= 0;
      data_out_valid <= 0;
    end
    if (empty) begin
      data_out_valid <= 0;
      r_ptr <= r_ptr;
    end else if (r_en) begin
      data_out_valid <= 0;
      /* verilator lint_off WIDTH */
      /* verilator lint_off UNSIGNED */
      if (r_ptr < DEPTH - 1) r_ptr <= r_ptr + 1;
      /* verilator lint_on WIDTH */
      /* verilator lint_on UNSIGNED */
      else
        r_ptr <= 0;
    end else begin
      data_out_valid <= 1;
      r_ptr <= r_ptr;
    end
  end
  /* verilator lint_off WIDTH */
  assign full = (count == DEPTH);
  /* verilator lint_on WIDTH */
  assign empty = (count == 0);
  assign data_in_ready = (!full);
endmodule

