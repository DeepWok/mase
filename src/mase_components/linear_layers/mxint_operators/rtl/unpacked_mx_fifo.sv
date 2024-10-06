`timescale 1 ns / 1 ps
/* verilator lint_off PINMISSING */
module unpacked_mx_fifo #(
    parameter DEPTH = 8,
    parameter MAN_WIDTH = 8,
    parameter EXP_WIDTH = 8,
    parameter IN_SIZE = 8
) (
    input clk,
    input rst,
    input [MAN_WIDTH-1:0] mdata_in[IN_SIZE - 1:0],
    input [EXP_WIDTH-1:0] edata_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output [MAN_WIDTH-1:0] mdata_out[IN_SIZE - 1:0],
    output [EXP_WIDTH-1:0] edata_out,
    output logic data_out_valid,
    input logic data_out_ready
);
  logic [MAN_WIDTH * IN_SIZE + EXP_WIDTH - 1:0] data_in_flatten;
  logic [MAN_WIDTH * IN_SIZE + EXP_WIDTH - 1:0] data_out_flatten;
  for (genvar i = 0; i < IN_SIZE; i++) begin : reshape
    assign data_in_flatten[i*MAN_WIDTH+MAN_WIDTH-1:i*MAN_WIDTH] = mdata_in[i];
  end
  assign data_in_flatten[MAN_WIDTH*IN_SIZE+EXP_WIDTH-1:MAN_WIDTH*IN_SIZE] = edata_in;
  fifo #(
      .DEPTH(DEPTH),
      .DATA_WIDTH(MAN_WIDTH * IN_SIZE + EXP_WIDTH)
  ) ff_inst (
      .clk(clk),
      .rst(rst),
      .in_data(data_in_flatten),
      .in_valid(data_in_valid),
      .in_ready(data_in_ready),
      .out_data(data_out_flatten),
      .out_valid(data_out_valid),
      .out_ready(data_out_ready),
      .empty(),
      .full()
  );
  for (genvar i = 0; i < IN_SIZE; i++) begin : unreshape
    assign mdata_out[i] = data_out_flatten[i*MAN_WIDTH+MAN_WIDTH-1:i*MAN_WIDTH];
  end
  assign edata_out = data_out_flatten[MAN_WIDTH*IN_SIZE+EXP_WIDTH-1:MAN_WIDTH*IN_SIZE];
endmodule
