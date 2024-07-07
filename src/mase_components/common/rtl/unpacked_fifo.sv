`timescale 1 ns / 1 ps
/* verilator lint_off PINMISSING */
module unpacked_fifo #(
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
  logic [DATA_WIDTH * IN_NUM - 1:0] data_in_flatten;
  logic [DATA_WIDTH * IN_NUM - 1:0] data_out_flatten;
  for (genvar i = 0; i < IN_NUM; i++) begin : reshape
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
  end
  fifo #(
      .DEPTH(DEPTH),
      .DATA_WIDTH(DATA_WIDTH * IN_NUM)
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
  for (genvar i = 0; i < IN_NUM; i++) begin : unreshape
    assign data_out[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
endmodule
