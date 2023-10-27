`timescale 1 ns / 1 ps
module unpacked_skid_buffer #(
    parameter DATA_WIDTH = 8,
    parameter IN_NUM = 512
) (
    input logic clk,
    input logic rst,
    input logic [DATA_WIDTH-1:0] data_in[IN_NUM - 1:0],
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [DATA_WIDTH-1:0] data_out[IN_NUM - 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  logic [DATA_WIDTH * IN_NUM - 1:0] data_in_flatten;
  logic [DATA_WIDTH * IN_NUM - 1:0] data_out_flatten;
  for (genvar i = 0; i < IN_NUM; i++) begin : reshape
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
  end
  skid_buffer #(
      .DATA_WIDTH(DATA_WIDTH * IN_NUM)
  ) buffer_inst (
      .data_in (data_in_flatten),
      .data_out(data_out_flatten),
      .*
  );
  for (genvar i = 0; i < IN_NUM; i++) begin : unreshape
    assign data_out[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
endmodule
