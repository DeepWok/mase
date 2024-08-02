`timescale 1ns / 1ps
module unpacked_register_slice #(
    parameter DATA_WIDTH = 32,
    parameter IN_SIZE = 16,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input MYDATA data_in[IN_SIZE-1:0],
    input logic data_in_valid,
    output logic data_in_ready,

    output MYDATA data_out[IN_SIZE-1:0],
    output logic data_out_valid,
    input logic data_out_ready
);
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_in_flatten;
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_out_flatten;
  for (genvar i = 0; i < IN_SIZE; i++) begin
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = data_in[i];
    assign data_out[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
  register_slice #(
      .DATA_WIDTH(DATA_WIDTH * IN_SIZE)
  ) register_slice (
      .clk           (clk),
      .rst           (rst),
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_in       (data_in_flatten),
      .data_out_valid(data_out_valid),
      .data_out_ready(data_out_ready),
      .data_out      (data_out_flatten)
  );
endmodule
