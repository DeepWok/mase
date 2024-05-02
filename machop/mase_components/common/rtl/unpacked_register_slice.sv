`timescale 1ns / 1ps
module unpacked_register_slice #(
    parameter DATA_WIDTH = 32,
    parameter IN_SIZE = 16,
    parameter type MYDATA = logic [DATA_WIDTH-1:0]
) (
    input logic clk,
    input logic rst,

    input  MYDATA in_data [IN_SIZE-1:0],
    input  logic  in_valid,
    output logic  in_ready,

    output MYDATA out_data [IN_SIZE-1:0],
    output logic  out_valid,
    input  logic  out_ready
);
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_in_flatten;
  logic [DATA_WIDTH * IN_SIZE - 1 : 0] data_out_flatten;
  for (genvar i = 0; i < IN_SIZE; i++) begin
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = in_data[i];
    assign out_data[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end
  register_slice #(
      .DATA_WIDTH(DATA_WIDTH * IN_SIZE)
  ) register_slice (
      .clk      (clk),
      .rst      (rst),
      .in_valid (in_valid),
      .in_ready (in_ready),
      .in_data  (data_in_flatten),
      .out_valid(out_valid),
      .out_ready(out_ready),
      .out_data (data_out_flatten)
  );
endmodule