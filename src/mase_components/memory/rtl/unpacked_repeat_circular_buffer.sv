/*
Module      : repeat_circular_buffer
Description : This module is a repeating circular buffer.
*/

`timescale 1ns / 1ps

module unpacked_repeat_circular_buffer #(
    parameter DATA_WIDTH = 32,
    parameter IN_NUM = 1,
    parameter REPEAT = 2,
    parameter SIZE = 4
) (
    input logic clk,
    input logic rst,

    // Input streaming port
    input  logic [DATA_WIDTH-1:0] in_data [IN_NUM-1:0],
    input  logic                  in_valid,
    output logic                  in_ready,

    // Output streaming port
    output logic [DATA_WIDTH-1:0] out_data [IN_NUM-1:0],
    output logic                  out_valid,
    input  logic                  out_ready
);

  logic [DATA_WIDTH * IN_NUM - 1:0] data_in_flatten;
  logic [DATA_WIDTH * IN_NUM - 1:0] data_out_flatten;

  for (genvar i = 0; i < IN_NUM; i++) begin : reshape
    assign data_in_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH] = in_data[i];
  end

  repeat_circular_buffer #(
      .DATA_WIDTH(DATA_WIDTH * IN_NUM),
      .REPEAT(REPEAT),
      .SIZE(SIZE)
  ) buffer_inst (
      .clk(clk),
      .rst(rst),

      // Input streaming port
      .in_data (data_in_flatten),
      .in_valid(in_valid),
      .in_ready(in_ready),

      // Output streaming port
      .out_data (data_out_flatten),
      .out_valid(out_valid),
      .out_ready(out_ready)
  );

  for (genvar i = 0; i < IN_NUM; i++) begin : unreshape
    assign out_data[i] = data_out_flatten[i*DATA_WIDTH+DATA_WIDTH-1:i*DATA_WIDTH];
  end

endmodule
