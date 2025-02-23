/*
Module      : split2_width_data
Description : This module implements a 1-to-2 streaming interface handshake.
*/

`timescale 1ns / 1ps
module split2_with_data #(
    parameter DATA_WIDTH = -1,
    parameter FIFO_DEPTH = -1
) (
    input logic clk,
    input logic rst,
    input logic [DATA_WIDTH - 1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,

    output logic [DATA_WIDTH - 1:0] fifo_data_out,
    output logic fifo_data_out_valid,
    input logic fifo_data_out_ready,

    output logic [DATA_WIDTH - 1:0] straight_data_out,
    output logic straight_data_out_valid,
    input logic straight_data_out_ready
);
  logic fifo_in_valid, fifo_in_ready;
  split2 #() data_out_n_split_i (
      .data_in_valid (data_in_valid),
      .data_in_ready (data_in_ready),
      .data_out_valid({fifo_in_valid, straight_data_out_valid}),
      .data_out_ready({fifo_in_ready, straight_data_out_ready})
  );
  fifo #(
      .DEPTH(FIFO_DEPTH),
      .DATA_WIDTH(DATA_WIDTH)
  ) ff_inst (
      .clk(clk),
      .rst(rst),
      .in_data(data_in),
      .in_valid(fifo_in_valid),
      .in_ready(fifo_in_ready),
      .out_data(fifo_data_out),
      .out_valid(fifo_data_out_valid),
      .out_ready(fifo_data_out_ready),
      .empty(),
      .full()
  );
  assign straight_data_out = data_in;

endmodule
