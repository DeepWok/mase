`timescale 1 ns / 1 ps
module unpacked_mx_split2_with_data #(
    parameter DEPTH = 8,
    parameter MAN_WIDTH = 8,
    parameter EXP_WIDTH = 8,
    parameter IN_SIZE = 8
) (
    input clk,
    input rst,
    // Input interface
    input [MAN_WIDTH-1:0] mdata_in[IN_SIZE - 1:0],
    input [EXP_WIDTH-1:0] edata_in,
    input logic data_in_valid,
    output logic data_in_ready,
    // FIFO output interface
    output [MAN_WIDTH-1:0] fifo_mdata_out[IN_SIZE - 1:0],
    output [EXP_WIDTH-1:0] fifo_edata_out,
    output logic fifo_data_out_valid,
    input logic fifo_data_out_ready,
    // Straight output interface
    output [MAN_WIDTH-1:0] straight_mdata_out[IN_SIZE - 1:0],
    output [EXP_WIDTH-1:0] straight_edata_out,
    output logic straight_data_out_valid,
    input logic straight_data_out_ready
);
  // Flatten the input data
  logic [MAN_WIDTH * IN_SIZE + EXP_WIDTH - 1:0] data_in_flatten;
  logic [MAN_WIDTH * IN_SIZE + EXP_WIDTH - 1:0] fifo_data_out_flatten;
  logic [MAN_WIDTH * IN_SIZE + EXP_WIDTH - 1:0] straight_data_out_flatten;
  // Add register slice at the end for FIFO outputs
  logic [MAN_WIDTH-1:0] fifo_mdata_out_unreg[IN_SIZE - 1:0];
  logic [EXP_WIDTH-1:0] fifo_edata_out_unreg;
  logic fifo_data_out_unreg_valid, fifo_data_out_unreg_ready;

  // Input flattening
  for (genvar i = 0; i < IN_SIZE; i++) begin : reshape
    assign data_in_flatten[i*MAN_WIDTH+MAN_WIDTH-1:i*MAN_WIDTH] = mdata_in[i];
  end
  assign data_in_flatten[MAN_WIDTH*IN_SIZE+EXP_WIDTH-1:MAN_WIDTH*IN_SIZE] = edata_in;

  // Split2 instance
  split2_with_data #(
      .DATA_WIDTH(MAN_WIDTH * IN_SIZE + EXP_WIDTH),
      .FIFO_DEPTH(DEPTH)
  ) split2_with_data_i (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_flatten),
      .data_in_valid(data_in_valid),
      .data_in_ready(data_in_ready),
      .fifo_data_out(fifo_data_out_flatten),
      .fifo_data_out_valid(fifo_data_out_unreg_valid),
      .fifo_data_out_ready(fifo_data_out_unreg_ready),
      .straight_data_out(straight_data_out_flatten),
      .straight_data_out_valid(straight_data_out_valid),
      .straight_data_out_ready(straight_data_out_ready)
  );

  // Unflatten FIFO output
  for (genvar i = 0; i < IN_SIZE; i++) begin : unreshape_fifo
    assign fifo_mdata_out_unreg[i] = fifo_data_out_flatten[i*MAN_WIDTH+MAN_WIDTH-1:i*MAN_WIDTH];
  end
  assign fifo_edata_out_unreg = fifo_data_out_flatten[MAN_WIDTH*IN_SIZE+EXP_WIDTH-1:MAN_WIDTH*IN_SIZE];

  mxint_skid_buffer #(
      .DATA_PRECISION_0(MAN_WIDTH),
      .DATA_PRECISION_1(EXP_WIDTH),
      .IN_NUM(IN_SIZE)
  ) fifo_out_reg_slice (
      .clk(clk),
      .rst(rst),
      .mdata_in(fifo_mdata_out_unreg),
      .edata_in(fifo_edata_out_unreg),
      .data_in_valid(fifo_data_out_unreg_valid),
      .data_in_ready(fifo_data_out_unreg_ready),
      .mdata_out(fifo_mdata_out),
      .edata_out(fifo_edata_out),
      .data_out_valid(fifo_data_out_valid),
      .data_out_ready(fifo_data_out_ready)
  );

  // Unflatten straight output
  for (genvar i = 0; i < IN_SIZE; i++) begin : unreshape_straight
    assign straight_mdata_out[i] = straight_data_out_flatten[i*MAN_WIDTH+MAN_WIDTH-1:i*MAN_WIDTH];
  end
  assign straight_edata_out = straight_data_out_flatten[MAN_WIDTH*IN_SIZE+EXP_WIDTH-1:MAN_WIDTH*IN_SIZE];

endmodule
