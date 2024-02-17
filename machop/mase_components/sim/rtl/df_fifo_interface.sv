`ifndef FIFO_INTF__SV
`define FIFO_INTF__SV
`timescale 1ns / 1ps
interface df_fifo_intf (
    input clock,
    input reset
);
  logic finish;
  logic rd_en;
  logic wr_en;
  logic fifo_rd_block;
  logic fifo_wr_block;
endinterface
`endif
