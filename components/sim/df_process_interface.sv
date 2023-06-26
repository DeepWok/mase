`ifndef PROCESS_INTF__SV
`define PROCESS_INTF__SV
`timescale 1ns / 1ps
interface df_process_intf (
    input clock,
    input reset
);
  logic finish;
  logic ap_start;
  logic ap_ready;
  logic ap_done;
  logic ap_continue;
  logic real_start;
  logic pin_stall;
  logic pout_stall;
endinterface
`endif
