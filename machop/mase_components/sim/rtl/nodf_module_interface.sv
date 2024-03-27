`ifndef MODULE_INTF__SV
`define MODULE_INTF__SV
`timescale 1ns / 1ps
interface nodf_module_intf (
    input clock,
    input reset
);
  logic finish;
  logic ap_start;
  logic ap_ready;
  logic ap_done;
  logic ap_continue;
endinterface
`endif
